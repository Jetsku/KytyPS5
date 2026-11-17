#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/queueSubmission.h"
#include "graphics/host_gpu/renderer/commandStream.h"
#include "graphics/host_gpu/renderer/drawEncode.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorEncode.h"
#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"
#include "graphics/host_gpu/renderer/drawPrep/drawRun.h"
#include "graphics/host_gpu/renderer/drawPrep/texDelta.h"
#include "graphics/host_gpu/renderer/hazardTracker.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorSetReuse.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/textureBindingMemo.h"
#include "graphics/host_gpu/renderer/pipeline/textureSetCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "common/profiler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <functional>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
class Buffer;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct PreparedVertexBuffers;
struct PreparedIndexBuffer;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
class CommandRecorder;
class CommandSink;
struct RenderExecutorTestAccess;

namespace DrawPrep {
struct PreparedDraw;
struct BindingPlan;
struct StagePlan;
class Engine;
namespace Segment {
struct EncodedDraw;
} // namespace Segment
} // namespace DrawPrep

class DispatchGuard;
namespace MeshIndirect {
class Converter;
} // namespace MeshIndirect

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	EopWrite,
	EopInterrupt,
	EopWriteBack,
	EopFlip,
	EopWriteBackFlip,
	EopOnlyFlip,
	DispatchIndirect,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// KYTY_ORDERED_APPEND (renderCompute.cpp).
[[nodiscard]] bool OrderedAppendSplitEnabled();

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// KYTY_GPU_INSTANCE_COUNT: nonzero, the instance count is the guest dword the GPU reads there
	// (instance_count is then 1, a placeholder).
	uint64_t         gpu_instance_address       = 0;
};

// GPU-resident draw arguments for vkCmdDraw*Indirect*. Guest DrawIndexedIndirect (20 bytes) and
// DrawIndirect (16 bytes) records are byte-identical to VkDrawIndexedIndirectCommand and
// VkDrawIndirectCommand, so the guest memory is consumed without CPU interpretation.
struct DrawIndirectSource {
	uint64_t args_addr  = 0;
	uint32_t stride     = 0;
	uint32_t max_count  = 1;
	uint64_t count_addr = 0; // Nonzero: draw min(*count_addr, max_count) records.
	bool     indexed    = false;
	// Index state at the packet (INDEX_BASE, INDEX_BUFFER_SIZE in elements, INDEX_TYPE).
	// firstIndex and vertexOffset come from each record.
	uint64_t index_base_addr     = 0;
	uint32_t index_buffer_size   = 0;
	uint32_t index_type_and_size = 0;

	[[nodiscard]] uint32_t RecordSize() const { return indexed ? 20u : 16u; }
	// Bytes covering every record that max_count allows.
	[[nodiscard]] uint64_t ArgsSize() const {
		return static_cast<uint64_t>(max_count - 1u) * stride + RecordSize();
	}
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// KYTY_GPU_INSTANCE_COUNT: as in DrawIndexArgs.
	uint64_t         gpu_instance_address       = 0;
};

// Barrier batcher (KYTY_BARRIER_BATCH, default on; KYTY_BARRIER_BATCH=0 restores the direct
// per-site barriers). Global memory dependencies, buffer barriers and image layout transitions
// requested through CommandBuffer are accumulated and recorded as ONE vkCmdPipelineBarrier2
// directly before the next command that may access memory:
//   - Handle(): every native recording site obtains the handle through it, so the pending
//     batch is recorded (ending an active rendering instance first) before anything the caller
//     records. StateHandle() is the exception for state-only commands (binds, dynamic state,
//     push constants/descriptors), which barriers do not order.
//   - BeginRendering(): the draw path records its draw right after it.
//   - End() / scheduler submission.
// Consecutive requests with nothing recorded between them are merged (the union of their
// scopes is at least as strong as the sequence). A memory request is elided when nothing was
// recorded since the previous flushed barrier and that barrier's memory dependency covers it.
// KYTY_BARRIER_SINK (default on, needs the batcher) additionally keeps a pending memory-only
// batch across a draw that continues the same rendering instance when the draw is proven
// hazard-free with respect to everything since the last full barrier; see
// CommandBuffer::CanSinkPending() for the exact conditions.
// KYTY_DRAW_WRITE_SINK (default on, needs the batcher): the shader-write barrier a draw with guest
// storage-buffer writes requests after itself no longer ends the rendering instance. It stays
// pending while later draws continue the same instance and is recorded at the next flush point
// (another instance, any Handle() command, End()). Guest-visible ordering between draws is what
// the GPU provides: none for shader storage accesses unless the guest synchronizes (PS/VS/CS
// partial flushes and cache actions become Guest requests, which are never sunk this way), so
// draws of one instance may overlap as they do on the hardware. Every non-draw consumer is still
// ordered after the writes. KYTY_DRAW_WRITE_SINK=0 ends the instance after such draws again.
// KYTY_UPLOAD_BATCH (default on, needs the batcher): CPU-dirty buffer uploads are queued with
// CommandBuffer::RequestUploadCopy and recorded at the next flush point as ONE barrier for all
// destination buffers, the copies, and their post-copy barriers merged into the batch being
// flushed (see CommandBuffer::RecordPendingUploads). KYTY_UPLOAD_BATCH=0 records each upload with
// its own EndRendering and barrier pair.
[[nodiscard]] bool BarrierBatchEnabled();
[[nodiscard]] bool BarrierSinkEnabled();
[[nodiscard]] bool DrawWriteSinkEnabled();
// KYTY_DEPTH_FEEDBACK_KEEP (default on, needs the batcher): a depth attachment that draws of the
// current rendering instance also sample keeps one tracked access (attachment + shader read)
// instead of toggling between the two with a barrier per draw, which ended the instance before
// every such draw. Taken only with an exact proof that no attachment write happened since the
// instance began (content serial of RenderExecutor::AcquireRenderTargets, no load clear, no
// depth/stencil write in any of its draws): reads after reads need no ordering. If the instance
// is not continued after all, BeginRendering() records the left-out barrier before the next
// one (CommandBuffer::NoteFeedbackKeep). KYTY_DEPTH_FEEDBACK_KEEP=0 restores the toggles.
[[nodiscard]] bool DepthFeedbackKeepEnabled();
// KYTY_DEPTH_FEEDBACK_LAZY (default on, needs KYTY_DEPTH_FEEDBACK_KEEP): the keep adopts the union
// (attachment + shader read) only when a draw samples the depth attachment or the image already
// holds the union. A draw that does not sample it while it holds the attachment access alone keeps
// that access: the union would avoid no barrier for it, and the change of tracked state made the
// next draw unable to continue the draw run (DrawRun "depth-promotions excluded": about 40% of the
// draws in Astro Bot's clock tower level, against 0.2% in the Sky Garden). The first sampling draw
// of the instance adopts the union instead, with the same proof. KYTY_DEPTH_FEEDBACK_LAZY=0
// restores the promotion by every unwritten draw.
[[nodiscard]] bool DepthFeedbackLazyEnabled();
// Whether the depth-feedback keep moves a depth attachment's tracked access to the union
// (AcquireRenderTargets), given the draw samples it (sampled), its tracked access is the attachment
// access alone (attachment_only) and the keep's proof holds (keep).
[[nodiscard]] constexpr bool DepthFeedbackAdoptsUnion(bool keep, bool lazy, bool sampled,
                                                      bool attachment_only) {
	return keep && !(lazy && !sampled && attachment_only);
}
// KYTY_DEPTH_LAYOUT_STABLE (default on): a depth target the draw does not sample keeps its
// current attachment layout while that layout allows the draw's writes, instead of taking the
// narrowest layout for each draw's write aspects (depth_stable_attachment_layout in
// depthRenderTarget.h). Draws alternating stencil writes no longer transition the image, and so no
// longer end the rendering instance, between them. KYTY_DEPTH_LAYOUT_STABLE=0 restores the
// per-draw layouts.
[[nodiscard]] bool DepthLayoutStableEnabled();
// Descriptor commit switches (default on; =0 restores the previous behaviour):
// KYTY_PUSH_CONSTANT_SHADOW skips push-constant updates identical to the one in effect;
// KYTY_DESCRIPTOR_SET_REUSE reuses descriptor sets (layouts beyond maxPushDescriptors) written
// earlier in the same command buffer with the same contents, and skips rebinding the bound set.
[[nodiscard]] bool PushConstantShadowEnabled();
[[nodiscard]] bool DescriptorSetReuseEnabled();
// KYTY_DESCRIPTOR_SET_REUSE_AUDIT=1 (default off, a measurement): every descriptor-set commit also
// counts whether an earlier commit of its command buffer had exactly its contents (the reuse an
// unbounded cache would find) and whether a 64-slot cache indexed by a full digest of the contents
// would still hold it (DescriptorSetAudit*; compare DescriptorSetsReused).
[[nodiscard]] bool DescriptorSetReuseAuditEnabled();
[[nodiscard]] bool UploadBatchEnabled();
// KYTY_DISPATCH_OVERLAP (live, default on, needs the batcher): consecutive direct guest dispatches
// that the guest does not order overlap on the GPU, as on the console. Every dispatch requests, after
// itself, a dependency from compute shader accesses to everything later (BarrierOrigin::
// ShaderAccess), and a dispatch with storage writes one from everything earlier to it
// (ShaderWriteHazard); between two dispatches they merge into one compute -> compute barrier that
// serialized every dispatch pair. With the switch on, a direct dispatch drops the previous
// dispatch's still-pending ShaderAccess request and skips its own ShaderWriteHazard when nothing at
// all happened between the two (CommandBuffer::TakeDispatchOverlap): no command recorded, no other
// barrier request, layout transition or upload queued, no guest packet other than register writes
// executed (a draw, any synchronization packet, a label, a flush or a CP read breaks the chain),
// and neither dispatch uses emulated LDS (one device buffer for every workgroup). The ShaderAccess
// request of the later dispatch then orders both dispatches (its first scope is every earlier
// compute shader access) before everything after them, so emulator copies, readbacks, draws and
// guest-synchronized dispatches see exactly what they saw before. Only the two dispatches lose
// their ordering against each other, which the guest never requested: its CS_PARTIAL_FLUSH,
// ACQUIRE_MEM, RELEASE_MEM, EVENT_WRITE and WAIT_REG_MEM still become barriers or end the chain.
// Indirect dispatches keep their barriers (their argument barrier serializes them anyway).
// KYTY_DISPATCH_OVERLAP=0 restores the barrier pair between every two dispatches.
[[nodiscard]] bool DispatchOverlapEnabled();
// KYTY_DISPATCH_OVERLAP_UPLOADS (live, default off; needs KYTY_DISPATCH_OVERLAP and the upload
// batcher): buffer uploads queued for a dispatch (pages the CPU wrote) do not end its overlap
// chain. The first such upload in a chain keeps the full pre-copy barrier; later ones only wait
// for earlier copies, so their dispatches overlap the chain's earlier ones (Ghost of Yotei's
// one-workgroup skinning dispatches each come with a few uploads).
[[nodiscard]] bool DispatchOverlapUploadsEnabled();
// KYTY_PM4_BARRIERS=guest (live, default off; the GT7 fork's barriers:guest, usa-ratt
// a67c978c5/31b711db6): the guest's own synchronization orders the GPU work. A full barrier at
// its ACQUIRE_MEMs, passed WAIT_REG_MEMs and the start of each queue's slice (another queue's
// labels it may have waited for), in addition to the partial flushes, cache actions and
// RELEASE_MEMs that already record one; and none before or after guest dispatches nor after draws
// that write buffers. Dispatches with emulated LDS (one device buffer) or BDA writes, and the
// ordered-append and split-dispatch paths, keep their barriers. Any other value: the renderer's
// barriers around every dispatch, as before.
[[nodiscard]] bool GuestSyncBarriersEnabled();
// KYTY_GPU_HAZARD (live, default 0; hazardTracker.h): resource-precise hazards between guest direct
// dispatches. The command buffer keeps the guest address ranges (and, as a conservative second
// check, the host buffer/image byte ranges) every dispatch of the current chain read and wrote; a
// chain starts at a recorded barrier that orders everything earlier before later compute shader
// accesses and ends at the first command recorded that is not a chain dispatch (a draw, a copy, a
// helper, an indirect dispatch, upload copies, image/buffer barriers not of the dispatch's own
// resources). A dispatch with no RAW/WAR/WAW overlap against the chain, in both checks, then skips
// its ShaderWriteHazard request and keeps the chain's pending ShaderAccess (and any guest global
// barrier queued since) pending past itself instead of recording it before itself: they are
// re-queued right after the dispatch, so every later command sees exactly the barrier it saw before.
// Dispatches whose accesses are not all in bindings (address writes, DMA/BDA, GDS, emulated LDS,
// bindless or runtime images, indirect buffer tables, ray tracing) never skip and end the chain.
//   0: off. 1: track and count only (no change in what is recorded). 2: skip when the pending batch
//   is memory dependencies of ShaderAccess/guest origin only (or empty). 3: as 2, and also when
//   image/buffer barriers or upload copies of the dispatch's own resources are pending: those are
//   recorded alone (without the memory dependency) before the dispatch.
//   4: as 3, and a dispatch with untracked accesses (imprecise; all its accesses are compute shader
//   accesses) keeps the chain valid but poisoned instead of ending it: nothing passes it, but the
//   ShaderAccess flush after it (compute -> all commands) restarts a clean chain.
// KYTY_GPU_HAZARD_COUNT=1 (live, default 0): per-flip counters on the live lines (" haz [..]"),
// also at mode 0 for the batch/pass counts.
[[nodiscard]] int64_t GpuHazardMode();
[[nodiscard]] bool    GpuHazardCountEnabled();
// KYTY_INDIRECT_ARGS_RANGES (live, default 0; renderDraw.cpp IndirectArgsBarrierNeeded): an indirect draw
// in an active rendering instance that began without its argument barrier keeps the instance (no
// barrier) when the guest ranges written by this instance's draws (sunk shader-write barriers) do not
// overlap its argument and count ranges. Writers before the instance are ordered by their own
// barriers, recorded before the instance began. Never with KYTY_PM4_BARRIERS=guest.
// Counters (KYTY_GPU_HAZARD_COUNT): isplit (argument barrier needed inside an active instance, as
// without the switch), iskip, iconf (overlap), iimp (an instance write not known), ihost (guest free,
// host ranges overlap: the verify count; the barrier is kept).
enum class HazIndirect : uint32_t { Split, Skip, Conflict, Imprecise, HostMismatch };
void GpuHazardCountIndirect(HazIndirect what);

// KYTY_HOLD_SKIPPED_DRAWS (videoOut.cpp, the GT7 fork's round 133, usa-ratt 65ed4150b): draws
// KYTY_PIPELINE_SKIP left out while their pipelines built in the background, since the last flip
// of the main display.
inline std::atomic<uint32_t> g_frame_pipeline_skips {0};

// KYTY_CUBE_EVERY (renderDraw.cpp): a guest frame starts (the command processor's frame boundary).
void NoteCubeFrame() noexcept;

// Attribution of batched barrier requests (gpuOpProfiler site of the recorded batch).
enum class BarrierOrigin : uint32_t {
	Guest,             // CP EmitGlobalBarrier: guest RELEASE_MEM / EVENT_WRITE flushes
	ShaderAccess,      // after a dispatch: shader accesses -> everything later
	ShaderWrite,       // after a draw with shader buffer writes
	ShaderWriteHazard, // before a dispatch with shader writes: everything earlier -> it
	IndirectArgs,      // shader/transfer writes -> indirect command fetch
	Gds,               // GDS buffer -> shader stages
	Image,             // Image::Transit layout/access transitions
	Upload,            // after queued buffer uploads: copy writes -> everything later
	Count,
};

// The guest packet behind a CP global barrier (CommandProcessor::EmitGlobalBarrier). Names the
// gpuOpProfiler site of the recorded batch (batch.guest.<kind>, batch.mixed.guest.<kind>) and selects
// the KYTY_GUEST_BARRIER_SCOPE mode.
enum class GuestBarrierKind : uint32_t {
	CsPartialFlush,    // EVENT_WRITE CS_PARTIAL_FLUSH
	GfxPartialFlush,   // EVENT_WRITE VS/PS_PARTIAL_FLUSH
	CbDbWriteback,     // EVENT_WRITE CB/DB data writeback-invalidate (0x16, 0x31)
	MetaWriteback,     // EVENT_WRITE DB data / DB, CB metadata writeback-invalidate (0x2a, 0x2c, 0x2e)
	ReleaseMem,        // RELEASE_MEM with an end-of-pipe event or cache actions
	Pm4Sync,           // KYTY_PM4_BARRIERS=guest waits, slice starts
	Count,
};
// KYTY_GUEST_BARRIER_SCOPE (live, default unset = every kind full): what a CP global barrier of each
// kind records. Comma list of <kind>=<mode>, kinds cs, gfx, cbdb, meta, rel, sync, all; modes full
// (ALL_COMMANDS -> ALL_COMMANDS, memory write -> memory read/write, as before), narrow (first scope
// only the stages the packet waits for: cs = compute shader, gfx = graphics shader and attachment
// stages, cbdb/meta = attachment stages; rel and sync stay full) and drop (nothing recorded: the
// renderer's own resource barriers order the work; TheCruZ's fork records none for these events).
// Shorthands: "narrow" = cs,gfx,cbdb,meta narrow; "drop" = cs,gfx,cbdb,meta drop.
enum class GuestBarrierMode : uint32_t { Full = 0, Narrow = 1, Drop = 2 };
[[nodiscard]] GuestBarrierMode GuestBarrierScope(GuestBarrierKind kind);
// Parser of KYTY_GUEST_BARRIER_SCOPE (2 bits per kind); -1 for a malformed value (then all full).
[[nodiscard]] int64_t ParseGuestBarrierScope(const char* value);

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	// Begins (or continues) the rendering instance for the draw recorded next. Pending batched
	// barriers are recorded outside rendering first, unless they can be sunk (see above).
	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;

	// Recorder-aware recording (KYTY_CP_RECORDER, commandRecorder.h). A CommandSink has
	// vk::CommandBuffer's method names; with the recorder off it records natively, exactly as the
	// native handle would, and with the recorder on it encodes packets for the recorder thread.
	//  - Sink(): Handle()'s semantics (pending batched barriers recorded first, a foreign command,
	//    the push-constant shadow forgotten) without the drain Handle() needs in recorder mode.
	//  - StateSink(): StateHandle()'s semantics, for state commands.
	//  - EmissionSink(): StateSink() at an emission safe point (the draw's emission tail, a
	//    dispatch): the caller holds no native handle obtained from Handle()/StateHandle(), so a
	//    direct window opened by one of them (below) closes and commands go to the recorder again.
	// In recorder mode Handle() and StateHandle() drain the recorder (it executes everything
	// encoded so far) and open a direct window: the CP records natively, and every
	// CommandBuffer method and sink records natively too, until the next safe point, Submit or
	// Begin. Native handles must stay function locals, never live across a safe point.
	[[nodiscard]] CommandSink Sink() const;
	[[nodiscard]] CommandSink StateSink() const;
	[[nodiscard]] CommandSink EmissionSink() const;
	//  - HelperSink(): Sink() for the cache/tiler/fault/readback helpers that recorded through
	//    Handle() before (each drained the recorder). KYTY_CP_SINK_ROUTES=0 (live; default 1)
	//    restores that: the recorder is drained and the sink records natively in the direct window.
	[[nodiscard]] CommandSink HelperSink() const;
	// Marks an emission safe point without a sink.
	void BeginEmission() const { CloseDirectWindow(); }
	// The native command buffer being recorded, for identity comparisons only (Image::Transit
	// targets, the dynamic-state shadow). Never record through it.
	[[nodiscard]] vk::CommandBuffer Identity() const noexcept { return m_buffer; }
	// The recorder of this buffer's scheduler (null with KYTY_CP_RECORDER off).
	[[nodiscard]] CommandRecorder* Recorder() const noexcept { return m_recorder; }

	// Queues a global memory dependency (synchronization2 masks).
	void RequestMemoryBarrier(vk::PipelineStageFlags2 src_stages, vk::AccessFlags2 src_access,
	                          vk::PipelineStageFlags2 dst_stages, vk::AccessFlags2 dst_access,
	                          BarrierOrigin origin) const;
	// Profiler attribution only: the Guest request just queued came from `kind`.
	void NoteGuestBarrierKind(GuestBarrierKind kind) const;
	// Queues a buffer memory barrier.
	void RequestBufferBarrier(const vk::BufferMemoryBarrier2& barrier, BarrierOrigin origin) const;
	// Routes image barriers built for `target` into the batch. Returns false (nothing done) when
	// the batcher is disabled or `target` is not this buffer; the caller then records them
	// itself. deferrable: the caller guarantees that it records no memory-accessing command
	// through a previously obtained native handle before the next flush point; otherwise the
	// batch (with these barriers) is recorded immediately.
	[[nodiscard]] bool BatchImageBarriers(std::span<const vk::ImageMemoryBarrier2> barriers,
	                                      vk::CommandBuffer target, bool deferrable) const;
	// Queues a copy of CPU-dirty guest data from a staging `source` into `destination`, recorded
	// at the next flush point (Handle(), BeginRendering(), End(), FlushBarriers()) after one
	// barrier ordering every earlier access of the destination buffers before the queued copies;
	// the copies' writes are then ordered before everything later by buffer barriers merged into
	// the batch flushed with them. A request whose regions overlap a queued copy into the same
	// buffer records the queue first (copies in one batch are unordered). Requires
	// UploadBatchEnabled(); the source must stay valid for this command buffer.
	void RequestUploadCopy(vk::Buffer source, vk::Buffer destination,
	                       std::span<const vk::BufferCopy> regions) const;
	// Records the pending batch now (no-op when empty).
	void FlushBarriers() const;
	// KYTY_DISPATCH_OVERLAP (see DispatchOverlapEnabled()). After a direct guest dispatch and its
	// ShaderAccess request: the chain is open while nothing else happens in this command buffer.
	// lds: the dispatch uses emulated LDS (never overlapped).
	void NoteGuestDispatch(bool lds) const;
	// Ends the open chain (the command processor executed a packet other than a register write or
	// a direct dispatch). Recording any command or flushing the batch ends it too.
	void BreakDispatchOverlap() const noexcept { m_overlap_open = false; }
	// At the next direct dispatch, after its bindings were committed and before its own barrier
	// requests: whether the chain is still open, nothing but the previous dispatch's ShaderAccess
	// request is pending, and neither dispatch uses emulated LDS. Always closes the chain. When it
	// returns true and 	ake is set, the pending request was dropped: the caller skips its
	// ShaderWriteHazard request and keeps its own ShaderAccess request after the dispatch.
	[[nodiscard]] bool TakeDispatchOverlap(bool lds, bool take) const;
	// KYTY_GPU_HAZARD (see GpuHazardMode()). A direct dispatch's accesses: guest ranges (space 0)
	// and host ranges (space = resource handle); precise = every access is in these.
	struct DispatchAccess {
		std::vector<Hazard::Range> guest;
		std::vector<Hazard::Range> host;
		std::vector<uint64_t>      handles; // host buffers and images of the bindings
		bool                       precise = true;
		void Clear() {
			guest.clear();
			host.clear();
			handles.clear();
			precise = true;
		}
	};
	// After TakeDispatchOverlap and before the dispatch's own barrier requests: true when the
	// dispatch must skip its ShaderWriteHazard request (the pending batch was then taken off: its
	// memory dependency is re-queued by NoteHazardDispatch after the dispatch). overlap: the overlap
	// chain already dropped the pending request.
	[[nodiscard]] bool DecideDispatchHazard(const DispatchAccess& access, bool overlap) const;
	// Right after the dispatch command was recorded.
	void NoteHazardDispatch(const DispatchAccess& access) const;
	// Whether anything is pending, and whether queued upload copies are (RequestUploadCopy).
	[[nodiscard]] bool HasPendingBarriers() const noexcept { return !m_pending.Empty(); }
	[[nodiscard]] bool HasPendingUploads() const noexcept { return !m_pending.uploads.empty(); }
	// KYTY_DISPATCH_OVERLAP_UPLOADS: set while a direct dispatch prepares its bindings
	// (renderCompute.cpp); uploads queued then stay pending for the dispatch's own flush while its
	// overlap chain is open (KeepUploadsForDispatch), instead of being recorded at once.
	void SetDispatchPrep(bool on) const noexcept { m_dispatch_prep = on; }
	[[nodiscard]] bool KeepUploadsForDispatch() const;
	// KYTY_DEPTH_FEEDBACK_KEEP: the draw being recorded left out an access-only barrier of its
	// depth attachment that no access inside the active rendering instance needs. Unless the
	// next BeginRendering() continues that instance, it queues `ordering` with the batch it
	// records before the new instance (the ended instance's attachment store and reads must
	// precede the new instance's accesses).
	void NoteFeedbackKeep(const vk::ImageMemoryBarrier2& ordering) const;

	// Brackets the draw recording path from the state commands through the draw itself. The
	// draw must be recorded right after BeginRendering(). safe: the draw writes only its own
	// color/depth attachments (no storage buffer/image writes, atomics, address writes, GDS,
	// fault/LOD counters, indirect arguments) and samples none of its attachments.
	class DrawScope {
	public:
		DrawScope(const CommandBuffer& buffer, bool safe) noexcept: m_buffer(buffer) {
			m_buffer.m_draw_scope = true;
			m_buffer.m_draw_safe  = safe;
		}
		~DrawScope() {
			m_buffer.m_draw_scope = false;
			m_buffer.m_draw_safe  = false;
		}
		DrawScope(const DrawScope&)            = delete;
		DrawScope& operator=(const DrawScope&) = delete;

	private:
		const CommandBuffer& m_buffer;
	};

	// Identifies the active rendering instance; 0 while no instance is active.
	[[nodiscard]] uint64_t ActiveRenderingSerial() const {
		return m_rendering ? m_rendering_serial : 0;
	}
	// The DB_COUNT_CONTROL the active rendering instance counts under (OcclusionCounter::Dump).
	[[nodiscard]] uint32_t OcclusionControl() const { return m_occlusion_control; }
	// Image barriers queued for the next flush point (inspection).
	[[nodiscard]] size_t PendingImageBarriers() const { return m_pending.images.size(); }
	void BindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline);
	// The pipeline last bound through BindPipeline since Begin (null after Begin).
	[[nodiscard]] vk::Pipeline BoundPipeline(vk::PipelineBindPoint point) const {
		return m_shadow.BoundPipeline(point);
	}
	// PushDescriptors result: the update was skipped (identical to the one still in effect), there
	// was nothing comparable (no earlier push in this command buffer for the bind point, another
	// layout, a bound set, or the dedup is off), or the binding list differs; otherwise the index
	// of the first write whose descriptors differ.
	static constexpr int32_t PushAvoided   = DrawEncode::Shadow::PushAvoided;
	static constexpr int32_t PushMissState = DrawEncode::Shadow::PushMissState;
	static constexpr int32_t PushMissShape = DrawEncode::Shadow::PushMissShape;
	// `known_miss`: the caller knows the update differs from every earlier one of this command
	// buffer (a descriptor refers to an upload made for this command, KYTY_PUSH_SHADOW_FRESH_SKIP):
	// recorded without the comparison and without keeping a copy for the next one.
	static constexpr int32_t PushMissFresh = DrawEncode::Shadow::PushMissFresh;
	int32_t PushDescriptors(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t set,
	                        uint32_t count, const vk::WriteDescriptorSet* writes,
	                        bool known_miss = false);
	void InvalidateDescriptors(vk::PipelineBindPoint point);
	// Advances whenever a descriptor command is recorded for the bind point (a push or a set bind),
	// its state is forgotten, or the command buffer begins. Unchanged since a push: the descriptors
	// it pushed are still in effect (descriptor commands go through this class only, see
	// BindDescriptorSet), so a later push with the same layout may update them incrementally
	// (KYTY_DRAW_RUN_PUSH).
	[[nodiscard]] uint64_t DescriptorEpoch(vk::PipelineBindPoint point) const {
		return m_shadow.DescriptorEpoch(point);
	}
	// Binds `set` as set 0 unless it is still the set bound there with this layout
	// (KYTY_DESCRIPTOR_SET_REUSE). Bound sets are disturbed only by another bind or a push
	// descriptor update of set 0 at that bind point, both of which go through this class.
	void BindDescriptorSet(vk::PipelineBindPoint point, vk::PipelineLayout layout,
	                       vk::DescriptorSet set);
	// Records vkCmdPushConstants(layout, stages, 0, size, data) unless the last update recorded
	// in this command buffer was exactly this one and no other update can have been recorded
	// since (KYTY_PUSH_CONSTANT_SHADOW). Other recorders reach the native handle only through
	// Handle(), which forgets the shadow; InvalidatePushConstants() is for a caller that records
	// push constants through StateHandle().
	void PushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t size,
	                   const void* data);
	void InvalidatePushConstants() const {
		SettleEncode();
		m_shadow.InvalidatePushConstants();
	}

	// The encode's view (drawEncode.h): the shadow the state commands above update, the switches
	// its comparisons depend on, and the ring the encoder writes (null while recording natively).
	[[nodiscard]] DrawEncode::Shadow& EncodeShadow() const {
		SettleEncode();
		return m_shadow;
	}
	[[nodiscard]] static DrawEncode::ShadowFlags EncodeFlags();
	[[nodiscard]] const CommandStream::Ring* EncoderRing() const noexcept {
		return Encoding() ? &m_encoder->GetRing() : nullptr;
	}
	// KYTY_ENC_OFFLOAD (encodeOffload.h): the command-stream encoder while encoding (null while
	// recording natively); the shadow without settling (the offload's own use: what the commit
	// thread owns while jobs are pending, or the shadow the jobs update); and the hook that settles
	// pending jobs before the shadow contents are touched (called once, then cleared).
	[[nodiscard]] CommandStream::Encoder* StreamEncoder() const noexcept {
		return Encoding() ? m_encoder : nullptr;
	}
	[[nodiscard]] DrawEncode::Shadow& EncodeShadowUnsettled() const noexcept { return m_shadow; }
	void SetSettleHook(void (*hook)(void* context), void* context) const noexcept {
		m_settle_hook         = hook;
		m_settle_hook_context = context;
	}
	void SettleEncode() const {
		if (m_settle_hook != nullptr) [[unlikely]] {
			const auto hook = m_settle_hook;
			m_settle_hook   = nullptr;
			hook(m_settle_hook_context);
		}
	}

	// Native handle for recording any command. Records pending batched barriers first.
	[[nodiscard]] vk::CommandBuffer Handle() const;
	// Native handle for state-only commands (binds, dynamic state, push constants/descriptors).
	// Never records pending barriers; nothing that accesses memory may be recorded through it
	// before the next flush point (Handle(), BeginRendering(), End()).
	[[nodiscard]] vk::CommandBuffer StateHandle() const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	// Read-only: a committed draw-prep draw points these at its register snapshot.
	[[nodiscard]] const HW::Context&    GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] const HW::UserConfig& GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] const HW::Shader&     GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	struct PendingUpload {
		vk::Buffer source;
		vk::Buffer destination;
		uint32_t   first_region = 0;
		uint32_t   region_count = 0;
	};
	struct PendingBarriers {
		vk::MemoryBarrier2                    memory;
		bool                                  has_memory = false;
		std::vector<vk::ImageMemoryBarrier2>  images;
		std::vector<vk::BufferMemoryBarrier2> buffers;
		// Queued upload copies (RequestUploadCopy), recorded first when the batch is flushed.
		std::vector<PendingUpload>  uploads;
		std::vector<vk::BufferCopy> upload_regions;
		uint32_t                              origins = 0; // bit per BarrierOrigin
		uint32_t                              memory_origins = 0; // origins of the memory requests
		uint32_t                              guest_kinds = 0; // bit per GuestBarrierKind (profiler)

		[[nodiscard]] bool Empty() const {
			return !has_memory && images.empty() && buffers.empty() && uploads.empty();
		}
		void Clear() {
			has_memory = false;
			memory     = vk::MemoryBarrier2 {};
			images.clear();
			buffers.clear();
			uploads.clear();
			upload_regions.clear();
			origins        = 0;
			memory_origins = 0;
			guest_kinds    = 0;
		}
	};
	// Records the queued upload copies behind one barrier and queues their post-copy barriers.
	void RecordPendingUploads() const;
	// Every native command other than the batch itself and the rendering bookkeeping below.
	void NoteForeignCommand() const {
		m_recorded_since_flush = true;
		m_epoch_clean          = false;
		m_overlap_open         = false;
		++m_haz_foreign;
	}
	// FlushBarriers' recording with the KYTY_GPU_HAZARD bookkeeping; side_own: the batch's
	// image/buffer barriers and uploads concern only the next dispatch's own resources.
	void FlushBarriersImpl(bool side_own) const;
	[[nodiscard]] bool CanSinkPending() const;
	[[nodiscard]] bool CanSinkDrawWrites() const;
	void               NoteDrawRecorded() const;
	void               ResetBarrierState() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	mutable uint64_t    m_rendering_serial  = 0;
	mutable uint32_t    m_occlusion_control = 0;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;
	// What the state commands recorded in this buffer left in effect (drawEncode.h): bound
	// pipelines, push-descriptor contents and epochs, push constants.
	mutable DrawEncode::Shadow m_shadow;
	mutable void (*m_settle_hook)(void* context) = nullptr;
	mutable void* m_settle_hook_context          = nullptr;

	// Barrier batcher state (see BarrierBatchEnabled()). Owned by the recording producer.
	mutable PendingBarriers m_pending;
	// Memory dependency of the last recorded batch in this command buffer, for elision.
	mutable vk::MemoryBarrier2 m_last_memory;
	mutable bool               m_last_memory_valid = false;
	// A command was recorded (or may have been) since the last recorded batch.
	mutable bool m_recorded_since_flush = true;
	// Sinking epoch: a full ALL_COMMANDS/MEMORY_WRITE -> ALL_COMMANDS/MEMORY_READ|WRITE batch
	// was recorded in this command buffer, and everything recorded since is state commands, the
	// begin of at most one rendering instance (m_epoch_instance) and safe draws inside it.
	mutable bool     m_epoch_clean    = false;
	mutable uint64_t m_epoch_instance = 0;
	// Nonzero while the batcher or the rendering bookkeeping records through Handle().
	mutable uint32_t m_internal_recording = 0;
	mutable bool     m_draw_scope         = false;
	mutable bool     m_draw_safe          = false;
	// NoteFeedbackKeep() since the last BeginRendering() (at most one depth attachment per draw).
	mutable std::optional<vk::ImageMemoryBarrier2> m_feedback_keep;
	// KYTY_DISPATCH_OVERLAP: NoteGuestDispatch() since anything else happened (TakeDispatchOverlap).
	mutable bool m_overlap_open = false;
	mutable bool m_overlap_lds  = false;
	// KYTY_DISPATCH_OVERLAP_UPLOADS: the open chain already recorded an upload with the full
	// pre-copy barrier; the next flush's uploads take the narrow one (m_upload_narrow).
	mutable bool m_overlap_upload_fenced = false;
	mutable bool m_upload_narrow         = false;
	mutable bool m_dispatch_prep         = false;
	// KYTY_GPU_HAZARD: commands recorded that are not chain dispatches (every NoteForeignCommand,
	// draw and upload copy), the chain, and the memory dependency a skipping dispatch took off the
	// batch (re-queued after it with its origins).
	mutable uint64_t           m_haz_foreign = 0;
	mutable Hazard::Chain      m_haz_chain;
	mutable vk::MemoryBarrier2 m_haz_stash;
	mutable bool               m_haz_stash_valid   = false;
	mutable uint32_t           m_haz_stash_origins = 0;
	mutable uint32_t           m_haz_stash_kinds   = 0;

	// KYTY_CP_RECORDER: the scheduler's recorder and its encoder (null when off), and whether a
	// direct window is open (the CP records natively after a drain; see Sink()).
	CommandRecorder*        m_recorder      = nullptr;
	CommandStream::Encoder* m_encoder       = nullptr;
	mutable bool            m_direct_window = false;
	// KYTY_EMIT_SPLIT: commands were recorded natively (a direct window) or the buffer began since
	// the last resolved draw took this flag: the recorder's shadow may be stale.
	mutable bool m_record_reset = true;

public:
	[[nodiscard]] bool TakeRecordReset() const noexcept {
		const bool reset = m_record_reset;
		m_record_reset   = false;
		return reset;
	}

private:
	[[nodiscard]] bool      Encoding() const noexcept {
		return m_encoder != nullptr && !m_direct_window;
	}
	void OpenDirectWindow(const void* caller) const;
	void CloseDirectWindow() const;

	friend class CommandScheduler;
	friend class CommandSink;
};

// Native command recording for the recorder-aware paths (see CommandBuffer::Sink()). Every method
// records into the native command buffer or encodes an identical packet (commandStream.h); the
// choice is made per call, so a sink obtained before a direct window opened follows it.
class CommandSink {
public:
	explicit CommandSink(const CommandBuffer& owner) noexcept: m_owner(&owner) {}

	[[nodiscard]] vk::CommandBuffer Identity() const noexcept { return m_owner->m_buffer; }

	void beginRendering(const vk::RenderingInfo& info) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->beginRendering(info);
		} else {
			m_owner->m_buffer.beginRendering(info);
		}
	}
	void endRendering() const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->endRendering();
		} else {
			m_owner->m_buffer.endRendering();
		}
	}
	void pipelineBarrier2(const vk::DependencyInfo& info) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->pipelineBarrier2(info);
		} else {
			m_owner->m_buffer.pipelineBarrier2(info);
		}
	}
	void pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst,
	                     vk::DependencyFlags flags, uint32_t memory_count,
	                     const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->pipelineBarrier(src, dst, flags, memory_count, memory, buffer_count,
			                                    buffers, image_count, images);
		} else {
			m_owner->m_buffer.pipelineBarrier(src, dst, flags, memory_count, memory, buffer_count,
			                                  buffers, image_count, images);
		}
	}
	void copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
	                const vk::BufferCopy* regions) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->copyBuffer(source, destination, count, regions);
		} else {
			m_owner->m_buffer.copyBuffer(source, destination, count, regions);
		}
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       uint32_t count, const vk::BufferImageCopy* regions) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->copyBufferToImage(source, destination, layout, count, regions);
		} else {
			m_owner->m_buffer.copyBufferToImage(source, destination, layout, count, regions);
		}
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       uint32_t count, const vk::BufferImageCopy* regions) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->copyImageToBuffer(source, layout, destination, count, regions);
		} else {
			m_owner->m_buffer.copyImageToBuffer(source, layout, destination, count, regions);
		}
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count,
	               const vk::ImageCopy* regions) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->copyImage(source, source_layout, destination, destination_layout,
			                              count, regions);
		} else {
			m_owner->m_buffer.copyImage(source, source_layout, destination, destination_layout,
			                            count, regions);
		}
	}
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                uint32_t value) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->fillBuffer(buffer, offset, size, value);
		} else {
			m_owner->m_buffer.fillBuffer(buffer, offset, size, value);
		}
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color,
	                     uint32_t count, const vk::ImageSubresourceRange* ranges) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->clearColorImage(image, layout, color, count, ranges);
		} else {
			m_owner->m_buffer.clearColorImage(image, layout, color, count, ranges);
		}
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue* value, uint32_t count,
	                            const vk::ImageSubresourceRange* ranges) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->clearDepthStencilImage(image, layout, value, count, ranges);
		} else {
			m_owner->m_buffer.clearDepthStencilImage(image, layout, value, count, ranges);
		}
	}
	void bindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->bindPipeline(point, pipeline);
		} else {
			m_owner->m_buffer.bindPipeline(point, pipeline);
		}
	}
	void bindDescriptorSets(vk::PipelineBindPoint point, vk::PipelineLayout layout,
	                        uint32_t first_set, uint32_t set_count, const vk::DescriptorSet* sets,
	                        uint32_t dynamic_count, const uint32_t* dynamic_offsets) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->bindDescriptorSets(point, layout, first_set, set_count, sets,
			                                       dynamic_count, dynamic_offsets);
		} else {
			m_owner->m_buffer.bindDescriptorSets(point, layout, first_set, set_count, sets,
			                                     dynamic_count, dynamic_offsets);
		}
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t set,
	                          uint32_t count, const vk::WriteDescriptorSet* writes) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->pushDescriptorSetKHR(point, layout, set, count, writes);
		} else {
			m_owner->m_buffer.pushDescriptorSetKHR(point, layout, set, count, writes);
		}
	}
	// vkUpdateDescriptorSets of `set` (every write's dstSet): a device call, made at once natively,
	// or by the recorder thread before the packets encoded after this one (a later bind of the set).
	void updateDescriptorSets(vk::Device device, vk::DescriptorSet set, uint32_t count,
	                          const vk::WriteDescriptorSet* writes) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->updateDescriptorSets(set, count, writes);
		} else {
			device.updateDescriptorSets(count, writes, 0, nullptr);
		}
	}
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
	                   uint32_t size, const void* data) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->pushConstants(layout, stages, offset, size, data);
		} else {
			m_owner->m_buffer.pushConstants(layout, stages, offset, size, data);
		}
	}
	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
	                        const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
		} else {
			m_owner->m_buffer.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
		}
	}
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->bindIndexBuffer(buffer, offset, type);
		} else {
			m_owner->m_buffer.bindIndexBuffer(buffer, offset, type);
		}
	}
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setViewportWithCount(count, viewports);
		} else {
			m_owner->m_buffer.setViewportWithCount(count, viewports);
		}
	}
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setScissorWithCount(count, scissors);
		} else {
			m_owner->m_buffer.setScissorWithCount(count, scissors);
		}
	}
	void setLineWidth(float width) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setLineWidth(width);
		} else {
			m_owner->m_buffer.setLineWidth(width);
		}
	}
	void setBlendConstants(const float constants[4]) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setBlendConstants(constants);
		} else {
			m_owner->m_buffer.setBlendConstants(constants);
		}
	}
	void setDepthTestEnable(vk::Bool32 enable) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setDepthTestEnable(enable);
		} else {
			m_owner->m_buffer.setDepthTestEnable(enable);
		}
	}
	void setDepthWriteEnable(vk::Bool32 enable) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setDepthWriteEnable(enable);
		} else {
			m_owner->m_buffer.setDepthWriteEnable(enable);
		}
	}
	void setDepthCompareOp(vk::CompareOp op) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setDepthCompareOp(op);
		} else {
			m_owner->m_buffer.setDepthCompareOp(op);
		}
	}
	void setDepthBiasEnable(vk::Bool32 enable) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setDepthBiasEnable(enable);
		} else {
			m_owner->m_buffer.setDepthBiasEnable(enable);
		}
	}
	void setDepthBias(float constant, float clamp, float slope) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setDepthBias(constant, clamp, slope);
		} else {
			m_owner->m_buffer.setDepthBias(constant, clamp, slope);
		}
	}
	void setStencilTestEnable(vk::Bool32 enable) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setStencilTestEnable(enable);
		} else {
			m_owner->m_buffer.setStencilTestEnable(enable);
		}
	}
	void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
	                  vk::StencilOp depth_fail, vk::CompareOp compare) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setStencilOp(faces, fail, pass, depth_fail, compare);
		} else {
			m_owner->m_buffer.setStencilOp(faces, fail, pass, depth_fail, compare);
		}
	}
	void setStencilCompareMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setStencilCompareMask(faces, mask);
		} else {
			m_owner->m_buffer.setStencilCompareMask(faces, mask);
		}
	}
	void setStencilWriteMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setStencilWriteMask(faces, mask);
		} else {
			m_owner->m_buffer.setStencilWriteMask(faces, mask);
		}
	}
	void setStencilReference(vk::StencilFaceFlags faces, uint32_t reference) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setStencilReference(faces, reference);
		} else {
			m_owner->m_buffer.setStencilReference(faces, reference);
		}
	}
	void setCullMode(vk::CullModeFlags mode) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setCullMode(mode);
		} else {
			m_owner->m_buffer.setCullMode(mode);
		}
	}
	void setFrontFace(vk::FrontFace face) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setFrontFace(face);
		} else {
			m_owner->m_buffer.setFrontFace(face);
		}
	}
	void setDepthBoundsTestEnable(vk::Bool32 enable) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setDepthBoundsTestEnable(enable);
		} else {
			m_owner->m_buffer.setDepthBoundsTestEnable(enable);
		}
	}
	void setDepthBounds(float min, float max) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setDepthBounds(min, max);
		} else {
			m_owner->m_buffer.setDepthBounds(min, max);
		}
	}
	void setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setColorWriteEnableEXT(count, enables);
		} else {
			m_owner->m_buffer.setColorWriteEnableEXT(count, enables);
		}
	}
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->setAttachmentFeedbackLoopEnableEXT(aspects);
		} else {
			m_owner->m_buffer.setAttachmentFeedbackLoopEnableEXT(aspects);
		}
	}
	void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
	          uint32_t first_instance) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->draw(vertex_count, instance_count, first_vertex, first_instance);
		} else {
			m_owner->m_buffer.draw(vertex_count, instance_count, first_vertex, first_instance);
		}
	}
	void drawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
	                 int32_t vertex_offset, uint32_t first_instance) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->drawIndexed(index_count, instance_count, first_index, vertex_offset,
			                                first_instance);
		} else {
			m_owner->m_buffer.drawIndexed(index_count, instance_count, first_index, vertex_offset,
			                              first_instance);
		}
	}
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->drawMeshTasksEXT(x, y, z);
		} else {
			m_owner->m_buffer.drawMeshTasksEXT(x, y, z);
		}
	}
	void drawMeshTasksIndirectEXT(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                              uint32_t stride) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->drawMeshTasksIndirectEXT(buffer, offset, draw_count, stride);
		} else {
			m_owner->m_buffer.drawMeshTasksIndirectEXT(buffer, offset, draw_count, stride);
		}
	}
	void drawMeshTasksIndirectCountEXT(vk::Buffer buffer, vk::DeviceSize offset,
	                                   vk::Buffer count_buffer, vk::DeviceSize count_offset,
	                                   uint32_t max_count, uint32_t stride) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->drawMeshTasksIndirectCountEXT(buffer, offset, count_buffer,
			                                                  count_offset, max_count, stride);
		} else {
			m_owner->m_buffer.drawMeshTasksIndirectCountEXT(buffer, offset, count_buffer,
			                                                count_offset, max_count, stride);
		}
	}
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                  uint32_t stride) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->drawIndirect(buffer, offset, draw_count, stride);
		} else {
			m_owner->m_buffer.drawIndirect(buffer, offset, draw_count, stride);
		}
	}
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                         uint32_t stride) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->drawIndexedIndirect(buffer, offset, draw_count, stride);
		} else {
			m_owner->m_buffer.drawIndexedIndirect(buffer, offset, draw_count, stride);
		}
	}
	void drawIndirectCount(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,
	                       vk::DeviceSize count_offset, uint32_t max_count, uint32_t stride) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->drawIndirectCount(buffer, offset, count_buffer, count_offset,
			                                      max_count, stride);
		} else {
			m_owner->m_buffer.drawIndirectCount(buffer, offset, count_buffer, count_offset,
			                                    max_count, stride);
		}
	}
	void drawIndexedIndirectCount(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,
	                              vk::DeviceSize count_offset, uint32_t max_count,
	                              uint32_t stride) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->drawIndexedIndirectCount(buffer, offset, count_buffer, count_offset,
			                                             max_count, stride);
		} else {
			m_owner->m_buffer.drawIndexedIndirectCount(buffer, offset, count_buffer, count_offset,
			                                           max_count, stride);
		}
	}
	void dispatch(uint32_t x, uint32_t y, uint32_t z) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->dispatch(x, y, z);
		} else {
			m_owner->m_buffer.dispatch(x, y, z);
		}
	}
	void dispatchBase(uint32_t bx, uint32_t by, uint32_t bz, uint32_t x, uint32_t y,
	                  uint32_t z) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->dispatchBase(bx, by, bz, x, y, z);
		} else {
			m_owner->m_buffer.dispatchBase(bx, by, bz, x, y, z);
		}
	}
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->dispatchIndirect(buffer, offset);
		} else {
			m_owner->m_buffer.dispatchIndirect(buffer, offset);
		}
	}
	void resetQueryPool(vk::QueryPool pool, uint32_t first, uint32_t count) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->resetQueryPool(pool, first, count);
		} else {
			m_owner->m_buffer.resetQueryPool(pool, first, count);
		}
	}
	void beginQuery(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->beginQuery(pool, query, flags);
		} else {
			m_owner->m_buffer.beginQuery(pool, query, flags);
		}
	}
	void endQuery(vk::QueryPool pool, uint32_t query) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->endQuery(pool, query);
		} else {
			m_owner->m_buffer.endQuery(pool, query);
		}
	}
	void copyQueryPoolResults(vk::QueryPool pool, uint32_t first, uint32_t count,
	                          vk::Buffer destination, vk::DeviceSize offset, vk::DeviceSize stride,
	                          vk::QueryResultFlags flags) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->copyQueryPoolResults(pool, first, count, destination, offset,
			                                         stride, flags);
		} else {
			m_owner->m_buffer.copyQueryPoolResults(pool, first, count, destination, offset, stride,
			                                       flags);
		}
	}
	void writeTimestamp2(vk::PipelineStageFlags2 stage, vk::QueryPool pool, uint32_t query) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->writeTimestamp2(stage, pool, query);
		} else {
			m_owner->m_buffer.writeTimestamp2(stage, pool, query);
		}
	}
	// A VK_EXT_debug_utils label before the next command (PassCensus::LabelsEnabled(): --rd or
	// KYTY_TRACE_VK_LABELS=1). An inserted label, not a region: a RenderDoc torii capture with
	// begin/end regions around every draw and dispatch lost the device (rd3b), one without labels
	// did not (rd4).
	void insertDebugLabel(const char* name) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->insertDebugLabel(name);
		} else if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdInsertDebugUtilsLabelEXT != nullptr) {
			vk::DebugUtilsLabelEXT label {};
			label.pLabelName = name;
			m_owner->m_buffer.insertDebugUtilsLabelEXT(label);
		}
	}
	void writeBufferMarkerAMD(vk::PipelineStageFlagBits stage, vk::Buffer buffer, vk::DeviceSize offset,
	                          uint32_t marker) const {
		if (m_owner->Encoding()) {
			m_owner->m_encoder->writeBufferMarkerAMD(stage, buffer, offset, marker);
		} else {
			m_owner->m_buffer.writeBufferMarkerAMD(stage, buffer, offset, marker);
		}
	}
	// KYTY_STREAM_COPY_DEFER: encodes a host copy of guest bytes for the recorder thread (in stream
	// order, before the next Submit). False (nothing encoded) outside recorder encoding (recorder
	// off or a direct window open): the caller then copies itself. Not a command: construct the
	// sink directly (CommandSink(buffer)); Sink() would also flush the barrier batch.
	[[nodiscard]] bool hostCopy(void* destination, uint64_t source, uint64_t size) const {
		if (!m_owner->Encoding()) {
			return false;
		}
		m_owner->m_encoder->hostCopy(destination, source, size);
		return true;
	}

private:
	const CommandBuffer* m_owner;
};

// Graphics dynamic state last recorded by a draw (KYTY_DYNAMIC_STATE_SHADOW). Dynamic state
// persists in a command buffer until it is set again or a pipeline with that state static is
// bound. It is therefore reusable only in the same command buffer while the graphics pipeline
// bound there is still the one the recording draw bound: every renderer pipeline declares the
// same dynamic states (color-write enable only with color attachments, tracked separately), and
// any other graphics pipeline bind (blits) or a Begin changes the bound pipeline.
struct GraphicsDynamicStateShadow {
	static constexpr uint32_t MaxViewports = 16;

	vk::CommandBuffer command  = nullptr;
	vk::Pipeline      pipeline = nullptr;
	bool              valid    = false;
	uint32_t          viewport_count = 0;
	std::array<vk::Viewport, MaxViewports> viewports {};
	std::array<vk::Rect2D, MaxViewports>   scissors {};
	float                line_width = 0.0f;
	std::array<float, 4> blend_constants {};
	vk::Bool32           depth_test_enable  = VK_FALSE;
	vk::Bool32           depth_write_enable = VK_FALSE;
	vk::CompareOp        depth_compare_op   = vk::CompareOp::eNever;
	vk::Bool32           depth_bias_enable  = VK_FALSE;
	bool                 depth_bias_valid   = false;
	std::array<float, 3> depth_bias {};
	vk::Bool32           stencil_test_enable = VK_FALSE;
	bool                 stencil_valid       = false;
	vk::StencilOpState   stencil_front {};
	vk::StencilOpState   stencil_back {};
	bool                 color_write_valid = false;
	uint32_t             color_write_count = 0;
	std::array<vk::Bool32, RENDER_COLOR_ATTACHMENTS_MAX> color_write {};
	bool                 feedback_valid = false;
	vk::ImageAspectFlags feedback;
	// KYTY_PIPELINE_DYNAMIC_STATE (PipelineDynamicRasterStateEnabled).
	vk::CullModeFlags    cull_mode;
	vk::FrontFace        front_face               = vk::FrontFace::eCounterClockwise;
	vk::Bool32           depth_bounds_test_enable = VK_FALSE;
	bool                 depth_bounds_valid       = false;
	std::array<float, 2> depth_bounds {};
};

// KYTY_REBIND_AFTER_JOIN (descriptors.cpp).
[[nodiscard]] bool RebindAfterJoinEnabled();
void               NoteRebindAfterJoin(const char* kind, const char* name);

namespace Pssr {
struct Request;
} // namespace Pssr

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context);
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	// Experimental Pro mode (renderPssr.cpp, upstream 651ef73a0): libPsml's PSSR request as a
	// bilinear upscale of the input T#'s active rectangle into the output T# (graphics queue).
	void ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode,
	                    bool async_compute = false);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute = false);
	// KYTY_INDIRECT_THREADS_GPU (renderCompute.cpp): thread-dimension indirect dispatches go to
	// DispatchIndirect, converted on the GPU, instead of a CPU read of their counts.
	[[nodiscard]] static bool IndirectThreadsOnGpu();
	// KYTY_GPU_INSTANCE_COUNT (renderDraw.cpp): draws inheriting a native indirect draw's instance
	// count read it on the GPU (DrawIndexArgs/DrawAutoArgs::gpu_instance_address).
	[[nodiscard]] static bool GpuInstanceCountEnabled();

	// plan: the committed draw's binding plan for this stage (KYTY_DRAW_PREP_BINDINGS), or null.
	// keep_images (KYTY_DRAW_RUN continuation): the stage's texture and sampler bindings are the
	// previous draw's, kept as they are; only the per-draw data is prepared.
	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared,
	                     DrawPrep::StagePlan* plan = nullptr, bool keep_images = false);
	void                           FindBuffers(PreparedBindings& bindings);
	// Each stage's FindBuffers in order (upstream's form of the call).
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	// keep_images (KYTY_DRAW_RUN continuation): no image transition is recorded; the images are in
	// the states the previous draw left them in (DrawRunImagesUnchanged).
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings, bool keep_images = false);
	// CommitBindings' cache and barrier work, which runs ahead of its encode (descriptors.cpp).
	void CommitBindingsSideEffects(CommandBuffer& buffer, const PipelineCache::Pipeline& pipeline,
	                               std::span<PreparedBindings* const> bindings, bool keep_images);
	// The encode after the side effects, through the builder (descriptorEncode.h).
	void CommitBindingsEncode(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                          const PipelineCache::Pipeline&     pipeline,
	                          std::span<PreparedBindings* const> bindings, bool keep_images);
	template <typename Sink>
	void EncodeCommit(Sink& sink, DrawEncode::Shadow& shadow, CommandBuffer& buffer,
	                  vk::PipelineBindPoint point, const PipelineCache::Pipeline& pipeline,
	                  std::span<PreparedBindings* const> stages, bool partial,
	                  DrawEncode::DescriptorScratch& scratch, DrawEncode::BuiltDescriptors& built);
	[[nodiscard]] DrawEncode::DescriptorEnv MakeDescriptorEnv(std::span<PreparedBindings* const> stages);
	void NoteDescriptorOffsetAudit(const PipelineCache::Pipeline&          pipeline,
	                               std::span<const vk::WriteDescriptorSet> writes);

	// Draw-prep: hands the committed draw's preparation to its program refresh (once).
	[[nodiscard]] DrawPrep::PreparedDraw* TakePreparedDraw() noexcept {
		auto* prepared  = m_prepared_draw;
		m_prepared_draw = nullptr;
		return prepared;
	}
	// KYTY_DRAW_PREP_BINDINGS: the committed draw's binding plan while its preparation awaits
	// validation (null: none), and its activation once DrawPrep::Validate accepted the
	// preparation (RefreshShaders).
	[[nodiscard]] const DrawPrep::BindingPlan* PendingBindingPlan() const noexcept {
		return m_binding_plan;
	}
	void ActivateBindingPlan() noexcept { m_binding_plan_active = m_binding_plan != nullptr; }
	// KYTY_DRAW_RUN: RefreshShaders accepted the committed draw's preparation (the structure key the
	// preparing thread computed describes the draw).
	void NotePreparedValidated() noexcept { m_prepared_validated = true; }
	// KYTY_DRAW_PREP_BINDINGS texturememo: the memo draw-prep threads read hints from (FindHint).
	[[nodiscard]] const TextureBindingMemo& GetTextureMemo() const noexcept { return m_texture_memo; }
	// Bindless engine method (Senaxx): at the start of a guest submission, settle the heap entries
	// the guest wrote since the previous one (GPU thread). Returns at once while no heap is watched.
	void SyncBindlessHeaps();

private:
	// Bindless heaps (Senaxx; descriptors.cpp): register the stage's heaps, patch their regions,
	// and once per frame resolve the keys draws asked for.
	void PrepareBindlessHeaps(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	// KYTY_BINDLESS_REPLACED_RESOLVE: PrepareBindlessHeaps' scratch list.
	std::vector<std::pair<BindlessHeap*, uint32_t>> m_bindless_replaced_keys;
	void PrepareBindlessSamplers(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void ResolveBindlessRequests();
	void SyncWrittenBindlessEntries();
	// Compares keys [first, last) of a heap with the T#s now in guest memory and settles the
	// changed ones again (SyncWrittenBindlessEntries; KYTY_BINDLESS_SYNC_RETIRE's recheck).
	void RecheckBindlessKeys(BindlessHeap& heap, uint32_t first, uint32_t last, uint32_t& view_changes,
	                         uint32_t& resource_changes, uint32_t& resettled);
	bool ResolveBindlessKey(BindlessHeap& heap, uint32_t key);
	// KYTY_BINDLESS_PREFILL: a small heap's keys settled at its first use, before the draw or
	// dispatch reads them (a T# another heap already resolved shares that key's slot).
	void PrefillBindlessHeap(BindlessHeap& heap, uint32_t entries);
	// KYTY_BINDLESS_REFILL: the heap's table hashed once per tick; changed, its new keys are settled.
	void                 RefillBindlessHeap(BindlessHeap& heap, uint32_t entries);
	std::vector<uint8_t> m_bindless_refill_bytes;
	// A settled key whose T# changed: released (its image unpinned when no key refers to it any
	// more) and resolved again; it keeps its old slot while the new texture cannot be resolved
	// yet (IDXTRI, KYTY_BINDLESS_KEEP_OLD).
	bool ResettleBindlessKey(BindlessHeap& heap, uint32_t key);
	// KYTY_BINDLESS_ENGINE=0: the rolling scan of every heap's entries (budget: keys per frame).
	uint32_t RevalidateBindlessKeys(uint32_t budget);
	uint32_t RevalidateBindlessHeap(BindlessHeap& heap, uint32_t budget);
	// KYTY_BINDLESS_RECORD_LAZY: a record pointer heap's requests, growth, retries and revalidation,
	// at its first use in a frame (its call table is the one in use, not one being rebuilt).
	void ProcessRecordHeap(BindlessHeap& heap);
	void ReresolveUnregisteredKeys(BindlessHeap& heap);
	std::vector<uint32_t> m_bindless_reresolve;
	uint32_t RetryBindlessRecords(BindlessHeap* heap, uint64_t frame, uint32_t budget);
	uint32_t ResolveFlaggedKeys(BindlessHeap& heap, uint32_t budget, uint32_t* requested);
	uint32_t m_bindless_budget_left     = 0;
	uint64_t m_bindless_budget_frame    = UINT64_MAX;
	uint64_t m_bindless_snapshot_serial = 0; // snapshots consumed
	// Engine method, E5 (CommitBindings): the stage's bindless images in a read-only layout.
	void MakeBindlessImagesReadable(std::span<BindlessHeap* const> heaps, vk::CommandBuffer target);
	[[nodiscard]] uint64_t BindlessFrame() const;
	// IDXTRI's kept old views: keys whose changed T# could not be resolved yet, retried each frame.
	std::vector<std::pair<BindlessHeap*, uint32_t>> m_bindless_retry;
	// ResolveBindlessKey's last failure may pass (an image being dropped, a full array).
	bool m_bindless_transient_failure = false;
	// KYTY_BINDLESS_RECORD_RETRY: the last ResolveBindlessKey read an unfilled record (null pointer or
	// all-zero T#); such keys are read again after a few frames (m_bindless_record_retry).
	bool m_bindless_record_unfilled = false;
	struct BindlessRecordRetry {
		BindlessHeap* heap    = nullptr;
		uint32_t      key     = 0;
		uint32_t      attempt = 0;
		uint64_t      due     = 0;
	};
	std::vector<BindlessRecordRetry> m_bindless_record_retry;
	uint32_t                         m_bindless_record_attempt = 0; // of the key being retried
	std::unordered_set<uint64_t>     m_bindless_record_queued;
	void QueueBindlessRecordRetry(BindlessHeap& heap, uint32_t key, uint32_t attempt);
	std::unordered_set<uint64_t>               m_bindless_surveyed; // base ^ table offset << 48
	uint64_t                                   m_bindless_frame          = UINT64_MAX;
	uint64_t                                   m_bindless_snapshot_frame = 0;
	// KYTY_BINDLESS_EVICT usage probe (UpdateBindlessUsageProbe): the frame it was armed in
	// (0 = idle) and the first frame the next may start.
	uint64_t                                   m_bindless_probe_frame = 0;
	uint64_t                                   m_bindless_next_probe  = 0;
	std::vector<ImageId>                       m_bindless_unused;
	void UpdateBindlessUsageProbe(uint64_t frame, uint64_t snapshot_frame);
	std::vector<uint32_t>                      m_bindless_requests;
	// KYTY_RUNTIME_IMAGES: T#s (and bindings) shaders requested, taken from the snapshot.
	std::vector<std::pair<std::array<uint32_t, 8>, uint32_t>> m_runtime_image_requests;
	std::vector<std::array<uint32_t, 8>>       m_bindless_window;
	std::vector<std::pair<uint64_t, uint64_t>> m_bindless_written;
	// KYTY_BINDLESS_SYNC_INDEX: heap serials by 1 MiB guest bucket of their key span, how many of
	// BindlessTable::Heaps() (append-only) are indexed, and scratch for the resized heaps.
	std::unordered_map<uint64_t, std::vector<uint32_t>> m_bindless_heap_buckets;
	size_t                                              m_bindless_heap_indexed = 0;
	std::vector<uint32_t>                               m_bindless_heap_resized;
	std::vector<uint32_t>                               m_bindless_heap_candidates;
	std::vector<uint64_t>                               m_bindless_heap_marks;
	std::vector<ImageId>                       m_bindless_changed;
	std::vector<uint32_t>                      m_bindless_srt;
	// KYTY_BINDLESS_REFRESH (Erlou93/KytyPS5-Yotei RefreshBindlessDescriptor): a resolved key's T# is
	// looked up again once per frame, and by every draw while its image is GPU-written; per slot of
	// each image array, the frame and draw it was last looked up and whether its image is GPU-written.
	struct BindlessRefreshState {
		uint64_t frame       = UINT64_MAX;
		uint64_t draw        = 0;
		bool     gpu_written = false;
	};
	void RefreshBindlessHeap(BindlessHeap& heap, uint32_t keys);
	bool RefreshBindlessSlot(BindlessHeap& heap, uint32_t key, BindlessRefreshState& state);
	std::array<std::vector<BindlessRefreshState>, 4> m_bindless_refresh;
	uint64_t                                         m_bindless_refresh_draw = 0;
	// KYTY_BINDLESS_IDLE_RELEASE (Erlou93/KytyPS5-Yotei ReleaseUnusedBindlessSlots): heaps no draw used
	// for that many frames give their slots and image references back (round robin, a budget per frame).
	void   ReleaseIdleBindlessHeaps(uint64_t frame);
	size_t m_bindless_idle_cursor = 0;

	void DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	void DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);
	// Records a GPU-sourced indirect draw. Returns false, before recording anything, when the
	// draw needs CPU-visible counts; the caller then reads the arguments and draws directly.
	[[nodiscard]] bool DrawIndirectNative(uint64_t submit_id, CommandBuffer& buffer,
	                                      const DrawIndirectSource& source);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	// Resolves into `binding` (image, description) in place; its view and mip views are left
	// for RebindImages. Writing in place avoids copying the ~0.5 KB description twice.
	// hash_hint: TextureBindingMemo::Hash of the binding's key, computed by a draw-prep worker;
	// tag_hint: the memo entry tag it found for the key (TextureBindingMemo::FindHint, 0: none).
	void ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	                    const ShaderRecompiler::IR::DescriptorValue& value, TextureBinding& binding,
	                    const uint64_t* hash_hint = nullptr, uint64_t tag_hint = 0);
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	// ResolveTexture's full resolution (no memo lookup); `memo` records the answer.
	void ResolveTextureFull(const ShaderRecompiler::IR::ImageResource& resource,
	                        const ShaderTextureResource& descriptor,
	                        const TextureBindingMemo::Key& memo_key, uint64_t hash, bool memo,
	                        TextureBinding& binding);
	// KYTY_DRAW_SEQUENCE_FAST (textures): PrepareBindings' texture resolution of a stage whose
	// program and T# words are those its bindings were last resolved from, when
	// TextureBindingMemo::TryRepeatResolve proves each ResolveTexture would give the same answer:
	// performs what those resolutions would do (their bookkeeping, BindImage) and returns true.
	[[nodiscard]] bool RepeatStageTextures(const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                                       const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
	                                       PreparedBindings&                               prepared);
	// KYTY_XFRAME_REUSE (drawPrep/xframeReuse.h), RepeatStageTextures for new words: moves the set
	// in `prepared` (the one that fell out of the stage's history; none: null program) into the
	// cross-frame store and, when the store holds a set of `program` with these T# words, makes it
	// the current set (true; not yet revalidated). Otherwise `prepared` holds storage vectors.
	[[nodiscard]] bool TakeStoredTextureSet(const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                                        const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
	                                        PreparedBindings&                               prepared);
	using XFrameTextureSets = TextureSetCache<ShaderRecompiler::IR::CompiledShaderInfo,
	                                          ShaderRecompiler::IR::DescriptorValue, TextureBinding>;
	std::unique_ptr<XFrameTextureSets> m_xframe_sets;
	[[nodiscard]] vk::Sampler NativeSampler(const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                                        uint32_t                                        index,
	                                        const ShaderRecompiler::IR::DescriptorValue&    value,
	                                        bool normalize_unnormalized = false);
	// keep_images (KYTY_DRAW_RUN continuation): the image rebinding and target rediscovery are
	// skipped (the previous draw's bindings are kept).
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors, bool keep_images = false);
	// KYTY_BDA_REACH: the guest index bytes [first, second) the mesh prologue of the next
	// PrepareGraphicsBindings may read (set by ExecutePreparedDraw for direct indexed mesh draws,
	// consumed there); {0, 0}: unknown.
	std::pair<uint64_t, uint64_t> m_bda_mesh_index {};
	bool                          m_bda_mesh_known = false; // the range above is the draw's (may be empty)
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable);
	// KYTY_CONT_KERNEL: PrepareBindings for a continuation stage (keep_images): no GDS, no bindless heaps, no verify.
	void PrepareBindingsContinuation(const ShaderStageRuntime& runtime, PreparedBindings& prepared,
	                                 DrawPrep::StagePlan* plan);
	// AcquireRenderTargets' depth decisions (no effects): the aspects the draw samples of its depth
	// attachment and the feedback aspects, the attachment layout and access, and the
	// KYTY_DEPTH_FEEDBACK_KEEP decision (keep; adopt: the keep notes its ordering and adopts the
	// union scope). views_ok false: a sampled view of the image is not one of its views.
	struct DepthAccessDecision {
		vk::ImageAspectFlags sampled;
		vk::ImageAspectFlags feedback;
		vk::ImageLayout      layout = vk::ImageLayout::eUndefined;
		vk::AccessFlags2     access;
		vk::AccessFlags2     sampled_access;
		bool                 keep           = false;
		bool                 adopt          = false;
		bool                 stable_avoided = false;
		bool                 views_ok       = true;
		Profiler::FrameEvent miss           = Profiler::FrameEvent::DepthFeedbackKeepMissWrite;
	};
	[[nodiscard]] DepthAccessDecision DecideDepthAccess(const CommandBuffer& buffer, const RenderDepthInfo& depth,
	                                                    const Image& image,
	                                                    std::span<PreparedBindings* const> stages) const;
	[[nodiscard]] static vk::ImageMemoryBarrier2 FeedbackKeepOrdering(const Image& image, vk::ImageLayout layout,
	                                                                  vk::AccessFlags2 sampled);
	// written: DrawScissorUnion of the draw (nullptr: the whole colour targets may be written).
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {},
	                                               const vk::Rect2D* written = nullptr);
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	// A fast-clear-eliminate / FMASK or DCC decompress draw (CB_COLOR_CONTROL.MODE): consumed
	// without drawing. KYTY_CB_METADATA_MATERIALIZE=1 first materializes the target's tracked DCC
	// fast clear (renderDraw.cpp).
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer,
	                                                        uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	// `site` names the caller's stage and table kind; it only selects the entry checked first.
	// *fresh (optional): set when the returned range was allocated by this call (no earlier upload
	// of the current recording had these bytes), i.e. no descriptor written before refers to it.
	[[nodiscard]] vk::DescriptorBufferInfo UploadShaderData(std::span<const uint32_t> data,
	                                                        uint32_t site, bool* fresh = nullptr);
	// Descriptor set for a layout beyond maxPushDescriptors: reused or written, then bound.
	// fresh: the writes refer to an upload made for this command (KYTY_SET_REUSE_FRESH).
	void CommitDescriptorSet(CommandBuffer& buffer, vk::PipelineBindPoint point,
	                         const PipelineCache::Pipeline& pipeline, bool fresh);
	// Records the metadata clears a dispatch's buffer writes establish; whether it recorded one
	// (renderCompute.cpp, KYTY_COMPUTE_META_CLEAR_DISPATCH).
	// keep_dispatch (optional): set when a cleared HTile is also a colour target's CMASK
	// (KYTY_CMASK_MSAA), whose dispatch must still run.
	bool                      UpdateComputeMetaClears(const ShaderComputeInputInfo& input,
	                                                  const CommandBuffer&          buffer,
	                                                  bool* keep_dispatch = nullptr);
	[[nodiscard]] bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                              CommandBuffer& command, uint32_t group_x,
	                                              uint32_t group_y, uint32_t group_z, uint32_t mode,
	                                              uint64_t code_address = 0);

	// KYTY_LDS_DEVICE_BUFFER: binds the LDS regions of a compute program whose LDS lives in a device
	// buffer (a SharedMemory binding): one region per workgroup of `groups` (null: an indirect
	// dispatch, whose count the CPU does not know; it gets the buffer as it is, at least
	// LdsIndirectBytes). Returns false when the program has no such binding.
	bool BindComputeLds(const ShaderComputeInputInfo& input, PreparedBindings& bindings,
	                    const uint32_t* groups);
	// KYTY_SHADER_CALLS_SPLIT (callSplitDispatch.cpp, shader/recompiler/CallSplit.h): a dispatch of
	// a split kernel as its chain of dispatches. `pristine`: the input info before the kernel's
	// program lookup; `run`: PipelineCache::TakeCallSplitRun.
	void DispatchCallSplit(CommandBuffer& buffer, const ShaderComputeInputInfo& pristine,
	                       const std::shared_ptr<const void>& run, const HW::ComputeShaderInfo& cs_regs,
	                       const HW::ShaderRegisters& sh_regs, uint32_t groups_x, uint32_t groups_y,
	                       uint32_t groups_z);
	struct CallSplitState;
	std::shared_ptr<CallSplitState> m_call_split; // shared_ptr: destroyed where it is complete
	// The PSSR pass's pipeline (renderPssr.cpp), created by the first request.
	struct PssrState;
	std::shared_ptr<PssrState> m_pssr;
	// KYTY_CALL_PARTITIONS (callPartitionDispatch.cpp, shader/recompiler/ShaderCalls.h): a dispatch of
	// a partitioned program runs once per partition. A step is the partition one run of
	// DispatchDirectOne / DispatchIndirectOne dispatches (its code, report set and target).
	struct PartitionStep {
		const void* code   = nullptr; // a CallSplit::Code
		uint64_t    report = 0;
		uint64_t    target = 0;
	};
	void DispatchDirectOne(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                       uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode,
	                       bool async_compute, const PartitionStep* step,
	                       std::shared_ptr<const void>* partition_run);
	void DispatchIndirectOne(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                         uint32_t mode, bool async_compute, const PartitionStep* step,
	                         std::shared_ptr<const void>* partition_run);
	void DispatchPartitions(CommandBuffer& buffer, const std::shared_ptr<const void>& run,
	                        const std::function<void(const PartitionStep&)>& dispatch);
	struct CallPartitionState;
	std::shared_ptr<CallPartitionState> m_call_partition;

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	// Reused by every draw (all draws hold the render mutex); DrawRenderState::Reset restores
	// only what the previous draw changed. Owns the draw's per-stage program preparation.
	std::unique_ptr<DrawRenderState>      m_draw_state;
	// Program preparation output of the current dispatch; its stage runtime points here.
	PipelineCache::StagePrep              m_compute_prep;
	// KYTY_LDS_DEVICE_BUFFER: device-local buffer of compute LDS regions (BindComputeLds), grown
	// on demand; a replaced one is released once the tick that used it has completed.
	std::unique_ptr<Buffer>               m_compute_lds;
	std::vector<ImageId>                  m_bound_images;
	// The descriptor writes the last CommitBindings built (drawEncode.h).
	DrawEncode::DescriptorScratch m_descriptor_scratch;
	DrawEncode::BuiltDescriptors  m_built_descriptors;
	// KYTY_EMIT_SPLIT (renderDraw.cpp, descriptors.cpp): the partial write list of the current run's
	// continuations (a function of the programs, as KYTY_CONT_EMIT's skeleton), and the counters.
public:
	struct EmitSplitTemplate;
	struct EmitSplitState;

private:
	EmitSplitState* m_emit_split = nullptr;
	[[nodiscard]] EmitSplitState& EmitSplit();
	// Whether this continuation can be committed as one resolved draw (fills the template for it);
	// counts the reason when not.
	[[nodiscard]] bool EmitSplitPrepare(const CommandBuffer& buffer, const PipelineCache::Pipeline& pipeline,
	                                    std::span<PreparedBindings* const> stages);
	// The resolved draw of the draw being committed: descriptors from the template, push data, and
	// the caller's vertex/index/draw fields; written to the buffer's stream encoder.
	void EmitSplitWrite(CommandBuffer& buffer, const PipelineCache::Pipeline& pipeline,
	                    std::span<PreparedBindings* const> stages, CommandStream::ResolvedDrawPacket& packet,
	                    const CommandStream::ResolvedDrawArrays& arrays_in, bool verify_only);
	void EmitSplitReport();
	// KYTY_ENC_OFFLOAD (renderDraw.cpp, encodeOffload.h).
	struct OffloadJob;
	struct OffloadState;
	OffloadState* m_offload = nullptr;
	OffloadJob*   OffloadAcquire(CommandBuffer& buffer);
	void          OffloadPost(OffloadJob* job, CommandBuffer& buffer);
	OffloadJob*   OffloadBindings(CommandBuffer& buffer, const PipelineCache::Pipeline& pipeline,
	                              std::span<PreparedBindings* const> stages, bool keep_images, bool mesh_active,
	                              bool run_verify_push, const PreparedVertexBuffers& vertex,
	                              const PreparedIndexBuffer& index);
	void          OffloadVerifyPart(OffloadJob* job, CommandBuffer& buffer, const CommandStream::Ring* ring,
	                                uint64_t real_begin);
	void          OffloadSettle(uint32_t reason);
	void          OffloadReport();
	static void   OffloadSettleNow(OffloadState& state, uint64_t& reason);
	struct ShaderUploadEntry {
		uint64_t tick = 0;
		uint64_t hash = 0;
		vk::DescriptorBufferInfo allocation;
		std::vector<uint32_t> words;
	};
	std::array<ShaderUploadEntry, 64> m_shader_uploads;
	// KYTY_UPLOAD_DEDUP: same-tick content dedup of shader-data/flattened-SRT uploads. By default
	// each upload site keeps its last upload (m_upload_site_last); KYTY_UPLOAD_DEDUP_TABLE=1 also
	// looks every upload up in a hashed content table, with the entry each site filled or matched
	// last checked before hashing (the U54 behaviour).
	std::array<ShaderUploadEntry, 32>  m_upload_site_last;
	std::array<ShaderUploadEntry, 256> m_upload_dedup;
	std::array<uint32_t, 32>           m_upload_last_slot {};
	// KYTY_DESCRIPTOR_SET_REUSE: sets written earlier in the current command buffer.
	DescriptorSetReuse m_descriptor_set_reuse;
	// KYTY_SET_REUSE_FRESH=verify: reuse hits of sets that refer to a fresh upload (must stay 0).
	uint64_t m_set_reuse_fresh_mismatches = 0;
	// Rendering instance begun right after the last indirect-argument barrier. Buffer writes
	// are recorded outside rendering, or end it (shader-write barrier), so while this instance
	// stays active the barrier still covers every argument write.
	uint64_t m_indirect_barrier_rendering = 0;
	// KYTY_INDIRECT_ARGS_COVERED: the rendering instance in which a draw last queued its shader-write
	// barrier without ending the instance (KYTY_DRAW_WRITE_SINK); 0 if none.
	uint64_t m_indirect_write_rendering = 0;
	// KYTY_INDIRECT_ARGS_RANGES: the guest ranges the draws of rendering instance
	// m_haz_write_instance wrote with sunk shader-write barriers (imprecise: some write not known).
	uint64_t                   m_haz_write_instance  = 0;
	bool                       m_haz_write_imprecise = false;
	std::vector<Hazard::Range> m_haz_instance_writes;
	std::vector<Hazard::Range> m_haz_instance_host_writes; // (VkBuffer, host offsets), the verify check
	void NoteInstanceWrites(uint64_t instance, std::span<PreparedBindings* const> stages);
	// Whether an indirect draw recorded in the active rendering instance needs its own argument
	// barrier (which ends the instance).
	// source (KYTY_INDIRECT_ARGS_RANGES): the draw's argument/count ranges; count: the call that
	// records the barrier (counters).
	// host: the argument/count buffers' host ranges (verify: a host overlap with the instance's written
	// host ranges keeps the barrier and counts ihost).
	[[nodiscard]] bool IndirectArgsBarrierNeeded(const CommandBuffer& buffer, const DrawIndirectSource* source = nullptr,
	                                             bool count = false, std::span<const Hazard::Range> host = {}) const;
	// KYTY_DEPTH_FEEDBACK_KEEP: the depth attachment of the draw being recorded and whether the
	// draw writes it, applied to Image::feedback_instance/_serial once BeginRendering() has
	// chosen the draw's rendering instance (NoteDepthFeedback).
	struct DepthFeedbackNote {
		ImageId id {};
		bool    writes = false;
		bool    valid  = false;
	};
	DepthFeedbackNote m_depth_feedback;
	void              NoteDepthFeedback(const CommandBuffer& buffer);
	// The ImageResource fields BuildTextureDescription reads. The shader-specific identity
	// (source slot, first use pc, indirect-image tables) is excluded, so one texture bound from
	// different shaders shares an entry.
	struct TextureDescriptionKey {
		ShaderRecompiler::IR::ImageResourceClass resource_class {};
		Prospero::TextureNumericClass            numeric_class {};
		ShaderRecompiler::Decoder::ImageDimension dimension {};
		ShaderRecompiler::IR::ImageMipMode       mip_mode {};
		uint32_t                                 mip_count         = 0;
		Prospero::BufferFormat                   conversion_format {};
		uint32_t                                 shader_swizzle    = 0;
		bool                                     read              = false;
		bool                                     written           = false;
		bool                                     atomic            = false;
		bool                                     depth_compare     = false;
		bool                                     cube              = false;
		bool                                     r128              = false;
		bool                                     atomic64          = false;
		uint32_t                                 max_array_index   = UINT32_MAX; // KYTY_TEXTURE_LAYER_BOUND

		bool operator==(const TextureDescriptionKey&) const = default;
	};
	struct TextureDescriptionEntry {
		bool valid = false;
		TextureDescriptionKey key;
		std::array<uint32_t, 8> words {};
		TextureCache::ImageDesc desc;
	};
	std::array<TextureDescriptionEntry, 4096> m_texture_descriptions;
	// KYTY_TEXTURE_BINDING_MEMO: (T# dwords, resource) -> resolved image, description and view.
	TextureBindingMemo m_texture_memo;
	// KYTY_DRAW_SEQUENCE_VERIFY: the views a repeated stage claimed (RebindImages).
	std::vector<vk::ImageView> m_claimed_views;
	// KYTY_DRAW_SEQUENCE_FAST outcomes, always counted (the DrawSequence* frame events need a
	// connected profiler); read by tests.
	struct DrawSequenceTotals {
		uint64_t target_repeats    = 0;
		uint64_t target_misses     = 0;
		uint64_t target_records    = 0;
		uint64_t texture_repeats   = 0;
		uint64_t texture_misses    = 0;
		uint64_t texture_history_hits = 0;
		uint64_t view_repeats      = 0;
		uint64_t verify_checks     = 0;
		uint64_t verify_mismatches = 0;
		uint64_t verify_races      = 0;
	};
	DrawSequenceTotals m_draw_sequence_totals;
	// KYTY_SAMPLER_MEMO: final sampler dwords -> native sampler. The sampler cache never evicts,
	// so a remembered handle stays the one GetSampler returns for those dwords. 64 sets of
	// SamplerMemoWays entries, most recently used first (NativeSampler).
	struct SamplerMemoEntry {
		std::array<uint32_t, 4> fields {};
		bool                    integer_border = false;
		vk::Sampler             sampler        = nullptr;
	};
	static constexpr uint32_t SamplerMemoWays = 4;
	std::array<SamplerMemoEntry, 64 * SamplerMemoWays> m_sampler_memo {};
	// KYTY_TARGET_DESC_MEMO: target descriptions are pure functions of the target registers
	// (and constant device format support). Keyed on the exact register bytes; FindImage and
	// everything after it still run for every draw, except that a lookup of the unchanged
	// description repeats the last one when that is proven (`lookup`, KYTY_DRAW_SEQUENCE_FAST).
	struct ColorTargetDescMemo {
		bool                            valid = false;
		HW::RenderTarget                registers {};
		uint32_t                        mask         = 0;
		uint32_t                        slice_offset = 0;
		TextureCache::ImageDesc         desc;
		uint32_t                        guest_mip_level   = 0;
		uint32_t                        guest_array_layer = 0;
		Prospero::ColorComponentMapping export_mapping;
		TextureCache::RepeatLookup      lookup;
	};
	std::array<ColorTargetDescMemo, RENDER_COLOR_ATTACHMENTS_MAX> m_color_target_memo {};
	struct DepthTargetDescMemo {
		bool                       valid = false;
		HW::DepthRenderTarget      registers {};
		TextureCache::ImageDesc    desc;
		TextureCache::RepeatLookup lookup;
	};
	DepthTargetDescMemo        m_depth_target_memo {};
	// A target lookup of `desc` (equal to the description `record` was made for, if it is valid):
	// the proven repeat of the recorded lookup (TextureCache::TryRepeatLookup), otherwise
	// FindImage, recording into `record`. Null `record`: FindImage (colorRenderTarget.cpp).
	[[nodiscard]] ImageId FindTargetImage(TextureCache::ImageDesc& desc, bool exact_format,
	                                      TextureCache::RepeatLookup* record);
	GraphicsDynamicStateShadow m_dynamic_state {};
	// Draw-prep (drawPrep.h): the preparation of the draw the engine is committing, taken by
	// RefreshShaders in place of GetGraphicsPrograms when its certificate holds. Null otherwise.
	DrawPrep::PreparedDraw* m_prepared_draw = nullptr;
	// KYTY_DRAW_PREP_DISPATCH: the preparation of the dispatch the engine is executing, taken by
	// DispatchDirect in place of GetComputeProgram when its certificate holds. Null otherwise.
	DrawPrep::PreparedDraw* m_prepared_dispatch = nullptr;
	// KYTY_DRAW_PREP_BINDINGS: the binding plan of the draw the engine is committing (null: none),
	// and whether it applies: set by RefreshShaders once DrawPrep::Validate accepted the same
	// slot's preparation (the plan was derived from it), cleared by the engine after the draw.
	DrawPrep::BindingPlan* m_binding_plan        = nullptr;
	bool                   m_binding_plan_active = false;
	// KYTY_DRAW_PREP_BINDINGS_VERIFY (P4b-2): the views a predicted view run claims (RebindImages).
	std::vector<vk::ImageView> m_claimed_run_views;
	// KYTY_NATIVE_INDIRECT_MESH (meshIndirect.h): created by the first native indirect mesh draw.
	std::unique_ptr<MeshIndirect::Converter> m_mesh_indirect;
	std::unique_ptr<DispatchGuard>           m_dispatch_guard;

	// KYTY_DRAW_RUN (drawPrep/drawRun.h): the run the last recorded draw can seed and the committed
	// draw's part in it (GPU thread; draws hold the render mutex).
	struct DrawRunImage {
		ImageId                 id {};
		vk::Image               image = nullptr;
		vk::PipelineStageFlags2 stage;
		vk::AccessFlags2        access;
		vk::ImageLayout         layout         = vk::ImageLayout::eUndefined;
		uint64_t                serial         = 0;
		uint64_t                address        = 0; // guest range (info.data)
		uint64_t                size           = 0;
		uint32_t                resident_first = 0;
		bool                    registered     = false;
		bool                    single_state   = false; // no per-subresource states
		bool                    texture        = false; // a sampled binding (else an attachment)
	};
	struct DrawRunRecord {
		bool              valid            = false;
		uint64_t          key              = 0;
		uint64_t          targets_key      = 0; // KYTY_DRAW_RUN_TARGETS
		uint64_t          swap_key         = 0; // KYTY_DRAW_RUN_PIPESWAP
		uint64_t          texfree_key      = 0; // KYTY_DRAW_RUN_TEXDELTA
		uint64_t          activity         = 0;
		vk::CommandBuffer command          = nullptr;
		uint64_t          tick             = 0;
		uint64_t          rendering_serial = 0;
		uint32_t          slice_offset     = 0;
		const void*       vertex_program   = nullptr;
		const void*       pixel_program    = nullptr;
		bool              ps_active        = false;
		uint32_t          color_count      = 0;
		uint32_t          color_slots      = 0;
		RenderState       rendering;
		// The textures of the recorded stages, then the attachments (from attachments_begin), as the
		// draw left them.
		std::vector<DrawRunImage> images;
		uint32_t                  attachments_begin = 0;
		// KYTY_DRAW_RUN_ACQUIRE and verify mode: the resolved targets (depth: one entry) and the
		// scissor union (KYTY_ALIAS_BYTES claims) the attachments were acquired for.
		std::vector<RenderColorInfo> colors;
		std::vector<RenderDepthInfo> depth;
		vk::Rect2D                   written {};
		bool                         bounded = false;
		// The record stored colors/depth/written (acquisition reuse) and the verify copies (the
		// flags are live switches: a later draw must not read what an earlier record left).
		bool acquire_valid = false;
		bool verify_valid  = false;
		// KYTY_DRAW_RUN_PUSH: the graphics descriptor epoch after the draw's push, and its layout.
		uint64_t           push_epoch  = 0;
		vk::PipelineLayout push_layout = nullptr;
		bool               push_valid  = false; // pushed (not a descriptor set)
		// KYTY_DRAW_RUN_WRITES: the recorded programs write buffers (recorded with the switch on).
		bool writes = false;
		// KYTY_DRAW_RUN_CONT_ELIGIBLE: the KYTY_DRAW_RUN_WRITES value the record was made under.
		bool record_writes_ok = false;
		// KYTY_DRAW_RUN_RECALL: the texture bindings each stage committed (the vertex stage's first)
		// and its samplers: a recall takes each stage's set back from its texture history only when it
		// still holds exactly these.
		struct RecallTexture {
			ImageId         id {};
			vk::ImageView   view   = nullptr;
			vk::ImageLayout layout = vk::ImageLayout::eUndefined;
			uint64_t        mips   = 0; // hash of the mip views
		};
		std::vector<RecallTexture> recall_textures;
		std::vector<vk::Sampler>   recall_samplers;
		uint32_t                   recall_vertex_textures = 0;
		uint32_t                   recall_vertex_samplers = 0;
		bool                       recall_valid           = false;
		// KYTY_DRAW_RUN_IMAGE_EPOCH: the image-state epoch at which the marks were last proven
		// current (0: never); refreshed by a check that passes.
		mutable uint64_t image_epoch = 0;
		// Verify mode: what the normal path must reproduce for a continuation.
		std::vector<TextureBinding>          textures;
		std::vector<vk::Sampler>             samplers;
		std::vector<vk::DescriptorImageInfo> pushed_images; // image and sampler descriptors
	};
	// DrawIndex/DrawAuto under the render mutex: the previous run's validity is taken for this draw,
	// and a draw the draw-prep engine does not commit counts as other command-processor work.
	void                       BeginDrawRun();
	[[nodiscard]] bool DrawRunCommandUnchanged(const CommandBuffer& buffer) const;
	[[nodiscard]] DrawRunImage MakeDrawRunImage(ImageId id, bool texture) const;
	// attachments_only: the textures of the record are not checked (KYTY_DRAW_RUN_ACQUIRE).
	// DrawRunImagesChange: 0 when unchanged, else the first difference (verify-mode detail).
	// attachment_serials: false compares the content serials of the textures only (a recall).
	[[nodiscard]] uint32_t DrawRunImagesChange(bool compare_serials,
	                                           bool attachments_only = false, bool log_change = false,
	                                           bool attachment_serials = true) const;
	// DrawRunImagesChange over any marks (the run's, or KYTY_IMG_ACQSEED's).
	[[nodiscard]] uint32_t DrawRunMarksChange(std::span<const DrawRunImage> marks, bool compare_serials,
	                                          bool attachments_only, bool log_change,
	                                          bool attachment_serials) const;
	[[nodiscard]] bool DrawRunImagesUnchanged(bool compare_serials,
	                                          bool attachments_only = false) const;
	// Whether a guest range lies over one of the marked attachments.
	[[nodiscard]] static bool DrawRunOverAttachment(std::span<const DrawRunImage> marks,
	                                                uint64_t address, uint64_t size);
	// KYTY_DRAW_RUN_ACQUIRE: a draw that does not continue the run but resolved the recorded targets
	// for the same scissor union, in the same rendering instance right after the recorded draw, with
	// no texture over an attachment's memory, keeps the recorded attachment acquisition.
	[[nodiscard]] bool DrawRunAcquireCandidate(const CommandBuffer&               buffer,
	                                           const DrawRenderState&             state,
	                                           std::span<PreparedBindings* const> stages,
	                                           const vk::Rect2D*                  written) const;
	// KYTY_IMG_STATS (diagnostics): the first condition of DrawRunAcquireCandidate that fails, as
	// ImgSide::C::AcqWhyCaller + n (n = 2 no record ... 13 images changed, 14 none fails).
	[[nodiscard]] uint8_t DrawRunAcquireRefusal(const CommandBuffer&               buffer,
	                                            const DrawRenderState&             state,
	                                            std::span<PreparedBindings* const> stages,
	                                            const vk::Rect2D*                  written) const;
	// The refusal of the draw whose acquisition runs next (0: not a draw's acquisition).
	uint8_t m_acq_why = 0;
	// KYTY_IMG_ACQSEED (live 0|1|verify, default 0; drawPrep/imgSide.h): the acquisition of a draw that
	// may not seed a draw run (a mesh draw, more than one vertex stage, no structure key, ...) but
	// whose acquisition is an ordinary one (no feedback, no clears, no texture over an attachment, no
	// shader writes, no storage image) seeds KYTY_DRAW_RUN_ACQUIRE's reuse for the next draw, under
	// the same proof (DrawRunAcquireCandidate's conditions against this record). Taken after the
	// draw, as DrawRunRecordDraw takes the run's; valid for the next draw only (BeginDrawRun).
	struct AcquireSeed {
		bool                         valid            = false;
		uint64_t                     activity         = 0;
		vk::CommandBuffer            command          = nullptr;
		uint64_t                     tick             = 0;
		uint64_t                     rendering_serial = 0;
		RenderState                  rendering;
		std::vector<DrawRunImage>    marks; // the attachments, as the draw left them
		std::vector<RenderColorInfo> colors;
		std::vector<RenderDepthInfo> depth; // one entry (an incomplete type here)
		vk::Rect2D                   written {};
		bool                         bounded = false;
	};
	AcquireSeed m_acq_seed;
	bool        m_acq_seed_prev = false; // m_acq_seed.valid when the committed draw began
	// KYTY_IMG_ACQFB (count build): the previous full acquisition's targets (signature), result and
	// feedback aspects, and the depth-feedback keep barriers AcquireRenderTargets noted so far.
	struct AcquireLast {
		bool                 valid     = false;
		uint64_t             signature = 0;
		RenderState          rendering;
		vk::ImageAspectFlags feedback;
	};
	AcquireLast m_acq_last;
	uint64_t    m_acq_keeps = 0;
	struct AcqFbProbe {
		uint64_t                    signature   = 0;
		uint64_t                    transitions = 0;
		uint64_t                    keeps       = 0;
		bool                        instance    = false;
		uint32_t                    count       = 0;
		std::array<DrawRunImage, 9> marks {}; // the attachments before the acquisition
	};
	[[nodiscard]] AcqFbProbe AcqFbBefore(const CommandBuffer& buffer, const DrawRenderState& state) const;
	void AcqFbAfter(const AcqFbProbe& probe, const RenderState& rendering, vk::ImageAspectFlags feedback);
	// KYTY_IMG_ACQFB (live 0|1|verify, default 0): the acquisition of a draw without a run record
	// (typically one that samples its own attachments: a colour target bound as a texture, the depth
	// target sampled in a feedback layout) seeds the next draw's acquisition. The next draw takes the
	// seed's RenderState and feedback aspects instead of AcquireRenderTargets when the acquisition
	// would be a repeat (exact condition: imgside DESIGN 9.4, renderDraw.cpp AcqFbCandidate):
	// what the acquisition reads is the same - targets, scissor union, command buffer, tick, rendering
	// instance, activity epoch, every attachment mark including content serials as the seeding draw
	// left them, each colour target's is_bound, the depth decision (DecideDepthAccess, evaluated
	// live: sampled/feedback aspects, layout, access, keep, adopt) and no HTILE clear pending - and
	// its effects that are not repeats are performed: touches, attachment_layout/access, the
	// feedback-keep ordering and AdoptState (adopt), m_depth_feedback, and the content serials (the
	// targets' GPU-write marks: fresh serials, the unwritten depth target's restored), so no image
	// that took an attachment's serial in this draw's texture resolution is left looking current.
	struct AcqFbSeed {
		bool                                             valid            = false;
		uint64_t                                         activity         = 0;
		vk::CommandBuffer                                command          = nullptr;
		uint64_t                                         tick             = 0;
		uint64_t                                         rendering_serial = 0;
		RenderState                                      rendering;
		vk::ImageAspectFlags                             feedback;
		std::vector<DrawRunImage>                        marks; // colour targets, then the depth target
		std::vector<RenderColorInfo>                     colors;
		std::vector<RenderDepthInfo>                     depth; // one entry
		vk::Rect2D                                       written {};
		bool                                             bounded = false;
		bool                                             has_depth = false;
		DepthAccessDecision                              decision;
		std::array<bool, RENDER_COLOR_ATTACHMENTS_MAX>   color_bound {};
	};
	AcqFbSeed m_acqfb;
	bool      m_acqfb_prev = false; // m_acqfb.valid when the committed draw began
	bool      m_acqfb_this = false; // the committed draw's acquisition was a full one or an ACQFB reuse
	// What the last acquisition decided (AcquireRenderTargets, or an ACQFB reuse), for the seed.
	DepthAccessDecision                            m_acq_decision;
	bool                                           m_acq_has_depth = false;
	std::array<bool, RENDER_COLOR_ATTACHMENTS_MAX> m_acq_color_bound {};
	void AcqFbRecord(const CommandBuffer& buffer, const DrawRenderState& state, const RenderState& rendering,
	                 vk::ImageAspectFlags feedback, std::span<PreparedBindings* const> stages, const vk::Rect2D* written);
	[[nodiscard]] bool AcqFbCandidate(const CommandBuffer& buffer, const DrawRenderState& state,
	                                  std::span<PreparedBindings* const> stages, const vk::Rect2D* written,
	                                  DepthAccessDecision& decision) const;
	void AcqFbApply(CommandBuffer& buffer, DrawRenderState& state, const DepthAccessDecision& decision);
	// KYTY_IMG_STATS: what the draw's seeding step did (ImgSide::C::SeedWhyNotReached + n), and the
	// previous draw's.
	uint8_t m_seed_why      = 0;
	uint8_t m_seed_why_prev = 0;
	// Returns the seed outcome (m_seed_why: 11 depth promotion, 12 a texture over an attachment,
	// 13 recorded).
	uint8_t DrawRunSeedAcquire(const CommandBuffer& buffer, const DrawRenderState& state,
	                        const RenderState& rendering, std::span<PreparedBindings* const> stages,
	                        const vk::Rect2D* written);
	[[nodiscard]] bool DrawRunSeedCandidate(const CommandBuffer& buffer, const DrawRenderState& state,
	                                        std::span<PreparedBindings* const> stages,
	                                        const vk::Rect2D* written) const;
	// A continuation's kept textures and attachments are marked bound for the draw as their
	// resolution marks them (BindImage, BindRenderTarget), before the draw's buffer work.
	// attachments_only (KYTY_DRAW_RUN_TEXDELTA): the textures are resolved by the draw itself.
	void                       DrawRunMarkBindings(bool attachments_only = false);
	// Whether the committed draw continues the recorded run (the certificate without the images),
	// PrepareDrawRenderState after the program refresh.
	[[nodiscard]] bool DrawRunCandidate(const CommandBuffer& buffer, const DrawRenderState& state,
	                                    uint32_t render_target_slice_offset);
	// KYTY_DRAW_RUN_WRITES: whether a stage of a candidate writes buffers (1) and whether a written
	// range lies over an image of the run (2; 0: no writes).
	[[nodiscard]] static uint32_t DrawRunWritesOverRun(const ShaderStageRuntime&     runtime,
	                                                   std::span<const DrawRunImage> images);
	void               DrawRunTargets(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                  uint32_t render_target_slice_offset, DrawRenderState& state);
	// written: the scissor union the attachments were acquired for (null: unbounded claims).
	void DrawRunRecordDraw(const CommandBuffer& buffer, const DrawRenderState& state,
	                       uint32_t render_target_slice_offset, const RenderState& rendering,
	                       std::span<PreparedBindings* const> stages, const vk::Rect2D* written,
	                       const PipelineCache::Pipeline& pipeline);
	// KYTY_DRAW_RUN_PUSH: whether a continuation's CommitBindings may push only its per-draw
	// descriptors (the previous draw's image and sampler descriptors still in effect).
	[[nodiscard]] bool DrawRunPartialPush(const CommandBuffer&           buffer,
	                                      const PipelineCache::Pipeline& pipeline) const;
	void DrawRunVerify(const DrawRenderState& state, const RenderState& rendering,
	                   vk::ImageAspectFlags feedback_aspects,
	                   std::span<PreparedBindings* const> stages);
	DrawRunRecord m_run;
	// KYTY_DRAW_RUN_RECALL: earlier records of the instance (DrawRunKeep), whether the committed draw
	// recalls one, and whether m_run holds a complete record that is kept when it is replaced.
	[[nodiscard]] bool DrawRunRecallCandidate(const CommandBuffer& buffer, DrawRenderState& state,
	                                          uint32_t render_target_slice_offset);
	void               DrawRunKeep();
	// A recall that falls back late: its record is not kept again, nor seeds an acquisition reuse.
	void DrawRunRecallFallback() {
		if (m_run_recall) {
			m_run_recall     = false;
			m_run_keepable   = false;
			m_run_prev_valid = false;
		}
	}
	std::array<DrawRunRecord, 32> m_run_kept;
	uint32_t                     m_run_kept_next = 0;
	bool                         m_run_recall    = false;
	bool                         m_run_keepable  = false;
	bool          m_run_prev_valid     = false; // m_run.valid when the committed draw began
	bool          m_run_active         = false; // the draw takes the delta path
	uint64_t      m_rebind_epoch       = 0;     // KYTY_REBIND_AFTER_JOIN: UnregisterEpoch at resolution
	bool          m_lazy_find          = false; // KYTY_LAZY_FIND_BUFFER: FindBuffers leaves read V#s' ids unset
	bool          m_lazy_find_draw     = false; // and the draw being committed did (rebind check on)
	bool          m_run_verify         = false; // verify mode: the draw would have continued
	uint64_t      m_run_key            = 0;     // the committed draw's structure key (the engine)
	uint64_t      m_run_targets_key    = 0;     // and its registers' part (KYTY_DRAW_RUN_TARGETS)
	uint64_t      m_run_swap_key       = 0;     // and the key without the vertex program (PIPESWAP)
	bool          m_run_pipeswap       = false; // KYTY_DRAW_RUN_PIPESWAP: the continuation swaps
	uint64_t      m_run_texfree_key    = 0;     // the key without the T#/S# words (TEXDELTA)
	bool          m_run_texdelta       = false; // KYTY_DRAW_RUN_TEXDELTA: the continuation resolves its textures
	bool          m_texdelta_verify    = false; // KYTY_DRAW_RUN_TEXDELTA=2: the draw would have been a delta
	static void TexDeltaCount(DrawRun::TexDelta::Decision decision, bool taken = false);
	void        DrawRunRecordTextures(std::span<PreparedBindings* const> stages);
	// KYTY_DRAW_RUN_TEXDELTA: whether a delta's resolved textures may keep the record's acquisition.
	[[nodiscard]] uint32_t TexDeltaTextureVerdict(std::span<PreparedBindings* const> stages) const;
	// KYTY_DRAW_RUN_TEXDELTA=2: what a delta keeps against the normal path's results.
	void TexDeltaVerify(const DrawRenderState& state, const RenderState& rendering,
	                    vk::ImageAspectFlags feedback_aspects, bool dynamic_emitted,
	                    std::span<PreparedBindings* const> stages);
	// KYTY_CONT_KERNEL (renderDraw.cpp): what the draw that last recorded m_run (not through the
	// kernel) resolved, which a kernel continuation of the same key and rendering instance reuses.
	struct ContKernelRecord {
		const PipelineCache::Pipeline* pipeline          = nullptr;
		uint64_t                       generation        = 0;
		uint64_t                       key               = 0;
		uint64_t                       serial            = 0;
		vk::PrimitiveTopology          topology          = vk::PrimitiveTopology::ePointList;
		uint32_t                       element_size      = 0; // 2 or 4: an indexed draw without host indices
		bool                           primitive_restart = false;
		bool                           plain_pixel       = false;
		bool                           barrier_safe      = false;
		bool                           barrier_sink      = false;
		bool                           target_ops_quiet  = false; // colour mode 0/1, no depth-stencil copy
	} m_kernel_record;
	bool m_kernel        = false; // KYTY_CONT_KERNEL=1: the committed draw takes the kernel
	bool m_kernel_verify = false; // KYTY_CONT_KERNEL=2: the committed draw would have
	// KYTY_BK_HANDOVER (drawPrep/bookkeep.h): the committed draw was a kernel continuation (the
	// engine reads and clears it for the run counters).
	bool m_bk_kernel_taken = false;
	// KYTY_CP_COMMIT_STATS: the kernel's outcome for the committed draw (Taken is m_kernel itself).
	CommitStats::KernelOutcome    m_stats_kernel    = CommitStats::KernelOutcome::Off;
	CommitStats::CandidateOutcome m_stats_candidate = CommitStats::CandidateOutcome::NotReached;
	uint8_t m_stats_inelig      = 0; // DrawRun::KeyDiag::Inelig of the committed draw's record
	uint8_t m_stats_prev_inelig = 0; // and of the previous draw's
	// KYTY_SPEC_SEGMENTS (drawPrep/segment.h): the committed slot's encoded draw (the engine sets
	// it while the switch is on), whether the kernel validated it for this draw, and the draw the
	// draw command is spliced from (mode 1).
	const DrawPrep::Segment::EncodedDraw* m_segment           = nullptr;
	bool                                  m_segment_validated = false;
	const DrawPrep::Segment::EncodedDraw* m_segment_splice    = nullptr;
	// The kernel-taken continuation's encoded draw validated (counts; sets m_segment_validated);
	// the draw to splice in mode 1, else null. Verify: the serial values against the encoded ones.
	[[nodiscard]] const DrawPrep::Segment::EncodedDraw* SegmentValidate(const DrawIndexArgs& args);
	void SegmentVerify(const DrawIndexArgs& args, const DrawEmitInfo& emit,
	                   const DrawIndexBufferSource& index_source);
	[[nodiscard]] bool ContinuationKernelEntry(const DrawIndexArgs& args) const;
	[[nodiscard]] bool ContinuationKernel(uint64_t submit_id, CommandBuffer& buffer,
	                                      const DrawIndexArgs& args);
	// KYTY_DRAW_RUN_PIPESWAP: the kept vertex-stage bindings and claims fit the new vertex program.
	[[nodiscard]] static bool DrawRunPipeSwapCompatible(
	    const ShaderRecompiler::IR::CompiledShaderInfo& recorded,
	    const ShaderRecompiler::IR::CompiledShaderInfo& program);
	// KYTY_SKIP_INVALID: the last ResolveRenderDepthTarget found a depth/stencil state the host
	// cannot draw with (left without a depth target); PrepareDrawRenderState leaves the draw out.
	bool          m_depth_unsupported  = false;
	bool          m_run_targets_kept   = false; // KYTY_DRAW_RUN_TARGETS: the draw kept the record's
	// KYTY_DRAW_RUN_TARGETS: whether the committed draw may keep the record's resolved targets.
	[[nodiscard]] bool DrawRunTargetsCandidate(const CommandBuffer& buffer, const DrawRenderState& state,
	                                           uint32_t render_target_slice_offset) const;
	// KYTY_DRAW_RUN_KEY_DIAG (drawRun.h): the committed draw's key parts (the engine; null when off)
	// and the diagnostic's state (GPU thread).
	const DrawRun::KeyParts* m_run_parts = nullptr;
	struct DrawRunKeyDiagState {
		struct Recent {
			uint64_t          key      = 0;
			uint64_t          activity = 0;
			vk::CommandBuffer command  = nullptr;
			uint64_t          tick     = 0;
			uint64_t          serial   = 0;
		};
		DrawRun::KeyParts               record_parts {}; // m_run's
		bool                            record_parts_valid = false;
		uint64_t                        record_pipeline    = 0; // m_run's resolved pipeline
		uint64_t                        record_structure   = 0; // and structure
		bool                            record_resolved    = false;
		uint64_t                        last_key           = 0; // the previous committed draw's
		DrawRun::KeyDiag::Inelig        reason      = DrawRun::KeyDiag::Inelig::NotReached; // this draw's
		DrawRun::KeyDiag::Inelig        prev_reason = DrawRun::KeyDiag::Inelig::NotReached;
		int                             combo       = -2; // this draw's key-difference combination
		std::array<Recent, 16>          recent {};
		uint32_t                        recent_next = 0;
	};
	DrawRunKeyDiagState m_key_diag;
	// KYTY_DRAW_RUN_KEY_DIAG: the draw resolved its structure (eligibility check point).
	void DrawRunKeyDiagResolved(const CommandBuffer& buffer, std::span<PreparedBindings* const> stages,
	                            const PipelineCache::Pipeline& pipeline, const RenderState& rendering,
	                            bool recorded);
	bool          m_in_engine_commit   = false;
	bool          m_prepared_validated = false; // RefreshShaders accepted the draw's preparation
	uint32_t      m_run_slice_offset   = 0;
	// KYTY_MRT_SPLIT (renderDraw.cpp AliasedColorSlots): the colour slots this pass of a draw may bind
	// (~0: all), whether the draw may split (DrawIndex, DrawAuto), and the slots its first pass left
	// out for a second pass.
	uint32_t      m_mrt_filter         = ~0u;
	bool          m_mrt_split_allowed  = false;
	uint32_t      m_mrt_split_slots    = 0;

	friend class CommandProcessor;
	friend class DrawPrep::Engine;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
