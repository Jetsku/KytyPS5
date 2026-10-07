#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/queueSubmission.h"
#include "graphics/host_gpu/renderer/commandStream.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorSetReuse.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/textureBindingMemo.h"
#include "graphics/host_gpu/renderer/pipeline/textureSetCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
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
} // namespace DrawPrep

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

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
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
	// Whether anything is pending, and whether queued upload copies are (RequestUploadCopy).
	[[nodiscard]] bool HasPendingBarriers() const noexcept { return !m_pending.Empty(); }
	[[nodiscard]] bool HasPendingUploads() const noexcept { return !m_pending.uploads.empty(); }
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
		return m_bound_pipelines[point == vk::PipelineBindPoint::eCompute ? 1u : 0u];
	}
	// PushDescriptors result: the update was skipped (identical to the one still in effect), there
	// was nothing comparable (no earlier push in this command buffer for the bind point, another
	// layout, a bound set, or the dedup is off), or the binding list differs; otherwise the index
	// of the first write whose descriptors differ.
	static constexpr int32_t PushAvoided   = -1;
	static constexpr int32_t PushMissState = -2;
	static constexpr int32_t PushMissShape = -3;
	// `known_miss`: the caller knows the update differs from every earlier one of this command
	// buffer (a descriptor refers to an upload made for this command, KYTY_PUSH_SHADOW_FRESH_SKIP):
	// recorded without the comparison and without keeping a copy for the next one.
	static constexpr int32_t PushMissFresh = -4;
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
		return m_descriptor_epochs[point == vk::PipelineBindPoint::eCompute ? 1u : 0u];
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
	void InvalidatePushConstants() const { m_push_constants.valid = false; }

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
			origins = 0;
		}
	};
	// Records the queued upload copies behind one barrier and queues their post-copy barriers.
	void RecordPendingUploads() const;
	// Every native command other than the batch itself and the rendering bookkeeping below.
	void NoteForeignCommand() const {
		m_recorded_since_flush = true;
		m_epoch_clean          = false;
	}
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
	std::array<vk::Pipeline, 2> m_bound_pipelines {};
	struct DescriptorState {
		vk::PipelineLayout layout = nullptr;
		// Non-null: set 0 is this bound descriptor set (the push contents below are unused).
		vk::DescriptorSet bound_set = nullptr;
		std::vector<vk::WriteDescriptorSet> writes;
		std::vector<vk::DescriptorBufferInfo> buffers;
		std::vector<vk::DescriptorImageInfo> images;
	};
	std::array<DescriptorState, 2> m_descriptor_states;
	std::array<uint64_t, 2>        m_descriptor_epochs {};
	struct PushConstantShadow {
		bool                     valid  = false;
		vk::PipelineLayout       layout = nullptr;
		vk::ShaderStageFlags     stages;
		uint32_t                 size = 0;
		std::array<uint32_t, 64> dwords {};
	};
	mutable PushConstantShadow m_push_constants;

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

	// KYTY_CP_RECORDER: the scheduler's recorder and its encoder (null when off), and whether a
	// direct window is open (the CP records natively after a drain; see Sink()).
	CommandRecorder*        m_recorder      = nullptr;
	CommandStream::Encoder* m_encoder       = nullptr;
	mutable bool            m_direct_window = false;
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

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context);
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode);

	// plan: the committed draw's binding plan for this stage (KYTY_DRAW_PREP_BINDINGS), or null.
	// keep_images (KYTY_DRAW_RUN continuation): the stage's texture and sampler bindings are the
	// previous draw's, kept as they are; only the per-draw data is prepared.
	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared,
	                     DrawPrep::StagePlan* plan = nullptr, bool keep_images = false);
	void                           FindBuffers(PreparedBindings& bindings);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	// keep_images (KYTY_DRAW_RUN continuation): no image transition is recorded; the images are in
	// the states the previous draw left them in (DrawRunImagesUnchanged).
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings, bool keep_images = false);

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

private:
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
	                                        const ShaderRecompiler::IR::DescriptorValue&    value);
	// keep_images (KYTY_DRAW_RUN continuation): the image rebinding and target rediscovery are
	// skipped (the previous draw's bindings are kept).
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors, bool keep_images = false);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable);
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
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                              CommandBuffer& command, uint32_t group_x,
	                                              uint32_t group_y, uint32_t group_z, uint32_t mode);

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	// Reused by every draw (all draws hold the render mutex); DrawRenderState::Reset restores
	// only what the previous draw changed. Owns the draw's per-stage program preparation.
	std::unique_ptr<DrawRenderState>      m_draw_state;
	// Program preparation output of the current dispatch; its stage runtime points here.
	PipelineCache::StagePrep              m_compute_prep;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorBufferInfo> m_descriptor_buffers;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	std::vector<uint32_t>                 m_image_occurrences;
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
	// KYTY_DRAW_PREP_BINDINGS: the binding plan of the draw the engine is committing (null: none),
	// and whether it applies: set by RefreshShaders once DrawPrep::Validate accepted the same
	// slot's preparation (the plan was derived from it), cleared by the engine after the draw.
	DrawPrep::BindingPlan* m_binding_plan        = nullptr;
	bool                   m_binding_plan_active = false;
	// KYTY_DRAW_PREP_BINDINGS_VERIFY (P4b-2): the views a predicted view run claims (RebindImages).
	std::vector<vk::ImageView> m_claimed_run_views;
	// KYTY_NATIVE_INDIRECT_MESH (meshIndirect.h): created by the first native indirect mesh draw.
	std::unique_ptr<MeshIndirect::Converter> m_mesh_indirect;

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
	[[nodiscard]] uint32_t DrawRunImagesChange(bool compare_serials,
	                                           bool attachments_only = false, bool log_change = false) const;
	[[nodiscard]] bool     DrawRunImagesUnchanged(bool compare_serials,
	                                              bool attachments_only = false) const {
		return DrawRunImagesChange(compare_serials, attachments_only) == 0;
	}
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
	// A continuation's kept textures and attachments are marked bound for the draw as their
	// resolution marks them (BindImage, BindRenderTarget), before the draw's buffer work.
	void                       DrawRunMarkBindings();
	// Whether the committed draw continues the recorded run (the certificate without the images),
	// PrepareDrawRenderState after the program refresh.
	[[nodiscard]] bool DrawRunCandidate(const CommandBuffer& buffer, const DrawRenderState& state,
	                                    uint32_t render_target_slice_offset);
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
	bool          m_run_prev_valid     = false; // m_run.valid when the committed draw began
	bool          m_run_active         = false; // the draw takes the delta path
	bool          m_run_verify         = false; // verify mode: the draw would have continued
	uint64_t      m_run_key            = 0;     // the committed draw's structure key (the engine)
	bool          m_in_engine_commit   = false;
	bool          m_prepared_validated = false; // RefreshShaders accepted the draw's preparation
	uint32_t      m_run_slice_offset   = 0;

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
