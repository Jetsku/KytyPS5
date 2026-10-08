#include "common/assert.h"
#include "common/common.h"
#include "common/hangTrace.h"
#include "common/profiler.h"
#include "common/rendererBatch.h"
#include "common/threads.h"
#include "graphics/host_gpu/gpuBreadcrumbs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/memoryStats.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#define KYTY_RECORDER_CALLER() _ReturnAddress()
#else
#define KYTY_RECORDER_CALLER() __builtin_return_address(0)
#endif

namespace Libs::Graphics {

// ------------------------------------------------------------------------------------------------
// Barrier batcher

static bool EnvSwitch(const char* name, bool default_value) {
	const auto* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return default_value;
	}
	return !(std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
	         std::strcmp(value, "off") == 0);
}

bool BarrierBatchEnabled() {
	static const bool enabled = EnvSwitch("KYTY_BARRIER_BATCH", true);
	return enabled;
}

bool BarrierSinkEnabled() {
	static const bool enabled = BarrierBatchEnabled() && EnvSwitch("KYTY_BARRIER_SINK", true);
	return enabled;
}

bool UploadBatchEnabled() {
	static const bool enabled = BarrierBatchEnabled() && EnvSwitch("KYTY_UPLOAD_BATCH", true);
	return enabled;
}

bool DrawWriteSinkEnabled() {
	static const bool enabled = BarrierBatchEnabled() && EnvSwitch("KYTY_DRAW_WRITE_SINK", true);
	return enabled;
}

bool DepthFeedbackKeepEnabled() {
	static const bool enabled =
	    BarrierBatchEnabled() && EnvSwitch("KYTY_DEPTH_FEEDBACK_KEEP", true);
	return enabled;
}

bool DepthFeedbackLazyEnabled() {
	static const bool enabled =
	    DepthFeedbackKeepEnabled() && EnvSwitch("KYTY_DEPTH_FEEDBACK_LAZY", true);
	return enabled;
}

bool DepthLayoutStableEnabled() {
	static const bool enabled = EnvSwitch("KYTY_DEPTH_LAYOUT_STABLE", true);
	return enabled;
}

bool PushConstantShadowEnabled() {
	static const bool enabled = EnvSwitch("KYTY_PUSH_CONSTANT_SHADOW", true);
	return enabled;
}

bool DescriptorSetReuseEnabled() {
	static const bool enabled = EnvSwitch("KYTY_DESCRIPTOR_SET_REUSE", true);
	return enabled;
}

bool DescriptorSetReuseAuditEnabled() {
	static const bool enabled = EnvSwitch("KYTY_DESCRIPTOR_SET_REUSE_AUDIT", false);
	return enabled;
}

namespace {

using Stage2  = vk::PipelineStageFlagBits2;
using Access2 = vk::AccessFlagBits2;

// Access bits that MEMORY_READ / MEMORY_WRITE stand for in a queue (non-host) dependency. Host
// accesses and every unlisted bit must be matched literally.
constexpr vk::AccessFlags2 KnownDeviceReads =
    Access2::eIndirectCommandRead | Access2::eIndexRead | Access2::eVertexAttributeRead |
    Access2::eUniformRead | Access2::eInputAttachmentRead | Access2::eShaderRead |
    Access2::eColorAttachmentRead | Access2::eDepthStencilAttachmentRead |
    Access2::eTransferRead | Access2::eMemoryRead | Access2::eShaderSampledRead |
    Access2::eShaderStorageRead;
constexpr vk::AccessFlags2 KnownDeviceWrites =
    Access2::eShaderWrite | Access2::eColorAttachmentWrite |
    Access2::eDepthStencilAttachmentWrite | Access2::eTransferWrite | Access2::eMemoryWrite |
    Access2::eShaderStorageWrite;

// True when a dependency on `have` stages includes every stage in `want`. ALL_COMMANDS covers
// every queue stage; HOST is not a queue stage and must be present literally.
bool StagesCover(vk::PipelineStageFlags2 have, vk::PipelineStageFlags2 want) {
	if ((want & Stage2::eHost) && !(have & Stage2::eHost)) {
		return false;
	}
	want &= ~vk::PipelineStageFlags2 {Stage2::eHost};
	if (have & Stage2::eAllCommands) {
		return true;
	}
	return !(want & ~have);
}

bool AccessCovers(vk::AccessFlags2 have, vk::AccessFlags2 want) {
	if (have & Access2::eMemoryRead) {
		want &= ~KnownDeviceReads;
	}
	if (have & Access2::eMemoryWrite) {
		want &= ~KnownDeviceWrites;
	}
	return !(want & ~have);
}

bool MemoryCovers(const vk::MemoryBarrier2& have, const vk::MemoryBarrier2& want) {
	return StagesCover(have.srcStageMask, want.srcStageMask) &&
	       StagesCover(have.dstStageMask, want.dstStageMask) &&
	       AccessCovers(have.srcAccessMask, want.srcAccessMask) &&
	       AccessCovers(have.dstAccessMask, want.dstAccessMask);
}

// Orders every earlier command (all stages, all writes made available) before every later
// command (all stages, all reads and writes made visible) - what the guest global barrier is.
bool IsFullBarrier(const vk::MemoryBarrier2& barrier) {
	return (barrier.srcStageMask & Stage2::eAllCommands) &&
	       (barrier.dstStageMask & Stage2::eAllCommands) &&
	       (barrier.srcAccessMask & Access2::eMemoryWrite) &&
	       (barrier.dstAccessMask & Access2::eMemoryRead) &&
	       (barrier.dstAccessMask & Access2::eMemoryWrite);
}

void CountBatch(GpuOpProfiler::BarrierBatchEvent event, uint64_t amount = 1) {
	GpuOpProfiler::CountBarrierBatch(event, amount);
}

// gpuOpProfiler sites of recorded batches: the single origin, or "mixed".
constinit GpuOpProfiler::Site g_batch_sites[static_cast<size_t>(BarrierOrigin::Count) + 1] = {
    GpuOpProfiler::Site {"batch.guest_global"},  GpuOpProfiler::Site {"batch.shader_access"},
    GpuOpProfiler::Site {"batch.shader_write"},  GpuOpProfiler::Site {"batch.shader_hazard"},
    GpuOpProfiler::Site {"batch.indirect_args"}, GpuOpProfiler::Site {"batch.gds"},
    GpuOpProfiler::Site {"batch.image"},         GpuOpProfiler::Site {"batch.upload"},
    GpuOpProfiler::Site {"batch.mixed"},
};

GpuOpProfiler::Site& BatchSite(uint32_t origins) {
	if (origins != 0 && std::has_single_bit(origins)) {
		const auto index = static_cast<size_t>(std::countr_zero(origins));
		if (index < static_cast<size_t>(BarrierOrigin::Count)) {
			return g_batch_sites[index];
		}
	}
	return g_batch_sites[static_cast<size_t>(BarrierOrigin::Count)];
}

constexpr uint32_t OriginBit(BarrierOrigin origin) {
	return 1u << static_cast<uint32_t>(origin);
}

// Aggregate profile: one queued barrier request of `origin` (FrameEvent.BarrierRequests*).
void CountOriginRequest(BarrierOrigin origin) {
	static_assert(static_cast<uint32_t>(Profiler::FrameEvent::BarrierRequestsUpload) -
	                      static_cast<uint32_t>(Profiler::FrameEvent::BarrierRequestsGuest) + 1 ==
	                  static_cast<uint32_t>(BarrierOrigin::Count),
	              "one BarrierRequests counter per BarrierOrigin, in order");
	Profiler::CountFrameEvent(static_cast<Profiler::FrameEvent>(
	    static_cast<uint32_t>(Profiler::FrameEvent::BarrierRequestsGuest) +
	    static_cast<uint32_t>(origin)));
}

} // namespace

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

bool CommandBuffer::IsInvalid() const {
	return m_buffer == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	if (m_internal_recording == 0 && BarrierBatchEnabled()) {
		// The caller may record anything through the returned handle.
		FlushBarriers();
		NoteForeignCommand();
	}
	// Including push-constant updates with other layouts (helper passes, occlusion reductions).
	m_push_constants.valid = false;
	if (Encoding()) {
		// KYTY_CP_RECORDER: the recorder executes everything encoded so far (including the batch
		// flushed above); the caller then records natively at exactly this position.
		OpenDirectWindow(KYTY_RECORDER_CALLER());
	}
	return m_buffer;
}

vk::CommandBuffer CommandBuffer::StateHandle() const {
	EXIT_IF(IsInvalid());
	if (Encoding()) {
		OpenDirectWindow(KYTY_RECORDER_CALLER());
	}
	return m_buffer;
}

CommandSink CommandBuffer::Sink() const {
	EXIT_IF(IsInvalid());
	// Handle()'s logical effects, in the same order.
	if (m_internal_recording == 0 && BarrierBatchEnabled()) {
		FlushBarriers();
		NoteForeignCommand();
	}
	m_push_constants.valid = false;
	return CommandSink(*this);
}

CommandSink CommandBuffer::StateSink() const {
	EXIT_IF(IsInvalid());
	return CommandSink(*this);
}

CommandSink CommandBuffer::EmissionSink() const {
	EXIT_IF(IsInvalid());
	CloseDirectWindow();
	return CommandSink(*this);
}

void CommandBuffer::OpenDirectWindow(const void* caller) const {
	EXIT_IF(m_recorder == nullptr || m_direct_window);
	// Drain attribution: the GpuOpProfiler site when active, else the caller of Handle().
	const void* site  = nullptr;
	const void* scope = nullptr;
	if (GpuOpProfiler::Active()) {
		GpuOpProfiler::Detail::CurrentSites(&site, &scope);
	}
	m_recorder->Drain(site != nullptr ? site : caller, site != nullptr);
	m_direct_window = true;
	m_recorder->SetWindowOpen(true);
}

void CommandBuffer::CloseDirectWindow() const {
	if (m_direct_window) {
		m_direct_window = false;
		m_recorder->SetWindowOpen(false);
	}
}

void CommandBuffer::ResetBarrierState() const {
	m_pending.Clear();
	m_last_memory_valid    = false;
	m_recorded_since_flush = true;
	m_epoch_clean          = false;
	m_epoch_instance       = 0;
	m_internal_recording   = 0;
	m_feedback_keep.reset();
}

void CommandBuffer::NoteFeedbackKeep(const vk::ImageMemoryBarrier2& ordering) const {
	EXIT_IF(!BarrierBatchEnabled() || !m_rendering);
	m_feedback_keep = ordering;
}

void CommandBuffer::RequestMemoryBarrier(vk::PipelineStageFlags2 src_stages,
                                         vk::AccessFlags2        src_access,
                                         vk::PipelineStageFlags2 dst_stages,
                                         vk::AccessFlags2 dst_access, BarrierOrigin origin) const {
	EXIT_IF(IsInvalid() || !src_stages || !dst_stages);
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = src_stages;
	barrier.srcAccessMask = src_access;
	barrier.dstStageMask  = dst_stages;
	barrier.dstAccessMask = dst_access;
	if (!BarrierBatchEnabled()) {
		// Callers keep their own legacy paths; this is only a safe fallback.
		EndRendering();
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		m_buffer.pipelineBarrier2(dependency);
		return;
	}
	CountBatch(GpuOpProfiler::BarrierBatchEvent::Requests);
	if (!m_pending.Empty()) {
		// Nothing was recorded since the pending requests: one barrier with the union of the
		// scopes orders everything each of them (and their chain) ordered.
		CountBatch(GpuOpProfiler::BarrierBatchEvent::Merged);
	} else if (!m_recorded_since_flush && m_last_memory_valid &&
	           MemoryCovers(m_last_memory, barrier)) {
		// Nothing was recorded since the last batch, whose memory dependency already orders
		// every earlier command against every later one in these scopes.
		CountBatch(GpuOpProfiler::BarrierBatchEvent::Elided);
		return;
	}
	if (m_pending.has_memory) {
		auto& memory = m_pending.memory;
		memory.srcStageMask |= barrier.srcStageMask;
		memory.srcAccessMask |= barrier.srcAccessMask;
		memory.dstStageMask |= barrier.dstStageMask;
		memory.dstAccessMask |= barrier.dstAccessMask;
	} else {
		m_pending.memory     = barrier;
		m_pending.has_memory = true;
	}
	m_pending.origins |= OriginBit(origin);
	CountOriginRequest(origin);
}

void CommandBuffer::RequestBufferBarrier(const vk::BufferMemoryBarrier2& barrier,
                                         BarrierOrigin                   origin) const {
	EXIT_IF(IsInvalid());
	if (!BarrierBatchEnabled()) {
		EndRendering();
		vk::DependencyInfo dependency {};
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &barrier;
		m_buffer.pipelineBarrier2(dependency);
		return;
	}
	CountBatch(GpuOpProfiler::BarrierBatchEvent::Requests);
	if (!m_pending.Empty()) {
		CountBatch(GpuOpProfiler::BarrierBatchEvent::Merged);
	}
	m_pending.buffers.push_back(barrier);
	m_pending.origins |= OriginBit(origin);
	CountOriginRequest(origin);
}

bool CommandBuffer::BatchImageBarriers(std::span<const vk::ImageMemoryBarrier2> barriers,
                                       vk::CommandBuffer target, bool deferrable) const {
	if (!BarrierBatchEnabled() || IsInvalid() || target != m_buffer) {
		return false;
	}
	if (barriers.empty()) {
		return true;
	}
	CountBatch(GpuOpProfiler::BarrierBatchEvent::Requests);
	if (!m_pending.Empty()) {
		CountBatch(GpuOpProfiler::BarrierBatchEvent::Merged);
	}
	for (const auto& barrier: barriers) {
		// Layout transitions of one image stay in separate, ordered barrier commands.
		if (std::ranges::any_of(m_pending.images, [&barrier](const auto& pending) {
			    return pending.image == barrier.image;
		    })) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ImageBarrierSameImageFlushes);
			if (m_rendering) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::ImageBarrierSameImageRenderEnds);
			}
			FlushBarriers();
		}
		m_pending.images.push_back(barrier);
		m_pending.origins |= OriginBit(BarrierOrigin::Image);
		CountOriginRequest(BarrierOrigin::Image);
	}
	if (!deferrable) {
		FlushBarriers();
	}
	return true;
}

void CommandBuffer::RequestUploadCopy(vk::Buffer source, vk::Buffer destination,
                                      std::span<const vk::BufferCopy> regions) const {
	EXIT_IF(IsInvalid() || !UploadBatchEnabled() || source == nullptr || destination == nullptr);
	if (regions.empty()) {
		return;
	}
	// Copies recorded together are not ordered against each other: a rewrite of a queued
	// destination range (a page re-dirtied between two uploads) goes into the next batch.
	const bool overlaps = std::ranges::any_of(m_pending.uploads, [&](const PendingUpload& queued) {
		if (queued.destination != destination) {
			return false;
		}
		for (uint32_t index = 0; index < queued.region_count; index++) {
			const auto& old = m_pending.upload_regions[queued.first_region + index];
			for (const auto& region: regions) {
				if (region.dstOffset < old.dstOffset + old.size &&
				    old.dstOffset < region.dstOffset + region.size) {
					return true;
				}
			}
		}
		return false;
	});
	if (overlaps) {
		FlushBarriers();
	}
	CountBatch(GpuOpProfiler::BarrierBatchEvent::Requests);
	if (!m_pending.Empty()) {
		CountBatch(GpuOpProfiler::BarrierBatchEvent::Merged);
	}
	m_pending.uploads.push_back({source, destination,
	                             static_cast<uint32_t>(m_pending.upload_regions.size()),
	                             static_cast<uint32_t>(regions.size())});
	m_pending.upload_regions.insert(m_pending.upload_regions.end(), regions.begin(),
	                                regions.end());
	m_pending.origins |= OriginBit(BarrierOrigin::Upload);
	CountOriginRequest(BarrierOrigin::Upload);
}

void CommandBuffer::RecordPendingUploads() const {
	const GpuOpProfiler::ScopedSite site(
	    g_batch_sites[static_cast<size_t>(BarrierOrigin::Upload)]);
	std::vector<vk::BufferMemoryBarrier2> destinations;
	for (const auto& upload: m_pending.uploads) {
		if (std::ranges::any_of(destinations, [&upload](const auto& barrier) {
			    return barrier.buffer == upload.destination;
		    })) {
			continue;
		}
		vk::BufferMemoryBarrier2 barrier {};
		// Every earlier access of the destination (including an earlier upload copy) before
		// the copies' writes, as SynchronizeBuffer's own pre-copy barrier.
		barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
		barrier.srcAccessMask       = vk::AccessFlagBits2::eMemoryRead |
		                        vk::AccessFlagBits2::eMemoryWrite |
		                        vk::AccessFlagBits2::eTransferRead |
		                        vk::AccessFlagBits2::eTransferWrite;
		barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
		barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer              = upload.destination;
		barrier.offset              = 0;
		barrier.size                = VK_WHOLE_SIZE;
		destinations.push_back(barrier);
	}
	// Many destinations (a BDA synchronization pass) use one global dependency with the same
	// scopes instead: it orders at least everything the buffer barriers order.
	constexpr size_t   MaxBufferBarriers = 8;
	const bool         global            = destinations.size() > MaxBufferBarriers;
	vk::MemoryBarrier2 memory {};
	memory.srcStageMask  = destinations.front().srcStageMask;
	memory.srcAccessMask = destinations.front().srcAccessMask;
	memory.dstStageMask  = destinations.front().dstStageMask;
	memory.dstAccessMask = destinations.front().dstAccessMask;
	++m_internal_recording;
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags = vk::DependencyFlagBits::eByRegion;
	if (global) {
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &memory;
	} else {
		dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(destinations.size());
		dependency.pBufferMemoryBarriers    = destinations.data();
	}
	const auto sink = StateSink();
	sink.pipelineBarrier2(dependency);
	for (const auto& upload: m_pending.uploads) {
		sink.copyBuffer(upload.source, upload.destination, upload.region_count,
		                m_pending.upload_regions.data() + upload.first_region);
	}
	--m_internal_recording;
	MemoryStats::Count(MemoryStats::Counter::UploadCopies, m_pending.uploads.size());
	MemoryStats::Count(MemoryStats::Counter::UploadBarriers, 2);
	// The copies' writes before every later access. These join the batch recorded right after
	// the copies: its other barriers ordered earlier commands against later ones, which the
	// copies (writing only their destinations, reading host-written staging data) do not need.
	constexpr auto post_src_stage  = vk::PipelineStageFlagBits2::eTransfer;
	constexpr auto post_src_access = vk::AccessFlagBits2::eTransferWrite;
	constexpr auto post_dst_stage  = vk::PipelineStageFlagBits2::eAllCommands;
	const auto     post_dst_access =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	if (global) {
		if (m_pending.has_memory) {
			m_pending.memory.srcStageMask |= post_src_stage;
			m_pending.memory.srcAccessMask |= post_src_access;
			m_pending.memory.dstStageMask |= post_dst_stage;
			m_pending.memory.dstAccessMask |= post_dst_access;
		} else {
			m_pending.memory               = vk::MemoryBarrier2 {};
			m_pending.memory.srcStageMask  = post_src_stage;
			m_pending.memory.srcAccessMask = post_src_access;
			m_pending.memory.dstStageMask  = post_dst_stage;
			m_pending.memory.dstAccessMask = post_dst_access;
			m_pending.has_memory           = true;
		}
	} else {
		for (auto& barrier: destinations) {
			barrier.srcStageMask  = post_src_stage;
			barrier.srcAccessMask = post_src_access;
			barrier.dstStageMask  = post_dst_stage;
			barrier.dstAccessMask = post_dst_access;
			m_pending.buffers.push_back(barrier);
		}
	}
	m_pending.uploads.clear();
	m_pending.upload_regions.clear();
}

void CommandBuffer::FlushBarriers() const {
	if (m_pending.Empty()) {
		return;
	}
	EXIT_IF(IsInvalid());
	if (m_rendering && !m_pending.uploads.empty()) {
		MemoryStats::Count(MemoryStats::Counter::UploadRenderSplits);
	}
	if (m_rendering) {
		// Pipeline barriers cannot be recorded inside dynamic rendering. (A command recorded
		// through Handle() mostly ends rendering anyway; only a draw that has to restart its own
		// instance counts as a barrier split, in BeginRendering().) Ended before the batch site
		// is entered, so the end is attributed to the site whose command needed the flush.
		EndRendering();
	}
	if (!m_pending.uploads.empty()) {
		RecordPendingUploads();
	}
	const GpuOpProfiler::ScopedSite site(BatchSite(m_pending.origins));
	++m_internal_recording;
	if (m_pending.has_memory) {
		// The requests were issued in some order with nothing recorded between them. Widening
		// every buffer/image barrier by the global dependency reproduces the execution and
		// memory dependency chains the separate barrier commands formed through each other.
		const auto& memory = m_pending.memory;
		for (auto& image: m_pending.images) {
			image.srcStageMask |= memory.srcStageMask;
			image.srcAccessMask |= memory.srcAccessMask;
			image.dstStageMask |= memory.dstStageMask;
			image.dstAccessMask |= memory.dstAccessMask;
		}
		for (auto& buffer: m_pending.buffers) {
			buffer.srcStageMask |= memory.srcStageMask;
			buffer.srcAccessMask |= memory.srcAccessMask;
			buffer.dstStageMask |= memory.dstStageMask;
			buffer.dstAccessMask |= memory.dstAccessMask;
		}
	}
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount       = m_pending.has_memory ? 1u : 0u;
	dependency.pMemoryBarriers          = m_pending.has_memory ? &m_pending.memory : nullptr;
	dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(m_pending.buffers.size());
	dependency.pBufferMemoryBarriers    = m_pending.buffers.data();
	dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(m_pending.images.size());
	dependency.pImageMemoryBarriers     = m_pending.images.data();
	StateSink().pipelineBarrier2(dependency);
	--m_internal_recording;

	m_last_memory_valid = m_pending.has_memory;
	if (m_pending.has_memory) {
		m_last_memory = m_pending.memory;
	}
	m_recorded_since_flush = false;
	// A full batch orders everything before it against everything after it; its buffer/image
	// barriers were widened by the same scopes above.
	m_epoch_clean    = m_pending.has_memory && IsFullBarrier(m_pending.memory);
	m_epoch_instance = 0;
	m_pending.Clear();
}

bool CommandBuffer::CanSinkPending() const {
	// Sinking records the pending barrier B after the draw D2 that follows it instead of before.
	// Everything recorded after the flush point still sees B (it covers all earlier commands), so
	// only the P -> D2 orderings for commands P before B are lost. They are not needed when:
	//  - commands before the epoch's full barrier F are ordered against D2 by F itself;
	//  - everything since F is state commands, the begin of this rendering instance and safe
	//    draws in it (m_epoch_clean / m_epoch_instance): their only writes are this instance's
	//    attachments, which D2 accesses only as attachments, and attachment accesses of draws in
	//    one rendering instance are ordered by rasterization order without barriers;
	//  - D2 is safe: it writes only this instance's attachments (no WAR against reads of P) and
	//    samples none of them (no RAW against P's attachment writes);
	//  - the batch has no buffer/image barriers: layout transitions must precede D2.
	return BarrierSinkEnabled() && m_draw_scope && m_draw_safe && m_rendering && m_epoch_clean &&
	       m_epoch_instance != 0 && m_epoch_instance == m_rendering_serial &&
	       m_pending.has_memory && m_pending.images.empty() && m_pending.buffers.empty() &&
	       m_pending.uploads.empty();
}

bool CommandBuffer::CanSinkDrawWrites() const {
	// A draw continuing its rendering instance may pass pending barriers that consist only of
	// the post-draw shader-write requests of earlier draws. Every such request was made inside
	// this instance: a new instance records the pending batch before it begins. Any other origin
	// (guest synchronization, layout transitions, buffer barriers, indirect arguments, GDS)
	// keeps the ordinary rules.
	return DrawWriteSinkEnabled() && m_draw_scope && m_rendering && m_pending.has_memory &&
	       m_pending.images.empty() && m_pending.buffers.empty() && m_pending.uploads.empty() &&
	       m_pending.origins == OriginBit(BarrierOrigin::ShaderWrite);
}

void CommandBuffer::NoteDrawRecorded() const {
	m_recorded_since_flush = true;
	if (!(m_draw_scope && m_draw_safe && m_rendering && m_epoch_instance == m_rendering_serial)) {
		m_epoch_clean = false;
	}
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	m_bound_pipelines = {};
	for (auto& state: m_descriptor_states) {
		state.layout    = nullptr;
		state.bound_set = nullptr;
	}
	for (auto& epoch: m_descriptor_epochs) {
		epoch++;
	}
	// Push constants are undefined at the start of a command buffer.
	m_push_constants.valid = false;
	// Commands of other submissions can precede this buffer on the queue: no epoch, no elision.
	ResetBarrierState();
	m_direct_window = false;
	if (m_recorder != nullptr) {
		m_recorder->SetWindowOpen(false);
		// KYTY_CP_RECORDER: the recorder begins the native buffer (CommandScheduler::BeginCommand
		// encodes the Begin packet right after this).
		return;
	}
	auto buffer = StateHandle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	FlushBarriers();
	auto buffer = StateHandle();
	if (GpuBreadcrumbs::Active()) {
		// The command buffer's last breadcrumb: at the bottom of the pipe, everything in it completed.
		const auto tick = m_context.GetCommandScheduler().CurrentTick();
		GpuBreadcrumbs::Write(buffer, GpuBreadcrumbs::Queue::Graphics,
		                      GpuBreadcrumbs::Note(GpuBreadcrumbs::Queue::Graphics,
		                                           {.op = GpuBreadcrumbs::OpCommandBufferEnd, .tick = tick}));
	}

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

static size_t BindingPointIndex(vk::PipelineBindPoint point) {
	EXIT_IF(point != vk::PipelineBindPoint::eGraphics && point != vk::PipelineBindPoint::eCompute);
	return point == vk::PipelineBindPoint::eCompute ? 1u : 0u;
}

void CommandBuffer::BindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline) {
	auto& current = m_bound_pipelines[BindingPointIndex(point)];
	if (Common::RendererBatchEnabled() && current == pipeline) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::PipelineBindsAvoided);
		return;
	}
	// State commands: not ordered by barriers, so not a batch flush point (render.h).
	StateSink().bindPipeline(point, pipeline);
	current = pipeline;
}

void CommandBuffer::InvalidateDescriptors(vk::PipelineBindPoint point) {
	auto& state     = m_descriptor_states[BindingPointIndex(point)];
	state.layout    = nullptr;
	state.bound_set = nullptr;
	m_descriptor_epochs[BindingPointIndex(point)]++;
}

void CommandBuffer::BindDescriptorSet(vk::PipelineBindPoint point, vk::PipelineLayout layout,
                                      vk::DescriptorSet set) {
	auto& state = m_descriptor_states[BindingPointIndex(point)];
	EXIT_IF(set == nullptr || layout == nullptr);
	if (DescriptorSetReuseEnabled() && state.bound_set == set && state.layout == layout) {
		// Still bound as set 0 with this layout: nothing since disturbed it (every other bind and
		// every push descriptor update of this bind point passes through this class), and binding
		// a pipeline never disturbs descriptor sets.
		Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorSetBindsAvoided);
		return;
	}
	StateSink().bindDescriptorSets(point, layout, 0, 1, &set, 0, nullptr);
	m_descriptor_epochs[BindingPointIndex(point)]++;
	state.layout    = DescriptorSetReuseEnabled() ? layout : nullptr;
	state.bound_set = DescriptorSetReuseEnabled() ? set : nullptr;
	state.writes.clear();
	state.buffers.clear();
	state.images.clear();
}

void CommandBuffer::PushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages,
                                  uint32_t size, const void* data) {
	auto& shadow = m_push_constants;
	// Push-constant values persist in the command buffer across pipeline binds; values set with
	// this very layout are valid for every pipeline created with it.
	if (PushConstantShadowEnabled() && shadow.valid && shadow.layout == layout &&
	    shadow.stages == stages && shadow.size == size &&
	    std::memcmp(shadow.dwords.data(), data, size) == 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::PushConstantUpdatesAvoided);
		return;
	}
	StateSink().pushConstants(layout, stages, 0, size, data);
	Profiler::CountFrameEvent(Profiler::FrameEvent::PushConstantUpdates);
	shadow.valid = PushConstantShadowEnabled() && size <= sizeof(shadow.dwords);
	if (shadow.valid) {
		shadow.layout = layout;
		shadow.stages = stages;
		shadow.size   = size;
		std::memcpy(shadow.dwords.data(), data, size);
	}
}

int32_t CommandBuffer::PushDescriptors(vk::PipelineBindPoint point, vk::PipelineLayout layout,
                                       uint32_t set, uint32_t count,
                                       const vk::WriteDescriptorSet* writes, bool known_miss) {
	auto& state = m_descriptor_states[BindingPointIndex(point)];
	if (known_miss) {
		// The comparison below could only miss. No copy is kept for the next update either: one
		// equal to this (its upload deduplicated to the same range) is then recorded again, which
		// is only redundant; the Sky Garden start avoids no push at all (DescriptorPushesAvoided).
		StateSink().pushDescriptorSetKHR(point, layout, set, count, writes);
		Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorPushes);
		state.layout    = nullptr;
		state.bound_set = nullptr;
		return PushMissFresh;
	}
	bool supported = Common::RendererBatchEnabled() && set == 0;
	size_t buffer_count = 0, image_count = 0;
	for (uint32_t i = 0; supported && i < count; ++i) {
		const auto& write = writes[i];
		const bool buffer_type = write.descriptorType == vk::DescriptorType::eStorageBuffer ||
		                         write.descriptorType == vk::DescriptorType::eUniformBuffer;
		const bool image_type = write.descriptorType == vk::DescriptorType::eSampler ||
		                        write.descriptorType == vk::DescriptorType::eSampledImage ||
		                        write.descriptorType == vk::DescriptorType::eStorageImage ||
		                        write.descriptorType == vk::DescriptorType::eCombinedImageSampler;
		supported = write.pNext == nullptr && write.pTexelBufferView == nullptr &&
		            ((buffer_type && write.pBufferInfo != nullptr) ||
		             (image_type && write.pImageInfo != nullptr));
		buffer_count += buffer_type ? write.descriptorCount : 0u;
		image_count += image_type ? write.descriptorCount : 0u;
	}
	int32_t result = PushMissState;
	if (supported && state.layout == layout && state.bound_set == nullptr) {
		result = state.writes.size() == count && state.buffers.size() == buffer_count &&
		                 state.images.size() == image_count
		             ? PushAvoided
		             : PushMissShape;
		size_t buffer_index = 0, image_index = 0;
		for (uint32_t i = 0; result == PushAvoided && i < count; ++i) {
			const auto& write = writes[i];
			const auto& old   = state.writes[i];
			if (write.dstBinding != old.dstBinding || write.dstArrayElement != old.dstArrayElement ||
			    write.descriptorCount != old.descriptorCount ||
			    write.descriptorType != old.descriptorType) {
				result = PushMissShape;
				break;
			}
			const bool buffer_type = write.descriptorType == vk::DescriptorType::eStorageBuffer ||
			                         write.descriptorType == vk::DescriptorType::eUniformBuffer;
			for (uint32_t j = 0; j < write.descriptorCount; ++j) {
				const bool equal = buffer_type ? write.pBufferInfo[j] == state.buffers[buffer_index++]
				                               : write.pImageInfo[j] == state.images[image_index++];
				if (!equal) {
					result = static_cast<int32_t>(i);
					break;
				}
			}
		}
	}
	if (result == PushAvoided) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorPushesAvoided);
		return result;
	}
	StateSink().pushDescriptorSetKHR(point, layout, set, count, writes);
	m_descriptor_epochs[BindingPointIndex(point)]++;
	Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorPushes);
	state.layout    = nullptr;
	state.bound_set = nullptr;
	if (!supported) return result;
	state.writes.assign(writes, writes + count);
	state.buffers.clear();
	state.images.clear();
	state.buffers.reserve(buffer_count);
	state.images.reserve(image_count);
	for (auto& write: state.writes) {
		const bool buffer_type = write.descriptorType == vk::DescriptorType::eStorageBuffer ||
		                         write.descriptorType == vk::DescriptorType::eUniformBuffer;
		if (buffer_type) state.buffers.insert(state.buffers.end(), write.pBufferInfo,
		                                     write.pBufferInfo + write.descriptorCount);
		else state.images.insert(state.images.end(), write.pImageInfo,
		                         write.pImageInfo + write.descriptorCount);
		write.pBufferInfo = nullptr;
		write.pImageInfo = nullptr;
	}
	state.layout = layout;
	return result;
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
	const bool breadcrumbs = GpuBreadcrumbs::Active();
	if ((m_graphics.diagnostic_checkpoints_enabled || breadcrumbs) && m_buffer) {
		const DiagnosticCheckpoint checkpoint {
		    .op        = op,
		    .submit_id = submit_id,
		    .arg0      = arg0,
		    .arg1      = arg1,
		    .arg2      = arg2,
		    .arg3      = arg3,
		    .arg4      = arg4,
		    .tick      = m_context.GetCommandScheduler().CurrentTick(),
		    .vs        = m_shaders != nullptr ? m_shaders->GetVs().es_regs.data_addr : 0,
		    .ps        = m_shaders != nullptr ? m_shaders->GetPs().ps_regs.data_addr : 0,
		    .cs        = m_shaders != nullptr ? m_shaders->GetCs().cs_regs.data_addr : 0};
		if (m_graphics.diagnostic_checkpoints_enabled) {
			const auto* marker = RecordDiagnosticCheckpoint(checkpoint);
			if (marker != nullptr) {
				Handle().setCheckpointNV(marker);
			}
		}
		if (breadcrumbs) {
			// KYTY_GPU_BREADCRUMBS: a state-like command (it orders nothing the batcher tracks), so it
			// neither flushes pending barriers nor ends the rendering instance.
			const auto marker = GpuBreadcrumbs::Note(GpuBreadcrumbs::Queue::Graphics, checkpoint);
			const auto sink   = StateSink();
			sink.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, GpuBreadcrumbs::Buffer(),
			                          GpuBreadcrumbs::TopOffset(GpuBreadcrumbs::Queue::Graphics), marker);
			sink.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, GpuBreadcrumbs::Buffer(),
			                          GpuBreadcrumbs::BottomOffset(GpuBreadcrumbs::Queue::Graphics), marker);
		}
	}
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	const auto count_control = GetRegisters().GetDepthCountControl();
	auto&      occlusion     = m_context.GetOcclusionCounter();
	// A DB_COUNT_CONTROL change matters only when this instance or the next one is counted
	// (KYTY_OCCLUSION_GATE, occlusion.h): with neither counted, no sample can reach a guest value.
	const bool same_control =
	    m_occlusion_control == count_control ||
	    (OcclusionCounter::GateEnabled() && !occlusion.Active() && !occlusion.WouldCount(count_control));
	const bool same_instance = m_rendering && m_render_state == state && same_control;
	if (same_instance) {
		m_occlusion_control = count_control;
	}
	bool barrier_split = false;
	if (!BarrierBatchEnabled()) {
		if (same_instance) {
			return;
		}
	} else if (same_instance) {
		// The instance continues: a depth access kept by this draw needs no barrier.
		if (m_pending.Empty()) {
			m_feedback_keep.reset();
			NoteDrawRecorded();
			return;
		}
		if (CanSinkPending()) {
			CountBatch(GpuOpProfiler::BarrierBatchEvent::Sunk);
			m_feedback_keep.reset();
			NoteDrawRecorded();
			return;
		}
		if (CanSinkDrawWrites()) {
			CountBatch(GpuOpProfiler::BarrierBatchEvent::DrawWriteSinks);
			m_feedback_keep.reset();
			NoteDrawRecorded();
			return;
		}
		// The pending barrier cannot move past this draw: end the instance to record it.
		CountBatch(GpuOpProfiler::BarrierBatchEvent::RenderSplits);
		barrier_split = true;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	if (m_rendering) {
		// Attribution of the end (GpuOps.EndRendering.<site>): a pending barrier that could not
		// be sunk, or different targets/state/occlusion control.
		if (barrier_split) {
			KYTY_GPU_OP_SITE("render.barrier_split");
			EndRendering();
		} else if (m_render_state == state) {
			KYTY_GPU_OP_SITE("render.occlusion_control");
			EndRendering();
		} else {
			KYTY_GPU_OP_SITE("render.state_change");
			EndRendering();
		}
	}
	if (m_feedback_keep) {
		// Unreachable (NoteFeedbackKeep needs an active instance, and ending it queues this
		// barrier); kept so a new instance can never begin without the left-out ordering.
		m_pending.images.push_back(*m_feedback_keep);
		m_pending.origins |= OriginBit(BarrierOrigin::Image);
		m_feedback_keep.reset();
	}
	if (BarrierBatchEnabled()) {
		FlushBarriers();
	}
	// Query bookkeeping recorded by the occlusion counter below does not access memory that
	// draws access; it is neither a flush point nor a foreign command.
	++m_internal_recording;
	m_context.GetOcclusionCounter().Prepare(count_control);

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	StateSink().beginRendering(rendering);
	m_context.GetOcclusionCounter().Begin();
	--m_internal_recording;
	if (HangTrace::Enabled() || OcclusionCounter::SyncProxyDumps()) {
		const auto& db        = GetRegisters().GetDepthRenderTarget();
		auto&       occlusion = m_context.GetOcclusionCounter();
		// A query a split dump begins inside this instance reports the same scope
		// (KYTY_OCCLUSION_SPLIT).
		occlusion.NoteInstance(db.z_read_base_addr, state.width, state.height,
		                       state.num_color_attachments, depth_stencil.has_depth,
		                       static_cast<uint32_t>(db.z_info.format));
		if (occlusion.Active()) {
			occlusion.NoteScope(db.z_read_base_addr, state.width, state.height,
			                    state.num_color_attachments, depth_stencil.has_depth,
			                    static_cast<uint32_t>(db.z_info.format));
		}
	}
	m_render_state = state;
	m_rendering    = true;
	++m_rendering_serial;
	m_occlusion_control = count_control;
	if (BarrierBatchEnabled()) {
		// The attachment load operations of the first instance begun after a full barrier belong
		// to the sinking epoch; a second instance does not (its loads may read what the first
		// one stored).
		if (m_epoch_clean && m_epoch_instance == 0 && m_draw_scope) {
			m_epoch_instance = m_rendering_serial;
		} else {
			m_epoch_clean = false;
		}
		NoteDrawRecorded();
	}
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	GpuOpProfiler::CountEndRendering();
	// The occlusion counter may also record a query reduction here (Accumulate): foreign work.
	++m_internal_recording;
	m_context.GetOcclusionCounter().End();
	StateSink().endRendering();
	m_rendering    = false;
	m_render_state = {};
	m_context.GetOcclusionCounter().Accumulate();
	--m_internal_recording;
	if (BarrierBatchEnabled()) {
		NoteForeignCommand();
	}
	if (m_feedback_keep) {
		// The instance a kept depth access relied on ended before the draw that kept it: order its
		// attachment store and reads before whatever follows (KYTY_DEPTH_FEEDBACK_KEEP). A barrier
		// of this image already pending was built from the kept state and orders the same.
		const auto image = m_feedback_keep->image;
		if (std::ranges::none_of(m_pending.images,
		                         [image](const auto& pending) { return pending.image == image; })) {
			m_pending.images.push_back(*m_feedback_keep);
			m_pending.origins |= OriginBit(BarrierOrigin::Image);
		}
		m_feedback_keep.reset();
	}
}

} // namespace Libs::Graphics
