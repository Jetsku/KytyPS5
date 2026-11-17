#include "common/assert.h"
#include "common/staticOnce.h"
#include "common/common.h"
#include "common/hangTrace.h"
#include "common/liveSwitch.h"
#include "common/onceValue.h"
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
#include "graphics/host_gpu/renderer/drawPrep/prepStats.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#if defined(_MSC_VER)
#include <intrin.h>
#define KYTY_RECORDER_CALLER() _ReturnAddress()
#else
#define KYTY_RECORDER_CALLER() __builtin_return_address(0)
#endif

namespace Libs::Graphics {

// KYTY_CP_SINK_ROUTES (default 1, live): CommandBuffer::HelperSink() encodes for the recorder; 0
// drains it and records natively, as the helpers' former Handle() did.
static Live::Switch g_sink_routes("KYTY_CP_SINK_ROUTES", Live::ParseDefaultOn);
// KYTY_DISPATCH_OVERLAP (render.h DispatchOverlapEnabled): live; default on.
static Live::Switch g_dispatch_overlap("KYTY_DISPATCH_OVERLAP", Live::ParseDefaultOn);
// KYTY_DISPATCH_OVERLAP_UPLOADS (render.h DispatchOverlapUploadsEnabled): live; default off.
static Live::Switch g_dispatch_overlap_uploads("KYTY_DISPATCH_OVERLAP_UPLOADS", Live::ParseDefaultOff);
// KYTY_PM4_BARRIERS=guest (render.h GuestSyncBarriersEnabled): live; default off.
static Live::Switch g_pm4_guest_barriers("KYTY_PM4_BARRIERS", [](const char* value) -> int64_t {
	return value != nullptr && std::strcmp(value, "guest") == 0 ? 1 : 0;
});

// KYTY_GPU_HAZARD / KYTY_GPU_HAZARD_COUNT (render.h GpuHazardMode): live; default 0.
static Live::Switch g_gpu_hazard("KYTY_GPU_HAZARD", [](const char* value) -> int64_t {
	return value != nullptr ? std::clamp<int64_t>(std::strtoll(value, nullptr, 10), 0, 4) : 0;
});
static Live::Switch g_gpu_hazard_count("KYTY_GPU_HAZARD_COUNT", Live::ParseDefaultOff);

namespace HazCount {
enum : uint32_t {
	Disp,        // direct dispatches seen with KYTY_GPU_HAZARD != 0
	Ovl,         // the dispatch-overlap chain already dropped the pending request
	Impre,       // accesses not all in bindings
	Inval,       // no intact chain (something else recorded since the last reset)
	Raw,         // read-after-write overlap with the chain (guest ranges)
	War,         // write-after-read
	Waw,         // write-after-write
	HostMis,     // guest ranges free, host ranges overlap (the verify count)
	FreeEmpty,   // free, nothing pending
	FreeMem,     // free, only ShaderAccess/guest memory dependencies pending
	FreeSide,    // free, plus the dispatch's own image/buffer barriers or uploads
	FreeForeign, // free, but other image/buffer barriers or uploads pending
	FreeOther,   // free, but memory dependencies of other origins pending
	Skips,       // the dispatch skipped (mode >= 2)
	Flushes,     // batches recorded
	Full,        // ... with a full ALL -> ALL memory dependency
	ImageBarriers, // image barriers recorded in batches
	ImageOnly,   // batches of image barriers only
	Begins,      // rendering instances begun
	ImgRoLayout, // image barrier requests: read-only before and after, layout change only
	ImgWr,       // write before, read after
	ImgRw,       // read before, write after
	ImgWw,       // write before and after
	ImgSameLayout, // ... of all requests: old layout == new layout
	ISplit,      // KYTY_INDIRECT_ARGS_RANGES (render.h HazIndirect)
	ISkip,
	IConf,
	IImp,
	IHost,
	FullGuest,   // full batches whose memory origins include a guest barrier
	FullDispatch, // ... include ShaderAccess or ShaderWriteHazard
	FullDraw,    // ... include a draw's ShaderWrite
	FullOther,   // ... include none of these
	InvalNone,   // inval split: no chain since the command buffer began / an invalidation
	Count
};
constexpr const char* Names[Count] = {"disp", "ovl", "impre", "inval", "raw", "war", "waw", "hostmis", "fempty",
                                      "fmem", "fside", "fforeign", "fother", "skip", "flush", "full", "imgb",
                                      "imgonly", "rbeg", "iro", "iwr", "irw", "iww", "isame", "isplit", "iskip", "iconf",
                                      "iimp", "ihost", "fguest", "fdisp", "fdraw", "fother2",
                                      "invnone"};
std::atomic<uint64_t> g_values[Count] {};
inline void Add(uint32_t index, uint64_t amount = 1) {
	g_values[index].fetch_add(amount, std::memory_order_relaxed);
}
std::string Line() {
	uint64_t values[Count];
	uint64_t any = 0;
	for (uint32_t i = 0; i < Count; i++) {
		values[i] = g_values[i].load(std::memory_order_relaxed);
		any |= values[i];
	}
	// emitsplit measurement: the recorder thread's spin (TSC ticks; the cpidle tsc converts) and the
	// KYTY_EMIT_SPLIT counters (resolved draws expanded, verify-only records checked, mismatches).
	const auto r        = CommandStream::ResolvedDrawPublished();
	uint64_t   mismatch = 0;
	for (const auto m: r.mismatches) {
		mismatch += m;
	}
	std::string text = " rec [busy_us " + std::to_string(CommandRecorder::TotalBusyNs() / 1000u) + ", spin_tsc " +
	                   std::to_string(DrawPrep::PrepStats::g_spin.recorder.load(std::memory_order_relaxed)) +
	                   ", es " + std::to_string(r.records) + ", espush " + std::to_string(r.pushes) + ", esvchk " +
	                   std::to_string(r.verify_checks) + ", esvmis " + std::to_string(mismatch) + "]";
	if (any == 0) {
		return text;
	}
	text += " haz [";
	for (uint32_t i = 0; i < Count; i++) {
		text += Names[i];
		text += ' ';
		text += std::to_string(values[i]);
		text += i + 1 < Count ? ", " : "]";
	}
	return text;
}
[[maybe_unused]] const bool g_line_registered = (Live::SetLineSuffixProvider(&Line), true);
} // namespace HazCount

int64_t GpuHazardMode() {
	return BarrierBatchEnabled() ? g_gpu_hazard.Get() : 0;
}

bool GpuHazardCountEnabled() {
	return g_gpu_hazard_count.On();
}

void GpuHazardCountIndirect(HazIndirect what) {
	if (g_gpu_hazard_count.On()) {
		HazCount::Add(HazCount::ISplit + static_cast<uint32_t>(what));
	}
}

// KYTY_GUEST_BARRIER_SCOPE (render.h GuestBarrierScope): live; default every kind full.
int64_t ParseGuestBarrierScope(const char* value) {
	if (value == nullptr || value[0] == '\0') {
		return 0;
	}
	constexpr auto Kinds = static_cast<uint32_t>(GuestBarrierKind::Count);
	auto set = [](int64_t bits, uint32_t kind, uint32_t mode) {
		return (bits & ~(int64_t {3} << (2 * kind))) | (int64_t {mode} << (2 * kind));
	};
	auto mode_of = [](std::string_view text) -> int {
		if (text == "full") return 0;
		if (text == "narrow") return 1;
		if (text == "drop") return 2;
		return -1;
	};
	int64_t          bits = 0;
	std::string_view rest(value);
	while (!rest.empty()) {
		const auto comma = rest.find(',');
		auto       item  = rest.substr(0, comma);
		rest             = comma == std::string_view::npos ? std::string_view {} : rest.substr(comma + 1);
		if (item.empty()) {
			continue;
		}
		const auto equals = item.find('=');
		if (equals == std::string_view::npos) {
			// Shorthand: the four EVENT_WRITE kinds.
			const int mode = mode_of(item);
			if (mode < 0) {
				return -1;
			}
			for (uint32_t kind = 0; kind <= static_cast<uint32_t>(GuestBarrierKind::MetaWriteback); kind++) {
				bits = set(bits, kind, static_cast<uint32_t>(mode));
			}
			continue;
		}
		const auto name = item.substr(0, equals);
		const int  mode = mode_of(item.substr(equals + 1));
		if (mode < 0) {
			return -1;
		}
		static constexpr std::string_view Names[Kinds] = {"cs", "gfx", "cbdb", "meta", "rel", "sync"};
		bool found = false;
		for (uint32_t kind = 0; kind < Kinds; kind++) {
			if (name == Names[kind] || name == "all") {
				bits  = set(bits, kind, static_cast<uint32_t>(mode));
				found = true;
			}
		}
		if (!found) {
			return -1;
		}
	}
	return bits;
}

static Live::Switch g_guest_barrier_scope("KYTY_GUEST_BARRIER_SCOPE", [](const char* value) -> int64_t {
	const auto bits = ParseGuestBarrierScope(value);
	return bits < 0 ? 0 : bits; // malformed: every kind full
});

GuestBarrierMode GuestBarrierScope(GuestBarrierKind kind) {
	const auto bits = g_guest_barrier_scope.Get();
	return static_cast<GuestBarrierMode>((bits >> (2 * static_cast<uint32_t>(kind))) & 3);
}

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

// KYTY_DEPTH_RO_STORE_NONE=1 (default off; chenxiao07/KytyPS5 ef46df82f): a depth or stencil
// aspect the rendering's attachment layout makes read-only is stored with STORE_OP_NONE. The draws
// of such a pass cannot write it, and STORE_OP_STORE is a write for synchronization (a pass that
// tests against the depth buffer it also samples, then the next transition of it: a
// write-after-write the barrier made for the shader reads did not cover) and a write-back for the
// driver.
static vk::AttachmentStoreOp DepthStoreOp(vk::ImageLayout layout, bool stencil) {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_DEPTH_RO_STORE_NONE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	if (enabled) {
		switch (layout) {
			case vk::ImageLayout::eDepthStencilReadOnlyOptimal: return vk::AttachmentStoreOp::eNone;
			case vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal:
			case vk::ImageLayout::eDepthReadOnlyOptimal:
				if (!stencil) return vk::AttachmentStoreOp::eNone;
				break;
			case vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal:
			case vk::ImageLayout::eStencilReadOnlyOptimal:
				if (stencil) return vk::AttachmentStoreOp::eNone;
				break;
			default: break;
		}
	}
	return vk::AttachmentStoreOp::eStore;
}

bool BarrierBatchEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [&] {
		return EnvSwitch("KYTY_BARRIER_BATCH", true);
	}));
	return enabled;
}

bool BarrierSinkEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [&] {
		return BarrierBatchEnabled() && EnvSwitch("KYTY_BARRIER_SINK", true);
	}));
	return enabled;
}

bool UploadBatchEnabled() {
	static constinit Common::StaticOnce<bool> enabled_once;
	const bool& enabled = enabled_once.Get([&] { return BarrierBatchEnabled() && EnvSwitch("KYTY_UPLOAD_BATCH", true); });
	return enabled;
}

bool DrawWriteSinkEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [&] {
		return BarrierBatchEnabled() && EnvSwitch("KYTY_DRAW_WRITE_SINK", true);
	}));
	return enabled;
}

bool DepthFeedbackKeepEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [&] {
		return BarrierBatchEnabled() && EnvSwitch("KYTY_DEPTH_FEEDBACK_KEEP", true);
	}));
	return enabled;
}

bool DepthFeedbackLazyEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [&] {
		return DepthFeedbackKeepEnabled() && EnvSwitch("KYTY_DEPTH_FEEDBACK_LAZY", true);
	}));
	return enabled;
}

bool DepthLayoutStableEnabled() {
	static const bool enabled = EnvSwitch("KYTY_DEPTH_LAYOUT_STABLE", true);
	return enabled;
}

bool PushConstantShadowEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [&] {
		return EnvSwitch("KYTY_PUSH_CONSTANT_SHADOW", true);
	}));
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

// Batches with exactly one GuestBarrierKind: guest-only, and mixed with other origins.
constinit GpuOpProfiler::Site g_guest_kind_sites[2][static_cast<size_t>(GuestBarrierKind::Count)] = {
    {GpuOpProfiler::Site {"batch.guest.cs_flush"}, GpuOpProfiler::Site {"batch.guest.gfx_flush"},
     GpuOpProfiler::Site {"batch.guest.cbdb_wb"}, GpuOpProfiler::Site {"batch.guest.meta_wb"},
     GpuOpProfiler::Site {"batch.guest.release_mem"}, GpuOpProfiler::Site {"batch.guest.pm4_sync"}},
    {GpuOpProfiler::Site {"batch.mixed.guest.cs_flush"}, GpuOpProfiler::Site {"batch.mixed.guest.gfx_flush"},
     GpuOpProfiler::Site {"batch.mixed.guest.cbdb_wb"}, GpuOpProfiler::Site {"batch.mixed.guest.meta_wb"},
     GpuOpProfiler::Site {"batch.mixed.guest.release_mem"}, GpuOpProfiler::Site {"batch.mixed.guest.pm4_sync"}},
};

GpuOpProfiler::Site& BatchSite(uint32_t origins, uint32_t guest_kinds) {
	constexpr uint32_t GuestBit = 1u << static_cast<uint32_t>(BarrierOrigin::Guest);
	if ((origins & GuestBit) != 0 && guest_kinds != 0 && std::has_single_bit(guest_kinds)) {
		const auto kind = static_cast<size_t>(std::countr_zero(guest_kinds));
		if (kind < static_cast<size_t>(GuestBarrierKind::Count)) {
			return g_guest_kind_sites[origins == GuestBit ? 0 : 1][kind];
		}
	}
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
	SettleEncode();
	m_shadow.InvalidatePushConstants();
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
	SettleEncode();
	m_shadow.InvalidatePushConstants();
	return CommandSink(*this);
}

CommandSink CommandBuffer::HelperSink() const {
	if (Encoding() && !g_sink_routes.On()) {
		// The former Handle() path: drain, then record natively in the direct window.
		(void)Handle();
	}
	return Sink();
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
	m_record_reset  = true; // KYTY_EMIT_SPLIT: native commands follow
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
	m_overlap_open         = false;
	m_overlap_upload_fenced = false;
	m_upload_narrow         = false;
	m_dispatch_prep         = false;
	// KYTY_GPU_HAZARD: commands of earlier submissions precede this buffer; no chain until a reset.
	++m_haz_foreign;
	m_haz_chain.Invalidate();
	m_haz_stash_valid = false;
}

bool DispatchOverlapEnabled() {
	return BarrierBatchEnabled() && g_dispatch_overlap.Get() != 0;
}

bool DispatchOverlapUploadsEnabled() {
	return DispatchOverlapEnabled() && UploadBatchEnabled() && g_dispatch_overlap_uploads.Get() != 0;
}

bool GuestSyncBarriersEnabled() {
	return g_pm4_guest_barriers.Get() != 0;
}

void CommandBuffer::NoteGuestDispatch(bool lds) const {
	// Called right after the dispatch's ShaderAccess request: nothing may have been recorded or
	// requested since the dispatch except that request.
	m_overlap_open = BarrierBatchEnabled() && m_pending.has_memory && m_pending.images.empty() &&
	                 m_pending.buffers.empty() && m_pending.uploads.empty() &&
	                 m_pending.origins == OriginBit(BarrierOrigin::ShaderAccess);
	m_overlap_lds = lds;
}

bool CommandBuffer::KeepUploadsForDispatch() const {
	return m_dispatch_prep && m_overlap_open && DispatchOverlapUploadsEnabled() &&
	       m_pending.has_memory && m_pending.images.empty() && m_pending.buffers.empty() &&
	       (m_pending.origins & ~(OriginBit(BarrierOrigin::ShaderAccess) | OriginBit(BarrierOrigin::Upload))) == 0;
}

bool CommandBuffer::TakeDispatchOverlap(bool lds, bool take) const {
	const bool open = m_overlap_open;
	m_overlap_open  = false;
	// KYTY_DISPATCH_OVERLAP_UPLOADS: upload copies queued for this dispatch (CPU-written pages its
	// bindings need) do not end the chain.
	const bool with_uploads = !m_pending.uploads.empty() && DispatchOverlapUploadsEnabled();
	const uint32_t allowed  = OriginBit(BarrierOrigin::ShaderAccess) |
	                         (with_uploads ? OriginBit(BarrierOrigin::Upload) : 0u);
	if (!open || lds || m_overlap_lds || !m_pending.has_memory || !m_pending.images.empty() ||
	    !m_pending.buffers.empty() || (!m_pending.uploads.empty() && !with_uploads) ||
	    m_pending.origins != allowed) {
		m_overlap_upload_fenced = false;
		return false;
	}
	if (take) {
		// The next dispatch's own ShaderAccess request covers the dropped one: its first scope is
		// every earlier compute shader access, the dropped dispatch's included.
		if (with_uploads) {
			// The first upload of a chain keeps its full pre-copy barrier (everything earlier,
			// the chain's first dispatch included, before the copies). Later ones only wait for
			// earlier copies (RecordPendingUploads): the CPU wrote those pages while no guest
			// synchronization separated the chain's dispatches, which on the console run
			// concurrently with such writes too. The copies' post barrier still orders them
			// before this dispatch.
			m_pending.memory     = vk::MemoryBarrier2 {};
			m_pending.has_memory = false;
			m_pending.origins    = OriginBit(BarrierOrigin::Upload);
			m_pending.memory_origins = 0;
			m_upload_narrow         = m_overlap_upload_fenced;
			m_overlap_upload_fenced = true;
		} else {
			m_pending.Clear();
		}
	}
	return true;
}

template <typename Handle>
static uint64_t HandleBits(Handle handle) {
	return std::bit_cast<uint64_t>(static_cast<typename Handle::CType>(handle));
}

bool CommandBuffer::DecideDispatchHazard(const DispatchAccess& access, bool overlap) const {
	const auto mode = GpuHazardMode();
	if (mode == 0) {
		return false;
	}
	const bool count = GpuHazardCountEnabled();
	auto       note  = [count](uint32_t index) {
        if (count) {
            HazCount::Add(index);
        }
	};
	note(HazCount::Disp);
	if (overlap) {
		note(HazCount::Ovl);
		return false;
	}
	if (!access.precise) {
		note(HazCount::Impre);
		return false;
	}
	if (!m_haz_chain.Intact(m_haz_foreign) || m_haz_chain.guest.Poisoned()) {
		note(HazCount::Inval);
		if (!m_haz_chain.valid) {
			note(HazCount::InvalNone);
		}
		return false;
	}
	switch (m_haz_chain.guest.Check(access.guest)) {
		case Hazard::Verdict::Raw: note(HazCount::Raw); return false;
		case Hazard::Verdict::War: note(HazCount::War); return false;
		case Hazard::Verdict::Waw: note(HazCount::Waw); return false;
		case Hazard::Verdict::Free: break;
	}
	// Verify: the conservative host-range check of the same bindings (a mismatch keeps the barrier).
	if (m_haz_chain.host.Check(access.host) != Hazard::Verdict::Free) {
		note(HazCount::HostMis);
		return false;
	}
	if (m_pending.Empty()) {
		note(HazCount::FreeEmpty);
		if (mode < 2) {
			return false;
		}
		note(HazCount::Skips);
		return true;
	}
	constexpr uint32_t Allowed = OriginBit(BarrierOrigin::ShaderAccess) | OriginBit(BarrierOrigin::Guest) |
	                             OriginBit(BarrierOrigin::ShaderWriteHazard);
	if (m_pending.has_memory && (m_pending.memory_origins & ~Allowed) != 0) {
		note(HazCount::FreeOther);
		return false;
	}
	const bool side = !m_pending.images.empty() || !m_pending.buffers.empty() || !m_pending.uploads.empty();
	auto       own  = [&access](uint64_t handle) {
        return std::ranges::find(access.handles, handle) != access.handles.end();
	};
	if (side) {
		bool foreign = false;
		for (const auto& barrier: m_pending.images) {
			foreign |= !own(HandleBits(barrier.image));
		}
		for (const auto& barrier: m_pending.buffers) {
			foreign |= !own(HandleBits(barrier.buffer));
		}
		uint32_t destinations = 0;
		for (size_t i = 0; i < m_pending.uploads.size(); i++) {
			const auto destination = m_pending.uploads[i].destination;
			foreign |= !own(HandleBits(destination));
			bool seen = false;
			for (size_t j = 0; j < i; j++) {
				seen |= m_pending.uploads[j].destination == destination;
			}
			destinations += seen ? 0u : 1u;
		}
		// More destinations make RecordPendingUploads merge the post-copy barrier into the memory
		// dependency, which the dispatch needs before it.
		if (foreign || destinations > 8) {
			note(HazCount::FreeForeign);
			return false;
		}
		note(HazCount::FreeSide);
		if (mode < 3) {
			return false;
		}
	} else {
		note(HazCount::FreeMem);
		if (mode < 2) {
			return false;
		}
	}
	// Take the memory dependency off the batch; NoteHazardDispatch re-queues it after the dispatch.
	if (m_pending.has_memory) {
		if (m_haz_stash_valid) {
			m_haz_stash.srcStageMask |= m_pending.memory.srcStageMask;
			m_haz_stash.srcAccessMask |= m_pending.memory.srcAccessMask;
			m_haz_stash.dstStageMask |= m_pending.memory.dstStageMask;
			m_haz_stash.dstAccessMask |= m_pending.memory.dstAccessMask;
		} else {
			m_haz_stash = m_pending.memory;
		}
		m_haz_stash_valid = true;
		m_haz_stash_origins |= m_pending.memory_origins;
		m_haz_stash_kinds |= m_pending.guest_kinds;
		m_pending.origins &= ~m_pending.memory_origins;
		m_pending.memory         = vk::MemoryBarrier2 {};
		m_pending.has_memory     = false;
		m_pending.memory_origins = 0;
		m_pending.guest_kinds    = 0;
	}
	if (side) {
		// The dispatch's own transitions, buffer barriers and uploads, each complete in itself, before it.
		FlushBarriersImpl(true);
	}
	note(HazCount::Skips);
	return true;
}

void CommandBuffer::NoteHazardDispatch(const DispatchAccess& access) const {
	if (const auto mode = GpuHazardMode(); mode != 0) {
		m_haz_chain.NoteDispatch(m_haz_foreign - 1, m_haz_foreign, access.precise, access.guest, access.host,
		                         mode >= 4);
	} else if (m_haz_chain.valid) {
		m_haz_chain.Invalidate();
	}
	if (m_haz_stash_valid) {
		// The memory dependency taken off before the dispatch, queued again after it: every later
		// command sees it as it would have (it also covers the dispatch, which recorded no barrier).
		if (m_pending.has_memory) {
			m_pending.memory.srcStageMask |= m_haz_stash.srcStageMask;
			m_pending.memory.srcAccessMask |= m_haz_stash.srcAccessMask;
			m_pending.memory.dstStageMask |= m_haz_stash.dstStageMask;
			m_pending.memory.dstAccessMask |= m_haz_stash.dstAccessMask;
		} else {
			m_pending.memory     = m_haz_stash;
			m_pending.has_memory = true;
		}
		m_pending.origins |= m_haz_stash_origins;
		m_pending.memory_origins |= m_haz_stash_origins;
		m_pending.guest_kinds |= m_haz_stash_kinds;
		m_haz_stash_valid   = false;
		m_haz_stash_origins = 0;
		m_haz_stash_kinds   = 0;
	}
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
	m_pending.memory_origins |= OriginBit(origin);
	CountOriginRequest(origin);
}

void CommandBuffer::NoteGuestBarrierKind(GuestBarrierKind kind) const {
	if (GpuOpProfiler::Active() && (m_pending.origins & OriginBit(BarrierOrigin::Guest)) != 0) {
		m_pending.guest_kinds |= 1u << static_cast<uint32_t>(kind);
	}
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
	if (GpuHazardCountEnabled()) {
		constexpr vk::AccessFlags2 Writes = Access2::eShaderWrite | Access2::eShaderStorageWrite |
		                                    Access2::eColorAttachmentWrite | Access2::eDepthStencilAttachmentWrite |
		                                    Access2::eTransferWrite | Access2::eMemoryWrite | Access2::eHostWrite;
		for (const auto& barrier: barriers) {
			const bool before = static_cast<bool>(barrier.srcAccessMask & Writes);
			const bool after  = static_cast<bool>(barrier.dstAccessMask & Writes);
			HazCount::Add(before ? (after ? HazCount::ImgWw : HazCount::ImgWr)
			                     : (after ? HazCount::ImgRw : HazCount::ImgRoLayout));
			if (barrier.oldLayout == barrier.newLayout) {
				HazCount::Add(HazCount::ImgSameLayout);
			}
		}
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
		if (m_upload_narrow) {
			// KYTY_DISPATCH_OVERLAP_UPLOADS (TakeDispatchOverlap): a later upload of an open
			// dispatch chain waits for the earlier copies only.
			barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
			barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
		}
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
		m_pending.memory_origins |= OriginBit(BarrierOrigin::Upload);
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
	FlushBarriersImpl(false);
}

void CommandBuffer::FlushBarriersImpl(bool side_own) const {
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
	const bool side = !m_pending.uploads.empty() || !m_pending.images.empty() || !m_pending.buffers.empty();
	if (!m_pending.uploads.empty()) {
		RecordPendingUploads();
	}
	const GpuOpProfiler::ScopedSite site(BatchSite(m_pending.origins, m_pending.guest_kinds));
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

	// KYTY_GPU_HAZARD: the chain restarts at a barrier that orders everything (or the chain's
	// dispatches) before later compute accesses; other recorded barriers end it.
	if (GpuHazardMode() != 0) {
		const auto& memory = m_pending.memory;
		m_haz_chain.OnFlush(m_haz_foreign, m_pending.has_memory,
		                    static_cast<VkPipelineStageFlags2>(memory.srcStageMask),
		                    static_cast<VkAccessFlags2>(memory.srcAccessMask),
		                    static_cast<VkPipelineStageFlags2>(memory.dstStageMask),
		                    static_cast<VkAccessFlags2>(memory.dstAccessMask), side && !side_own);
	} else if (m_haz_chain.valid) {
		m_haz_chain.Invalidate();
	}
	if (GpuHazardCountEnabled()) {
		HazCount::Add(HazCount::Flushes);
		if (m_pending.has_memory && IsFullBarrier(m_pending.memory)) {
			HazCount::Add(HazCount::Full);
			const auto o = m_pending.memory_origins;
			bool       any = false;
			if (o & OriginBit(BarrierOrigin::Guest)) {
				HazCount::Add(HazCount::FullGuest);
				any = true;
			}
			if (o & (OriginBit(BarrierOrigin::ShaderAccess) | OriginBit(BarrierOrigin::ShaderWriteHazard))) {
				HazCount::Add(HazCount::FullDispatch);
				any = true;
			}
			if (o & OriginBit(BarrierOrigin::ShaderWrite)) {
				HazCount::Add(HazCount::FullDraw);
				any = true;
			}
			if (!any) {
				HazCount::Add(HazCount::FullOther);
			}
		}
		if (!m_pending.images.empty()) {
			HazCount::Add(HazCount::ImageBarriers, m_pending.images.size());
			if (!m_pending.has_memory && m_pending.buffers.empty()) {
				HazCount::Add(HazCount::ImageOnly);
			}
		}
	}

	m_last_memory_valid = m_pending.has_memory;
	if (m_pending.has_memory) {
		m_last_memory = m_pending.memory;
	}
	m_recorded_since_flush = false;
	// A full batch orders everything before it against everything after it; its buffer/image
	// barriers were widened by the same scopes above.
	m_epoch_clean    = m_pending.has_memory && IsFullBarrier(m_pending.memory);
	m_epoch_instance = 0;
	m_overlap_open   = false;
	m_upload_narrow  = false;
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
	m_overlap_open         = false;
	++m_haz_foreign;
	if (!(m_draw_scope && m_draw_safe && m_rendering && m_epoch_instance == m_rendering_serial)) {
		m_epoch_clean = false;
	}
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	// Nothing bound, every descriptor epoch advances, push constants are undefined.
	SettleEncode();
	m_shadow.Begin();
	m_record_reset = true;
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

DrawEncode::ShadowFlags CommandBuffer::EncodeFlags() {
	return {Common::RendererBatchEnabled(), PushConstantShadowEnabled(), DescriptorSetReuseEnabled()};
}

// State commands: not ordered by barriers, so not batch flush points (render.h).
void CommandBuffer::BindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline) {
	auto sink = StateSink();
	m_shadow.BindPipeline(sink, EncodeFlags(), point, pipeline);
}

void CommandBuffer::InvalidateDescriptors(vk::PipelineBindPoint point) {
	SettleEncode();
	m_shadow.InvalidateDescriptors(point);
}

void CommandBuffer::BindDescriptorSet(vk::PipelineBindPoint point, vk::PipelineLayout layout,
                                      vk::DescriptorSet set) {
	SettleEncode();
	auto sink = StateSink();
	m_shadow.BindDescriptorSet(sink, EncodeFlags(), point, layout, set);
}

void CommandBuffer::PushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages,
                                  uint32_t size, const void* data) {
	SettleEncode();
	auto sink = StateSink();
	m_shadow.PushConstants(sink, EncodeFlags(), layout, stages, size, data);
}

int32_t CommandBuffer::PushDescriptors(vk::PipelineBindPoint point, vk::PipelineLayout layout,
                                       uint32_t set, uint32_t count,
                                       const vk::WriteDescriptorSet* writes, bool known_miss) {
	SettleEncode();
	auto sink = StateSink();
	return m_shadow.PushDescriptors(sink, EncodeFlags(), point, layout, set, count, writes, known_miss);
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

// The same rendering: the size, the layers, the depth/stencil attachment and the color attachments
// up to the count (those past it are no part of a rendering; the defaulted comparison read all eight
// at every draw; chenxiao07/KytyPS5 7bce89917).
static bool SameRendering(const RenderState& a, const RenderState& b) {
	if (a.num_color_attachments != b.num_color_attachments || a.width != b.width ||
	    a.height != b.height || a.num_layers != b.num_layers ||
	    !(a.depth_stencil_attachment == b.depth_stencil_attachment)) {
		return false;
	}
	const auto count = std::min<uint32_t>(a.num_color_attachments, RENDER_COLOR_ATTACHMENTS_MAX);
	for (uint32_t i = 0; i < count; i++) {
		if (!(a.color_attachments[i] == b.color_attachments[i])) {
			return false;
		}
	}
	return true;
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	const auto count_control = GetRegisters().GetDepthCountControl();
	auto&      occlusion     = m_context.GetOcclusionCounter();
	// A DB_COUNT_CONTROL change matters only when this instance or the next one is counted
	// (KYTY_OCCLUSION_GATE, occlusion.h): with neither counted, no sample can reach a guest value.
	const bool same_control =
	    m_occlusion_control == count_control ||
	    (OcclusionCounter::GateEnabled() && !occlusion.Active() && !occlusion.WouldCount(count_control));
	const bool same_instance = m_rendering && SameRendering(m_render_state, state) && same_control;
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
	if (m_rendering) {
		// Attribution of the end (GpuOps.EndRendering.<site>): a pending barrier that could not
		// be sunk, or different targets/state/occlusion control.
		if (barrier_split) {
			KYTY_GPU_OP_SITE("render.barrier_split");
			EndRendering();
		} else if (SameRendering(m_render_state, state)) {
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
	depth.storeOp = depth_stencil.depth_clear ? vk::AttachmentStoreOp::eStore
	                                          : DepthStoreOp(depth_stencil.image_layout, false);
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp = depth_stencil.stencil_clear ? vk::AttachmentStoreOp::eStore
	                                              : DepthStoreOp(depth_stencil.image_layout, true);
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	if (DeadViewScrubEnabled()) {
		// KYTY_DEAD_VIEW_SCRUB: also checked when an encoded rendering runs (commandRecorder.cpp).
		ScrubDeadRenderingViews(rendering, "a draw's rendering");
	}
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
	// The open pass holds the cleared contents, so a later clear begins a new pass.
	m_render_state = state;
	for (auto& color: m_render_state.color_attachments) {
		color.is_clear = false;
	}
	m_render_state.depth_stencil_attachment.depth_clear   = false;
	m_render_state.depth_stencil_attachment.stencil_clear = false;
	m_rendering    = true;
	++m_rendering_serial;
	if (GpuHazardCountEnabled()) {
		HazCount::Add(HazCount::Begins);
	}
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
