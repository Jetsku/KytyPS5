#include "graphics/host_gpu/renderer/gpuOpProfiler.h"

#include "common/cpuPlacement.h"
#include "common/hangTrace.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineFeedback.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fmt/format.h>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define KYTY_GPU_OP_RETURN_ADDRESS() reinterpret_cast<uintptr_t>(_ReturnAddress())
#else
#define KYTY_GPU_OP_RETURN_ADDRESS() reinterpret_cast<uintptr_t>(__builtin_return_address(0))
#endif

namespace Libs::Graphics::GpuOpProfiler {

namespace Detail {
bool g_active = false;
// The innermost and outermost KYTY_GPU_OP_SITE of the calling thread (ScopedSite, EnterSite).
constinit thread_local Site* t_site  = nullptr;
constinit thread_local Site* t_scope = nullptr;
} // namespace Detail

namespace {

using Detail::t_scope;
using Detail::t_site;

// ------------------------------------------------------------------------------------------------
// Configuration

// KYTY_GPU_OP_PROFILE_STAMPS: where a capture writes its timestamps.
//   ops (default): after every hooked operation (the GPU runs serialized; totals inflate).
//   passes: only around render passes (before the begin, after the end), at each change between
//     guest dispatches and emulator work outside passes, and before the command buffer ends. Draws
//     and consecutive guest dispatches keep their overlap.
//   alternate: even captures ops, odd captures passes.
enum class StampMode : uint8_t { Ops, Passes, Alternate };

struct Config {
	bool      counters     = false; // publish per-flip counters
	bool      draw_counters = false; // KYTY_GPU_OP_COUNTERS=2: draws by pipeline fragment stage
	bool      capture      = false; // sampled timestamp captures
	double    period_s     = 0.0;
	uint64_t  max_captures = 0;
	uint32_t  queries      = 262144;
	StampMode stamps       = StampMode::Ops;
};

bool EnvSet(const char* name) {
	const auto* value = std::getenv(name);
	return value != nullptr && value[0] != '\0';
}

const Config& GetConfig() {
	static const Config config = [] {
		Config c;
		if (const auto* value = std::getenv("KYTY_GPU_OP_PROFILE"); value != nullptr) {
			const double seconds = std::strtod(value, nullptr);
			if (std::isfinite(seconds) && seconds > 0.0) {
				c.capture  = true;
				c.period_s = seconds;
			}
		}
		if (const auto* value = std::getenv("KYTY_GPU_OP_PROFILE_MAX"); value != nullptr) {
			c.max_captures = std::strtoull(value, nullptr, 10);
		}
		if (const auto* value = std::getenv("KYTY_GPU_OP_PROFILE_QUERIES"); value != nullptr) {
			const auto queries = std::strtoull(value, nullptr, 10);
			if (queries != 0) {
				c.queries = static_cast<uint32_t>(
				    std::clamp<unsigned long long>(queries, 4096ull, 1ull << 22u));
			}
		}
		if (const auto* value = std::getenv("KYTY_GPU_OP_PROFILE_STAMPS"); value != nullptr) {
			if (std::strcmp(value, "passes") == 0) {
				c.stamps = StampMode::Passes;
			} else if (std::strcmp(value, "alternate") == 0) {
				c.stamps = StampMode::Alternate;
			}
		}
		if (EnvSet("KYTY_GPU_OP_COUNTERS")) {
			c.counters      = std::strcmp(std::getenv("KYTY_GPU_OP_COUNTERS"), "0") != 0;
			c.draw_counters = std::strcmp(std::getenv("KYTY_GPU_OP_COUNTERS"), "2") == 0;
		} else {
			c.counters = c.capture || HangTrace::Enabled() || Profiler::AggregateEnabled();
		}
		return c;
	}();
	return config;
}

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

template <typename Handle>
uint64_t HandleBits(Handle handle) noexcept {
	static_assert(sizeof(Handle) == sizeof(uint64_t));
	return std::bit_cast<uint64_t>(handle);
}

// ------------------------------------------------------------------------------------------------
// Sites

std::atomic<Site*> g_sites {nullptr};
constinit Site     g_unknown_site {"?"};
// KYTY_GPU_OP_PROFILE_STAMPS=passes: the site of a segment stamp names the work it closes.
constinit Site g_segment_compute {"segment.compute"};
constinit Site g_segment_emulator {"segment.emulator"};

} // namespace

// Links `site` into the published list (once; ScopedSite checks `linked` first).
void Detail::LinkSite(Site& site) noexcept {
	if (site.linked.load(std::memory_order_acquire)) {
		return;
	}
	bool expected = false;
	if (!site.linked.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
		return;
	}
	// Leaked on purpose: Tracy keeps plot name pointers for the whole session.
	auto* plot_name = new std::string(fmt::format("GpuOps.Barriers.{}", site.name));
	site.plot_name  = plot_name->c_str();
	auto* end_plot_name = new std::string(fmt::format("GpuOps.EndRendering.{}", site.name));
	site.end_plot_name  = end_plot_name->c_str();
	Site* head      = g_sites.load(std::memory_order_relaxed);
	do {
		site.next.store(head, std::memory_order_relaxed);
	} while (!g_sites.compare_exchange_weak(head, &site, std::memory_order_release,
	                                        std::memory_order_relaxed));
}

namespace {

using Detail::LinkSite;

// ------------------------------------------------------------------------------------------------
// Counters

struct Counters {
	std::atomic<uint64_t> render_passes {0};
	std::atomic<uint64_t> barriers {0};
	std::atomic<uint64_t> transitions {0};
	std::atomic<uint64_t> command_buffers {0};
	std::atomic<uint64_t> rendering_ends {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(BarrierBatchEvent::Count)> batch {};
};
Counters g_counters;

std::atomic<VkCommandBuffer> g_guest_cb {nullptr};   // guest scheduler's current buffer
std::atomic<VkCommandBuffer> g_capture_cb {nullptr}; // same buffer while it is being captured
std::atomic<uint64_t>        g_flip_seq {0};
std::atomic<bool>            g_capture_window {false};
std::atomic<uint64_t>        g_submit_calls {0};
std::atomic<uint64_t>        g_submit_batches {0};
std::atomic<uint64_t>        g_submit_cbs {0};

bool IsGuest(VkCommandBuffer cb) noexcept {
	return cb == g_guest_cb.load(std::memory_order_relaxed);
}
bool IsCapture(VkCommandBuffer cb) noexcept {
	return cb == g_capture_cb.load(std::memory_order_relaxed);
}

// Cumulative totals for gpucounts.csv (counters on), never reset.
struct Totals {
	std::atomic<uint64_t> submit_calls {0};
	std::atomic<uint64_t> submit_cbs {0};
	std::atomic<uint64_t> full_barriers {0}; // a memory barrier ALL_COMMANDS -> ALL_COMMANDS
	std::array<std::atomic<uint64_t>, static_cast<size_t>(FlushCause::Count)> flush {};
	// KYTY_GPU_OP_COUNTERS=2: guest draws by the bound pipeline's fragment stage
	// (PipelineClass), switches between a pipeline with and one without a fragment stage between
	// consecutive draws of one rendering instance, and graphics pipelines created per class.
	std::array<std::atomic<uint64_t>, 4> draws {};
	std::atomic<uint64_t>                fragment_switches {0};
	std::array<std::atomic<uint64_t>, 4> pipelines {};
	// Sums of the per-flip counters published by OnGuestFlip.
	uint64_t guest_cbs = 0, render_passes = 0, rendering_ends = 0, barriers = 0;
};
Totals g_totals;

constexpr std::array<const char*, static_cast<size_t>(FlushCause::Count)> kFlushCauseNames = {
    "slice_end",          "slice_stopped",           "eop_batch",
    "eop_packet_bound",   "idle_gpu",                "eager_readback",
    "periodic",           "batched",                 "deferred_label_wait",
    "proxy_label",        "flip_wait",               "before_gc",
    "defer_held",         "defer_declined_priority", "defer_declined_backlog",
    "defer_released_backlog", "defer_released_idle"};

// KYTY_GPU_OP_COUNTERS=2: a graphics pipeline's fragment stage.
enum class PipelineClass : uint8_t { Unknown, GuestFragment, NoFragment, EmptyFragment };
constexpr std::array<const char*, 4> kPipelineClassNames = {"unknown", "guest_fs", "no_fs", "empty_fs"};
std::mutex                                  g_pipeline_class_mutex;
std::unordered_map<uint64_t, PipelineClass> g_pipeline_class;
// The guest buffer's bound graphics pipeline class and the class of the previous draw of the
// current rendering instance (Unknown after a begin). Only the guest buffer's recording thread.
std::atomic<uint8_t> g_bound_class {0};
std::atomic<uint8_t> g_pass_draw_class {0};

void CountDraw(VkCommandBuffer cb) noexcept {
	if (!IsGuest(cb)) {
		return;
	}
	const auto bound = g_bound_class.load(std::memory_order_relaxed);
	g_totals.draws[bound & 3u].fetch_add(1, std::memory_order_relaxed);
	const auto previous = g_pass_draw_class.exchange(bound, std::memory_order_relaxed);
	const auto has_fs   = [](uint8_t c) {
		return c == static_cast<uint8_t>(PipelineClass::GuestFragment) ||
		       c == static_cast<uint8_t>(PipelineClass::EmptyFragment);
	};
	const auto known = [](uint8_t c) { return c != static_cast<uint8_t>(PipelineClass::Unknown); };
	if (known(previous) && known(bound) && has_fs(previous) != has_fs(bound)) {
		g_totals.fragment_switches.fetch_add(1, std::memory_order_relaxed);
	}
}

void CountBarrier(uint64_t transitions) noexcept {
	g_counters.barriers.fetch_add(1, std::memory_order_relaxed);
	if (transitions != 0) {
		g_counters.transitions.fetch_add(transitions, std::memory_order_relaxed);
	}
	auto* site = t_site != nullptr ? t_site : &g_unknown_site;
	site->barriers.fetch_add(1, std::memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------
// Original dispatcher entries

struct RealFunctions {
	PFN_vkCmdBindPipeline                  vkCmdBindPipeline                  = nullptr;
	PFN_vkCmdDraw                          vkCmdDraw                          = nullptr;
	PFN_vkCmdDrawIndexed                   vkCmdDrawIndexed                   = nullptr;
	PFN_vkCmdDrawIndirect                  vkCmdDrawIndirect                  = nullptr;
	PFN_vkCmdDrawIndexedIndirect           vkCmdDrawIndexedIndirect           = nullptr;
	PFN_vkCmdDrawIndirectCount             vkCmdDrawIndirectCount             = nullptr;
	PFN_vkCmdDrawIndexedIndirectCount      vkCmdDrawIndexedIndirectCount      = nullptr;
	PFN_vkCmdDrawMeshTasksEXT              vkCmdDrawMeshTasksEXT              = nullptr;
	PFN_vkCmdDrawMeshTasksIndirectEXT      vkCmdDrawMeshTasksIndirectEXT      = nullptr;
	PFN_vkCmdDrawMeshTasksIndirectCountEXT vkCmdDrawMeshTasksIndirectCountEXT = nullptr;
	PFN_vkCmdDispatch                      vkCmdDispatch                      = nullptr;
	PFN_vkCmdDispatchIndirect              vkCmdDispatchIndirect              = nullptr;
	PFN_vkCmdCopyBuffer                    vkCmdCopyBuffer                    = nullptr;
	PFN_vkCmdCopyImage                     vkCmdCopyImage                     = nullptr;
	PFN_vkCmdCopyBufferToImage             vkCmdCopyBufferToImage             = nullptr;
	PFN_vkCmdCopyImageToBuffer             vkCmdCopyImageToBuffer             = nullptr;
	PFN_vkCmdBlitImage                     vkCmdBlitImage                     = nullptr;
	PFN_vkCmdResolveImage                  vkCmdResolveImage                  = nullptr;
	PFN_vkCmdClearColorImage               vkCmdClearColorImage               = nullptr;
	PFN_vkCmdClearDepthStencilImage        vkCmdClearDepthStencilImage        = nullptr;
	PFN_vkCmdClearAttachments              vkCmdClearAttachments              = nullptr;
	PFN_vkCmdFillBuffer                    vkCmdFillBuffer                    = nullptr;
	PFN_vkCmdUpdateBuffer                  vkCmdUpdateBuffer                  = nullptr;
	PFN_vkCmdPipelineBarrier               vkCmdPipelineBarrier               = nullptr;
	PFN_vkCmdPipelineBarrier2              vkCmdPipelineBarrier2              = nullptr;
	PFN_vkCmdBeginRendering                vkCmdBeginRendering                = nullptr;
	PFN_vkCmdEndRendering                  vkCmdEndRendering                  = nullptr;
	PFN_vkCmdBeginQuery                    vkCmdBeginQuery                    = nullptr;
	PFN_vkCmdEndQuery                      vkCmdEndQuery                      = nullptr;
	PFN_vkCmdCopyQueryPoolResults          vkCmdCopyQueryPoolResults          = nullptr;
	PFN_vkEndCommandBuffer                 vkEndCommandBuffer                 = nullptr;
	PFN_vkQueueSubmit                      vkQueueSubmit                      = nullptr;
	PFN_vkQueueSubmit2                     vkQueueSubmit2                     = nullptr;
	PFN_vkCreateImageView                  vkCreateImageView                  = nullptr;
	PFN_vkDestroyImageView                 vkDestroyImageView                 = nullptr;
	PFN_vkCreateGraphicsPipelines          vkCreateGraphicsPipelines          = nullptr;
	PFN_vkDestroyPipeline                  vkDestroyPipeline                  = nullptr;
	// Not hooked; used to record the profiler's own commands.
	PFN_vkCmdWriteTimestamp2 vkCmdWriteTimestamp2 = nullptr;
	PFN_vkCmdWriteTimestamp  vkCmdWriteTimestamp  = nullptr;
	PFN_vkCmdResetQueryPool  vkCmdResetQueryPool  = nullptr;
};
RealFunctions g_real;

// ------------------------------------------------------------------------------------------------
// Operation records

enum class OpKind : uint8_t {
	CbBegin,
	Draw,
	DrawIndexed,
	DrawIndirect,
	DrawIndexedIndirect,
	DrawIndirectCount,
	DrawIndexedIndirectCount,
	DrawMesh,
	DrawMeshIndirect,
	DrawMeshIndirectCount,
	Dispatch,
	DispatchIndirect,
	CopyBuffer,
	CopyImage,
	CopyBufferToImage,
	CopyImageToBuffer,
	BlitImage,
	ResolveImage,
	ClearColorImage,
	ClearDepthStencilImage,
	ClearAttachments,
	FillBuffer,
	UpdateBuffer,
	Barrier,
	Barrier2,
	BeginRendering,
	EndRendering,
	BeginQuery,
	EndQuery,
	CopyQueryResults,
	Segment, // KYTY_GPU_OP_PROFILE_STAMPS=passes: closes a stretch of work outside render passes
	Count,
};

struct KindInfo {
	const char* name;
	const char* category;
};
constexpr std::array<KindInfo, static_cast<size_t>(OpKind::Count)> kKinds {{
    {"cb_begin", "cb"},
    {"draw", "draw"},
    {"draw_indexed", "draw"},
    {"draw_indirect", "draw"},
    {"draw_indexed_indirect", "draw"},
    {"draw_indirect_count", "draw"},
    {"draw_indexed_indirect_count", "draw"},
    {"draw_mesh", "draw"},
    {"draw_mesh_indirect", "draw"},
    {"draw_mesh_indirect_count", "draw"},
    {"dispatch", "dispatch"},
    {"dispatch_indirect", "dispatch"},
    {"copy_buffer", "transfer"},
    {"copy_image", "transfer"},
    {"copy_buffer_to_image", "transfer"},
    {"copy_image_to_buffer", "transfer"},
    {"blit_image", "transfer"},
    {"resolve_image", "transfer"},
    {"clear_color_image", "clear"},
    {"clear_depth_stencil_image", "clear"},
    {"clear_attachments", "clear"},
    {"fill_buffer", "transfer"},
    {"update_buffer", "transfer"},
    {"barrier", "barrier"},
    {"barrier2", "barrier"},
    {"begin_rendering", "render_pass"},
    {"end_rendering", "render_pass"},
    {"begin_query", "query"},
    {"end_query", "query"},
    {"copy_query_results", "query"},
    {"segment", "segment"},
}};

constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();

bool IsDraw(OpKind kind) {
	return kind >= OpKind::Draw && kind <= OpKind::DrawMeshIndirectCount;
}
bool IsDispatch(OpKind kind) {
	return kind == OpKind::Dispatch || kind == OpKind::DispatchIndirect;
}

struct OpRecord {
	uint32_t    query       = kNone;
	uint32_t    cb          = 0;
	uint32_t    pass        = kNone;
	OpKind      kind        = OpKind::Count;
	const Site* site        = nullptr;
	const Site* scope       = nullptr;
	uintptr_t   caller      = 0;
	uint64_t    pipeline    = 0;
	// Kind-specific counts: draw (vertex/index count, instances), indirect (draw count, stride),
	// mesh/dispatch (groups x, y, z).
	uint64_t    v[3]        = {};
	uint64_t    bytes       = 0;
	uint64_t    texels      = 0;
	uint32_t    regions     = 0;
	uint64_t    src_stages  = 0;
	uint64_t    dst_stages  = 0;
	uint32_t    mem_barriers = 0;
	uint32_t    buf_barriers = 0;
	uint32_t    img_barriers = 0;
	uint32_t    transitions  = 0;
};

struct CbRecord {
	uint64_t tick  = 0;
	uint32_t query = kNone;
};

struct PassRecord {
	uint32_t width        = 0;
	uint32_t height       = 0;
	uint32_t layers       = 0;
	uint32_t view_mask    = 0;
	uint32_t colors       = 0;
	uint32_t clear_mask   = 0; // bit i: color i cleared, bit 8 depth, bit 9 stencil
	VkFormat color0       = VK_FORMAT_UNDEFINED;
	VkFormat depth        = VK_FORMAT_UNDEFINED;
};

struct CaptureData {
	uint64_t                index      = 0;
	uint64_t                flip       = 0;
	uint64_t                start_ns   = 0;
	uint64_t                end_ns     = 0;
	uint32_t                capacity   = 0;
	uint32_t                used       = 0;
	uint64_t                overflow   = 0;
	bool                    truncated  = false;
	bool                    read_error = false;
	bool                    pass_stamps = false; // KYTY_GPU_OP_PROFILE_STAMPS=passes
	double                  period_ns  = 1.0;
	uint32_t                valid_bits = 64;
	uint64_t                submit_calls   = 0;
	uint64_t                submit_batches = 0;
	uint64_t                submit_cbs     = 0;
	std::vector<OpRecord>   ops;
	std::vector<CbRecord>   cbs;
	std::vector<PassRecord> passes;
	std::vector<uint64_t>   results; // {timestamp, availability} per query
};

// ------------------------------------------------------------------------------------------------
// Registries (shader programs, pipelines, image view formats)

struct ShaderInfo {
	const char* stage = "?";
	uint64_t    hash  = 0;
};
struct PipelineInfo {
	std::array<uint64_t, 3> vertex {};
	uint32_t                vertex_count = 0;
	uint64_t                pixel        = 0;
	uint64_t                compute      = 0;
};
std::mutex                                 g_registry_mutex;
std::unordered_map<uint64_t, ShaderInfo>   g_shaders;
std::unordered_map<uint64_t, PipelineInfo> g_pipelines;

std::mutex                             g_view_mutex;
std::unordered_map<uint64_t, VkFormat> g_view_formats;

VkFormat ViewFormat(VkImageView view) {
	if (view == VK_NULL_HANDLE) {
		return VK_FORMAT_UNDEFINED;
	}
	std::lock_guard lock(g_view_mutex);
	const auto      iter = g_view_formats.find(HandleBits(view));
	return iter != g_view_formats.end() ? iter->second : VK_FORMAT_UNDEFINED;
}

// ------------------------------------------------------------------------------------------------
// Capture state. Phase transitions run on the guest scheduler's recording producer
// (OnBeginCommand); hooks append to the capture under the mutex while recording the captured
// buffer, which Vulkan already requires to be externally synchronized.

enum class Phase : uint8_t { Idle, Active, Pending };

// The stretch of work outside render passes since the last stamp (pass stamps).
enum class SegmentClass : uint8_t { None, Compute, Emulator };

struct State {
	std::mutex                   mutex;
	Phase                        phase      = Phase::Idle;
	uint64_t                     seen_flip  = 0;
	uint64_t                     start_flip = 0;
	uint64_t                     next_ns    = 0;
	uint64_t                     last_tick  = 0;
	uint64_t                     captures   = 0;
	bool                         disabled   = false;
	std::unique_ptr<CaptureData> data;
	uint32_t                     next_query   = 0;
	uint32_t                     current_cb   = 0;
	uint32_t                     current_pass = kNone;
	uint32_t                     views        = 1;
	SegmentClass                 segment      = SegmentClass::None;
	std::array<uint64_t, 2>      bound {}; // graphics, compute
	// Query pool, created on the first capture.
	vk::Device    device     = nullptr;
	vk::QueryPool pool       = nullptr;
	uint32_t      capacity   = 0;
	double        period_ns  = 1.0;
	uint32_t      valid_bits = 64;
};
State g_state;

void WriteTimestamp(VkCommandBuffer cb, uint32_t query) {
	const auto pool = static_cast<VkQueryPool>(g_state.pool);
	if (g_real.vkCmdWriteTimestamp2 != nullptr) {
		g_real.vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool, query);
	} else {
		g_real.vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, query);
	}
}

// Caller holds g_state.mutex. Inside a multiview render pass a timestamp occupies one query
// per view; the first one is read.
uint32_t AllocateStamp(VkCommandBuffer cb) {
	auto& s     = g_state;
	auto* data  = s.data.get();
	const auto n = s.views;
	if (s.next_query + n > s.capacity) {
		++data->overflow;
		return kNone;
	}
	const auto query = s.next_query;
	s.next_query += n;
	WriteTimestamp(cb, query);
	return query;
}

template <typename Fill>
void Record(VkCommandBuffer cb, OpKind kind, uintptr_t caller, Fill&& fill) {
	auto&           s = g_state;
	std::lock_guard lock(s.mutex);
	auto*           data = s.data.get();
	if (data == nullptr || !IsCapture(cb)) {
		return;
	}
	OpRecord op;
	op.kind   = kind;
	op.cb     = s.current_cb;
	op.pass   = s.current_pass;
	op.site   = t_site;
	op.scope  = t_scope;
	op.caller = caller;
	if (IsDraw(kind)) {
		op.pipeline = s.bound[0];
	} else if (IsDispatch(kind)) {
		op.pipeline = s.bound[1];
	}
	fill(op, s);
	// Pass stamps: after a render pass ends; everything else is stamped by SegmentBoundary.
	op.query = !data->pass_stamps || kind == OpKind::EndRendering ? AllocateStamp(cb) : kNone;
	data->ops.push_back(op);
}

// Pass stamps, before an op outside a render pass: when it starts a different kind of work than
// the stretch since the last stamp, a stamp closes that stretch. `close`: before a render pass
// begins and before the command buffer ends, whatever follows.
void SegmentBoundary(VkCommandBuffer cb, SegmentClass next, bool close = false) {
	auto&           s = g_state;
	std::lock_guard lock(s.mutex);
	auto*           data = s.data.get();
	if (data == nullptr || !data->pass_stamps || !IsCapture(cb) || s.current_pass != kNone) {
		return;
	}
	if (s.segment == SegmentClass::None || (!close && s.segment == next)) {
		s.segment = close ? SegmentClass::None : next;
		return;
	}
	OpRecord op;
	op.kind  = OpKind::Segment;
	op.cb    = s.current_cb;
	op.site  = s.segment == SegmentClass::Compute ? &g_segment_compute : &g_segment_emulator;
	op.scope = op.site;
	op.query = AllocateStamp(cb);
	data->ops.push_back(op);
	s.segment = close ? SegmentClass::None : next;
}

// A guest dispatch (renderCompute's "dispatch" sites) or emulator work.
SegmentClass DispatchClass() noexcept {
	const auto* site = t_site;
	return site != nullptr && (std::strcmp(site->name, "dispatch") == 0 ||
	                           std::strcmp(site->name, "dispatch.indirect") == 0)
	           ? SegmentClass::Compute
	           : SegmentClass::Emulator;
}

void Record(VkCommandBuffer cb, OpKind kind, uintptr_t caller) {
	Record(cb, kind, caller, [](OpRecord&, State&) {});
}

// ------------------------------------------------------------------------------------------------
// Output

std::filesystem::path ResolveOutputDirectory() {
	if (auto dir = HangTrace::OutputDirectory(); !dir.empty()) {
		return std::filesystem::path(dir);
	}
	if (const auto* dir = std::getenv("KYTY_GPU_OP_PROFILE_DIR"); dir != nullptr && dir[0] != 0) {
		return std::filesystem::path(dir);
	}
	const auto now  = std::chrono::system_clock::now();
	const auto secs = std::chrono::system_clock::to_time_t(now);
	std::tm    tm_v {};
#ifdef _WIN32
	localtime_s(&tm_v, &secs);
	const auto pid = static_cast<uint64_t>(GetCurrentProcessId());
#else
	localtime_r(&secs, &tm_v);
	const auto pid = static_cast<uint64_t>(getpid());
#endif
	char stamp[32] {};
	std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm_v);
	return std::filesystem::path("_Profiling") / fmt::format("gpuops-{}-pid{}", stamp, pid);
}

uintptr_t ModuleBase() {
#ifdef _WIN32
	static const auto base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
	return base;
#else
	return 0;
#endif
}

std::string CallerText(uintptr_t caller) {
	if (caller == 0) {
		return {};
	}
	const auto base = ModuleBase();
	if (base != 0 && caller >= base && caller - base < 0x40000000u) {
		return fmt::format("exe+0x{:x}", caller - base);
	}
	return fmt::format("0x{:x}", caller);
}

std::string FormatName(VkFormat format) {
	if (format == VK_FORMAT_UNDEFINED) {
		return {};
	}
	return vk::to_string(static_cast<vk::Format>(format));
}

const char* SiteName(const Site* site) {
	return site != nullptr ? site->name : "?";
}

constexpr const char* kReadme = R"(KytyPS5 sampled GPU operation profile (KYTY_GPU_OP_PROFILE=<seconds>)

Each capture covers the guest command buffers begun between two guest flips. An ALL_COMMANDS
timestamp is written at the start of every guest command buffer and directly after every
recorded operation (draw, dispatch, copy/blit/clear/fill/resolve, pipeline barrier, dynamic
rendering begin/end, query op). Operations are hooked at the Vulkan dispatcher, so every
recording site is covered; the presenter's command buffers are not captured.

ATTRIBUTION CAVEAT
  delta_ns = (this op's completion stamp) - (previous stamp in the same command buffer).
  GPUs overlap consecutive work, so this is the incremental completion time, not the isolated
  cost of the op. A cheap op after an expensive one can absorb part of its tail; barriers
  typically show ~0 because the previous op already drained. Per-category/site/pipeline totals
  are a heuristic of where GPU time goes. Writing a timestamp after every op reduces overlap, so
  a captured frame's op total is usually larger than an unprofiled frame's gpu_busy_us.
  cb_begin rows carry the gap since the previous captured command buffer's last stamp (idle,
  presenter work, CPU starvation); it is excluded from op totals.

PASS STAMPS (KYTY_GPU_OP_PROFILE_STAMPS=passes, or alternate: every second capture)
  Stamps only before each render pass begins, after it ends, at each change between guest
  dispatches and other work outside passes ("segment" rows: site segment.compute or
  segment.emulator names the work they close), and before the command buffer ends. Draws and
  consecutive guest dispatches keep their overlap, so totals stay close to gpu_busy_us. Other
  rows have an empty delta_ns: they are timed together by the next stamped row of their command
  buffer (end_rendering: the whole pass from its begin). totals: pass_stamps=1, unstamped_ops.

gpuops-<flip>.csv (one row per op; <flip> = guest flip counter at capture start)
  seq                 recording order within the capture
  cb, tick            capture-local command buffer index and its scheduler tick
  kind, category      op kind (draw_indexed, barrier2, begin_rendering, ...) and category
                      (cb, draw, dispatch, transfer, clear, barrier, render_pass, query)
  site                innermost KYTY_GPU_OP_SITE tag active while recording ("?" = untagged)
  scope               outermost tag (e.g. draw.index for everything recorded during a draw)
  caller              return address of the vkCmd call, exe+0x<rva>; symbolize with
                      analyze_gpuops.py --exe kyty_emulator.exe (llvm-symbolizer)
  delta_ns            see caveat; empty when the stamp was unavailable or over capacity
  cb_offset_ns        stamp minus the command buffer's start stamp
  pipeline, shaders   bound pipeline handle for draws/dispatches, and its guest shader hashes
                      (vs/ls/hs/ds/ms/ps/cs:0x<hash>), "internal" for emulator pipelines
  vertex_count, index_count, instance_count, draw_count, groups_x, groups_y, groups_z
  bytes               buffer copy/fill/update size (fill with VK_WHOLE_SIZE: empty)
  texels              image copy/blit/resolve/clear_attachments texel count (w*h*d*layers)
  regions             copy regions / clear ranges or rects
  src_stages, dst_stages   barrier stage masks (hex; barrier2 = OR over all barriers)
  mem_barriers, buf_barriers, img_barriers, layout_transitions (image barriers old != new layout)
  rt_pass             capture-local render pass index the op was recorded in
  rt_width, rt_height, rt_layers, rt_view_mask, rt_colors, rt_color0_format, rt_depth_format,
  rt_clear_mask       render target of that pass (clear mask: bit i color i, bit 8 depth,
                      bit 9 stencil load-op clear)

gpuops-summary.csv (appended per capture; long format)
  capture, flip, section, key, count, gpu_ns, share_pct, extra
  sections: totals, category, kind, site, scope, scope_site, pipeline (top 50), render_target
  (top 50, ops recorded inside passes on that target), caller (top 50), barrier_site (count =
  barrier calls, extra = layout transitions), render_pass_site (count = begins, gpu_ns = begin
  deltas), end_rendering_site (who ended passes).
  totals keys: ops, stamped_ops, unavailable_stamps, pass_stamps, unstamped_ops, overflow_ops,
  command_buffers, vk_queue_submit_calls, vk_submit_batches, vk_submitted_cbs, render_pass_begins,
  barriers,
  layout_transitions, draws, dispatches, op_gpu_ns, cb_span_gpu_ns, cb_gap_gpu_ns,
  first_to_last_gpu_ns, queries_used, query_capacity, capture_cpu_ns, truncated.
  vk_queue_submit_* count every vkQueueSubmit on the device during the capture window
  (including the presenter's).
)";

class Writer {
public:
	~Writer() { Stop(); }

	void Push(std::unique_ptr<CaptureData> capture) {
		{
			std::lock_guard lock(m_mutex);
			if (m_stopped) {
				return;
			}
			m_queue.push_back(std::move(capture));
			if (!m_thread.joinable()) {
				m_thread = std::thread([this] { Run(); });
			}
		}
		m_cv.notify_one();
	}

	void Stop() {
		{
			std::lock_guard lock(m_mutex);
			m_stopped = true;
		}
		m_cv.notify_one();
		if (m_thread.joinable()) {
			m_thread.join();
		}
	}

private:
	void Run() {
		Common::PlaceCurrentThreadCold();
		for (;;) {
			std::unique_ptr<CaptureData> capture;
			{
				std::unique_lock lock(m_mutex);
				m_cv.wait(lock, [this] { return m_stopped || !m_queue.empty(); });
				if (m_queue.empty()) {
					return;
				}
				capture = std::move(m_queue.front());
				m_queue.pop_front();
			}
			Write(*capture);
		}
	}

	bool EnsureDirectory() {
		if (m_dir_ready) {
			return true;
		}
		m_dir = ResolveOutputDirectory();
		std::error_code error;
		std::filesystem::create_directories(m_dir, error);
		if (error) {
			std::printf("GPU op profiler: cannot create %s: %s\n", m_dir.string().c_str(),
			            error.message().c_str());
			std::fflush(stdout);
			return false;
		}
		if (auto* file = std::fopen((m_dir / "gpuops-README.txt").string().c_str(), "wb");
		    file != nullptr) {
			std::fputs(kReadme, file);
			std::fclose(file);
		}
		m_dir_ready = true;
		return true;
	}

	void Write(CaptureData& capture);

	std::mutex                               m_mutex;
	std::condition_variable                  m_cv;
	std::deque<std::unique_ptr<CaptureData>> m_queue;
	std::thread                              m_thread;
	bool                                     m_stopped   = false;
	bool                                     m_dir_ready = false;
	std::filesystem::path                    m_dir;
	std::unordered_map<uint64_t, std::string> m_pipeline_keys;
};

struct Aggregate {
	uint64_t count = 0;
	uint64_t ns    = 0;
	uint64_t extra = 0;
};
using AggregateMap = std::unordered_map<std::string, Aggregate>;

void Accumulate(AggregateMap& map, const std::string& key, uint64_t ns, uint64_t extra = 0) {
	auto& entry = map[key];
	++entry.count;
	entry.ns += ns;
	entry.extra += extra;
}

void Writer::Write(CaptureData& capture) {
	if (!EnsureDirectory()) {
		return;
	}
	const uint64_t mask = capture.valid_bits >= 64 ? std::numeric_limits<uint64_t>::max()
	                                               : (uint64_t {1} << capture.valid_bits) - 1u;
	const auto stamp = [&](uint32_t query, uint64_t& value) {
		if (query == kNone || query >= capture.used || capture.read_error ||
		    size_t {query} * 2u + 1u >= capture.results.size() ||
		    capture.results[size_t {query} * 2u + 1u] == 0) {
			return false;
		}
		value = capture.results[size_t {query} * 2u] & mask;
		return true;
	};
	// Delta a - b in nanoseconds, modulo the valid bits; negative deltas (reordering across
	// submissions without calibrated ordering) clamp to 0.
	const auto delta_ns = [&](uint64_t a, uint64_t b) -> uint64_t {
		uint64_t delta = (a - b) & mask;
		if (capture.valid_bits < 64 && ((delta >> (capture.valid_bits - 1)) & 1u) != 0) {
			return 0;
		}
		if (capture.valid_bits >= 64 && static_cast<int64_t>(delta) < 0) {
			return 0;
		}
		return static_cast<uint64_t>(static_cast<double>(delta) * capture.period_ns);
	};

	const auto pipeline_key = [&](uint64_t handle) -> const std::string& {
		auto iter = m_pipeline_keys.find(handle);
		if (iter != m_pipeline_keys.end()) {
			return iter->second;
		}
		std::string key;
		{
			std::lock_guard lock(g_registry_mutex);
			const auto      pipeline = g_pipelines.find(handle);
			const auto shader = [&](uint64_t id) {
				const auto found = g_shaders.find(id);
				if (found == g_shaders.end()) {
					return fmt::format("id{}", id);
				}
				return fmt::format("{}:0x{:016x}", found->second.stage, found->second.hash);
			};
			if (handle == 0) {
				key = "none";
			} else if (pipeline == g_pipelines.end()) {
				key = "internal";
			} else if (pipeline->second.compute != 0) {
				key = shader(pipeline->second.compute);
			} else {
				for (uint32_t i = 0; i < pipeline->second.vertex_count; ++i) {
					if (!key.empty()) {
						key += '|';
					}
					key += shader(pipeline->second.vertex[i]);
				}
				if (pipeline->second.pixel != 0) {
					key += '|';
					key += shader(pipeline->second.pixel);
				}
			}
		}
		return m_pipeline_keys.emplace(handle, std::move(key)).first->second;
	};
	const auto pass_key = [&](uint32_t pass) {
		const auto& p = capture.passes[pass];
		return fmt::format("{}x{}x{} c{}:{} d:{}", p.width, p.height, p.layers, p.colors,
		                   FormatName(p.color0), FormatName(p.depth));
	};

	// Per-op rows.
	const auto op_path = m_dir / fmt::format("gpuops-{}.csv", capture.flip);
	auto*      op_file = std::fopen(op_path.string().c_str(), "wb");
	std::string buffer;
	buffer.reserve(size_t {1} << 20u);
	const auto flush = [&](bool force) {
		if (op_file != nullptr && (force || buffer.size() > (size_t {1} << 20u))) {
			std::fwrite(buffer.data(), 1, buffer.size(), op_file);
			buffer.clear();
		} else if (op_file == nullptr) {
			buffer.clear();
		}
	};
	buffer +=
	    "seq,cb,tick,kind,category,site,scope,caller,delta_ns,cb_offset_ns,pipeline,shaders,"
	    "vertex_count,index_count,instance_count,draw_count,groups_x,groups_y,groups_z,bytes,"
	    "texels,regions,src_stages,dst_stages,mem_barriers,buf_barriers,img_barriers,"
	    "layout_transitions,rt_pass,rt_width,rt_height,rt_layers,rt_view_mask,rt_colors,"
	    "rt_color0_format,rt_depth_format,rt_clear_mask\n";

	AggregateMap by_kind, by_category, by_site, by_scope, by_scope_site, by_pipeline, by_target,
	    by_caller, barrier_sites, pass_sites, end_sites;
	uint64_t stamped = 0, unavailable = 0, unstamped = 0, draws = 0, dispatches = 0, barriers = 0;
	uint64_t transitions = 0, pass_begins = 0, op_ns = 0, span_ns = 0, gap_ns = 0;
	bool     have_first = false, have_last = false;
	uint64_t first_stamp = 0, last_stamp = 0;

	// Command-buffer start stamps and the cb currently walked.
	uint32_t current_cb = kNone;
	bool     have_prev  = false;
	uint64_t prev_stamp = 0, cb_start = 0;
	bool     have_cb_start = false;
	bool     have_cb_last  = false;
	uint64_t cb_last       = 0;
	const auto close_cb = [&] {
		if (have_cb_start && have_cb_last) {
			span_ns += delta_ns(cb_last, cb_start);
		}
	};
	const auto open_cb = [&](uint32_t cb, uint64_t seq) {
		close_cb();
		current_cb    = cb;
		have_cb_start = false;
		have_cb_last  = false;
		have_prev     = false;
		uint64_t start = 0;
		const auto& record = capture.cbs[cb];
		std::string gap_text;
		if (stamp(record.query, start)) {
			++stamped;
			have_cb_start = true;
			cb_start      = start;
			have_prev     = true;
			prev_stamp    = start;
			if (have_last) {
				const auto gap = delta_ns(start, last_stamp);
				gap_ns += gap;
				gap_text = fmt::format("{}", gap);
			}
			if (!have_first) {
				have_first  = true;
				first_stamp = start;
			}
			have_last  = true;
			last_stamp = start;
		} else {
			++unavailable;
		}
		fmt::format_to(std::back_inserter(buffer), "{},{},{},cb_begin,cb,,,,{},0", seq, cb,
		               record.tick, gap_text);
		buffer.append(27, ','); // remaining columns are empty for cb_begin rows
		buffer += '\n';
	};

	uint64_t seq = 0;
	for (const auto& op: capture.ops) {
		if (op.cb != current_cb) {
			// Every captured buffer gets its cb_begin row, including ones without ops.
			for (uint32_t cb = current_cb == kNone ? 0 : current_cb + 1; cb <= op.cb; ++cb) {
				open_cb(cb, seq++);
			}
		}
		uint64_t   value = 0;
		const bool valid = stamp(op.query, value);
		uint64_t   delta = 0;
		if (valid) {
			++stamped;
			if (have_prev) {
				delta = delta_ns(value, prev_stamp);
			}
			have_prev    = true;
			prev_stamp   = value;
			have_cb_last = true;
			cb_last      = value;
			if (!have_first) {
				have_first  = true;
				first_stamp = value;
			}
			have_last  = true;
			last_stamp = value;
			op_ns += delta;
		} else if (op.query == kNone && capture.pass_stamps) {
			++unstamped; // pass stamps: timed by the next stamp of its command buffer
		} else {
			++unavailable;
		}
		const auto& kind   = kKinds[static_cast<size_t>(op.kind)];
		const auto* site   = SiteName(op.site);
		const auto* scope  = SiteName(op.scope);
		const auto  caller = CallerText(op.caller);
		const bool  has_pipeline = IsDraw(op.kind) || IsDispatch(op.kind);
		const std::string& shaders = has_pipeline ? pipeline_key(op.pipeline) : std::string {};

		Accumulate(by_kind, kind.name, delta);
		Accumulate(by_category, kind.category, delta);
		Accumulate(by_site, site, delta);
		Accumulate(by_scope, scope, delta);
		Accumulate(by_scope_site, fmt::format("{}>{}", scope, site), delta);
		if (!caller.empty()) {
			Accumulate(by_caller, caller, delta);
		}
		if (has_pipeline) {
			Accumulate(by_pipeline, shaders, delta, 0);
			by_pipeline[shaders].extra = op.pipeline;
		}
		if (op.pass != kNone && op.kind != OpKind::BeginRendering &&
		    op.kind != OpKind::EndRendering) {
			Accumulate(by_target, pass_key(op.pass), delta);
		}
		if (IsDraw(op.kind)) {
			++draws;
		} else if (IsDispatch(op.kind)) {
			++dispatches;
		} else if (op.kind == OpKind::Barrier || op.kind == OpKind::Barrier2) {
			++barriers;
			transitions += op.transitions;
			Accumulate(barrier_sites, fmt::format("{}>{}", scope, site), delta, op.transitions);
		} else if (op.kind == OpKind::BeginRendering) {
			++pass_begins;
			Accumulate(pass_sites, fmt::format("{}>{}", scope, site), delta);
		} else if (op.kind == OpKind::EndRendering) {
			Accumulate(end_sites, fmt::format("{}>{}", scope, site), delta);
		}

		// vertex_count,index_count,instance_count,draw_count,groups_x,groups_y,groups_z
		uint64_t counts[7] {};
		switch (op.kind) {
			case OpKind::Draw:
				counts[0] = op.v[0];
				counts[2] = op.v[1];
				break;
			case OpKind::DrawIndexed:
				counts[1] = op.v[0];
				counts[2] = op.v[1];
				break;
			case OpKind::DrawIndirect:
			case OpKind::DrawIndexedIndirect:
			case OpKind::DrawIndirectCount:
			case OpKind::DrawIndexedIndirectCount:
			case OpKind::DrawMeshIndirect:
			case OpKind::DrawMeshIndirectCount: counts[3] = op.v[0]; break;
			case OpKind::DrawMesh:
			case OpKind::Dispatch:
				counts[4] = op.v[0];
				counts[5] = op.v[1];
				counts[6] = op.v[2];
				break;
			default: break;
		}
		fmt::format_to(std::back_inserter(buffer), "{},{},{},{},{},{},{},{},", seq++, op.cb,
		               capture.cbs[op.cb].tick, kind.name, kind.category, site, scope, caller);
		if (valid) {
			fmt::format_to(std::back_inserter(buffer), "{},{},", delta,
			               have_cb_start ? delta_ns(value, cb_start) : 0);
		} else {
			buffer += ",,";
		}
		if (has_pipeline) {
			fmt::format_to(std::back_inserter(buffer), "0x{:x},{},", op.pipeline, shaders);
		} else {
			buffer += ",,";
		}
		for (auto count: counts) {
			if (count != 0) {
				fmt::format_to(std::back_inserter(buffer), "{}", count);
			}
			buffer += ',';
		}
		if (op.bytes == std::numeric_limits<uint64_t>::max()) {
			buffer += ',';
		} else if (op.bytes != 0) {
			fmt::format_to(std::back_inserter(buffer), "{},", op.bytes);
		} else {
			buffer += ',';
		}
		if (op.texels != 0) {
			fmt::format_to(std::back_inserter(buffer), "{}", op.texels);
		}
		buffer += ',';
		if (op.regions != 0) {
			fmt::format_to(std::back_inserter(buffer), "{}", op.regions);
		}
		buffer += ',';
		if (op.kind == OpKind::Barrier || op.kind == OpKind::Barrier2) {
			fmt::format_to(std::back_inserter(buffer), "0x{:x},0x{:x},{},{},{},{},", op.src_stages,
			               op.dst_stages, op.mem_barriers, op.buf_barriers, op.img_barriers,
			               op.transitions);
		} else {
			buffer += ",,,,,,";
		}
		if (op.pass != kNone) {
			const auto& p = capture.passes[op.pass];
			fmt::format_to(std::back_inserter(buffer), "{},{},{},{},0x{:x},{},{},{},0x{:x}\n",
			               op.pass, p.width, p.height, p.layers, p.view_mask, p.colors,
			               FormatName(p.color0), FormatName(p.depth), p.clear_mask);
		} else {
			buffer += ",,,,,,,,\n";
		}
		flush(false);
	}
	// Trailing captured command buffers without any op.
	if (!capture.cbs.empty()) {
		for (uint32_t cb = current_cb == kNone ? 0 : current_cb + 1; cb < capture.cbs.size();
		     ++cb) {
			open_cb(cb, seq++);
		}
	}
	close_cb();
	flush(true);
	if (op_file != nullptr) {
		std::fclose(op_file);
	}

	// Summary rows.
	const auto summary_path = m_dir / "gpuops-summary.csv";
	std::error_code error;
	const bool      exists = std::filesystem::exists(summary_path, error) &&
	                    std::filesystem::file_size(summary_path, error) > 0;
	auto* summary = std::fopen(summary_path.string().c_str(), "ab");
	if (summary == nullptr) {
		return;
	}
	std::string out;
	if (!exists) {
		out += "capture,flip,section,key,count,gpu_ns,share_pct,extra\n";
	}
	const auto total = std::max<uint64_t>(op_ns, 1);
	const auto row = [&](const char* section, const std::string& key, uint64_t count, uint64_t ns,
	                     const std::string& extra) {
		fmt::format_to(std::back_inserter(out), "{},{},{},{},{},{},{:.3f},{}\n", capture.index,
		               capture.flip, section, key, count, ns,
		               100.0 * static_cast<double>(ns) / static_cast<double>(total), extra);
	};
	const auto totals = [&](const char* key, uint64_t value) {
		fmt::format_to(std::back_inserter(out), "{},{},totals,{},{},,,\n", capture.index,
		               capture.flip, key, value);
	};
	totals("ops", capture.ops.size());
	totals("stamped_ops", stamped);
	totals("unavailable_stamps", unavailable);
	totals("pass_stamps", capture.pass_stamps ? 1 : 0);
	totals("unstamped_ops", unstamped);
	totals("overflow_ops", capture.overflow);
	totals("command_buffers", capture.cbs.size());
	totals("vk_queue_submit_calls", capture.submit_calls);
	totals("vk_submit_batches", capture.submit_batches);
	totals("vk_submitted_cbs", capture.submit_cbs);
	totals("render_pass_begins", pass_begins);
	totals("barriers", barriers);
	totals("layout_transitions", transitions);
	totals("draws", draws);
	totals("dispatches", dispatches);
	totals("op_gpu_ns", op_ns);
	totals("cb_span_gpu_ns", span_ns);
	totals("cb_gap_gpu_ns", gap_ns);
	totals("first_to_last_gpu_ns",
	       have_first && have_last ? delta_ns(last_stamp, first_stamp) : 0);
	totals("queries_used", capture.used);
	totals("query_capacity", capture.capacity);
	totals("capture_cpu_ns", capture.end_ns - capture.start_ns);
	totals("truncated", capture.truncated ? 1 : 0);

	const auto emit = [&](const char* section, const AggregateMap& map, size_t limit,
	                      bool extra_hex, bool with_extra) {
		std::vector<std::pair<const std::string*, const Aggregate*>> sorted;
		sorted.reserve(map.size());
		for (const auto& [key, value]: map) {
			sorted.emplace_back(&key, &value);
		}
		std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
			return a.second->ns != b.second->ns ? a.second->ns > b.second->ns
			                                    : a.second->count > b.second->count;
		});
		if (sorted.size() > limit) {
			sorted.resize(limit);
		}
		for (const auto& [key, value]: sorted) {
			std::string extra;
			if (with_extra) {
				extra = extra_hex ? fmt::format("0x{:x}", value->extra)
				                  : fmt::format("{}", value->extra);
			}
			row(section, *key, value->count, value->ns, extra);
		}
	};
	constexpr size_t kAll = std::numeric_limits<size_t>::max();
	emit("category", by_category, kAll, false, false);
	emit("kind", by_kind, kAll, false, false);
	emit("site", by_site, kAll, false, false);
	emit("scope", by_scope, kAll, false, false);
	emit("scope_site", by_scope_site, kAll, false, false);
	emit("pipeline", by_pipeline, 50, true, true);
	emit("render_target", by_target, 50, false, false);
	emit("caller", by_caller, 50, false, false);
	emit("barrier_site", barrier_sites, kAll, false, true);
	emit("render_pass_site", pass_sites, kAll, false, false);
	emit("end_rendering_site", end_sites, kAll, false, false);
	std::fwrite(out.data(), 1, out.size(), summary);
	std::fclose(summary);

	std::printf("GPU op profiler: capture %" PRIu64 " (flip %" PRIu64 "): %zu ops, %zu cmdbufs, "
	            "%" PRIu64 " render passes, %" PRIu64 " barriers, op total %.3f ms -> %s\n",
	            capture.index, capture.flip, capture.ops.size(), capture.cbs.size(), pass_begins,
	            barriers, static_cast<double>(op_ns) / 1e6, op_path.string().c_str());
	std::fflush(stdout);
}

Writer& GetWriter() {
	static Writer writer;
	return writer;
}

// ------------------------------------------------------------------------------------------------
// Hooks

#define KYTY_GPU_OP_CALLER KYTY_GPU_OP_RETURN_ADDRESS()

VKAPI_ATTR void VKAPI_CALL HookCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint point,
                                               VkPipeline pipeline) {
	g_real.vkCmdBindPipeline(cb, point, pipeline);
	if (GetConfig().draw_counters && point == VK_PIPELINE_BIND_POINT_GRAPHICS && IsGuest(cb)) {
		auto pipeline_class = PipelineClass::Unknown;
		{
			std::lock_guard lock(g_pipeline_class_mutex);
			if (const auto found = g_pipeline_class.find(HandleBits(pipeline)); found != g_pipeline_class.end()) {
				pipeline_class = found->second;
			}
		}
		g_bound_class.store(static_cast<uint8_t>(pipeline_class), std::memory_order_relaxed);
	}
	if (IsCapture(cb)) [[unlikely]] {
		std::lock_guard lock(g_state.mutex);
		if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
			g_state.bound[0] = HandleBits(pipeline);
		} else if (point == VK_PIPELINE_BIND_POINT_COMPUTE) {
			g_state.bound[1] = HandleBits(pipeline);
		}
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdDraw(VkCommandBuffer cb, uint32_t vertex_count,
                                       uint32_t instance_count, uint32_t first_vertex,
                                       uint32_t first_instance) {
	g_real.vkCmdDraw(cb, vertex_count, instance_count, first_vertex, first_instance);
	CountDraw(cb);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::Draw, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.v[0] = vertex_count;
			op.v[1] = instance_count;
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdDrawIndexed(VkCommandBuffer cb, uint32_t index_count,
                                              uint32_t instance_count, uint32_t first_index,
                                              int32_t vertex_offset, uint32_t first_instance) {
	g_real.vkCmdDrawIndexed(cb, index_count, instance_count, first_index, vertex_offset,
	                        first_instance);
	CountDraw(cb);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::DrawIndexed, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.v[0] = index_count;
			op.v[1] = instance_count;
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdDrawIndirect(VkCommandBuffer cb, VkBuffer buffer,
                                               VkDeviceSize offset, uint32_t draw_count,
                                               uint32_t stride) {
	g_real.vkCmdDrawIndirect(cb, buffer, offset, draw_count, stride);
	CountDraw(cb);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::DrawIndirect, KYTY_GPU_OP_CALLER,
		       [&](OpRecord& op, State&) { op.v[0] = draw_count; });
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer buffer,
                                                      VkDeviceSize offset, uint32_t draw_count,
                                                      uint32_t stride) {
	g_real.vkCmdDrawIndexedIndirect(cb, buffer, offset, draw_count, stride);
	CountDraw(cb);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::DrawIndexedIndirect, KYTY_GPU_OP_CALLER,
		       [&](OpRecord& op, State&) { op.v[0] = draw_count; });
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer buffer,
                                                    VkDeviceSize offset, VkBuffer count_buffer,
                                                    VkDeviceSize count_offset,
                                                    uint32_t max_draw_count, uint32_t stride) {
	g_real.vkCmdDrawIndirectCount(cb, buffer, offset, count_buffer, count_offset, max_draw_count,
	                              stride);
	CountDraw(cb);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::DrawIndirectCount, KYTY_GPU_OP_CALLER,
		       [&](OpRecord& op, State&) { op.v[0] = max_draw_count; });
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer buffer,
                                                           VkDeviceSize offset,
                                                           VkBuffer     count_buffer,
                                                           VkDeviceSize count_offset,
                                                           uint32_t     max_draw_count,
                                                           uint32_t     stride) {
	g_real.vkCmdDrawIndexedIndirectCount(cb, buffer, offset, count_buffer, count_offset,
	                                     max_draw_count, stride);
	CountDraw(cb);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::DrawIndexedIndirectCount, KYTY_GPU_OP_CALLER,
		       [&](OpRecord& op, State&) { op.v[0] = max_draw_count; });
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdDrawMeshTasksEXT(VkCommandBuffer cb, uint32_t x, uint32_t y,
                                                   uint32_t z) {
	g_real.vkCmdDrawMeshTasksEXT(cb, x, y, z);
	CountDraw(cb);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::DrawMesh, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.v[0] = x;
			op.v[1] = y;
			op.v[2] = z;
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdDrawMeshTasksIndirectEXT(VkCommandBuffer cb, VkBuffer buffer,
                                                           VkDeviceSize offset,
                                                           uint32_t draw_count, uint32_t stride) {
	g_real.vkCmdDrawMeshTasksIndirectEXT(cb, buffer, offset, draw_count, stride);
	CountDraw(cb);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::DrawMeshIndirect, KYTY_GPU_OP_CALLER,
		       [&](OpRecord& op, State&) { op.v[0] = draw_count; });
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdDrawMeshTasksIndirectCountEXT(
    VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkBuffer count_buffer,
    VkDeviceSize count_offset, uint32_t max_draw_count, uint32_t stride) {
	g_real.vkCmdDrawMeshTasksIndirectCountEXT(cb, buffer, offset, count_buffer, count_offset,
	                                          max_draw_count, stride);
	CountDraw(cb);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::DrawMeshIndirectCount, KYTY_GPU_OP_CALLER,
		       [&](OpRecord& op, State&) { op.v[0] = max_draw_count; });
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y,
                                           uint32_t z) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, DispatchClass());
	}
	g_real.vkCmdDispatch(cb, x, y, z);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::Dispatch, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.v[0] = x;
			op.v[1] = y;
			op.v[2] = z;
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer buffer,
                                                   VkDeviceSize offset) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, DispatchClass());
	}
	g_real.vkCmdDispatchIndirect(cb, buffer, offset);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::DispatchIndirect, KYTY_GPU_OP_CALLER);
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst,
                                             uint32_t count, const VkBufferCopy* regions) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdCopyBuffer(cb, src, dst, count, regions);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::CopyBuffer, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.regions = count;
			for (uint32_t i = 0; i < count; ++i) {
				op.bytes += regions[i].size;
			}
		});
	}
}

uint64_t Texels(const VkExtent3D& extent, uint32_t layers) {
	return uint64_t {extent.width} * extent.height * std::max(extent.depth, 1u) *
	       std::max(layers, 1u);
}

VKAPI_ATTR void VKAPI_CALL HookCmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout src_layout,
                                            VkImage dst, VkImageLayout dst_layout, uint32_t count,
                                            const VkImageCopy* regions) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdCopyImage(cb, src, src_layout, dst, dst_layout, count, regions);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::CopyImage, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.regions = count;
			for (uint32_t i = 0; i < count; ++i) {
				op.texels += Texels(regions[i].extent, regions[i].srcSubresource.layerCount);
			}
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer src, VkImage dst,
                                                    VkImageLayout layout, uint32_t count,
                                                    const VkBufferImageCopy* regions) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdCopyBufferToImage(cb, src, dst, layout, count, regions);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::CopyBufferToImage, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.regions = count;
			for (uint32_t i = 0; i < count; ++i) {
				op.texels += Texels(regions[i].imageExtent, regions[i].imageSubresource.layerCount);
			}
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage src,
                                                    VkImageLayout layout, VkBuffer dst,
                                                    uint32_t count,
                                                    const VkBufferImageCopy* regions) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdCopyImageToBuffer(cb, src, layout, dst, count, regions);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::CopyImageToBuffer, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.regions = count;
			for (uint32_t i = 0; i < count; ++i) {
				op.texels += Texels(regions[i].imageExtent, regions[i].imageSubresource.layerCount);
			}
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdBlitImage(VkCommandBuffer cb, VkImage src, VkImageLayout src_layout,
                                            VkImage dst, VkImageLayout dst_layout, uint32_t count,
                                            const VkImageBlit* regions, VkFilter filter) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdBlitImage(cb, src, src_layout, dst, dst_layout, count, regions, filter);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::BlitImage, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.regions = count;
			for (uint32_t i = 0; i < count; ++i) {
				const auto& o = regions[i].dstOffsets;
				const VkExtent3D extent {
				    static_cast<uint32_t>(std::abs(o[1].x - o[0].x)),
				    static_cast<uint32_t>(std::abs(o[1].y - o[0].y)),
				    static_cast<uint32_t>(std::abs(o[1].z - o[0].z))};
				op.texels += Texels(extent, regions[i].dstSubresource.layerCount);
			}
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdResolveImage(VkCommandBuffer cb, VkImage src,
                                               VkImageLayout src_layout, VkImage dst,
                                               VkImageLayout dst_layout, uint32_t count,
                                               const VkImageResolve* regions) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdResolveImage(cb, src, src_layout, dst, dst_layout, count, regions);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::ResolveImage, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.regions = count;
			for (uint32_t i = 0; i < count; ++i) {
				op.texels += Texels(regions[i].extent, regions[i].srcSubresource.layerCount);
			}
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdClearColorImage(VkCommandBuffer cb, VkImage image,
                                                  VkImageLayout layout,
                                                  const VkClearColorValue* color, uint32_t count,
                                                  const VkImageSubresourceRange* ranges) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdClearColorImage(cb, image, layout, color, count, ranges);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::ClearColorImage, KYTY_GPU_OP_CALLER,
		       [&](OpRecord& op, State&) { op.regions = count; });
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdClearDepthStencilImage(
    VkCommandBuffer cb, VkImage image, VkImageLayout layout, const VkClearDepthStencilValue* value,
    uint32_t count, const VkImageSubresourceRange* ranges) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdClearDepthStencilImage(cb, image, layout, value, count, ranges);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::ClearDepthStencilImage, KYTY_GPU_OP_CALLER,
		       [&](OpRecord& op, State&) { op.regions = count; });
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdClearAttachments(VkCommandBuffer cb, uint32_t attachment_count,
                                                   const VkClearAttachment* attachments,
                                                   uint32_t rect_count, const VkClearRect* rects) {
	g_real.vkCmdClearAttachments(cb, attachment_count, attachments, rect_count, rects);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::ClearAttachments, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.regions = rect_count;
			for (uint32_t i = 0; i < rect_count; ++i) {
				const auto& rect = rects[i];
				op.texels += Texels({rect.rect.extent.width, rect.rect.extent.height, 1},
				                    rect.layerCount) *
				             attachment_count;
			}
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdFillBuffer(VkCommandBuffer cb, VkBuffer buffer,
                                             VkDeviceSize offset, VkDeviceSize size,
                                             uint32_t data) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdFillBuffer(cb, buffer, offset, size, data);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::FillBuffer, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.bytes   = size == VK_WHOLE_SIZE ? std::numeric_limits<uint64_t>::max() : size;
			op.regions = 1;
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdUpdateBuffer(VkCommandBuffer cb, VkBuffer buffer,
                                               VkDeviceSize offset, VkDeviceSize size,
                                               const void* data) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdUpdateBuffer(cb, buffer, offset, size, data);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::UpdateBuffer, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.bytes   = size;
			op.regions = 1;
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdPipelineBarrier(
    VkCommandBuffer cb, VkPipelineStageFlags src_stages, VkPipelineStageFlags dst_stages,
    VkDependencyFlags flags, uint32_t memory_count, const VkMemoryBarrier* memory,
    uint32_t buffer_count, const VkBufferMemoryBarrier* buffers, uint32_t image_count,
    const VkImageMemoryBarrier* images) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdPipelineBarrier(cb, src_stages, dst_stages, flags, memory_count, memory,
	                            buffer_count, buffers, image_count, images);
	if (!IsGuest(cb)) {
		return;
	}
	if (memory_count != 0 && src_stages == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT &&
	    dst_stages == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) {
		g_totals.full_barriers.fetch_add(1, std::memory_order_relaxed);
	}
	uint32_t transitions = 0;
	for (uint32_t i = 0; i < image_count; ++i) {
		transitions += images[i].oldLayout != images[i].newLayout ? 1u : 0u;
	}
	CountBarrier(transitions);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::Barrier, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			op.src_stages   = src_stages;
			op.dst_stages   = dst_stages;
			op.mem_barriers = memory_count;
			op.buf_barriers = buffer_count;
			op.img_barriers = image_count;
			op.transitions  = transitions;
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdPipelineBarrier2(VkCommandBuffer         cb,
                                                   const VkDependencyInfo* dependency) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdPipelineBarrier2(cb, dependency);
	if (!IsGuest(cb) || dependency == nullptr) {
		return;
	}
	for (uint32_t i = 0; i < dependency->memoryBarrierCount; ++i) {
		const auto& memory = dependency->pMemoryBarriers[i];
		if (memory.srcStageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT &&
		    memory.dstStageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) {
			g_totals.full_barriers.fetch_add(1, std::memory_order_relaxed);
			break;
		}
	}
	uint32_t transitions = 0;
	for (uint32_t i = 0; i < dependency->imageMemoryBarrierCount; ++i) {
		const auto& image = dependency->pImageMemoryBarriers[i];
		transitions += image.oldLayout != image.newLayout ? 1u : 0u;
	}
	CountBarrier(transitions);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::Barrier2, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State&) {
			for (uint32_t i = 0; i < dependency->memoryBarrierCount; ++i) {
				op.src_stages |= dependency->pMemoryBarriers[i].srcStageMask;
				op.dst_stages |= dependency->pMemoryBarriers[i].dstStageMask;
			}
			for (uint32_t i = 0; i < dependency->bufferMemoryBarrierCount; ++i) {
				op.src_stages |= dependency->pBufferMemoryBarriers[i].srcStageMask;
				op.dst_stages |= dependency->pBufferMemoryBarriers[i].dstStageMask;
			}
			for (uint32_t i = 0; i < dependency->imageMemoryBarrierCount; ++i) {
				op.src_stages |= dependency->pImageMemoryBarriers[i].srcStageMask;
				op.dst_stages |= dependency->pImageMemoryBarriers[i].dstStageMask;
			}
			op.mem_barriers = dependency->memoryBarrierCount;
			op.buf_barriers = dependency->bufferMemoryBarrierCount;
			op.img_barriers = dependency->imageMemoryBarrierCount;
			op.transitions  = transitions;
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* info) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::None, true);
	}
	g_real.vkCmdBeginRendering(cb, info);
	if (!IsGuest(cb) || info == nullptr) {
		return;
	}
	g_counters.render_passes.fetch_add(1, std::memory_order_relaxed);
	g_pass_draw_class.store(static_cast<uint8_t>(PipelineClass::Unknown), std::memory_order_relaxed);
	if (IsCapture(cb)) [[unlikely]] {
		PassRecord pass;
		pass.width     = info->renderArea.extent.width;
		pass.height    = info->renderArea.extent.height;
		pass.layers    = info->layerCount;
		pass.view_mask = info->viewMask;
		pass.colors    = info->colorAttachmentCount;
		for (uint32_t i = 0; i < info->colorAttachmentCount && i < 8; ++i) {
			const auto& color = info->pColorAttachments[i];
			if (color.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
				pass.clear_mask |= 1u << i;
			}
			if (pass.color0 == VK_FORMAT_UNDEFINED) {
				pass.color0 = ViewFormat(color.imageView);
			}
		}
		if (info->pDepthAttachment != nullptr) {
			pass.depth = ViewFormat(info->pDepthAttachment->imageView);
			if (info->pDepthAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
				pass.clear_mask |= 1u << 8u;
			}
		}
		if (info->pStencilAttachment != nullptr) {
			if (pass.depth == VK_FORMAT_UNDEFINED) {
				pass.depth = ViewFormat(info->pStencilAttachment->imageView);
			}
			if (info->pStencilAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
				pass.clear_mask |= 1u << 9u;
			}
		}
		Record(cb, OpKind::BeginRendering, KYTY_GPU_OP_CALLER, [&](OpRecord& op, State& s) {
			s.current_pass = static_cast<uint32_t>(s.data->passes.size());
			s.data->passes.push_back(pass);
			s.views = std::max(1u, static_cast<uint32_t>(std::popcount(pass.view_mask)));
			op.pass = s.current_pass;
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdEndRendering(VkCommandBuffer cb) {
	g_real.vkCmdEndRendering(cb);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::EndRendering, KYTY_GPU_OP_CALLER, [&](OpRecord&, State& s) {
			s.current_pass = kNone;
			s.views        = 1;
			s.segment      = SegmentClass::None; // this stamp closes the pass
		});
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdBeginQuery(VkCommandBuffer cb, VkQueryPool pool, uint32_t query,
                                             VkQueryControlFlags flags) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdBeginQuery(cb, pool, query, flags);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::BeginQuery, KYTY_GPU_OP_CALLER);
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdEndQuery(VkCommandBuffer cb, VkQueryPool pool, uint32_t query) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdEndQuery(cb, pool, query);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::EndQuery, KYTY_GPU_OP_CALLER);
	}
}

VKAPI_ATTR void VKAPI_CALL HookCmdCopyQueryPoolResults(VkCommandBuffer cb, VkQueryPool pool,
                                                       uint32_t first, uint32_t count,
                                                       VkBuffer buffer, VkDeviceSize offset,
                                                       VkDeviceSize stride,
                                                       VkQueryResultFlags flags) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::Emulator);
	}
	g_real.vkCmdCopyQueryPoolResults(cb, pool, first, count, buffer, offset, stride, flags);
	if (IsCapture(cb)) [[unlikely]] {
		Record(cb, OpKind::CopyQueryResults, KYTY_GPU_OP_CALLER,
		       [&](OpRecord& op, State&) { op.regions = count; });
	}
}

VKAPI_ATTR VkResult VKAPI_CALL HookEndCommandBuffer(VkCommandBuffer cb) {
	if (IsCapture(cb)) [[unlikely]] {
		SegmentBoundary(cb, SegmentClass::None, true);
	}
	return g_real.vkEndCommandBuffer(cb);
}

VKAPI_ATTR VkResult VKAPI_CALL HookQueueSubmit(VkQueue queue, uint32_t count,
                                               const VkSubmitInfo* submits, VkFence fence) {
	{
		uint64_t cbs = 0;
		for (uint32_t i = 0; i < count; ++i) {
			cbs += submits[i].commandBufferCount;
		}
		g_totals.submit_calls.fetch_add(1, std::memory_order_relaxed);
		g_totals.submit_cbs.fetch_add(cbs, std::memory_order_relaxed);
	}
	if (g_capture_window.load(std::memory_order_relaxed)) [[unlikely]] {
		uint64_t cbs = 0;
		for (uint32_t i = 0; i < count; ++i) {
			cbs += submits[i].commandBufferCount;
		}
		g_submit_calls.fetch_add(1, std::memory_order_relaxed);
		g_submit_batches.fetch_add(count, std::memory_order_relaxed);
		g_submit_cbs.fetch_add(cbs, std::memory_order_relaxed);
	}
	return g_real.vkQueueSubmit(queue, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL HookQueueSubmit2(VkQueue queue, uint32_t count,
                                                const VkSubmitInfo2* submits, VkFence fence) {
	{
		uint64_t cbs = 0;
		for (uint32_t i = 0; i < count; ++i) {
			cbs += submits[i].commandBufferInfoCount;
		}
		g_totals.submit_calls.fetch_add(1, std::memory_order_relaxed);
		g_totals.submit_cbs.fetch_add(cbs, std::memory_order_relaxed);
	}
	if (g_capture_window.load(std::memory_order_relaxed)) [[unlikely]] {
		uint64_t cbs = 0;
		for (uint32_t i = 0; i < count; ++i) {
			cbs += submits[i].commandBufferInfoCount;
		}
		g_submit_calls.fetch_add(1, std::memory_order_relaxed);
		g_submit_batches.fetch_add(count, std::memory_order_relaxed);
		g_submit_cbs.fetch_add(cbs, std::memory_order_relaxed);
	}
	return g_real.vkQueueSubmit2(queue, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL HookCreateImageView(VkDevice device,
                                                   const VkImageViewCreateInfo* create,
                                                   const VkAllocationCallbacks* allocator,
                                                   VkImageView*                 view) {
	const auto result = g_real.vkCreateImageView(device, create, allocator, view);
	if (result == VK_SUCCESS && create != nullptr && view != nullptr) {
		std::lock_guard lock(g_view_mutex);
		g_view_formats[HandleBits(*view)] = create->format;
	}
	return result;
}

VKAPI_ATTR void VKAPI_CALL HookDestroyImageView(VkDevice device, VkImageView view,
                                                const VkAllocationCallbacks* allocator) {
	if (view != VK_NULL_HANDLE) {
		std::lock_guard lock(g_view_mutex);
		g_view_formats.erase(HandleBits(view));
	}
	g_real.vkDestroyImageView(device, view, allocator);
}

// KYTY_GPU_OP_COUNTERS=2: the fragment stage of every graphics pipeline made (libraries are not
// bound; a pipeline linked from libraries has no stages here and stays Unknown).
PipelineClass ClassOf(const VkGraphicsPipelineCreateInfo& info) {
	if (info.stageCount == 0 || info.pStages == nullptr) {
		return PipelineClass::Unknown;
	}
	for (uint32_t i = 0; i < info.stageCount; ++i) {
		if (info.pStages[i].stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
			const auto empty = EmptyFragment::Module(); // KYTY_EMPTY_FS
			return empty != nullptr && info.pStages[i].module == static_cast<VkShaderModule>(empty)
			           ? PipelineClass::EmptyFragment
			           : PipelineClass::GuestFragment;
		}
	}
	return PipelineClass::NoFragment;
}

VKAPI_ATTR VkResult VKAPI_CALL HookCreateGraphicsPipelines(VkDevice device, VkPipelineCache cache,
                                                           uint32_t                            count,
                                                           const VkGraphicsPipelineCreateInfo* infos,
                                                           const VkAllocationCallbacks*        allocator,
                                                           VkPipeline*                         pipelines) {
	const auto result = g_real.vkCreateGraphicsPipelines(device, cache, count, infos, allocator, pipelines);
	if (pipelines == nullptr || infos == nullptr) {
		return result;
	}
	std::lock_guard lock(g_pipeline_class_mutex);
	for (uint32_t i = 0; i < count; ++i) {
		if (pipelines[i] == VK_NULL_HANDLE || (infos[i].flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) != 0) {
			continue;
		}
		const auto pipeline_class = ClassOf(infos[i]);
		g_pipeline_class[HandleBits(pipelines[i])] = pipeline_class;
		g_totals.pipelines[static_cast<size_t>(pipeline_class)].fetch_add(1, std::memory_order_relaxed);
	}
	return result;
}

VKAPI_ATTR void VKAPI_CALL HookDestroyPipeline(VkDevice device, VkPipeline pipeline,
                                               const VkAllocationCallbacks* allocator) {
	if (pipeline != VK_NULL_HANDLE) {
		std::lock_guard lock(g_pipeline_class_mutex);
		g_pipeline_class.erase(HandleBits(pipeline));
	}
	g_real.vkDestroyPipeline(device, pipeline, allocator);
}

// gpucounts.csv (counters on, hang trace on): cumulative totals, appended at most once a second
// from OnGuestFlip. Per-window values are differences of two snapshots (the same t_ms groups one).
void WriteCountsFile(uint64_t flips) {
	static uint64_t last_ns = 0;
	static bool     failed  = false;
	const auto      now     = NowNs();
	if (failed || now - last_ns < 1'000'000'000ull) {
		return;
	}
	last_ns         = now;
	const auto dir  = HangTrace::OutputDirectory();
	if (dir.empty()) {
		return;
	}
	const auto path   = std::filesystem::path(dir) / "gpucounts.csv";
	static bool header = false;
	auto*       file   = std::fopen(path.string().c_str(), header ? "ab" : "wb");
	if (file == nullptr) {
		failed = true;
		return;
	}
	std::string out;
	if (!header) {
		out += "t_ms,flips,kind,name,value\n";
		header = true;
	}
	const auto t_ms = static_cast<unsigned long long>(HangTrace::NowNs() / 1'000'000ull);
	const auto row  = [&](const char* kind, std::string_view name, uint64_t value) {
		out += fmt::format("{},{},{},{},{}\n", t_ms, flips, kind, name, value);
	};
	const auto load = [](const std::atomic<uint64_t>& value) { return value.load(std::memory_order_relaxed); };
	row("total", "submit_calls", load(g_totals.submit_calls));
	row("total", "submit_cbs", load(g_totals.submit_cbs));
	row("total", "guest_cbs", g_totals.guest_cbs);
	row("total", "render_passes", g_totals.render_passes);
	row("total", "rendering_ends", g_totals.rendering_ends);
	row("total", "barriers", g_totals.barriers);
	row("total", "full_barriers", load(g_totals.full_barriers));
	for (size_t i = 0; i < kFlushCauseNames.size(); ++i) {
		row("flush", kFlushCauseNames[i], load(g_totals.flush[i]));
	}
	if (GetConfig().draw_counters) {
		for (size_t i = 0; i < kPipelineClassNames.size(); ++i) {
			row("draws", kPipelineClassNames[i], load(g_totals.draws[i]));
			row("pipelines", kPipelineClassNames[i], load(g_totals.pipelines[i]));
		}
		row("total", "fragment_switches", load(g_totals.fragment_switches));
	}
	for (auto* site = g_sites.load(std::memory_order_acquire); site != nullptr;
	     site       = site->next.load(std::memory_order_relaxed)) {
		if (const auto barriers = site->barriers.load(std::memory_order_relaxed); barriers != 0) {
			row("site_barriers", site->name, barriers);
		}
		if (const auto ends = site->end_renderings.load(std::memory_order_relaxed); ends != 0) {
			row("site_ends", site->name, ends);
		}
	}
	std::fwrite(out.data(), 1, out.size(), file);
	std::fclose(file);
}

// ------------------------------------------------------------------------------------------------
// Capture lifecycle (guest scheduler producer)

bool EnsurePool(GraphicContext& graphics) {
	auto& s = g_state;
	if (s.pool != nullptr) {
		return true;
	}
	if (s.disabled) {
		return false;
	}
	uint32_t family_count = 0;
	graphics.physical_device.getQueueFamilyProperties(&family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> families(family_count);
	graphics.physical_device.getQueueFamilyProperties(&family_count, families.data());
	const uint32_t valid_bits =
	    graphics.queue_family < family_count ? families[graphics.queue_family].timestampValidBits
	                                         : 0u;
	const double period = graphics.physical_device_properties.limits.timestampPeriod;
	if (valid_bits == 0 || valid_bits > 64 || !std::isfinite(period) || period <= 0.0 ||
	    (g_real.vkCmdWriteTimestamp2 == nullptr && g_real.vkCmdWriteTimestamp == nullptr) ||
	    g_real.vkCmdResetQueryPool == nullptr) {
		std::printf("GPU op profiler disabled: no usable timestamps (%u bits, %g ns)\n",
		            valid_bits, period);
		std::fflush(stdout);
		s.disabled = true;
		return false;
	}
	vk::QueryPoolCreateInfo create {};
	create.queryType  = vk::QueryType::eTimestamp;
	create.queryCount = GetConfig().queries;
	vk::QueryPool pool = nullptr;
	if (graphics.device.createQueryPool(&create, nullptr, &pool) != vk::Result::eSuccess ||
	    pool == nullptr) {
		std::printf("GPU op profiler disabled: query pool creation failed\n");
		std::fflush(stdout);
		s.disabled = true;
		return false;
	}
	s.device     = graphics.device;
	s.pool       = pool;
	s.capacity   = create.queryCount;
	s.period_ns  = period;
	s.valid_bits = valid_bits;
	return true;
}

// Caller holds g_state.mutex.
void BeginCapturedBuffer(VkCommandBuffer cb, uint64_t tick) {
	auto& s          = g_state;
	s.current_cb     = static_cast<uint32_t>(s.data->cbs.size());
	s.current_pass   = kNone;
	s.views          = 1;
	s.bound          = {};
	s.segment        = SegmentClass::None;
	CbRecord record;
	record.tick  = tick;
	record.query = AllocateStamp(cb);
	s.data->cbs.push_back(record);
	g_capture_cb.store(cb, std::memory_order_relaxed);
}

void StartCapture(VkCommandBuffer cb, uint64_t tick, uint64_t flip) {
	auto&           s = g_state;
	std::lock_guard lock(s.mutex);
	auto            data = std::make_unique<CaptureData>();
	data->index      = s.captures;
	data->flip       = flip;
	const auto stamps = GetConfig().stamps;
	data->pass_stamps =
	    stamps == StampMode::Passes || (stamps == StampMode::Alternate && (s.captures & 1u) != 0);
	data->start_ns   = NowNs();
	data->capacity   = s.capacity;
	data->period_ns  = s.period_ns;
	data->valid_bits = s.valid_bits;
	data->ops.reserve(std::min<size_t>(s.capacity, 65536));
	data->cbs.reserve(1024);
	data->passes.reserve(1024);
	s.data       = std::move(data);
	s.next_query = 0;
	s.start_flip = flip;
	s.phase      = Phase::Active;
	g_submit_calls.store(0, std::memory_order_relaxed);
	g_submit_batches.store(0, std::memory_order_relaxed);
	g_submit_cbs.store(0, std::memory_order_relaxed);
	g_capture_window.store(true, std::memory_order_relaxed);
	// Outside rendering, before any stamp of this capture. Later submissions on the queue are
	// ordered after the reset by the query-reset execution dependency.
	g_real.vkCmdResetQueryPool(cb, static_cast<VkQueryPool>(s.pool), 0, s.capacity);
	BeginCapturedBuffer(cb, tick);
}

// Stops stamping; the capture waits for its last tick.
void CloseCapture(uint64_t last_tick) {
	auto&           s = g_state;
	std::lock_guard lock(s.mutex);
	g_capture_cb.store(nullptr, std::memory_order_relaxed);
	g_capture_window.store(false, std::memory_order_relaxed);
	s.last_tick = last_tick;
	s.phase     = Phase::Pending;
	if (s.data) {
		s.data->end_ns         = NowNs();
		s.data->submit_calls   = g_submit_calls.load(std::memory_order_relaxed);
		s.data->submit_batches = g_submit_batches.load(std::memory_order_relaxed);
		s.data->submit_cbs     = g_submit_cbs.load(std::memory_order_relaxed);
	}
}

// All captured ticks completed: read without waiting and hand off to the writer.
void CollectCapture() {
	auto&                        s = g_state;
	std::unique_ptr<CaptureData> data;
	{
		std::lock_guard lock(s.mutex);
		data = std::move(s.data);
		if (!data) {
			s.phase = Phase::Idle;
			return;
		}
		data->used = s.next_query;
		data->results.assign(size_t {data->used} * 2u, 0);
		if (data->used != 0) {
			const auto result = s.device.getQueryPoolResults(
			    s.pool, 0, data->used, data->results.size() * sizeof(uint64_t),
			    data->results.data(), sizeof(uint64_t) * 2u,
			    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
			data->read_error = result != vk::Result::eSuccess && result != vk::Result::eNotReady;
		}
		s.next_query = 0;
		++s.captures;
		s.phase     = Phase::Idle;
		s.next_ns   = NowNs() + static_cast<uint64_t>(GetConfig().period_s * 1e9);
		s.seen_flip = g_flip_seq.load(std::memory_order_acquire);
	}
	GetWriter().Push(std::move(data));
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Public API

namespace Detail {

void EnterSite(Site& site, Site*& previous, bool& scope_owner) noexcept {
	LinkSite(site);
	previous = t_site;
	t_site   = &site;
	if (t_scope == nullptr) {
		t_scope     = &site;
		scope_owner = true;
	}
}

void LeaveSite(Site* previous, bool scope_owner) noexcept {
	t_site = previous;
	if (scope_owner) {
		t_scope = nullptr;
	}
}

void CurrentSites(const void** site, const void** scope) noexcept {
	*site  = t_site;
	*scope = t_scope;
}

void CountBarrierBatch(BarrierBatchEvent event, uint64_t amount) noexcept {
	const auto index = static_cast<size_t>(event);
	if (index < g_counters.batch.size() && amount != 0) {
		g_counters.batch[index].fetch_add(amount, std::memory_order_relaxed);
	}
}

void CountFlush(FlushCause cause) noexcept {
	const auto index = static_cast<size_t>(cause);
	if (index < g_totals.flush.size()) {
		g_totals.flush[index].fetch_add(1, std::memory_order_relaxed);
	}
}

void CountEndRendering() noexcept {
	g_counters.rendering_ends.fetch_add(1, std::memory_order_relaxed);
	auto* site = t_site != nullptr ? t_site : &g_unknown_site;
	site->end_renderings.fetch_add(1, std::memory_order_relaxed);
}

} // namespace Detail

bool Enabled() {
	const auto& config = GetConfig();
	return config.capture || config.counters;
}

bool CaptureEnabled() {
	return GetConfig().capture;
}

void InstallHooks(GraphicContext& /*graphics*/) {
	if (!Enabled() || Detail::g_active) {
		return;
	}
	auto&      d       = VULKAN_HPP_DEFAULT_DISPATCHER;
	const bool capture = CaptureEnabled();
#define KYTY_GPU_OP_HOOK(member, hook)                                                             \
	if (d.member != nullptr) {                                                                     \
		g_real.member = d.member;                                                                  \
		d.member      = hook;                                                                      \
	}
	// Counters.
	KYTY_GPU_OP_HOOK(vkCmdPipelineBarrier, &HookCmdPipelineBarrier);
	KYTY_GPU_OP_HOOK(vkCmdPipelineBarrier2, &HookCmdPipelineBarrier2);
	KYTY_GPU_OP_HOOK(vkCmdBeginRendering, &HookCmdBeginRendering);
	// gpucounts.csv submit totals (the capture also counts its own window through them).
	KYTY_GPU_OP_HOOK(vkQueueSubmit, &HookQueueSubmit);
	KYTY_GPU_OP_HOOK(vkQueueSubmit2, &HookQueueSubmit2);
	if (GetConfig().draw_counters) {
		KYTY_GPU_OP_HOOK(vkCreateGraphicsPipelines, &HookCreateGraphicsPipelines);
		KYTY_GPU_OP_HOOK(vkDestroyPipeline, &HookDestroyPipeline);
	}
	if (capture || GetConfig().draw_counters) {
		KYTY_GPU_OP_HOOK(vkCmdBindPipeline, &HookCmdBindPipeline);
		KYTY_GPU_OP_HOOK(vkCmdDraw, &HookCmdDraw);
		KYTY_GPU_OP_HOOK(vkCmdDrawIndexed, &HookCmdDrawIndexed);
		KYTY_GPU_OP_HOOK(vkCmdDrawIndirect, &HookCmdDrawIndirect);
		KYTY_GPU_OP_HOOK(vkCmdDrawIndexedIndirect, &HookCmdDrawIndexedIndirect);
		KYTY_GPU_OP_HOOK(vkCmdDrawIndirectCount, &HookCmdDrawIndirectCount);
		KYTY_GPU_OP_HOOK(vkCmdDrawIndexedIndirectCount, &HookCmdDrawIndexedIndirectCount);
		KYTY_GPU_OP_HOOK(vkCmdDrawMeshTasksEXT, &HookCmdDrawMeshTasksEXT);
		KYTY_GPU_OP_HOOK(vkCmdDrawMeshTasksIndirectEXT, &HookCmdDrawMeshTasksIndirectEXT);
		KYTY_GPU_OP_HOOK(vkCmdDrawMeshTasksIndirectCountEXT,
		                 &HookCmdDrawMeshTasksIndirectCountEXT);
	}
	if (capture) {
		g_real.vkCmdWriteTimestamp2 = d.vkCmdWriteTimestamp2;
		g_real.vkCmdWriteTimestamp  = d.vkCmdWriteTimestamp;
		g_real.vkCmdResetQueryPool  = d.vkCmdResetQueryPool;
		KYTY_GPU_OP_HOOK(vkCmdDispatch, &HookCmdDispatch);
		KYTY_GPU_OP_HOOK(vkCmdDispatchIndirect, &HookCmdDispatchIndirect);
		KYTY_GPU_OP_HOOK(vkCmdCopyBuffer, &HookCmdCopyBuffer);
		KYTY_GPU_OP_HOOK(vkCmdCopyImage, &HookCmdCopyImage);
		KYTY_GPU_OP_HOOK(vkCmdCopyBufferToImage, &HookCmdCopyBufferToImage);
		KYTY_GPU_OP_HOOK(vkCmdCopyImageToBuffer, &HookCmdCopyImageToBuffer);
		KYTY_GPU_OP_HOOK(vkCmdBlitImage, &HookCmdBlitImage);
		KYTY_GPU_OP_HOOK(vkCmdResolveImage, &HookCmdResolveImage);
		KYTY_GPU_OP_HOOK(vkCmdClearColorImage, &HookCmdClearColorImage);
		KYTY_GPU_OP_HOOK(vkCmdClearDepthStencilImage, &HookCmdClearDepthStencilImage);
		KYTY_GPU_OP_HOOK(vkCmdClearAttachments, &HookCmdClearAttachments);
		KYTY_GPU_OP_HOOK(vkCmdFillBuffer, &HookCmdFillBuffer);
		KYTY_GPU_OP_HOOK(vkCmdUpdateBuffer, &HookCmdUpdateBuffer);
		KYTY_GPU_OP_HOOK(vkCmdEndRendering, &HookCmdEndRendering);
		KYTY_GPU_OP_HOOK(vkCmdBeginQuery, &HookCmdBeginQuery);
		KYTY_GPU_OP_HOOK(vkCmdEndQuery, &HookCmdEndQuery);
		KYTY_GPU_OP_HOOK(vkCmdCopyQueryPoolResults, &HookCmdCopyQueryPoolResults);
		KYTY_GPU_OP_HOOK(vkEndCommandBuffer, &HookEndCommandBuffer);
		KYTY_GPU_OP_HOOK(vkCreateImageView, &HookCreateImageView);
		KYTY_GPU_OP_HOOK(vkDestroyImageView, &HookDestroyImageView);
	}
#undef KYTY_GPU_OP_HOOK
	LinkSite(g_unknown_site);
	g_state.next_ns   = NowNs() + static_cast<uint64_t>(GetConfig().period_s * 1e9);
	Detail::g_active = true;
	std::printf("GPU op profiler: counters %s, sampled capture %s (period %.3f s, %u queries)\n",
	            GetConfig().counters ? "on" : "off", capture ? "on" : "off",
	            GetConfig().period_s, GetConfig().queries);
	std::fflush(stdout);
}

void OnBeginCommand(GraphicContext& graphics, vk::CommandBuffer buffer, uint64_t current_tick,
                    uint64_t known_gpu_tick) {
	if (!Detail::g_active) {
		return;
	}
	const auto cb = static_cast<VkCommandBuffer>(buffer);
	g_guest_cb.store(cb, std::memory_order_relaxed);
	g_counters.command_buffers.fetch_add(1, std::memory_order_relaxed);
	if (!CaptureEnabled()) {
		return;
	}
	auto& s = g_state;
	switch (s.phase) {
		case Phase::Idle: {
			const auto flip = g_flip_seq.load(std::memory_order_acquire);
			if (flip == s.seen_flip) {
				return; // start only at a frame boundary
			}
			s.seen_flip = flip;
			const auto max = GetConfig().max_captures;
			if (NowNs() < s.next_ns || (max != 0 && s.captures >= max) ||
			    !EnsurePool(graphics)) {
				return;
			}
			StartCapture(cb, current_tick, flip);
			return;
		}
		case Phase::Active: {
			bool overflow = false;
			{
				std::lock_guard lock(s.mutex);
				overflow = s.data != nullptr && s.data->overflow != 0;
				if (g_flip_seq.load(std::memory_order_acquire) == s.start_flip && !overflow) {
					BeginCapturedBuffer(cb, current_tick);
					return;
				}
				if (overflow) {
					s.data->truncated = true;
				}
			}
			// Every earlier buffer was submitted with a smaller tick.
			CloseCapture(current_tick - 1);
			[[fallthrough]];
		}
		case Phase::Pending:
			if (known_gpu_tick >= s.last_tick) {
				CollectCapture();
			}
			return;
	}
}

void OnSchedulerShutdown(GraphicContext& graphics) {
	if (!Detail::g_active) {
		return;
	}
	g_guest_cb.store(nullptr, std::memory_order_relaxed);
	if (!CaptureEnabled()) {
		return;
	}
	auto& s = g_state;
	if (s.phase == Phase::Active) {
		CloseCapture(0);
	}
	if (s.phase == Phase::Pending) {
		// The scheduler waited for every submitted tick before calling this.
		CollectCapture();
	}
	if (s.pool != nullptr) {
		graphics.device.destroyQueryPool(s.pool, nullptr);
		s.pool     = nullptr;
		s.disabled = true;
	}
	GetWriter().Stop();
}

void OnGuestFlip() {
	g_flip_seq.fetch_add(1, std::memory_order_acq_rel);
	if (!Detail::g_active || !GetConfig().counters) {
		return;
	}
	HangTrace::GpuOpCounts counts;
	counts.render_passes      = g_counters.render_passes.exchange(0, std::memory_order_relaxed);
	counts.barriers           = g_counters.barriers.exchange(0, std::memory_order_relaxed);
	counts.layout_transitions = g_counters.transitions.exchange(0, std::memory_order_relaxed);
	counts.command_buffers    = g_counters.command_buffers.exchange(0, std::memory_order_relaxed);
	const auto take_batch     = [](BarrierBatchEvent event) {
		return g_counters.batch[static_cast<size_t>(event)].exchange(0, std::memory_order_relaxed);
	};
	counts.barrier_requests      = take_batch(BarrierBatchEvent::Requests);
	counts.barriers_merged       = take_batch(BarrierBatchEvent::Merged);
	counts.barriers_elided       = take_batch(BarrierBatchEvent::Elided);
	counts.barriers_sunk         = take_batch(BarrierBatchEvent::Sunk);
	counts.barrier_render_splits = take_batch(BarrierBatchEvent::RenderSplits);
	counts.draw_write_sinks      = take_batch(BarrierBatchEvent::DrawWriteSinks);
	counts.rendering_ends = g_counters.rendering_ends.exchange(0, std::memory_order_relaxed);
	HangTrace::RecordGpuOpCounts(counts);
	if (HangTrace::Enabled()) {
		// One flip thread at a time publishes (flips are serialized by the presenter).
		static std::mutex           totals_mutex;
		std::lock_guard             lock(totals_mutex);
		g_totals.guest_cbs += counts.command_buffers;
		g_totals.render_passes += counts.render_passes;
		g_totals.rendering_ends += counts.rendering_ends;
		g_totals.barriers += counts.barriers;
		WriteCountsFile(g_flip_seq.load(std::memory_order_relaxed));
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuRenderingEnds, counts.rendering_ends);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuDrawWriteSinks, counts.draw_write_sinks);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuBarrierRequests, counts.barrier_requests);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuBarriersMerged, counts.barriers_merged);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuBarriersElided, counts.barriers_elided);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuBarriersSunk, counts.barriers_sunk);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuBarrierRenderSplits,
	                          counts.barrier_render_splits);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuRenderPassBegins, counts.render_passes);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuPipelineBarriers, counts.barriers);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuImageLayoutTransitions,
	                          counts.layout_transitions);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuGuestCommandBuffers,
	                          counts.command_buffers);
	if (Profiler::AggregateEnabled() && tracy::ProfilerAvailable() && TracyIsConnected) {
		for (auto* site = g_sites.load(std::memory_order_acquire); site != nullptr;
		     site       = site->next.load(std::memory_order_relaxed)) {
			TracyPlot(site->plot_name,
			          static_cast<int64_t>(site->barriers.load(std::memory_order_relaxed)));
			const auto ends = site->end_renderings.load(std::memory_order_relaxed);
			if (ends != 0) {
				TracyPlot(site->end_plot_name, static_cast<int64_t>(ends));
			}
		}
	}
}

void RegisterShader(uint64_t program_id, const char* stage, uint64_t guest_hash) {
	if (!CaptureEnabled()) {
		return;
	}
	std::lock_guard lock(g_registry_mutex);
	g_shaders[program_id] = {stage != nullptr ? stage : "?", guest_hash};
}

void RegisterGraphicsPipeline(vk::Pipeline pipeline, const uint64_t* vertex_program_ids,
                              uint32_t vertex_program_count, uint64_t pixel_program_id) {
	if (!CaptureEnabled() || pipeline == nullptr) {
		return;
	}
	PipelineInfo info;
	for (uint32_t i = 0; i < vertex_program_count && i < info.vertex.size(); ++i) {
		if (vertex_program_ids[i] != 0) {
			info.vertex[info.vertex_count++] = vertex_program_ids[i];
		}
	}
	info.pixel = pixel_program_id;
	std::lock_guard lock(g_registry_mutex);
	g_pipelines[HandleBits(static_cast<VkPipeline>(pipeline))] = info;
}

void RegisterComputePipeline(vk::Pipeline pipeline, uint64_t compute_program_id) {
	if (!CaptureEnabled() || pipeline == nullptr) {
		return;
	}
	PipelineInfo info;
	info.compute = compute_program_id;
	std::lock_guard lock(g_registry_mutex);
	g_pipelines[HandleBits(static_cast<VkPipeline>(pipeline))] = info;
}

} // namespace Libs::Graphics::GpuOpProfiler
