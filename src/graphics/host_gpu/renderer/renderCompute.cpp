#include "graphics/host_gpu/watchRange.h"
#include "common/staticOnce.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/renderer/nativeRayTracing.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hangTrace.h"
#include "common/liveSwitch.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/dispatchGuard.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/renderer/drawPrep/drawPrep.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/guest_gpu/command_processor/passCensus.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/rtBvhDump.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"

#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include <algorithm>
#include <bit>
#include <tuple>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <limits>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

// A dispatch without a compute program: the pass census drop line names the shader and the reason
// (PipelineCache::TakeEmptyProgramReason); the console gets one line per shader and reason.
static void DropProgramless(PipelineCache& cache) {
	(void)cache;
	uint64_t   hash   = 0;
	const auto reason = PipelineCache::TakeEmptyProgramReason(&hash);
	const auto text   = reason.empty() ? std::string("dropped: no compute program")
	                                   : "dropped: no compute program: " + reason;
	PassCensus::Drop('c', hash, text.c_str());
	static std::mutex                   mutex;
	static std::unordered_set<std::string> logged;
	const auto key = fmt::format("{:016x} {}", hash, reason);
	{
		std::scoped_lock lock(mutex);
		if (logged.size() > 256 || !logged.insert(key).second) return;
	}
	std::printf("Dispatch dropped: CS 0x%016llx has no program: %s\n", static_cast<unsigned long long>(hash),
	            reason.empty() ? "reason unknown" : reason.c_str());
	std::fflush(stdout);
}
// KYTY_DISPATCH_OVERLAP accounting, printed every 10 s once a direct dispatch ran:
//   DispatchOverlap 10s (on|off): <direct> direct dispatches, <pairs> unordered pairs (<pct>%),
//   <overlapped> overlapped, <indirect> indirect (<empty> empty skipped, KYTY_SKIP_EMPTY_INDIRECT)
// An unordered pair is a direct dispatch that followed another with nothing in between (it would
// overlap with the switch on); counted with the switch off too, for sizing.
static std::atomic<uint64_t> g_overlap_direct {0}, g_overlap_pairs {0}, g_overlap_taken {0},
    g_overlap_indirect {0}, g_overlap_empty_indirect {0};
static Live::Switch g_skip_empty_indirect("KYTY_SKIP_EMPTY_INDIRECT", Live::ParseDefaultOn);
// KYTY_INDIRECT_THREADS_GPU=1 (default off, live; GT7 fork 633e00962): DISPATCH_INDIRECT with the
// thread-dimension initiator carries thread counts, not workgroup counts. The command processor
// reads them on the CPU and dispatches directly, but the previous dispatch usually writes them on
// the GPU, so each read drains the GPU. With the switch the dispatch stays indirect: the dispatch
// guard's pass (dispatchGuard.h) converts the counts to workgroup counts on the GPU and keeps a copy
// of the thread counts, which the shader reads through a device address to bound its lanes
// (ShaderComputeInputInfo::dispatch_dimensions_indirect).
static Live::Switch g_indirect_threads_gpu("KYTY_INDIRECT_THREADS_GPU", Live::ParseDefaultOff);

bool RenderExecutor::IndirectThreadsOnGpu() {
	return g_indirect_threads_gpu.On();
}
static std::atomic<int64_t> g_overlap_last_print_ns {0};

// KYTY_SKIP_EMPTY_LOG=1 (diagnostic, default off): which programs KYTY_SKIP_EMPTY_INDIRECT drops. Per program code
// address: the first 4 skips, then every 256th, with the time since start and how many dispatches of that code ran
// before; and once, the shader hash of that code when one of its dispatches runs.
namespace SkipEmptyLog {
static bool Enabled() {
	static const bool on = [] {
		const auto* value = std::getenv("KYTY_SKIP_EMPTY_LOG");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}
struct Entry {
	uint64_t skips  = 0;
	uint64_t runs   = 0;
	bool     mapped = false;
};
static std::mutex                             g_mutex;
static std::unordered_map<uint64_t, Entry>    g_entries;
static double Seconds() {
	static const auto start = std::chrono::steady_clock::now();
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
static void NoteSkip(uint64_t code, uint64_t args) {
	if (!Enabled()) return;
	std::scoped_lock lock(g_mutex);
	auto& e = g_entries[code];
	const auto n = e.skips++;
	if (n < 4 || n % 256 == 0) {
		std::printf("SkipEmpty: t=%.1f code 0x%llx args 0x%llx skip #%llu (runs before %llu)\n", Seconds(),
		            static_cast<unsigned long long>(code), static_cast<unsigned long long>(args),
		            static_cast<unsigned long long>(n + 1), static_cast<unsigned long long>(e.runs));
		std::fflush(stdout);
	}
}
static void NoteRun(uint64_t code, uint64_t hash, uint64_t args) {
	if (!Enabled()) return;
	std::scoped_lock lock(g_mutex);
	auto& e = g_entries[code];
	e.runs++;
	if (!e.mapped) {
		e.mapped = true;
		std::printf("SkipEmpty: t=%.1f code 0x%llx = CS 0x%016llx (args 0x%llx; %llu skips, run #%llu)\n", Seconds(),
		            static_cast<unsigned long long>(code), static_cast<unsigned long long>(hash),
		            static_cast<unsigned long long>(args), static_cast<unsigned long long>(e.skips),
		            static_cast<unsigned long long>(e.runs));
		std::fflush(stdout);
	}
}
} // namespace SkipEmptyLog

// KYTY_SKIP_EMPTY_STRICT (live, default on): the guest copy of an indirect dispatch's arguments is
// read (KYTY_SKIP_EMPTY_INDIRECT, KYTY_WORKGROUP_IMAGES) only when it is current. HasGpuDirtyBytes
// alone misses two GPU-owned states: a queued download or side readback removes the exact dirty
// range when it is queued, before its backing publication lands, and a GPU-modified image over the
// range. In both the backing can still hold older bytes (the indirect draw path, TryDrawIndirectNative,
// already treats a pending publication as GPU-owned). Off: the HasGpuDirtyBytes test only, as before.
static Live::Switch g_skip_empty_strict("KYTY_SKIP_EMPTY_STRICT", Live::ParseDefaultOn);
static bool IndirectArgsCleanForBackingRead(RenderContext& context, uint64_t address, uint64_t size) {
	auto& buffers = context.GetBufferCache();
	if (buffers.HasGpuDirtyBytes(address, size)) {
		return false;
	}
	if (g_skip_empty_strict.Get() == 0) {
		return true;
	}
	return !buffers.HasPendingBackingPublication(address, size) &&
	       !context.GetTextureCache().IsRegionGpuModified(address, size);
}

// KYTY_DISPATCH_TRACE=<hash>[,<hash>...] or all (hexadecimal guest shader hashes, diagnostic): logs the
// first KYTY_DISPATCH_TRACE_LIMIT (default 64) dispatches of each listed program: tick, groups or
// indirect argument address (with the argument bytes guest memory holds and whether a GPU write of
// them is pending), the first 16 user SGPRs, the bound buffers and the bound images.
// KYTY_DISPATCH_TRACE_IMAGE=<hex base>[,<hex base>...] (diagnostic): also traces every dispatch of any
// program that binds an image or buffer with one of those base addresses (who writes or reads a
// render target), within the same per-program limit.
struct DispatchTraceRequest {
	std::vector<uint64_t> hashes;
	std::vector<uint64_t> bases;
	uint32_t              limit = 64;
};
static const DispatchTraceRequest& DispatchTrace() {
	static const DispatchTraceRequest request = [] {
		DispatchTraceRequest parsed;
		if (const auto* value = std::getenv("KYTY_DISPATCH_TRACE"); value != nullptr) {
			std::string_view text(value);
			while (!text.empty()) {
				const auto comma = text.find(',');
				const auto item  = std::string(text.substr(0, comma));
				if (item == "all") {
					parsed.hashes.push_back(0);
				} else if (!item.empty()) {
					parsed.hashes.push_back(std::strtoull(item.c_str(), nullptr, 16));
				}
				if (comma == std::string_view::npos) break;
				text.remove_prefix(comma + 1);
			}
		}
		if (const auto* value = std::getenv("KYTY_DISPATCH_TRACE_IMAGE"); value != nullptr) {
			std::string_view text(value);
			while (!text.empty()) {
				const auto comma = text.find(',');
				const auto item  = std::string(text.substr(0, comma));
				if (!item.empty()) parsed.bases.push_back(std::strtoull(item.c_str(), nullptr, 16));
				if (comma == std::string_view::npos) break;
				text.remove_prefix(comma + 1);
			}
		}
		if (const auto* value = std::getenv("KYTY_DISPATCH_TRACE_LIMIT"); value != nullptr) {
			parsed.limit = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
		}
		return parsed;
	}();
	return request;
}

// KYTY_DUMP_CS_IMAGES=<hash>:<ordinal>[:<ordinal>...][,<hash>:...];<folder> (diagnostic): after the
// listed dispatches (0-based, counted per program) of a call-split kernel, every colour image it binds
// is copied out and written to <folder> as raw linear texels (TextureCache::DebugDumpImage), named
// cs<hash>_d<ordinal>_t<index>[w]_<address>_<w>x<h>_vk<format>_b<bytes>.raw. Local files only.
// Without ordinals (<hash>[,<hash>...];<folder>): trigger mode, see below (create <folder>/dump.now).
// KYTY_DUMP_IMAGE_EXTRA=<hex address>[,...]: also dumped with every KYTY_DUMP_CS_IMAGES dump.
struct CsImageDump {
	struct Item {
		std::string prefix;
		uint64_t    base   = 0;
		std::string what;
		uint32_t    width  = 0;
		uint32_t    height = 0;
	};
	std::vector<Item> items;
};
// Collected before the dispatch (a call-split chain reuses the resource state), dumped after it.
static CsImageDump PlanCsImageDump(const ShaderComputeInputInfo& input_info) {
	struct Request {
		std::unordered_map<uint64_t, std::vector<uint32_t>> ordinals;
		std::string                                         folder;
	};
	static const Request request = [] {
		Request parsed;
		const char* value = std::getenv("KYTY_DUMP_CS_IMAGES");
		if (value == nullptr) return parsed;
		const std::string text(value);
		const auto        split = text.find(';');
		if (split == std::string::npos) return parsed;
		parsed.folder = text.substr(split + 1);
		std::string_view list(text.data(), split);
		while (!list.empty()) {
			const auto comma = list.find(',');
			std::string item(list.substr(0, comma));
			char*       rest = nullptr;
			const auto  hash = std::strtoull(item.c_str(), &rest, 16);
			auto&       out  = parsed.ordinals[hash];
			while (rest != nullptr && *rest == ':') {
				out.push_back(static_cast<uint32_t>(std::strtoul(rest + 1, &rest, 10)));
			}
			if (comma == std::string_view::npos) break;
			list.remove_prefix(comma + 1);
		}
		return parsed;
	}();
	if (request.ordinals.empty() || input_info.stage.program == nullptr || input_info.stage.resources == nullptr) {
		return {};
	}
	const auto& program = *input_info.stage.program;
	const auto  wanted  = request.ordinals.find(program.shader_hash);
	if (wanted == request.ordinals.end()) return {};
	static std::unordered_map<uint64_t, uint32_t> counts; // GPU thread only
	const uint32_t ordinal = counts[program.shader_hash]++;
	if (wanted->second.empty()) {
		// No ordinals: trigger mode. While <folder>/dump.now exists, the next dispatch of each listed
		// program is dumped once; the file is removed when every listed program has been dumped.
		// A number in dump.now (e.g. 6) dumps that many consecutive dispatches of each listed program.
		static std::unordered_map<uint64_t, uint32_t> dumped;
		static uint32_t                               polls = 0;
		static uint32_t                               want  = 0;
		std::error_code                               error;
		const auto trigger = std::filesystem::path(request.folder) / "dump.now";
		if ((polls++ & 7u) != 0u && dumped.empty()) return {};
		if (!std::filesystem::exists(trigger, error)) {
			dumped.clear();
			want = 0;
			return {};
		}
		if (want == 0) {
			want = 1;
			if (std::FILE* file = std::fopen(trigger.string().c_str(), "rb"); file != nullptr) {
				unsigned value = 0;
				if (std::fscanf(file, "%u", &value) == 1 && value > 0) want = std::min(value, 64u);
				std::fclose(file);
			}
		}
		if (dumped[program.shader_hash] >= want) return {};
		dumped[program.shader_hash]++;
		if (std::ranges::all_of(request.ordinals, [&](const auto& item) { return dumped[item.first] >= want; })) {
			std::filesystem::remove(trigger, error);
		}
	} else if (std::ranges::find(wanted->second, ordinal) == wanted->second.end()) {
		return {};
	}
	CsImageDump dump;
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto r = DecodeNativeDescriptor<ShaderTextureResource>(input_info.stage.resources->images[i]);
		dump.items.push_back(
		    {fmt::format("{}/cs{:016x}_d{}_t{}{}", request.folder, program.shader_hash, ordinal, i,
		                 program.info.images[i].written ? "w" : ""),
		     r.Base40(),
		     fmt::format("CS 0x{:016x} dispatch {} image {}{} fmt {} {}x{}", program.shader_hash, ordinal, i,
		                 program.info.images[i].written ? " (written)" : "", static_cast<uint32_t>(r.Format()),
		                 static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u),
		     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u});
	}
	if (const char* extra = std::getenv("KYTY_DUMP_IMAGE_EXTRA"); extra != nullptr) {
		for (const char* at = extra; *at != '\0';) {
			char*      rest = nullptr;
			const auto base = std::strtoull(at, &rest, 16);
			if (rest == at) break;
			// <address>@<w>x<h>: prefer the cached image of that extent (aliased targets).
			uint32_t want_w = 0;
			uint32_t want_h = 0;
			if (*rest == '@') {
				want_w = static_cast<uint32_t>(std::strtoul(rest + 1, &rest, 10));
				if (*rest == 'x') want_h = static_cast<uint32_t>(std::strtoul(rest + 1, &rest, 10));
			}
			dump.items.push_back({fmt::format("{}/cs{:016x}_d{}_x", request.folder, program.shader_hash, ordinal),
			                      base, fmt::format("CS 0x{:016x} dispatch {} extra", program.shader_hash, ordinal),
			                      want_w, want_h});
			at = *rest == ',' ? rest + 1 : rest;
		}
	}
	return dump;
}

static void RunCsImageDump(RenderContext& context, const CsImageDump& dump) {
	for (const auto& item: dump.items) {
		const bool ok = item.base != 0 && context.GetTextureCache().DebugDumpImage(item.base, item.width, item.height, item.prefix);
		LOGF("KYTY_DUMP_CS_IMAGES: %s base 0x%" PRIx64 ": %s\n", item.what.c_str(), item.base,
		     ok ? "dumped" : "not cached");
	}
}

// KYTY_NAN_SCAN=<folder>[:<count>] (diagnostic): while <folder>/scan.now exists, the next <count>
// (default 8000) compute dispatches and draws have the images they read and write (dispatches) or
// their colour targets (draws) checked for NaN/Inf after they ran (TextureCache::DebugNanScan: a
// band of rows, float formats only); "NanScan:" lines in submission order name the op, the shader
// and the image. Then the file is removed. Returns the op's ordinal (0: not scanning).
uint32_t NanScanTake() {
	struct Request {
		std::filesystem::path trigger;
		uint32_t              count = 8000;
	};
	static constexpr auto request_init = [] {
		Request parsed;
		if (const char* value = std::getenv("KYTY_NAN_SCAN"); value != nullptr && value[0] != '\0') {
			std::string text(value);
			if (const auto colon = text.rfind(':'); colon != std::string::npos && colon > 1) {
				parsed.count = static_cast<uint32_t>(std::strtoul(text.c_str() + colon + 1, nullptr, 10));
				text.resize(colon);
			}
			parsed.trigger = std::filesystem::path(text) / "scan.now";
		}
		return parsed;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(request_init())>> request_once;
	const Request& request = request_once.Get(request_init);
	if (request.trigger.empty()) return 0;
	static uint32_t left  = 0; // GPU thread only
	static uint32_t polls = 0;
	static uint32_t scan  = 0;
	if (left == 0) {
		if ((polls++ & 255u) != 0u) return 0;
		std::error_code error;
		if (!std::filesystem::exists(request.trigger, error)) return 0;
		std::filesystem::remove(request.trigger, error);
		scan++;
		std::printf("NanScan: start of scan %u, %u operations\n", scan, request.count);
		left = request.count;
	}
	left--;
	if (left == 0) std::printf("NanScan: end of scan %u (recorded; results follow as submissions complete)\n", scan);
	return request.count - left;
}


// KYTY_PASS_DUMP=<folder>[:<step>] (diagnostic, default off; needs KYTY_TRACE_PASS_CENSUS=1 for the
// game's marker names, exact with KYTY_CP_SEQ=0 KYTY_CP_SEQ_PREFETCH=0). While <folder>/pass.now
// exists it is read ("<frames> [<thumb frames>]", default "30 1") and removed; from the next frame
// boundary on, for <frames> consecutive frames, every pass segment (a run of draws and dispatches
// under the same "outer>inner" marker) is followed by a readback of every image it wrote: colour
// targets, depth and stencil of the depth target, storage images written by dispatches. Each
// image gives one line in <folder>/passdump.txt (frame, segment, marker, the segment's draw and
// dispatch counts by shader hash, then TextureCache::DebugPassSample's record: format, per-channel
// mean/min/max, NaN/Inf, zero fraction, a hash and a 16x9 grid of region means). The first
// <thumb frames> frames also write the sampled texels (every <step>-th row and column, default 4)
// as float RGBA files <folder>/f<frame>_s<segment>_<address>_<aspect>_<w>x<h>.f32. Frames are slow
// while dumping (each segment boundary ends the rendering). Local files only.
namespace PassDump {
namespace {
struct Output {
	uint64_t address = 0;
	uint32_t width   = 0;
	uint32_t height  = 0;
	int      aspect  = 0;
};
struct State {
	std::string                                   folder;
	uint32_t                                      step        = 4;
	uint32_t                                      polls       = 0;
	uint32_t                                      want_frames = 0; // armed: frames to capture
	uint32_t                                      want_thumbs = 1;
	bool                                          capturing   = false;
	uint64_t                                      start_frame = 0;
	uint64_t                                      arm_frame   = 0;
	uint64_t                                      seg_frame   = 0;
	uint32_t                                      seg_index   = 0;
	std::string                                   seg_key;
	std::map<std::pair<char, uint64_t>, uint32_t> seg_ops;
	std::vector<Output>                           seg_outputs;
	std::FILE*                                    file = nullptr;
	std::mutex                                    file_mutex;
};
State& Get() {
	static State s;
	static constexpr auto parsed_init = [] {
		State& st = s;
		if (const char* value = std::getenv("KYTY_PASS_DUMP"); value != nullptr && value[0] != '\0') {
			std::string text(value);
			if (const auto colon = text.rfind(':'); colon != std::string::npos && colon > 1) {
				st.step = std::max(1u, static_cast<uint32_t>(std::strtoul(text.c_str() + colon + 1, nullptr, 10)));
				text.resize(colon);
			}
			st.folder = text;
			std::printf("KYTY_PASS_DUMP enabled (diagnostic): trigger %s/pass.now, step %u\n", st.folder.c_str(),
			            st.step);
		}
		return true;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(parsed_init())>> parsed_once;
	const bool& parsed = parsed_once.Get(parsed_init);
	(void)parsed;
	return s;
}
void Track(RenderContext& context);
} // namespace

// Before a draw or dispatch (GPU thread): true when the current segment ended and has outputs to
// read back; the caller then ends the rendering and calls Flush.
bool Before(RenderContext& context) {
	auto& s = Get();
	if (s.folder.empty()) return false;
	const uint64_t frame = context.GetGpu().GetFrameNum();
	if (!s.capturing && s.want_frames == 0) {
		if ((s.polls++ & 63u) != 0u) return false;
		std::error_code error;
		const auto      trigger = std::filesystem::path(s.folder) / "pass.now";
		if (!std::filesystem::exists(trigger, error)) return false;
		unsigned frames = 30, thumbs = 1;
		if (std::FILE* f = std::fopen(trigger.string().c_str(), "rb"); f != nullptr) {
			if (std::fscanf(f, "%u %u", &frames, &thumbs) < 1) frames = 30;
			std::fclose(f);
		}
		std::filesystem::remove(trigger, error);
		s.want_frames = std::max(1u, frames);
		s.want_thumbs = thumbs;
		s.arm_frame   = frame;
		std::printf("KYTY_PASS_DUMP: armed at frame %llu for %u frames (%u with texels)\n",
		            static_cast<unsigned long long>(frame), s.want_frames, s.want_thumbs);
		std::fflush(stdout);
		return false;
	}
	if (!s.capturing) {
		if (frame == s.arm_frame) return false;
		s.capturing   = true;
		s.start_frame = frame;
		s.seg_frame   = frame;
		s.seg_index   = 0;
		s.seg_key     = PassCensus::CurrentKey();
		s.seg_ops.clear();
		s.seg_outputs.clear();
		if (s.file == nullptr) {
			s.file = std::fopen((std::filesystem::path(s.folder) / "passdump.txt").string().c_str(), "a");
		}
		std::printf("KYTY_PASS_DUMP: capture starts at frame %llu\n", static_cast<unsigned long long>(frame));
		std::fflush(stdout);
		return false;
	}
	const auto key = PassCensus::CurrentKey();
	if (frame == s.seg_frame && key == s.seg_key) return false;
	if (!s.seg_outputs.empty()) return true;
	Track(context);
	return false;
}

// Reads back the ended segment's outputs and starts the next segment (rendering ended).
void Flush(RenderContext& context) {
	auto& s = Get();
	if (!s.capturing) return;
	const uint64_t frame     = context.GetGpu().GetFrameNum();
	const uint32_t rel_frame = static_cast<uint32_t>(s.seg_frame - s.start_frame);
	std::string    ops;
	for (const auto& [k, n]: s.seg_ops) {
		ops += fmt::format("{}{}:{:016x}x{}", ops.empty() ? "" : ",", k.first, k.second, n);
	}
	const auto prefix = fmt::format("PASSDUMP f={} gpu_frame={} seg={} key=\"{}\" ops={}", rel_frame, s.seg_frame,
	                                s.seg_index, s.seg_key, ops.empty() ? "-" : ops);
	for (const auto& o: s.seg_outputs) {
		const auto thumb = rel_frame < s.want_thumbs
		                       ? fmt::format("{}/f{}_s{}_{:x}_{}", s.folder, rel_frame, s.seg_index, o.address,
		                                     o.aspect == 0 ? "c" : o.aspect == 1 ? "d" : "s")
		                       : std::string();
		auto* file_state = &s;
		const auto head  = fmt::format("{} addr={:x} want={}x{} ", prefix, o.address, o.width, o.height);
		const bool ok    = context.GetTextureCache().DebugPassSample(
            o.address, o.width, o.height, o.aspect, s.step, thumb, [file_state, head](std::string line) {
                std::scoped_lock lock(file_state->file_mutex);
                if (file_state->file != nullptr) {
                    std::fprintf(file_state->file, "%s%s\n", head.c_str(), line.c_str());
                    std::fflush(file_state->file);
                }
            });
		if (!ok) {
			std::scoped_lock lock(s.file_mutex);
			if (s.file != nullptr) std::fprintf(s.file, "%snot cached (aspect %d)\n", head.c_str(), o.aspect);
		}
	}
	s.seg_outputs.clear();
	s.seg_ops.clear();
	s.seg_key = PassCensus::CurrentKey();
	if (frame != s.seg_frame) {
		s.seg_frame = frame;
		s.seg_index = 0;
		if (frame - s.start_frame >= s.want_frames) {
			s.capturing   = false;
			s.want_frames = 0;
			std::printf("KYTY_PASS_DUMP: capture ended at frame %llu\n", static_cast<unsigned long long>(frame));
			std::fflush(stdout);
		}
	} else {
		s.seg_index++;
	}
}

namespace {
// A segment without outputs ends: the next one starts.
void Track(RenderContext& context) {
	auto& s = Get();
	if (!s.capturing) return;
	const uint64_t frame = context.GetGpu().GetFrameNum();
	const auto     key   = PassCensus::CurrentKey();
	if (frame == s.seg_frame && key == s.seg_key) return;
	s.seg_ops.clear();
	s.seg_key = key;
	if (frame != s.seg_frame) {
		s.seg_frame = frame;
		s.seg_index = 0;
		if (frame - s.start_frame >= s.want_frames) {
			s.capturing   = false;
			s.want_frames = 0;
			std::printf("KYTY_PASS_DUMP: capture ended at frame %llu\n", static_cast<unsigned long long>(frame));
			std::fflush(stdout);
		}
	} else {
		s.seg_index++;
	}
}

} // namespace

bool Capturing() {
	return Get().capturing;
}

// After a draw ('d': hash = PS, or VS without PS) or dispatch ('c'): counts it and notes an image it wrote.
void NoteOp(char kind, uint64_t hash) {
	auto& s = Get();
	if (!s.capturing) return;
	s.seg_ops[{kind, hash}]++;
}

// KYTY_PASS_DUMP_ADDRS=<hex address>[,...] (diagnostic, live, default off: every image; at most 8):
// only images at these guest addresses are read back, so texels of one target can be kept for many
// consecutive frames.
namespace {
std::array<std::atomic<uint64_t>, 8> g_pass_dump_addrs {};
Live::Switch                         g_pass_dump_addr_count("KYTY_PASS_DUMP_ADDRS", [](const char* value) -> int64_t {
	uint32_t n = 0;
	while (value != nullptr && *value != '\0' && n < g_pass_dump_addrs.size()) {
		char*      end     = nullptr;
		const auto address = std::strtoull(value, &end, 16);
		if (end == value) break;
		if (address != 0) g_pass_dump_addrs[n++].store(address, std::memory_order_relaxed);
		if (*end != ',') break;
		value = end + 1;
	}
	return n;
}); // live, diagnostic
} // namespace

void NoteOutput(uint64_t address, uint32_t width, uint32_t height, int aspect) {
	auto& s = Get();
	if (!s.capturing || address == 0) return;
	if (const auto n = g_pass_dump_addr_count.Get(); n > 0) {
		bool listed = false;
		for (int64_t i = 0; i < n && !listed; i++) {
			listed = g_pass_dump_addrs[static_cast<size_t>(i)].load(std::memory_order_relaxed) == address;
		}
		if (!listed) return;
	}
	for (const auto& o: s.seg_outputs) {
		if (o.address == address && o.aspect == aspect) return;
	}
	s.seg_outputs.push_back({address, width, height, aspect});
}
} // namespace PassDump

// KYTY_PASS_DUMP: a dispatch and the images it wrote.
static void PassDumpNoteDispatch(const ShaderComputeInputInfo& input_info) {
	if (!PassDump::Capturing() || input_info.stage.program == nullptr || input_info.stage.resources == nullptr) return;
	const auto& program = *input_info.stage.program;
	PassDump::NoteOp('c', program.shader_hash);
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (!program.info.images[i].written) continue;
		const auto r = DecodeNativeDescriptor<ShaderTextureResource>(input_info.stage.resources->images[i]);
		PassDump::NoteOutput(r.Base40(), static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
		                     0);
	}
}

// KYTY_FRAME_TRACE=<folder>[:<count>] (diagnostic): while <folder>/trace.now exists, the next <count>
// (default 4000) compute dispatches and draws are logged as "FrameTrace" lines (programs, bound
// images and buffers, render targets; frameTrace.h), then the file is removed. One frame's producers
// and readers of any target, in submission order. Local logs only.
bool FrameTraceTake() {
	struct Request {
		std::filesystem::path trigger;
		uint32_t              count = 4000;
	};
	static constexpr auto request_init = [] {
		Request parsed;
		if (const char* value = std::getenv("KYTY_FRAME_TRACE"); value != nullptr && value[0] != '\0') {
			std::string text(value);
			// A ':' after the drive letter belongs to the path.
			if (const auto colon = text.rfind(':'); colon != std::string::npos && colon > 1) {
				parsed.count = static_cast<uint32_t>(std::strtoul(text.c_str() + colon + 1, nullptr, 10));
				text.resize(colon);
			}
			parsed.trigger = std::filesystem::path(text) / "trace.now";
		}
		return parsed;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(request_init())>> request_once;
	const Request& request = request_once.Get(request_init);
	if (request.trigger.empty()) return false;
	static std::atomic<uint32_t> left {0};
	static std::atomic<uint32_t> polls {0};
	if (left.load(std::memory_order_relaxed) == 0) {
		if ((polls.fetch_add(1, std::memory_order_relaxed) & 255u) != 0u) return false;
		std::error_code error;
		if (!std::filesystem::exists(request.trigger, error)) return false;
		std::filesystem::remove(request.trigger, error);
		LOGF("FrameTrace: start, %u operations\n", request.count);
		left.store(request.count, std::memory_order_relaxed);
	}
	auto value = left.load(std::memory_order_relaxed);
	while (value != 0 && !left.compare_exchange_weak(value, value - 1, std::memory_order_relaxed)) {
	}
	if (value == 1) LOGF("FrameTrace: end\n");
	return value != 0;
}

static void TraceDispatch(RenderContext& context, const ShaderComputeInputInfo& input_info,
                          const uint32_t* groups, uint64_t args_addr) {
	const auto& request = DispatchTrace();
	const bool  frame   = FrameTraceTake();
	if (!frame && request.hashes.empty() && request.bases.empty()) return;
	const auto& program = *input_info.stage.program;
	// 0 in the list (KYTY_DISPATCH_TRACE=all) traces every program.
	const bool all     = std::ranges::find(request.hashes, uint64_t {0}) != request.hashes.end();
	const auto found   = std::ranges::find(request.hashes, program.shader_hash);
	const auto binds   = [&] {
		const auto listed = [&](uint64_t base) { return std::ranges::find(request.bases, base) != request.bases.end(); };
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			if (listed(DecodeNativeDescriptor<ShaderTextureResource>(input_info.stage.resources->images[i]).Base40())) {
				return true;
			}
		}
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			if (listed(DecodeNativeDescriptor<ShaderBufferResource>(input_info.stage.resources->buffers[i]).Base48())) {
				return true;
			}
		}
		return false;
	};
	if (!frame && !all && found == request.hashes.end() && (request.bases.empty() || !binds())) return;
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, uint32_t> counts;
	uint32_t                                      ordinal = 0;
	{
		std::scoped_lock lock(mutex);
		ordinal = counts[program.shader_hash]++;
	}
	if (!frame && ordinal >= request.limit) return;
	std::string text = fmt::format("{}: hash=0x{:016x} #{} frame={} q={} seq={} tick={} gpu_tick={}",
	                               frame ? "FrameTrace CS" : "DispatchTrace",
	                               program.shader_hash, ordinal, context.GetGpu().GetFrameNum(),
	                               static_cast<int32_t>(HangTrace::CpContextQueue()),
	                               HangTrace::CpContextSequence(),
	                               context.GetCommandScheduler().CurrentTick(),
	                               context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick());
	if (groups != nullptr) {
		text += fmt::format(" groups={}x{}x{}", groups[0], groups[1], groups[2]);
	} else {
		uint32_t   args[3] {};
		const bool read  = LibKernel::Memory::TryReadBacking(args_addr, args, sizeof(args));
		const bool dirty = context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args));
		text += fmt::format(" indirect=0x{:x} guest_args={}{}x{}x{}{}", args_addr, read ? "" : "?",
		                    args[0], args[1], args[2], dirty ? " (gpu-dirty)" : "");
	}
	text += " ud:";
	const auto& user_data = input_info.stage.resources->user_data;
	for (size_t i = 0; i < std::min<size_t>(user_data.size(), 16); i++) {
		text += fmt::format(" {:08x}", user_data[i]);
	}
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto r =
		    DecodeNativeDescriptor<ShaderBufferResource>(input_info.stage.resources->buffers[i]);
		text += fmt::format(" | b{}{}=0x{:x}/s{}/n{}/f{}/d{:03x}", i, program.info.buffers[i].written ? "w" : "",
		                    r.Base48(), r.Stride(), r.NumRecords(), r.RawFormat(), r.DstSelXYZW());
	}
	// The images: t<i>[w][s] = written, sampled; base, format, extent, tile mode.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto& image = program.info.images[i];
		const auto  r =
		    DecodeNativeDescriptor<ShaderTextureResource>(input_info.stage.resources->images[i]);
		text += fmt::format(
		    " | t{}{}{}=0x{:x}/f{}/{}x{}x{}/tile{}", i, image.written ? "w" : "",
		    image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled ? "s" : "",
		    r.Base40(), static_cast<uint32_t>(r.Format()), static_cast<uint32_t>(r.Width5()) + 1u,
		    static_cast<uint32_t>(r.Height5()) + 1u, static_cast<uint32_t>(r.Depth()) + 1u,
		    static_cast<uint32_t>(r.TileMode()));
	}
	LOGF("%s\n", text.c_str());
}

// KYTY_DISPATCH_PEEK=<address>+<bytes>[,<address>+<bytes>...] (hexadecimal address, or s<N>[@<hex
// offset>] for the address in user SGPRs N and N + 1, at most 4
// ranges of at most 64 bytes, diagnostic, with KYTY_DISPATCH_TRACE): around every traced dispatch
// (within the trace limit), copies the GPU's bytes of each range before and after it into a
// download buffer and logs them once the tick completed ("DispatchPeek"). The copies wait for all
// earlier work and the dispatch, so the traced dispatch loses its overlap with its neighbours.
struct DispatchPeekRange {
	uint64_t address = 0;
	uint32_t bytes   = 0;
	// s<N>[@<offset>]: the 48-bit address in user SGPRs N and N + 1 (a V# base or a pointer),
	// plus offset; UINT32_MAX for a fixed address.
	uint32_t user_sgpr = UINT32_MAX;
	// b<N>[@<offset>]: the base of the program's buffer binding N (as DispatchTrace lists them),
	// plus offset; UINT32_MAX otherwise.
	uint32_t buffer_slot = UINT32_MAX;
};
static const std::vector<DispatchPeekRange>& DispatchPeekRanges() {
	static const std::vector<DispatchPeekRange> ranges = [] {
		std::vector<DispatchPeekRange> parsed;
		if (const auto* value = std::getenv("KYTY_DISPATCH_PEEK"); value != nullptr) {
			std::string_view text(value);
			while (!text.empty() && parsed.size() < 4) {
				const auto comma = text.find(',');
				const auto item  = std::string(text.substr(0, comma));
				const auto plus  = item.find('+');
				if (plus != std::string::npos) {
					DispatchPeekRange range;
					const auto base = item.substr(0, plus);
					if (!base.empty() && (base[0] == 's' || base[0] == 'b')) {
						const auto slot = static_cast<uint32_t>(std::strtoul(base.c_str() + 1, nullptr, 10));
						(base[0] == 's' ? range.user_sgpr : range.buffer_slot) = slot;
						const auto at   = base.find('@');
						range.address =
						    at == std::string::npos ? 0 : std::strtoull(base.c_str() + at + 1, nullptr, 16);
					} else {
						range.address = std::strtoull(base.c_str(), nullptr, 16);
					}
					range.bytes   = std::min<uint32_t>(
                        static_cast<uint32_t>(std::strtoul(item.substr(plus + 1).c_str(), nullptr, 0)),
                        64u) & ~3u;
					if ((range.address != 0 || range.user_sgpr != UINT32_MAX || range.buffer_slot != UINT32_MAX) && range.bytes != 0) {
						parsed.push_back(range);
					}
				}
				if (comma == std::string_view::npos) break;
				text.remove_prefix(comma + 1);
			}
		}
		return parsed;
	}();
	return ranges;
}

// KYTY_DISPATCH_DUMP_DIR=<folder> (diagnostic, with KYTY_DISPATCH_TRACE): around every traced dispatch
// (within the trace limit) the GPU's bytes of each bound buffer (at most 256 KiB each) are written to
// <folder>/d<ordinal>_<before|after>_<hash>_b<binding>_<base>.bin once the tick completed.
static const std::string& DispatchDumpDir() {
	static const std::string dir = [] {
		const char* value = std::getenv("KYTY_DISPATCH_DUMP_DIR");
		return value != nullptr ? std::string(value) : std::string();
	}();
	return dir;
}

static bool DispatchPeekWanted(const ShaderComputeInputInfo& input_info) {
	const auto& request = DispatchTrace();
	if ((DispatchPeekRanges().empty() && DispatchDumpDir().empty()) || request.hashes.empty()) return false;
	const auto hash = input_info.stage.program->shader_hash;
	const bool all  = std::ranges::find(request.hashes, uint64_t {0}) != request.hashes.end();
	return all || std::ranges::find(request.hashes, hash) != request.hashes.end();
}

static void DispatchDump(RenderContext& context, CommandBuffer& command,
                         const ShaderComputeInputInfo& input_info, uint32_t phase, uint32_t ordinal) {
	const auto& program = *input_info.stage.program;
	auto&       cache   = context.GetBufferCache();
	struct Piece {
		uint32_t binding = 0;
		uint64_t base    = 0;
		uint64_t offset  = 0;
		uint64_t bytes   = 0;
	};
	std::vector<Piece> pieces;
	uint64_t           total = 0;
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto r     = DecodeNativeDescriptor<ShaderBufferResource>(input_info.stage.resources->buffers[i]);
		const auto bytes = std::min<uint64_t>(r.GetSize(), 256u * 1024u) & ~uint64_t {3};
		if (r.Base48() < 0x100000000ull || bytes == 0) continue;
		pieces.push_back({i, r.Base48(), total, bytes});
		total += (bytes + 255u) & ~uint64_t {255};
	}
	if (pieces.empty()) return;
	auto download = std::make_shared<Buffer>(context.GetGraphics(), context.GetCommandScheduler(),
	                                         MemoryUsage::Download, 0, AllFlags, total);
	command.EndRendering();
	for (const auto& piece: pieces) {
		const auto [source, offset] = cache.ObtainBuffer(piece.base, piece.bytes, false);
		if (source == nullptr) continue;
		download->CopyFrom(command, *source, offset, piece.offset, piece.bytes, vk::AccessFlagBits::eMemoryWrite,
		                   vk::AccessFlagBits::eHostRead,
		                   vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                   vk::AccessFlagBits::eHostRead);
	}
	const auto hash = program.shader_hash;
	context.GetCommandScheduler().DeferOperation([download, pieces, hash, phase, ordinal, total] {
		download->Invalidate(0, total);
		for (const auto& piece: pieces) {
			const auto name = fmt::format("{}/d{:03}_{}_{:016x}_b{}_{:x}.bin", DispatchDumpDir(), ordinal,
			                              phase == 0 ? "before" : "after", hash, piece.binding, piece.base);
			if (FILE* file = std::fopen(name.c_str(), "wb"); file != nullptr) {
				std::fwrite(download->Mapped().data() + piece.offset, 1, piece.bytes, file);
				std::fclose(file);
			}
		}
	});
}

// KYTY_CS_VERIFY=<cs hash> (hexadecimal, diagnostic, default off): for every dispatch of this compute
// program, right before the dispatch command, the GPU's bytes (the registered cache buffer, no
// synchronization) of the job table its user data 0/1 point at (256 bytes) and of the source records
// at the table's first qword (up to 64 KiB, at most to the buffer's end) are copied to a host-visible
// ring and, once the tick completed, compared with the guest's bytes read when the dispatch was
// recorded and with the guest's bytes then. "CsVerify" lines: per frame a summary, per dispatch whose
// GPU bytes differ from the recorded guest bytes the first differing offset and counts. Finds a CPU
// write the GPU copy missed (stale source data read through BDA). Local logs only.
static uint64_t CsVerifyHash() {
	static const uint64_t hash = [] {
		const auto* value = std::getenv("KYTY_CS_VERIFY");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : uint64_t {0};
	}();
	return hash;
}
static void CsVerifyDispatch(RenderContext& context, CommandBuffer& command, const ShaderComputeInputInfo& input_info,
                             uint32_t groups) {
	constexpr uint64_t TableBytes = 256;
	constexpr uint64_t SrcBytes   = 64 * 1024;
	constexpr uint64_t SlotBytes  = TableBytes + SrcBytes;
	constexpr uint32_t Slots      = 1024;
	static std::unique_ptr<Buffer> download;
	static uint32_t                next_slot = 0;
	static int64_t                 last_frame = -1;
	static uint32_t                ordinal = 0, unregistered = 0, dispatches = 0;
	static std::shared_ptr<std::atomic<uint32_t>> mismatched = std::make_shared<std::atomic<uint32_t>>(0);
	const auto& ud = input_info.stage.resources->user_data;
	if (ud.size() < 2) return;
	const auto frame = static_cast<int64_t>(context.GetGpu().GetFrameNum());
	if (frame != last_frame) {
		if (last_frame >= 0) {
			LOGF("CsVerify: frame=%" PRId64 " dispatches=%u unregistered=%u mismatched_so_far=%u\n", last_frame, dispatches,
			     unregistered, mismatched->load());
		}
		last_frame = frame;
		ordinal = dispatches = unregistered = 0;
	}
	const auto index = ordinal++;
	dispatches++;
	const uint64_t table = ud[0] | (static_cast<uint64_t>(ud[1] & 0xffffu) << 32u);
	if (table == 0) return;
	const auto* table_alias = LibKernel::Memory::GuestBackingAlias(table, TableBytes);
	if (table_alias == nullptr) return;
	uint64_t src = 0;
	std::memcpy(&src, table_alias, sizeof(src));
	src &= 0xffffffffffffull;
	auto& cache = context.GetBufferCache();
	struct Piece {
		uint64_t address = 0, bytes = 0, slot_offset = 0;
		std::vector<uint8_t> guest;
		bool registered = false;
	};
	auto pieces = std::make_shared<std::array<Piece, 2>>();
	(*pieces)[0].address = table;
	(*pieces)[0].bytes   = TableBytes;
	(*pieces)[1].address = src;
	(*pieces)[1].bytes   = 0;
	if (src != 0) {
		uint64_t bytes = SrcBytes;
		while (bytes > 256 && cache.DebugPeekBuffer(src, bytes).first == nullptr) bytes /= 2;
		(*pieces)[1].bytes = bytes;
	}
	if (download == nullptr) {
		download = std::make_unique<Buffer>(context.GetGraphics(), context.GetCommandScheduler(), MemoryUsage::Download, 0,
		                                    AllFlags, SlotBytes * Slots);
	}
	const auto slot = next_slot++ % Slots;
	for (uint32_t k = 0; k < 2; k++) {
		auto& piece = (*pieces)[k];
		if (piece.address == 0 || piece.bytes == 0) continue;
		piece.slot_offset = uint64_t {slot} * SlotBytes + (k == 0 ? 0 : TableBytes);
		const auto* alias = LibKernel::Memory::GuestBackingAlias(piece.address, piece.bytes);
		if (alias == nullptr) continue;
		piece.guest.assign(static_cast<const uint8_t*>(alias), static_cast<const uint8_t*>(alias) + piece.bytes);
		const auto [source, offset] = cache.DebugPeekBuffer(piece.address, piece.bytes);
		if (source == nullptr) {
			unregistered++;
			continue;
		}
		piece.registered = true;
		download->CopyFrom(command, *source, offset, piece.slot_offset, piece.bytes, vk::AccessFlagBits::eMemoryWrite,
		                   vk::AccessFlagBits::eHostRead, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                   vk::AccessFlagBits::eHostRead);
	}
	const bool table_cpu_dirty = cache.DebugCpuDirty(table, TableBytes);
	const bool src_cpu_dirty   = src != 0 && (*pieces)[1].bytes != 0 && cache.DebugCpuDirty(src, (*pieces)[1].bytes);
	auto       counter         = mismatched;
	context.GetCommandScheduler().DeferOperation([pieces, frame, index, groups, table_cpu_dirty, src_cpu_dirty, counter] {
		std::string text;
		for (uint32_t k = 0; k < 2; k++) {
			const auto& piece = (*pieces)[k];
			if (!piece.registered || piece.guest.empty()) continue;
			download->Invalidate(piece.slot_offset, piece.bytes);
			const auto* gpu   = download->Mapped().data() + piece.slot_offset;
			uint32_t    diff  = 0;
			uint64_t    first = UINT64_MAX;
			for (uint64_t w = 0; w + 4 <= piece.bytes; w += 4) {
				if (std::memcmp(gpu + w, piece.guest.data() + w, 4) != 0) {
					if (first == UINT64_MAX) first = w;
					diff++;
				}
			}
			if (diff == 0) continue;
			uint32_t    now_diff = 0;
			const auto* alias    = static_cast<const uint8_t*>(LibKernel::Memory::GuestBackingAlias(piece.address, piece.bytes));
			if (alias != nullptr) {
				for (uint64_t w = 0; w + 4 <= piece.bytes; w += 4) {
					if (std::memcmp(gpu + w, alias + w, 4) != 0) now_diff++;
				}
			}
			uint32_t g = 0, h = 0;
			std::memcpy(&g, gpu + first, 4);
			std::memcpy(&h, piece.guest.data() + first, 4);
			text += fmt::format(" | {} 0x{:x}+0x{:x}: {} dwords differ (vs guest now {}), first +0x{:x} gpu {:08x} guest {:08x}",
			                    k == 0 ? "table" : "src", piece.address, piece.bytes, diff, now_diff, first, g, h);
		}
		if (!text.empty()) {
			counter->fetch_add(1);
			LOGF("CsVerify: frame=%" PRId64 " #%u groups=%u cpu_dirty_at_record table=%d src=%d%s\n", frame, index, groups,
			     table_cpu_dirty ? 1 : 0, src_cpu_dirty ? 1 : 0, text.c_str());
		}
	});
}

// phase 0 = before the dispatch, 1 = after it.
static void DispatchPeek(RenderContext& context, CommandBuffer& command,
                         const ShaderComputeInputInfo& input_info, uint32_t phase) {
	const auto hash = input_info.stage.program->shader_hash;
	if (!DispatchDumpDir().empty()) {
		static std::unordered_map<uint64_t, uint32_t> dumps;
		const auto                                    ordinal = dumps[hash]++;
		if (ordinal < DispatchTrace().limit * 2) DispatchDump(context, command, input_info, phase, ordinal / 2);
		if (DispatchPeekRanges().empty()) return;
	}
	constexpr uint32_t SlotBytes = 4 * 64;
	constexpr uint32_t Slots     = 512;
	static std::unique_ptr<Buffer>                 download;
	static uint32_t                                next_slot = 0;
	static std::unordered_map<uint64_t, uint32_t>  peeks;
	if (peeks[hash] >= DispatchTrace().limit * 2) return;
	const auto ordinal = peeks[hash]++;
	auto&      cache   = context.GetBufferCache();
	if (download == nullptr) {
		download = std::make_unique<Buffer>(context.GetGraphics(), context.GetCommandScheduler(),
		                                    MemoryUsage::Download, 0, AllFlags,
		                                    uint64_t {SlotBytes} * Slots);
	}
	const auto slot   = next_slot++ % Slots;
	auto ranges = DispatchPeekRanges();
	const auto& user_data = input_info.stage.resources->user_data;
	const auto& program = *input_info.stage.program;
	for (auto& range: ranges) {
		if (range.buffer_slot != UINT32_MAX) {
			range.address += range.buffer_slot < program.info.buffers.size()
			                     ? DecodeNativeDescriptor<ShaderBufferResource>(
			                           input_info.stage.resources->buffers[range.buffer_slot])
			                           .Base48()
			                     : 0;
			continue;
		}
		if (range.user_sgpr == UINT32_MAX) continue;
		const auto low  = range.user_sgpr < user_data.size() ? user_data[range.user_sgpr] : 0u;
		const auto high = range.user_sgpr + 1 < user_data.size() ? user_data[range.user_sgpr + 1] : 0u;
		range.address += low | (static_cast<uint64_t>(high & 0xffffu) << 32u);
	}
	command.EndRendering();
	for (uint32_t i = 0; i < ranges.size(); i++) {
		if (ranges[i].address < 0x100000000ull) continue; // a null binding
		const auto [source, offset] = cache.ObtainBuffer(ranges[i].address, ranges[i].bytes, false);
		if (source == nullptr) continue;
		download->CopyFrom(command, *source, offset, uint64_t {slot} * SlotBytes + i * 64u,
		                   ranges[i].bytes, vk::AccessFlagBits::eMemoryWrite,
		                   vk::AccessFlagBits::eHostRead, vk::AccessFlagBits::eMemoryRead |
		                   vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eHostRead);
	}
	const auto tick = context.GetCommandScheduler().CurrentTick();
	context.GetCommandScheduler().DeferOperation([slot, ranges, hash, phase, ordinal, tick] {
		download->Invalidate(uint64_t {slot} * SlotBytes, SlotBytes);
		const auto* bytes = download->Mapped().data() + uint64_t {slot} * SlotBytes;
		std::string text  = fmt::format("DispatchPeek: hash=0x{:016x} #{} {} tick={}", hash,
		                                ordinal / 2, phase == 0 ? "before" : "after", tick);
		for (uint32_t i = 0; i < ranges.size(); i++) {
			text += fmt::format(" | 0x{:x}:", ranges[i].address);
			for (uint32_t word = 0; word < ranges[i].bytes / 4; word++) {
				uint32_t value = 0;
				std::memcpy(&value, bytes + i * 64u + word * 4u, sizeof(value));
				text += fmt::format(" {:08x}", value);
			}
		}
		LOGF("%s\n", text.c_str());
	});
}

// KYTY_BDA_LIVE_PROBE_CS=<cs hash> (hexadecimal, diagnostic, default off; rtskip DESIGN.md 7): for every
// dispatch of this program, right before the dispatch command, the first 64 bytes of each bound buffer as
// the GPU holds them (the registered cache buffer, no synchronization) are copied to a host-visible ring.
// The guest's bytes and the pages' states (redirected / settled by KYTY_BDA_LIVE_PAGES, CPU-dirty,
// GPU-dirty) are taken when the dispatch is recorded. A device loss prints the newest records with the
// GPU bytes (copies ordered before the hung dispatch completed); the first records print at once.
namespace LiveProbe {
constexpr uint32_t Slots    = 64;
constexpr uint32_t Bindings = 8;
constexpr uint32_t Bytes    = 64;
struct Binding {
	uint64_t base = 0, size = 0;
	bool     written = false, copied = false;
	uint32_t pages = 0, live = 0, settled = 0, cpu = 0, gpu = 0;
	bool     sub_cpu = false, sub_gpu = false; // the 4 KiB tracker page of `base`
	uint32_t guest[Bytes / 4] {};
};
struct Record {
	uint64_t ordinal = 0, tick = 0, frame = 0;
	uint32_t count   = 0;
	Binding  b[Bindings];
};
std::mutex              g_mutex;
std::unique_ptr<Buffer> g_download;
Record                  g_records[Slots];
uint64_t                g_next = 0;
uint64_t Hash() {
	static const uint64_t hash = [] {
		const auto* value = std::getenv("KYTY_BDA_LIVE_PROBE_CS");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : uint64_t {0};
	}();
	return hash;
}
std::string Words(const uint32_t* words) {
	std::string text;
	for (uint32_t k = 0; k < Bytes / 4; ++k) text += fmt::format(" {:08x}", words[k]);
	return text;
}
void Print(const Record& r, bool gpu) {
	std::printf("BdaLiveProbe #%" PRIu64 " frame %" PRIu64 " tick %" PRIu64 ":\n", r.ordinal, r.frame, r.tick);
	const auto slot = r.ordinal % Slots;
	for (uint32_t i = 0; i < r.count; ++i) {
		const auto& b = r.b[i];
		std::printf("  b%u%s 0x%" PRIx64 "+0x%" PRIx64 ": caching pages %u, redirected %u (settled %u), tracker CPU-dirty %u, "
		            "GPU-dirty %u; 4 KiB page of the base: cpu %d gpu %d\n    guest:%s\n",
		            i, b.written ? "w" : "", b.base, b.size, b.pages, b.live, b.settled, b.cpu, b.gpu, b.sub_cpu ? 1 : 0,
		            b.sub_gpu ? 1 : 0, Words(b.guest).c_str());
		if (gpu && b.copied && g_download != nullptr) {
			uint32_t words[Bytes / 4] {};
			std::memcpy(words, g_download->Mapped().data() + (slot * Bindings + i) * Bytes, Bytes);
			std::printf("    gpu:  %s\n", Words(words).c_str());
		}
	}
}
void Note() {
	std::unique_lock lock(g_mutex, std::try_to_lock);
	if (!lock.owns_lock() || g_next == 0) return;
	if (g_download != nullptr) g_download->Invalidate(0, uint64_t {Slots} * Bindings * Bytes);
	std::printf("  BDA live probe at the loss: %" PRIu64 " dispatches recorded; the newest (GPU bytes: copies made "
	            "right before each dispatch):\n", g_next);
	for (uint64_t k = g_next > 4 ? g_next - 4 : 0; k < g_next; ++k) Print(g_records[k % Slots], true);
}
void Dispatch(RenderContext& context, CommandBuffer& command, const ShaderComputeInputInfo& input_info) {
	auto&            cache   = context.GetBufferCache();
	const auto&      program = *input_info.stage.program;
	std::scoped_lock lock(g_mutex);
	if (g_download == nullptr) {
		g_download = std::make_unique<Buffer>(context.GetGraphics(), context.GetCommandScheduler(), MemoryUsage::Download,
		                                      0, AllFlags, uint64_t {Slots} * Bindings * Bytes);
		SetDeviceLossNote(&Note);
	}
	const auto slot = g_next % Slots;
	auto&      r    = g_records[slot];
	r               = {};
	r.ordinal       = g_next++;
	r.tick          = context.GetCommandScheduler().CurrentTick();
	r.frame         = context.GetGpu().GetFrameNum();
	for (uint32_t i = 0; i < program.info.buffers.size() && r.count < Bindings; ++i) {
		const auto d    = DecodeNativeDescriptor<ShaderBufferResource>(input_info.stage.resources->buffers[i]);
		auto&      b    = r.b[r.count++];
		b.base          = d.Base48();
		b.size          = d.GetSize();
		b.written       = program.info.buffers[i].written;
		if (b.base < 0x100000000ull || b.size == 0) continue;
		const auto peek = std::min<uint64_t>(Bytes, b.size) & ~uint64_t {3};
		(void)LibKernel::Memory::TryReadBacking(b.base, b.guest, peek);
		b.sub_cpu        = cache.DebugCpuDirty(b.base & ~uint64_t {4095}, 4096);
		b.sub_gpu        = cache.IsRegionGpuModified(b.base & ~uint64_t {4095}, 4096);
		const auto first = b.base >> BufferCache::CACHING_PAGEBITS;
		const auto last  = (b.base + b.size - 1) >> BufferCache::CACHING_PAGEBITS;
		for (auto page = first; page <= last && b.pages < 4096; ++page) {
			const auto address = page << BufferCache::CACHING_PAGEBITS;
			b.pages++;
			b.live += cache.IsBdaLivePage(address) ? 1u : 0u;
			b.settled += cache.IsBdaLivePageSettled(address) ? 1u : 0u;
			b.cpu += cache.DebugCpuDirty(address, BufferCache::CACHING_PAGESIZE) ? 1u : 0u;
			b.gpu += cache.IsRegionGpuModified(address, BufferCache::CACHING_PAGESIZE) ? 1u : 0u;
		}
		const auto [source, offset] = cache.DebugPeekBuffer(b.base, peek);
		if (source == nullptr) continue;
		g_download->CopyFrom(command, *source, offset, (slot * Bindings + (r.count - 1)) * Bytes, peek,
		                     vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eHostRead,
		                     vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                     vk::AccessFlagBits::eHostRead);
		b.copied = true;
	}
	if (r.ordinal < 24) {
		Print(r, false);
		std::fflush(stdout);
	}
}
} // namespace LiveProbe

static void PrintDispatchOverlap(bool enabled) {
	const auto now = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                          std::chrono::steady_clock::now().time_since_epoch())
	                                          .count());
	auto last = g_overlap_last_print_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		g_overlap_last_print_ns.store(now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000ll ||
	    !g_overlap_last_print_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	const auto direct   = g_overlap_direct.exchange(0, std::memory_order_relaxed);
	const auto pairs    = g_overlap_pairs.exchange(0, std::memory_order_relaxed);
	const auto taken    = g_overlap_taken.exchange(0, std::memory_order_relaxed);
	const auto indirect = g_overlap_indirect.exchange(0, std::memory_order_relaxed);
	const auto empty    = g_overlap_empty_indirect.exchange(0, std::memory_order_relaxed);
	// Console and log (the launcher does not keep stdout).
	Log::WriteToConsoleAndLog(fmt::format(
	    "DispatchOverlap 10s ({}{}): {} direct dispatches, {} unordered pairs ({:.1f}%), {} overlapped, "
	    "{} indirect ({} empty skipped)\n",
	    enabled ? "on" : "off", DispatchOverlapUploadsEnabled() ? ", uploads" : "", direct, pairs,
	    direct != 0 ? 100.0 * static_cast<double>(pairs) / static_cast<double>(direct) : 0.0, taken,
	    indirect, empty));
}

static void CountDispatchOverlap(bool unordered, bool enabled) {
	g_overlap_direct.fetch_add(1, std::memory_order_relaxed);
	if (unordered) {
		g_overlap_pairs.fetch_add(1, std::memory_order_relaxed);
		if (enabled) {
			g_overlap_taken.fetch_add(1, std::memory_order_relaxed);
		}
	}
	PrintDispatchOverlap(enabled);
}

// KYTY_GPU_HAZARD (render.h): what a direct dispatch reads and writes, from its committed bindings.
// Guest ranges: each buffer binding's guest range (written if the program writes that buffer), each
// image binding's guest data, stencil and metadata ranges (written for storage writes). Host ranges:
// each buffer binding's host buffer byte range and each image's whole host image. Anything the
// program can reach outside its bindings makes the access imprecise.
static void CollectDispatchAccess(TextureCache& textures, const ShaderStageRuntime& stage,
                                  const PreparedBindings& bindings, CommandBuffer::DispatchAccess& out) {
	out.Clear();
	const auto& program = *stage.program;
	const auto& info    = program.info;
	if (program.has_address_writes || info.uses_dma || info.bda_writes || info.uses_bvh || info.ray_query_sites != 0 ||
	    bindings.shared_memory.buffer != nullptr || bindings.gds.buffer != nullptr || !bindings.bindless_heaps.empty() ||
	    !bindings.bindless_patches.empty() || bindings.buffer_sources.size() != info.buffers.size() ||
	    bindings.buffers.size() != info.buffers.size() || bindings.images.size() != info.images.size() ||
	    ShaderRecompiler::IR::FindBinding(program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr) {
		out.precise = false;
		return;
	}
	auto bits = [](auto handle) {
		return std::bit_cast<uint64_t>(static_cast<typename decltype(handle)::CType>(handle));
	};
	for (uint32_t i = 0; i < info.buffers.size(); i++) {
		const auto& resource = info.buffers[i];
		if (resource.indirect_root != ShaderRecompiler::IR::BufferResource::NoIndirectBuffer ||
		    !resource.indirect_resources.empty()) {
			out.precise = false;
			return;
		}
		const auto& source = bindings.buffer_sources[i];
		const auto& host   = bindings.buffers[i];
		if (source.address == 0 || source.size == 0 || host.buffer == nullptr) {
			continue;
		}
		const bool write = resource.written || resource.atomic;
		out.guest.push_back({0, source.address, source.address + source.size, write});
		const auto handle = bits(host.buffer);
		const auto end    = host.range == VK_WHOLE_SIZE ? UINT64_MAX : host.offset + host.range;
		out.host.push_back({handle, host.offset, end, write});
		out.handles.push_back(handle);
	}
	for (uint32_t i = 0; i < info.images.size(); i++) {
		const auto& resource = info.images[i];
		if (resource.bindless || resource.runtime || !resource.indirect_resources.empty() ||
		    resource.indirect_root != ShaderRecompiler::IR::ImageResource::NoIndirectImage) {
			out.precise = false;
			return;
		}
		const auto& binding = bindings.images[i];
		if (binding.image_view == nullptr) {
			continue;
		}
		auto* image = textures.TryGetImage(binding.image_id);
		if (image == nullptr || !image->backing.image) {
			// A null-image fallback (no guest memory) or an image the binding no longer names.
			if (resource.written) {
				out.precise = false;
				return;
			}
			continue;
		}
		const bool write = (resource.written || resource.atomic) &&
		                   resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
		const bool any_write = resource.written || resource.atomic;
		if (any_write && !write) {
			out.precise = false; // a write through a non-storage class: not modelled
			return;
		}
		auto add = [&](const GuestRange& range) {
			if (range.size != 0) {
				out.guest.push_back({0, range.address, range.address + range.size, write});
			}
		};
		add(image->info.data);
		if (image->info.HasStencil()) {
			add(image->info.stencil);
		}
		if (image->info.HasMetadata()) {
			add(image->info.metadata.range);
		}
		const auto handle = bits(image->backing.image);
		out.host.push_back({handle, 0, UINT64_MAX, write});
		out.handles.push_back(handle);
	}
}

static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

// KYTY_COMPUTE_META_CLEAR_DISPATCH=1 (default off; upstream 01ce97388's behaviour): a dispatch
// whose buffer writes recorded a metadata clear (UpdateComputeMetaClears) still runs, so its guest
// writes to the metadata memory are kept for readers that reuse it as an ordinary buffer. Off, the
// recorded clear consumes the dispatch, as before.
static bool ComputeMetaClearDispatchEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_COMPUTE_META_CLEAR_DISPATCH");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_HTILE_RMW_CLEAR=1 (default off, PPSA26344 preset; based on Erlou93/KytyPS5-Yotei, a port of
// peepostare dbeff606): a dispatch that reads and rewrites one depth surface's HTile in place still
// runs, and the clear is recorded for the next draw. Ghost of Yotei clears each shadow map's HTile
// this way (CS b25b01ff1361e795); UpdateComputeMetaClears never takes a read metadata buffer, so the
// shadow maps were never cleared and every caster failed the depth test (no moonlight).
static void RecordHtileRewriteClear(TextureCache& cache, const ShaderComputeInputInfo& input) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HTILE_RMW_CLEAR");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled) {
		return;
	}
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (program.info.has_bitwise_xor) {
		return;
	}
	const auto count = std::min(program.info.buffers.size(), resources.buffers.size());
	uint64_t   meta  = 0;
	for (size_t i = 0; i < count; i++) {
		if (!program.info.buffers[i].written) {
			continue;
		}
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		if (!cache.IsMeta(descriptor.Base48())) {
			continue;
		}
		// Several metadata surfaces, or a write-only clear (UpdateComputeMetaClears handles that one).
		if (meta != 0 || !program.info.buffers[i].read) {
			return;
		}
		meta = descriptor.Base48();
	}
	if (meta != 0 && cache.ClearMeta(meta)) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16u) {
			LOGF("HTILE_RMW_CLEAR: CS 0x%016" PRIx64 " meta 0x%" PRIx64 "\n", program.shader_hash, meta);
		}
	}
}

// Whether a metadata clear was recorded.
bool RenderExecutor::UpdateComputeMetaClears(const ShaderComputeInputInfo& input,
                                            const CommandBuffer&          buffer,
                                            bool*                         keep_dispatch) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	// KYTY_META_TRACE: the buffers of this dispatch that hold the traced metadata address.
	if (const auto traced = MetaTraceAddress(); traced != 0) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource   = program.info.buffers[i];
			const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			const auto  base       = descriptor.Base48();
			const auto  size       = descriptor.GetSize();
			if (traced >= base && traced - base < std::max<uint64_t>(size, 1) && MetaTraceLine()) {
				LOGF("MetaTrace: frame %d CS 0x%016" PRIx64 " buffer %u base 0x%" PRIx64 " size 0x%" PRIx64
				     " written %d read %d xor %d base_is_meta %d exact %d\n",
				     static_cast<int>(m_context.GetGpu().GetFrameNum()), program.shader_hash, i, base, size,
				     resource.written ? 1 : 0, resource.read ? 1 : 0, program.info.has_bitwise_xor ? 1 : 0,
				     cache.IsMeta(base) ? 1 : 0, base == traced ? 1 : 0);
			}
		}
	}
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// Reads and transforms of metadata do not establish a clear state.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	bool cleared = false;
	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				// KYTY_HTILE_SIZE_RULE: a write-only fill larger than the HTILE at its base is no clear.
				const bool oversized = !resource.read && TextureCache::HtileSizeRuleEnabled() &&
				                       cache.OversizedMetaWrite(descriptor.Base48(), descriptor.GetSize());
				const bool meta_cleared = oversized       ? false
				                          : resource.read ? cache.ClearMeta(descriptor.Base48())
				                                          : cache.ClearMetaRange(descriptor.Base48(), descriptor.GetSize());
				// KYTY_META_CLEAR_COVER: a write-only fill that starts below a metadata surface
				// and covers it clears it too; the dispatch still runs (only an exact clear is
				// consumed), so the bytes before the surface are written as before.
				if (!resource.read && MetaClearCoverEnabled()) {
					const auto covered = cache.ClearMetaCovered(descriptor.Base48(), descriptor.GetSize());
					if (covered != 0 && MetaTraceAddress() != 0 && MetaTraceLine()) {
						LOGF("MetaTrace: CS 0x%016" PRIx64 " buffer %u base 0x%" PRIx64 " covered %u metadata surfaces\n",
						     program.shader_hash, i, descriptor.Base48(), covered);
					}
				}
				cleared |= meta_cleared;
				// KYTY_CMASK_MSAA (usa-ratt/KytyPS5-GT7 8a0b6058f): a colour target keeps its
				// CMASK at the same address as this HTile: the fill still runs for it.
				if (meta_cleared && keep_dispatch != nullptr && CmaskMsaaEnabled() &&
				    cache.IsColorMeta(descriptor.Base48())) {
					*keep_dispatch = true;
				}
			}
		}
	}
	return cleared;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	if (invocations != descriptor.NumRecords() || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

// KYTY_SPARSE_FILLS=1 (default off, read once; adapted from usa-ratt/KytyPS5-GT7 4409d9020): a
// uniform compute buffer fill (ResolveComputeBufferFill) that no image took as a clear, over memory
// with holes in its mappings (sparse memory), fills each mapped piece on the GPU, as the hardware's
// writes land there, and the dispatch is skipped; the holes are left alone, where the hardware drops
// the writes. A binding stops at the first hole, so the dispatch never reached the mapped pages past
// it (none at all when the first page is a hole). False (the dispatch runs) for a range without
// holes or with more than 16384 mapped pieces.
static bool TryFillMappedParts(RenderContext& context, const ShaderComputeInputInfo& input,
                               uint32_t group_x, uint32_t group_y, uint32_t group_z, uint32_t mode) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SPARSE_FILLS");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled || input.stage.resources->uniform_fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	ShaderBufferResource descriptor;
	uint32_t             value = 0;
	uint64_t             size  = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, value, size)) {
		return false;
	}
	const auto base   = descriptor.Base48();
	const auto prefix = Libs::LibKernel::Memory::ClampRangeSizeQuiet(base, size);
	if (prefix == size || (base & 3u) != 0 || (size & 3u) != 0) {
		return false;
	}
	constexpr size_t                           MaxSpans = 16384;
	std::vector<std::pair<uint64_t, uint64_t>> spans;
	if (!Libs::LibKernel::Memory::CommittedSpans(base, size, MaxSpans, &spans)) {
		return false;
	}
	uint64_t mapped = 0;
	for (const auto& span: spans) {
		mapped += span.second;
	}
	context.GetBufferCache().FillMappedSpans(spans, value);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: buffer fill 0x%012" PRIx64 "+0x%" PRIx64
		     " with 0x%08x by cs %016" PRIx64 " over memory with holes: its %zu mapped pieces "
		     "(0x%" PRIx64 " bytes) filled, the holes left alone (a dispatch stops at the first, "
		     "0x%" PRIx64 " in; KYTY_SPARSE_FILLS)\n",
		     base, size, value, input.stage.program->shader_hash, spans.size(), mapped, prefix);
	}
	return true;
}

// KYTY_PATTERN_FILLS=1 (default off; GT7 fork 83707d403): the SDK's buffer fill compute shader is
// recognized from its guest code (compared once a program) instead of only by GT7's hash
// (KYTY_STENCIL_PLANE_FILL_CLEAR). With 64 threads a group, invocation i stores pattern[i & 3]
// (user data s4..s7) to element i of the V# in s[0:3] while i < s8, with buffer_store_format_x;
// each invocation picks its word with v_movrels, which keeps it out of AnalyzeUniformFill. A fill
// of one word whose range is exactly an image's memory clears that image on the GPU
// (ClearImageFromBuffer; a stencil plane needs KYTY_STENCIL_PLANE_FILL_CLEAR too).
constexpr std::array<uint32_t, 23> SdkPatternFillCode {
    0xbfa00001u,              // s_inst_prefetch 0x1
    0xd7460004u, 0x04010c09u, // v_lshl_add_u32 v4, s9, 6, v0
    0x7e000204u,              // v_mov_b32 v0, s4
    0x7e020205u,              // v_mov_b32 v1, s5
    0x7e040206u,              // v_mov_b32 v2, s6
    0x7e060207u,              // v_mov_b32 v3, s7
    0x7da80808u,              // v_cmpx_gt_u32 s8, v4
    0xbf88000du,              // s_cbranch_execz (to s_endpgm)
    0x360a0883u,              // v_and_b32 v5, 3, v4
    0xbe84047eu,              // s_mov_b64 s[4:5], exec
    0xbe86047eu,              // s_mov_b64 s[6:7], exec
    0x7e100505u,              // v_readfirstlane_b32 s8, v5
    0x7da40a08u,              // v_cmpx_eq_u32 s8, v5
    0xbefc0308u,              // s_mov_b32 m0, s8
    0x7e0c8700u,              // v_movrels_b32 v6, v0
    0x8a867e06u,              // s_andn2_b64 s[6:7], s[6:7], exec
    0xbefe0406u,              // s_mov_b64 exec, s[6:7]
    0xbf85fff9u,              // s_cbranch_scc1 (to v_readfirstlane_b32)
    0xbefe0404u,              // s_mov_b64 exec, s[4:5]
    0xe0102000u, 0x80000604u, // buffer_store_format_x v6, v4, s[0:3], 0 idxen
    0xbf810000u,              // s_endpgm
};

static bool PatternFillsOn() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PATTERN_FILLS");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// Whether the guest code at `code_address` (the program `shader_hash` was made from) is the SDK's
// pattern fill: compared once a program.
static bool IsSdkPatternFillCode(uint64_t shader_hash, uint64_t code_address) {
	static std::mutex                         mutex;
	static std::unordered_map<uint64_t, bool> checked;
	std::scoped_lock                          lock(mutex);
	if (const auto found = checked.find(shader_hash); found != checked.end()) {
		return found->second;
	}
	std::array<uint32_t, SdkPatternFillCode.size()> code {};
	const bool match = code_address != 0 &&
	                   LibKernel::Memory::TryReadBacking(code_address, code.data(), sizeof(code)) &&
	                   code == SdkPatternFillCode;
	if (checked.size() < 65536) {
		checked.emplace(shader_hash, match);
	}
	if (match) {
		LOGF("Dispatch: cs %016" PRIx64 " at 0x%012" PRIx64
		     " is the SDK's pattern fill (KYTY_PATTERN_FILLS): a fill of one word clears the image "
		     "whose memory it fills\n",
		     shader_hash, code_address);
	}
	return match;
}

// The SDK's pattern fill of one word in this dispatch: its address, byte size and word (as the
// fork's ResolveSdkPatternFill: the first min(s8, invocations, records) dwords of a plain dword
// array bounded by index, a one-dimensional dispatch of 64-thread groups).
static bool ResolveSdkPatternFill(const ShaderComputeInputInfo& input, uint64_t code_address,
                                  uint32_t group_x, uint32_t group_y, uint32_t group_z,
                                  uint32_t mode, uint64_t& address, uint64_t& size,
                                  uint32_t& word) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	// Its one buffer is stored to and never read; s9 is the group's index (after 9 user SGPRs).
	if (!PatternFillsOn() || code_address == 0 || program.info.buffers.size() != 1 ||
	    resources.buffers.size() != 1 || !program.info.buffers[0].written ||
	    program.info.buffers[0].read || program.info.buffers[0].atomic ||
	    !program.info.images.empty() || program.user_data_base != 0 ||
	    resources.user_data.size() < 9 || input.workgroup_register != 9 || !input.group_id[0] ||
	    input.threads_num[0] != 64 || input.threads_num[1] != 1 || input.threads_num[2] != 1 ||
	    input.dispatch_thread_dimensions || mode != 0x41u || group_x == 0 || group_y != 1 ||
	    group_z != 1) {
		return false;
	}
	const auto& user = resources.user_data;
	if (user[5] != user[4] || user[6] != user[4] || user[7] != user[4]) {
		return false; // a pattern of more than one word
	}
	const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[0]);
	// The V# bound for its store is the one in s[0:3].
	ShaderBufferResource user_vsharp {};
	std::copy_n(user.begin(), 4, user_vsharp.fields);
	const auto format = descriptor.Format();
	if (user_vsharp.Base48() != descriptor.Base48() || user_vsharp.Stride() != descriptor.Stride() ||
	    user_vsharp.NumRecords() != descriptor.NumRecords() ||
	    user_vsharp.RawFormat() != descriptor.RawFormat() || descriptor.Stride() != 4 ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0 || descriptor.OutOfBounds() > 1 || descriptor.Type() != 0 ||
	    (format != Prospero::BufferFormat::k32UInt && format != Prospero::BufferFormat::k32SInt &&
	     format != Prospero::BufferFormat::k32Float)) {
		return false;
	}
	const uint64_t elements = std::min(
	    {uint64_t {user[8]}, uint64_t {group_x} * 64u, uint64_t {descriptor.NumRecords()}});
	if (elements == 0 || elements * 4u > UINT32_MAX ||
	    !GuestRange {descriptor.Base48(), elements * 4u}.Valid() ||
	    !IsSdkPatternFillCode(program.shader_hash, code_address)) {
		return false;
	}
	address = descriptor.Base48();
	size    = elements * 4u;
	word    = user[4];
	return true;
}

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                CommandBuffer& command, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z, uint32_t mode,
                                                uint64_t code_address) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		// A rejected clear (reported) leaves the guest shader to run.
		return cache.ClearImage(command, binding.image_id, image.backing.format, range, clear,
		                        "compute-stencil-fill");
	}
	// KYTY_STENCIL_PLANE_FILL_CLEAR: GT7's pattern fill (CS 0xfdc68063b0791828: thread i < s8 stores
	// s[4 + (i & 3)] with BUFFER_STORE_FORMAT_X) is a uniform fill when the four pattern words agree. Its
	// waterfall loop keeps it out of AnalyzeUniformFill; it clears GT7's course-map depth and stencil.
	if (StencilPlaneFillClearEnabled() && program.shader_hash == 0xfdc68063b0791828ull &&
	    resources.buffers.size() == 1 && resources.user_data.size() >= 9 &&
	    resources.user_data[4] == resources.user_data[5] && resources.user_data[4] == resources.user_data[6] &&
	    resources.user_data[4] == resources.user_data[7] && resources.user_data[8] != 0) {
		const auto pattern_buffer = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[0]);
		const auto bytes          = uint64_t {resources.user_data[8]} * 4u;
		if (pattern_buffer.Stride() == 4 && group_x * 64ull >= resources.user_data[8] &&
		    cache.ClearImageFromBuffer(command, pattern_buffer.Base48(), bytes, resources.user_data[4])) {
			if (MetaTraceAddress() == pattern_buffer.Base48() && MetaTraceLine()) {
				LOGF("MetaTrace: pattern fill 0x%" PRIx64 " size 0x%" PRIx64 " value 0x%08x: image clear\n",
				     pattern_buffer.Base48(), bytes, resources.user_data[4]);
			}
			return true;
		}
	}
	// KYTY_PATTERN_FILLS: the same SDK fill shader recognized from its code, in any title.
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::None) {
		uint64_t pattern_address = 0;
		uint64_t pattern_size    = 0;
		uint32_t pattern_word    = 0;
		if (ResolveSdkPatternFill(input, code_address, group_x, group_y, group_z, mode,
		                          pattern_address, pattern_size, pattern_word) &&
		    cache.ClearImageFromBuffer(command, pattern_address, pattern_size, pattern_word)) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				LOGF("Dispatch: the SDK's pattern fill 0x%012" PRIx64 "+0x%" PRIx64
				     " with 0x%08x clears an image (KYTY_PATTERN_FILLS)\n",
				     pattern_address, pattern_size, pattern_word);
			}
			return true;
		}
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	const bool resolved_fill =
	    ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear, size);
	// KYTY_META_TRACE: what became of a dispatch writing the traced address (fill or not, image clear).
	const auto trace_fill = [&](const char* what) {
		const auto traced = MetaTraceAddress();
		if (traced == 0 || resources.buffers.empty()) return;
		const auto first = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[0]);
		if (first.Base48() != traced || !MetaTraceLine()) return;
		LOGF("MetaTrace: CS 0x%016" PRIx64 " fill kind %u resolved %d size 0x%" PRIx64 " value 0x%08x mode 0x%x groups %u %u %u: %s\n",
		     program.shader_hash, static_cast<unsigned>(fill.kind), resolved_fill ? 1 : 0, size, packed_clear, mode,
		     group_x, group_y, group_z, what);
	};
	if (!resolved_fill) {
		trace_fill("not a uniform buffer fill");
		return false;
	}
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		trace_fill("no image clear");
		return false;
	}
	trace_fill("image clear");
	if (FrameTraceTake()) {
		LOGF("FrameTrace FILL: hash=0x%016" PRIx64 " addr=0x%" PRIx64 " size=0x%" PRIx64 " value=0x%08x\n",
		     program.shader_hash, descriptor.Base48(), size, packed_clear);
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
	}
	return true;
}

// KYTY_BACK_ADDRESS_BASES=1 (default off; upstream c174ade18 for another title): global loads
// read pages without a buffer as zero, so the 32 KiB after each base a compute shader loads from
// scalar data (ShaderInfo::address_bases, flattened SRT slots) get a buffer before the dispatch.
static bool BackAddressBasesEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_BACK_ADDRESS_BASES");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

static void BackAddressBases(RenderContext& context, const ShaderStageRuntime& stage) {
	if (!BackAddressBasesEnabled() || !stage) {
		return;
	}
	constexpr uint64_t Page = BufferCache::CACHING_PAGESIZE;
	const auto&        flat = stage.resources->flattened_srt;
	for (const auto [low, high]: stage.program->info.address_bases) {
		if (low >= flat.size() || high >= flat.size()) {
			continue;
		}
		const auto base = flat[low] | uint64_t {flat[high]} << 32u;
		for (uint64_t index = 0; index <= 2; ++index) {
			const auto page = base + index * Page;
			if (context.IsMapped(page, 1)) {
				(void)context.GetBufferCache().FindBuffer(page, 1);
			}
		}
	}
}

// KYTY_LDS_DEVICE_BUFFER_MB (default 1024): the most memory the compute LDS device buffer may
// take; a direct dispatch with more workgroups than it holds runs its excess workgroups without
// LDS (their accesses read zero and drop writes), and says so once per shader.
static uint64_t ComputeLdsMaxBytes(const GraphicContext& graphics) {
	static const uint64_t configured = [] {
		const auto* value = std::getenv("KYTY_LDS_DEVICE_BUFFER_MB");
		const auto  mib   = value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 0) : 1024u;
		return std::max<uint64_t>(mib, 1u) << 20u;
	}();
	return std::min<uint64_t>(configured,
	                          graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
}

// An indirect dispatch's workgroup count is GPU data: it gets at least this much (1,365 regions
// of 48 KiB), or the buffer as a direct dispatch grew it.
static constexpr uint64_t LdsIndirectBytes = 64ull << 20u;

bool RenderExecutor::BindComputeLds(const ShaderComputeInputInfo& input, PreparedBindings& bindings,
                                    const uint32_t* groups) {
	const auto& program = *input.stage.program;
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory) == nullptr) {
		return false;
	}
	auto&          graphics = m_context.GetGraphics();
	const uint64_t region =
	    uint64_t {ShaderRecompiler::LdsStorageRegionDwords(input.lds_size_dwords)} * sizeof(uint32_t);
	EXIT_IF(region == 0);
	const uint64_t max_regions = std::max<uint64_t>(ComputeLdsMaxBytes(graphics) / region, 1u);
	uint64_t       regions     = 0;
	if (groups != nullptr) {
		const uint64_t count = uint64_t {groups[0]} * groups[1] * groups[2];
		regions              = std::min(count, max_regions);
		if (count > max_regions) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
				char text[256];
				std::snprintf(text, sizeof(text),
				              "Kyty LDS device buffer: compute shader 0x%016" PRIx64
				              " dispatches %" PRIu64 " workgroups of %u bytes of LDS; the buffer holds "
				              "%" PRIu64 " (KYTY_LDS_DEVICE_BUFFER_MB), the others run without LDS\n",
				              program.shader_hash, count, input.lds_size_dwords * 4u, max_regions);
				Log::WriteToConsoleAndLog(text);
			}
		}
	} else {
		const auto current = m_compute_lds != nullptr ? m_compute_lds->Size() / region : 0u;
		regions = std::min(std::max<uint64_t>(current, LdsIndirectBytes / region), max_regions);
	}
	regions            = std::max<uint64_t>(regions, 1u);
	const auto needed = regions * region;
	if (m_compute_lds == nullptr || m_compute_lds->Size() < needed) {
		auto& scheduler = m_context.GetCommandScheduler();
		const auto old_size = m_compute_lds != nullptr ? m_compute_lds->Size() : 0u;
		// Grows by at least half, so a growing series of dispatches reallocates a few times only.
		const auto size =
		    std::min(std::max(needed, old_size + old_size / 2u), max_regions * region);
		if (m_compute_lds != nullptr) {
			// Recorded commands of the current tick may still use the old buffer.
			scheduler.DeferOperation([old = std::move(m_compute_lds)]() mutable { old.reset(); });
		}
		m_compute_lds = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
		                                         vk::BufferUsageFlagBits::eStorageBuffer, size);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			char text[256];
			std::snprintf(text, sizeof(text),
			              "Kyty LDS device buffer (KYTY_LDS_DEVICE_BUFFER): %.1f MiB for compute shader "
			              "0x%016" PRIx64 " (%u bytes of LDS per workgroup, %" PRIu64 " workgroups%s)\n",
			              static_cast<double>(size) / (1024.0 * 1024.0), program.shader_hash,
			              input.lds_size_dwords * 4u, regions, groups == nullptr ? ", indirect" : "");
			Log::WriteToConsoleAndLog(text);
		}
	}
	// A direct dispatch binds its own regions; an indirect one every region the buffer holds.
	const auto range =
	    groups != nullptr ? needed : m_compute_lds->Size() / region * region;
	bindings.shared_memory = {m_compute_lds->Handle(), 0, range};
	return true;
}

// KYTY_ORDERED_APPEND=1 (default off; upstream d4465faa7): a dispatch with
// DISPATCH_INITIATOR.ORDERED_APPEND_ENABLE whose program uses GDS and whose workgroups are one
// guest wave without emulated LDS runs one workgroup per vkCmdDispatchBase, in launch order with a
// barrier between them, so ordered-append counters see the workgroups in order. Logged once.
bool OrderedAppendSplitEnabled() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_ORDERED_APPEND");
		return value != nullptr && std::strcmp(value, "1") == 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

// KYTY_DCC_KEY10_FILL=1 (default off; Ghost of Yotei preset): the AGC Toolkit's clear to pixel. CS
// 0x2bd3cd129405f9a9 (V# in s[0:3], element in s[4:7], one formatted store per thread) with a
// 256-byte record stride stores the packed element (1-4 dwords of a 32-bit buffer format) once at
// the start of every 256-byte block after the Toolkit filled the target's DCC keys with 0x10; the
// hardware then reads every texel of such a block as that element. The texture cache keeps 0x10
// keys as uncompressed data, so the dispatch is replaced by a fill of the whole store range with
// the element (one dword pattern: every channel equal). Elements with unequal words keep the store
// as written (logged). Adapted from Erlou93/KytyPS5-Yotei KYTY_FIX_DCC_KEY_0X10 (renderCompute.cpp,
// TextureCache::RegisterDccPixelFill), which clears the texture at its next use instead.
static bool DccKey10FillEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DCC_KEY10_FILL");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

static bool TryDccKey10Fill(RenderContext& context, const ShaderComputeInputInfo& input,
                            uint32_t group_x, uint32_t group_y, uint32_t group_z,
                            bool use_thread_dimensions) {
	constexpr uint64_t ClearToPixelHash = 0x2bd3cd129405f9a9ull;
	const auto&        program          = *input.stage.program;
	const auto&        resources        = *input.stage.resources;
	if (program.shader_hash != ClearToPixelHash || use_thread_dimensions ||
	    resources.user_data.size() < 8u) {
		return false;
	}
	ShaderBufferResource v {};
	std::copy_n(resources.user_data.begin(), 4, v.fields);
	// KYTY_DCC_FILL_CS_LOG (diagnostics): the first dispatches of this shader with another record
	// stride (Erlou93/KytyPS5-Yotei registers stride-16 ones as DCC key fills).
	if (v.Stride() != 256u && std::getenv("KYTY_DCC_FILL_CS_LOG") != nullptr) {
		static std::atomic<uint32_t> fill_logged {0};
		if (fill_logged.fetch_add(1, std::memory_order_relaxed) < 64) {
			std::printf("DccFillCs: addr=0x%016" PRIx64 " stride=%u format=%u records=%u groups=%ux%ux%u "
			            "value=%08x,%08x,%08x,%08x swizzle=%d addtid=%d\n",
			            v.Base48(), v.Stride(), v.RawFormat(), v.NumRecords(), group_x, group_y,
			            group_z, resources.user_data[4], resources.user_data[5],
			            resources.user_data[6], resources.user_data[7], v.SwizzleEnabled() ? 1 : 0,
			            v.AddTid() ? 1 : 0);
		}
	}
	if (v.Base48() == 0 || v.Stride() != 256u || v.SwizzleEnabled() || v.AddTid()) {
		return false;
	}
	// GFX10 buffer formats of 1-4 32-bit channels (UINT, SINT, FLOAT each): 20-22 (32), 62-64
	// (32_32), 72-74 (32_32_32), 75-77 (32_32_32_32).
	const uint32_t format   = v.RawFormat();
	const uint32_t channels = format >= 20u && format <= 22u   ? 1u
	                          : format >= 62u && format <= 64u ? 2u
	                          : format >= 72u && format <= 74u ? 3u
	                          : format >= 75u && format <= 77u ? 4u
	                                                           : 0u;
	const uint64_t threads = uint64_t {group_x} * group_y * group_z *
	                         std::max(input.threads_num[0], 1u) *
	                         std::max(input.threads_num[1], 1u) *
	                         std::max(input.threads_num[2], 1u);
	const uint64_t bytes = std::min<uint64_t>(v.NumRecords(), threads) * 256u;
	if (channels == 0 || bytes == 0) {
		return false;
	}
	const uint32_t value = resources.user_data[4];
	bool           equal = true;
	for (uint32_t i = 1; i < channels; i++) {
		equal = equal && resources.user_data[4u + i] == value;
	}
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("KYTY_DCC_KEY10_FILL: clear to pixel addr=0x%016" PRIx64 " bytes=0x%" PRIx64
		     " channels=%u element=%08x,%08x,%08x,%08x: %s\n",
		     v.Base48(), bytes, channels, resources.user_data[4],
		     channels > 1 ? resources.user_data[5] : 0u, channels > 2 ? resources.user_data[6] : 0u,
		     channels > 3 ? resources.user_data[7] : 0u,
		     equal ? "filled" : "unequal words, store kept as written");
	}
	if (!equal) {
		return false;
	}
	context.GetBufferCache().FillBuffer(v.Base48(), bytes, value, false);
	return true;
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode, bool async_compute) {
	std::shared_ptr<const void> run;
	DispatchDirectOne(submit_id, buffer, thread_group_x, thread_group_y, thread_group_z, mode, async_compute,
	                  nullptr, &run);
	if (run != nullptr) {
		// KYTY_CALL_PARTITIONS: once per partition (callPartitionDispatch.cpp).
		DispatchPartitions(buffer, run, [&](const PartitionStep& step) {
			DispatchDirectOne(submit_id, buffer, thread_group_x, thread_group_y, thread_group_z, mode,
			                  async_compute, &step, nullptr);
		});
	}
}

void RenderExecutor::DispatchDirectOne(uint64_t submit_id, CommandBuffer& buffer,
                                       uint32_t thread_group_x, uint32_t thread_group_y,
                                       uint32_t thread_group_z, uint32_t mode, bool async_compute,
                                       const PartitionStep*         step,
                                       std::shared_ptr<const void>* partition_run) {
	KYTY_GPU_OP_SITE("dispatch");
	// KYTY_CP_GAP_STATS: this scope's phases (drawPrep/cpGaps.h).
	const CpGaps::Scope gap(CpGaps::Cat::DispatchOp);
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopReadyOperations();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	RenderLockGuard lock(m_context.GetMutex());
	// KYTY_DRAW_RUN: a dispatch ends the run of the draws before it (drawPrep/drawRun.h).
	BeginDrawRun();
	if (PassDump::Before(m_context)) { // KYTY_PASS_DUMP: the previous pass segment's outputs
		buffer.EndRendering();
		PassDump::Flush(m_context);
	}
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	DrawPrep::MakeComputeInputInfo(cs_regs, {thread_group_x, thread_group_y, thread_group_z, mode},
	                               input_info);
	// After MakeComputeInputInfo (it resets the info). Part of the program key (GETREG HW_ID2).
	input_info.async_compute = async_compute;
	// KYTY_SHADER_CALLS_SPLIT: each program of a split kernel's chain is looked up from this.
	const auto pristine_info = input_info;
	CpGaps::Phase(CpGaps::Cat::DispatchProgram);
	ShaderProgram compute_program;
	// KYTY_DRAW_PREP_DISPATCH (drawPrep.h): the program a worker prepared from this dispatch's
	// register snapshot, when its certificate holds; otherwise the serial lookup. Workers prepare
	// graphics-queue dispatches only (MakeComputeInputInfo leaves async_compute false).
	auto* const prepared = std::exchange(m_prepared_dispatch, nullptr);
	if (step != nullptr) {
		PipelineCache::SelectCallSplitCode(step->code);
	}
	if (step == nullptr && prepared != nullptr && !async_compute &&
	    DrawPrep::ValidateDispatch(*prepared, input_info)) {
		if (DrawPrep::VerifyMode() != 0) {
			// The serial lookup on copies; it only reads memory the certificate proved clean.
			auto serial_prep = std::make_unique<PipelineCache::StagePrep>();
			auto serial_info = input_info;
			const auto serial_program = m_context.GetPipelineCache().GetComputeProgram(
			    cs_regs, sh_regs, serial_info, *serial_prep);
			(void)DrawPrep::VerifyDispatch(*prepared, serial_program, serial_info, *serial_prep);
		}
		// The prepared program is never a call kernel's (PrepareComputeProgramSpeculative): a run
		// left on this thread belongs to another dispatch.
		PipelineCache::DropCallRuns();
		std::swap(m_compute_prep, prepared->compute_prep);
		input_info                 = prepared->compute_info;
		input_info.stage.resources = &m_compute_prep.resources;
		compute_program            = prepared->compute_program;
	} else {
		compute_program = m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info,
		                                                                 m_compute_prep);
	}
	if (step != nullptr) {
		PipelineCache::SelectCallSplitCode(nullptr);
	}
	if (partition_run != nullptr) {
		if (auto run = PipelineCache::TakePartitionRun(); run != nullptr) {
			*partition_run = std::move(run);
			ResetBindings();
			return;
		}
	}
	CpGaps::Phase(CpGaps::Cat::DispatchOp);
	// KYTY_RT_DUMP_BVH (diagnostic): before any binding is prepared (its readbacks may flush the
	// recording). Programs with BVH instructions only.
	if (RtBvhDumpEnabled() && compute_program && input_info.stage.program->info.uses_bvh) {
		RtBvhDispatch dump {};
		dump.shader_hash    = input_info.stage.program->shader_hash;
		dump.shader_address = cs_regs.cs_regs.data_addr;
		dump.user_sgpr      = std::span<const uint32_t>(
            cs_regs.cs_user_sgpr.value,
            std::min<uint32_t>(cs_regs.cs_user_sgpr.count, HW::UserSgprInfo::SGPRS_MAX));
		dump.groups = {thread_group_x, thread_group_y, thread_group_z};
		dump.frame  = static_cast<int>(m_context.GetGpu().GetFrameNum());
		RtBvhDumpOnDispatch(dump);
	}
	if (WatchReadEnabled() && compute_program) {
		WatchReadOnDispatch(input_info.stage.program->shader_hash,
		                    std::span<const uint32_t>(cs_regs.cs_user_sgpr.value,
		                                              std::min<uint32_t>(cs_regs.cs_user_sgpr.count,
		                                                                 HW::UserSgprInfo::SGPRS_MAX)),
		                    static_cast<int>(m_context.GetGpu().GetFrameNum()));
	}
	if (!compute_program) {
		// Temporary until RT is implemented.
		DropProgramless(m_context.GetPipelineCache());
		return;
	}
	if (auto split = PipelineCache::TakeCallSplitRun(); split != nullptr) {
		// A kernel with call loops runs as its chain of dispatches (callSplitDispatch.cpp).
		ResetBindings();
		PassCensus::Exec('c', input_info.stage.program != nullptr ? input_info.stage.program->shader_hash : 0, 2);
		if (input_info.stage.program != nullptr && input_info.stage.resources != nullptr) {
			// KYTY_DISPATCH_TRACE / _IMAGE: split kernels too (their chain never reaches the trace below).
			const uint32_t traced_groups[3] {thread_group_x, thread_group_y, thread_group_z};
			TraceDispatch(m_context, input_info, traced_groups, 0);
		}
		if (!use_thread_dimensions) {
			const auto image_dump = PlanCsImageDump(input_info);
			buffer.EndRendering();
			DispatchCallSplit(buffer, pristine_info, split, cs_regs, sh_regs, thread_group_x,
			                  thread_group_y, thread_group_z);
			RunCsImageDump(m_context, image_dump);
			PassDumpNoteDispatch(input_info); // KYTY_PASS_DUMP
		}
		return;
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	PassCensus::Exec('c', program.shader_hash, 0);
	WatchRange::SetCurrentOp("CS", program.shader_hash);
	WatchRange::NoteDispatch(program.shader_hash, async_compute, thread_group_x, thread_group_y, thread_group_z, false);
	WatchRange::SetCurrentOpDetail(static_cast<uint32_t>(m_context.GetGpu().GetFrameNum()),
	                               input_info.stage.resources->user_data.data(),
	                               static_cast<uint32_t>(input_info.stage.resources->user_data.size()), thread_group_x,
	                               thread_group_y, thread_group_z);
	const auto& resources = *input_info.stage.resources;
	// A recorded metadata clear consumes the dispatch unless KYTY_COMPUTE_META_CLEAR_DISPATCH keeps
	// its guest writes (upstream 01ce97388).
	bool keep_meta_dispatch = false;
	RecordHtileRewriteClear(buffer.GetContext().GetTextureCache(), input_info);
	if (UpdateComputeMetaClears(input_info, buffer, &keep_meta_dispatch) &&
	    !ComputeMetaClearDispatchEnabled() && !keep_meta_dispatch) {
		// KYTY_STENCIL_PLANE_SYNC=2: the consumed fill may also cover images at the same address (Ghost
		// of Yotei zero-fills its z-fail volume stencil plane 0x50615a0000, also an HTile there, every
		// frame); give them the clear too, or the plane keeps last frame's counts.
		if (TextureCache::StencilPlaneSyncLevel() >= 2 && TextureCache::PreserveStencilEnabled()) {
			(void)TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y, thread_group_z,
			                                  mode);
		}
		ResetBindings();
		return;
	}
	if (TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
	                                thread_group_z, mode, cs_regs.cs_regs.data_addr)) {
		ResetBindings();
		return;
	}
	if (step == nullptr &&
	    TryFillMappedParts(m_context, input_info, thread_group_x, thread_group_y, thread_group_z, mode)) {
		ResetBindings();
		return;
	}
	if (DccKey10FillEnabled() &&
	    TryDccKey10Fill(m_context, input_info, thread_group_x, thread_group_y, thread_group_z,
	                    use_thread_dimensions)) {
		ResetBindings();
		return;
	}

	if (use_thread_dimensions) {
		thread_group_x = input_info.workgroup_counts[0];
		thread_group_y = input_info.workgroup_counts[1];
		thread_group_z = input_info.workgroup_counts[2];
	}

	buffer.EndRendering();
	CpGaps::Phase(CpGaps::Cat::DispatchPipeline);
	// A software ray-tracing kernel whose pipeline is still being built, or any RT kernel after one
	// failed to build (rtSession.h), is skipped: the lighting it feeds lags or stays as it was.
	auto* pipeline_or_null =
	    m_context.GetPipelineCache().TryGetComputePipeline(input_info, compute_program);
	if (pipeline_or_null == nullptr) {
		ResetBindings();
		return;
	}
	auto& pipeline = *pipeline_or_null;
	const bool peek = DispatchPeekWanted(input_info);
	if (peek) DispatchPeek(m_context, buffer, input_info, 0);
	auto& bindings = m_compute_bindings;
	CpGaps::Phase(CpGaps::Cat::DispatchBind);
	PrepareBindings(input_info.stage, bindings);
	if (step != nullptr) {
		// KYTY_CALL_PARTITIONS: the first steps, to see that the parameters reach the program.
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_CALL_PARTITIONS: step of CS 0x{:016x}: call-state dword {}, report 0x{:x}, target 0x{:x}, bvh {}\n",
			    input_info.stage.program->shader_hash, input_info.stage.program->bindings.call_state_dword,
			    step->report, step->target, input_info.stage.program->info.uses_bvh ? 1 : 0));
		}
	}
	if (step != nullptr && input_info.stage.program->bindings.call_state_dword !=
	                         ShaderRecompiler::IR::PushData::NoStart) {
		const auto dword = input_info.stage.program->bindings.call_state_dword;
		const uint32_t params[ShaderRecompiler::IR::BindingLayout::CallStateDwords] = {
		    static_cast<uint32_t>(step->report), static_cast<uint32_t>(step->report >> 32u),
		    static_cast<uint32_t>(step->target), static_cast<uint32_t>(step->target >> 32u), 0u, 0u, 0u};
		std::copy(std::begin(params), std::end(params), bindings.shader_data.begin() + dword);
	}
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		// The requested thread extents initialize EXEC (upstream e4aca7a60).
		std::copy(std::begin(input_info.dispatch_threads_num),
		          std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	// The native dispatch size bounds workgroup ids for write-range proofs.
	bindings.dispatch_groups     = {thread_group_x, thread_group_y, thread_group_z};
	bindings.has_dispatch_groups = true;
	if (program.info.uses_dma) {
		PrefaultDmaPointers(m_context.GetBufferCache(), input_info.stage);
	}
	// KYTY_DISPATCH_OVERLAP_UPLOADS: uploads of this preparation stay pending for the dispatch's
	// flush while the overlap chain is open (CommandBuffer::KeepUploadsForDispatch).
	struct DispatchPrepScope {
		const CommandBuffer& buffer;
		explicit DispatchPrepScope(const CommandBuffer& b): buffer(b) { buffer.SetDispatchPrep(true); }
		~DispatchPrepScope() { buffer.SetDispatchPrep(false); }
		void End() const { buffer.SetDispatchPrep(false); }
	} dispatch_prep(buffer);
	if (program.info.bda_writes) {
		m_context.GetBufferCache().SnapshotBdaMergePages(program.shader_hash); // KYTY_BDA_CPU_MERGE
	}
	BackAddressBases(m_context, input_info.stage);
	FindBuffers(bindings);
	CpGaps::Phase(CpGaps::Cat::DispatchBda);
	if (program.info.uses_dma) {
		// KYTY_BDA_REACH: the dispatch's reach (workgroup ids bounded by a direct dispatch's size).
		const RenderContext::BdaReachStage reach {.runtime          = &input_info.stage,
		                                          .bindless_patches = bindings.bindless_patches,
		                                          .groups           = bindings.dispatch_groups,
		                                          .has_groups       = bindings.has_dispatch_groups};
		m_context.PrepareBda(std::span<const RenderContext::BdaReachStage>(&reach, 1));
	}
	// KYTY_RT_HW: the host acceleration structures of the program's guest TLAS.
	if (program.info.ray_query_sites != 0) {
		PrepareNativeRayTracingDispatch(m_context.GetNativeRayTracing(), buffer.Handle(), program.info,
		                                input_info.stage.resources->user_data, m_context.GetGpu().GetFrameNum(),
		                                [this](uint64_t address, uint64_t size) { return m_context.IsMapped(address, size); });
	}
	if (program.info.uses_bvh) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::RtStubDispatches);
	}
	if (program.info.bda_writes) {
		// KYTY_BDA_WRITES: settled synchronously right after the dispatch below. The settle relies
		// on PrepareBda's pass above having uploaded every CPU-dirty page first.
		EXIT_IF(!program.info.uses_dma);
		m_context.GetBufferCache().PrepareBdaWrites();
	}
	CpGaps::Phase(CpGaps::Cat::DispatchRebind);
	RebindImages(bindings);
	BindComputeLds(input_info, bindings, input_info.workgroup_counts);
	m_rebind_epoch = m_context.GetBufferCache().UnregisterEpoch();
	RebindBuffers(bindings);
	// KYTY_REBIND_AFTER_JOIN: a later binding's lookup joined a buffer an earlier one resolved.
	if (RebindAfterJoinEnabled()) {
		auto& cache = m_context.GetBufferCache();
		for (uint32_t pass = 0; pass < 4 && cache.UnregisterEpoch() != m_rebind_epoch; pass++) {
			NoteRebindAfterJoin("dispatch", nullptr);
			m_rebind_epoch = cache.UnregisterEpoch();
			RebindBuffers(bindings);
		}
	}
	CpGaps::Phase(CpGaps::Cat::DispatchEmit);

	PreparedBindings* descriptor_stage = &bindings;
	// Emission safe point (KYTY_CP_RECORDER, render.h): no native handle from the preparation above
	// is alive. Binding commits and the barrier requests below record state commands only (image
	// transitions and dependencies are batched); the dispatch goes through Sink(), which records
	// them first as one barrier, exactly as Handle() did.
	buffer.BeginEmission();
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	                          HasShaderBufferWrites(input_info.stage) || program.info.bda_writes;
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written &&
		                       image.resource_class ==
		                           ShaderRecompiler::IR::ImageResourceClass::Storage;
	                }) ||
	    has_storage_writes;
	// KYTY_DISPATCH_OVERLAP (render.h): nothing happened since the previous dispatch, so the guest
	// did not order the two; its pending ShaderAccess request is dropped and this dispatch skips
	// its own hazard request. Emulated LDS (one device buffer) and BDA-write settles never overlap.
	const bool no_overlap      = bindings.shared_memory.buffer != nullptr || program.info.bda_writes;
	const bool overlap_enabled = DispatchOverlapEnabled();
	const bool overlap         = buffer.TakeDispatchOverlap(no_overlap, overlap_enabled);
	dispatch_prep.End();
	CountDispatchOverlap(overlap, overlap_enabled);
	// KYTY_GPU_HAZARD (render.h): no barrier against the chain when no range of this dispatch
	// overlaps a range the chain wrote (or reads one it writes).
	static thread_local CommandBuffer::DispatchAccess hazard_access;
	const bool hazard_on   = GpuHazardMode() != 0;
	bool       hazard_skip = false;
	if (hazard_on) {
		CollectDispatchAccess(m_context.GetTextureCache(), input_info.stage, bindings, hazard_access);
		if (no_overlap) {
			hazard_access.precise = false; // emulated LDS, BDA-write settles
		}
		hazard_skip = buffer.DecideDispatchHazard(hazard_access, overlap && overlap_enabled);
	}
	// KYTY_PM4_BARRIERS=guest (render.h): the guest's own sync points order dispatches instead.
	const bool guest_barriers = GuestSyncBarriersEnabled() && !no_overlap;
	if (has_storage_writes && !(overlap && overlap_enabled) && !guest_barriers && !hazard_skip) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		ShaderWriteHazardBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64 " completed_tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     m_context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}
	{
		const uint32_t traced_groups[3] {thread_group_x, thread_group_y, thread_group_z};
		TraceDispatch(m_context, input_info, traced_groups, 0);
	}
	// KYTY_DUMP_CS_IMAGES: ordinary dispatches too (the bound images after the dispatch).
	const auto image_dump = PlanCsImageDump(input_info);
	buffer.BindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	// Vulkan debug labels (--rd / KYTY_TRACE_VK_LABELS): the dispatch's shader and game pass.
	const bool dispatch_label = PassCensus::LabelsEnabled();
	if (dispatch_label) {
		char text[256];
		PassCensus::LabelText(text, sizeof(text), 'c', program.shader_hash, 0);
		buffer.Sink().insertDebugLabel(text);
	}
	constexpr uint32_t DispatchInitiatorOrderedAppendEnable = 1u << 3u;
	const uint64_t     group_threads = uint64_t {std::max(input_info.threads_num[0], 1u)} *
	                               std::max(input_info.threads_num[1], 1u) *
	                               std::max(input_info.threads_num[2], 1u);
	if (OrderedAppendSplitEnabled() && (mode & DispatchInitiatorOrderedAppendEnable) != 0 &&
	    (ShaderRecompiler::IR::FindBinding(program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr) && group_threads <= input_info.wave_size && !input_info.lds_storage) {
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			std::printf("KYTY_ORDERED_APPEND: shader 0x%016" PRIx64
			            " runs %ux%ux%u ordered-append workgroups one by one\n",
			            program.shader_hash, thread_group_x, thread_group_y, thread_group_z);
			std::fflush(stdout);
		}
		for (uint32_t z = 0; z < thread_group_z; ++z) {
			for (uint32_t y = 0; y < thread_group_y; ++y) {
				for (uint32_t x = 0; x < thread_group_x; ++x) {
					if ((x | y | z) != 0u) {
						ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
					}
					buffer.Sink().dispatchBase(x, y, z, 1, 1, 1);
				}
			}
		}
	} else {
		if (CsVerifyHash() != 0 && program.shader_hash == CsVerifyHash()) [[unlikely]] {
			CsVerifyDispatch(m_context, buffer, input_info, thread_group_x * thread_group_y * thread_group_z);
		}
		if (LiveProbe::Hash() != 0 && program.shader_hash == LiveProbe::Hash()) [[unlikely]] {
			LiveProbe::Dispatch(m_context, buffer, input_info);
		}
		buffer.Sink().dispatch(thread_group_x, thread_group_y, thread_group_z);
	}
	if (hazard_on) {
		buffer.NoteHazardDispatch(hazard_access);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	if (auto& cache = m_context.GetTextureCache(); cache.HasStencilStorageShadows()) {
		cache.FlushStencilStorageShadows(); // KYTY_STENCIL_STORAGE_SHADOW
	}
	if (!image_dump.items.empty()) {
		ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
		RunCsImageDump(m_context, image_dump);
	}
	PassDumpNoteDispatch(input_info); // KYTY_PASS_DUMP
	if (const auto op = NanScanTake(); op != 0) {
		ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderTextureResource>(input_info.stage.resources->images[i]);
			(void)m_context.GetTextureCache().DebugNanScan(
			    r.Base40(), static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			    fmt::format("op {} CS 0x{:016x} groups {}x{}x{} image {} {}", op, program.shader_hash,
			                thread_group_x, thread_group_y, thread_group_z, i,
			                program.info.images[i].written ? "written" : "read"));
		}
	}

	// The removed host fence also ordered read-only dispatches before later writers.
	if (!guest_barriers) {
		ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	buffer.NoteGuestDispatch(no_overlap);
	if (program.info.bda_writes) {
		m_context.GetBufferCache().SettleBdaWrites(program.shader_hash);
	}
	if (peek) DispatchPeek(m_context, buffer, input_info, 1);
	// A proven uniform buffer fill leaves a known value (e.g. DCC fast-clear codes). Recording
	// it lets consumers skip reading the range back while nothing else writes it.
	{
		ShaderBufferResource known_descriptor;
		uint32_t             known_value = 0;
		uint64_t             known_size  = 0;
		const auto groups_x = use_thread_dimensions ? input_info.dispatch_threads_num[0]
		                                           : thread_group_x;
		if (ResolveComputeBufferFill(input_info, groups_x,
		                             use_thread_dimensions ? input_info.dispatch_threads_num[1]
		                                                   : thread_group_y,
		                             use_thread_dimensions ? input_info.dispatch_threads_num[2]
		                                                   : thread_group_z,
		                             mode, known_descriptor, known_value, known_size)) {
			m_context.GetBufferCache().RecordKnownFill(known_descriptor.Base48(), known_size,
			                                           known_value);
		}
	}
	// Observe only dispatches that actually reached the native path. This reuses the
	// already materialized descriptor/value proof and performs no guest-memory reads.
	if (Profiler::DetailedEnabled() && tracy::ProfilerAvailable() && TracyIsConnected) {
		ShaderBufferResource fill_descriptor;
		uint32_t fill_value = 0;
		uint64_t fill_size = 0;
		const auto groups_x = use_thread_dimensions ? input_info.dispatch_threads_num[0]
		                                           : thread_group_x;
		const auto groups_y = use_thread_dimensions ? input_info.dispatch_threads_num[1]
		                                           : thread_group_y;
		const auto groups_z = use_thread_dimensions ? input_info.dispatch_threads_num[2]
		                                           : thread_group_z;
		if (ResolveComputeBufferFill(input_info, groups_x, groups_y, groups_z, mode,
		                             fill_descriptor, fill_value, fill_size)) {
			KYTY_PROFILER_DETAIL_BLOCK("Compute::UniformFillEligible");
			static std::atomic<uint32_t> fill_count {0};
			constexpr uint32_t MaxFillMessages = 4096;
			const auto ordinal = fill_count.fetch_add(1, std::memory_order_relaxed);
			if (ordinal < MaxFillMessages) {
				char text[512];
				const uint32_t byte = fill_value & 0xffu;
				const auto written = std::snprintf(
				    text, sizeof(text),
				    "COMPUTE_UNIFORM_FILL id=%u shader=0x%" PRIx64 " address=0x%" PRIx64
				    " bytes=%" PRIu64 " value=0x%08x byte_uniform=%u tick=%" PRIu64,
				    ordinal + 1u, program.shader_hash, fill_descriptor.Base48(), fill_size,
				    fill_value, fill_value == byte * 0x01010101u ? 1u : 0u,
				    m_context.GetCommandScheduler().CurrentTick());
				if (written > 0 && static_cast<size_t>(written) < sizeof(text)) {
					TracyMessageS(text, static_cast<size_t>(written), 0);
				}
			} else if (ordinal == MaxFillMessages) {
				TracyMessageLS("COMPUTE_UNIFORM_FILL_LIMIT messages=4096", 0);
			}
		}
	}
	ResetBindings();
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer,
                                      uint64_t args_addr, uint32_t mode, bool async_compute) {
	std::shared_ptr<const void> run;
	DispatchIndirectOne(submit_id, buffer, args_addr, mode, async_compute, nullptr, &run);
	if (run != nullptr) {
		DispatchPartitions(buffer, run, [&](const PartitionStep& step) {
			DispatchIndirectOne(submit_id, buffer, args_addr, mode, async_compute, &step, nullptr);
		});
	}
}

void RenderExecutor::DispatchIndirectOne(uint64_t submit_id, CommandBuffer& buffer,
                                         uint64_t args_addr, uint32_t mode, bool async_compute,
                                         const PartitionStep*         step,
                                         std::shared_ptr<const void>* partition_run) {
	KYTY_GPU_OP_SITE("dispatch.indirect");
	g_overlap_indirect.fetch_add(1, std::memory_order_relaxed);
	const CpGaps::Scope gap(CpGaps::Cat::DispatchOp);
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0);
	// KYTY_INDIRECT_THREADS_GPU: the arguments are thread counts (the command processor sends these
	// here only with the switch on).
	const bool thread_counts = (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	m_context.GetCommandScheduler().PopReadyOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	RenderLockGuard lock(m_context.GetMutex());
	// KYTY_DRAW_RUN: a dispatch ends the run of the draws before it (drawPrep/drawRun.h).
	BeginDrawRun();
	if (PassDump::Before(m_context)) { // KYTY_PASS_DUMP: the previous pass segment's outputs
		buffer.EndRendering();
		PassDump::Flush(m_context);
	}
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.async_compute                = async_compute;
	input_info.dispatch_thread_dimensions   = thread_counts;
	input_info.dispatch_dimensions_indirect = thread_counts;
	CpGaps::Phase(CpGaps::Cat::DispatchProgram);
	ShaderProgram compute_program;
	// KYTY_DRAW_PREP_DISPATCH (drawPrep.h): as in DispatchDirect, from no workgroup counts.
	auto* const prepared = std::exchange(m_prepared_dispatch, nullptr);
	// KYTY_SKIP_EMPTY_INDIRECT (live, default on; upstream be5915c24): arguments no GPU write is
	// pending for, read from guest memory, with a zero group count: the dispatch does nothing.
	if (g_skip_empty_indirect.Get() != 0) {
		vk::DispatchIndirectCommand args {};
		if (IndirectArgsCleanForBackingRead(m_context, args_addr, sizeof(args)) &&
		    LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args)) &&
		    (args.x == 0 || args.y == 0 || args.z == 0)) {
			g_overlap_empty_indirect.fetch_add(1, std::memory_order_relaxed);
			SkipEmptyLog::NoteSkip(cs_regs.cs_regs.data_addr, args_addr);
			return;
		}
	}
	if (step != nullptr) {
		PipelineCache::SelectCallSplitCode(step->code);
	}
	if (step == nullptr && prepared != nullptr && !async_compute && !thread_counts &&
	    DrawPrep::ValidateDispatch(*prepared, input_info)) {
		if (DrawPrep::VerifyMode() != 0) {
			auto serial_prep = std::make_unique<PipelineCache::StagePrep>();
			auto serial_info = input_info;
			const auto serial_program = m_context.GetPipelineCache().GetComputeProgram(
			    cs_regs, buffer.GetRegisters().GetShaderRegisters(), serial_info, *serial_prep);
			(void)DrawPrep::VerifyDispatch(*prepared, serial_program, serial_info, *serial_prep);
		}
		PipelineCache::DropCallRuns(); // as in DispatchDirect
		std::swap(m_compute_prep, prepared->compute_prep);
		input_info                 = prepared->compute_info;
		input_info.stage.resources = &m_compute_prep.resources;
		compute_program            = prepared->compute_program;
	} else {
		// KYTY_WORKGROUP_IMAGES: images enumerated per workgroup need the dispatch's counts. Read
		// them from guest memory when no GPU write of them is pending; when one is (or they were
		// not read), and the program asked for them, synchronize the arguments and retry once.
		const bool workgroup_images = ShaderRecompiler::GetCodegenOptions().workgroup_images;
		const auto read_counts = [&] {
			vk::DispatchIndirectCommand args {};
			if (!IndirectArgsCleanForBackingRead(m_context, args_addr, sizeof(args)) ||
			    !LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args))) {
				return false;
			}
			input_info.workgroup_counts[0] = args.x;
			input_info.workgroup_counts[1] = args.y;
			input_info.workgroup_counts[2] = args.z;
			if (thread_counts) {
				const uint32_t local[3] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
				                           cs_regs.cs_regs.num_thread_z};
				for (uint32_t i = 0; i < 3u; i++) {
					const auto threads = input_info.workgroup_counts[i];
					input_info.workgroup_counts[i] =
					    threads == 0 ? 0u : (threads - 1u) / std::max(local[i], 1u) + 1u;
				}
			}
			return true;
		};
		bool counts_read = false;
		if (workgroup_images) {
			counts_read = read_counts();
			(void)ShaderRecompiler::IR::TakeWorkgroupCountsMissing();
		}
		compute_program = m_context.GetPipelineCache().GetComputeProgram(
		    cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info, m_compute_prep);
		if (workgroup_images && !compute_program &&
		    ShaderRecompiler::IR::TakeWorkgroupCountsMissing() && !counts_read &&
		    m_context.SynchronizeGpuBackingForRead(args_addr, sizeof(vk::DispatchIndirectCommand)) &&
		    read_counts()) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "KYTY_WORKGROUP_IMAGES: indirect dispatch of CS 0x{:x}: GPU-written arguments "
				    "{}x{}x{} read back for its workgroup images\n",
				    cs_regs.cs_regs.data_addr, input_info.workgroup_counts[0],
				    input_info.workgroup_counts[1], input_info.workgroup_counts[2]));
			}
			// Zero groups: nothing to run (the program stays empty and the dispatch is dropped).
			if (std::ranges::find(input_info.workgroup_counts, 0u) == std::end(input_info.workgroup_counts)) {
				compute_program = m_context.GetPipelineCache().GetComputeProgram(
				    cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info, m_compute_prep);
			}
		}
	}
	if (step != nullptr) {
		PipelineCache::SelectCallSplitCode(nullptr);
	}
	if (input_info.stage.program != nullptr) {
		SkipEmptyLog::NoteRun(cs_regs.cs_regs.data_addr, input_info.stage.program->shader_hash, args_addr);
	}
	if (partition_run != nullptr) {
		if (auto run = PipelineCache::TakePartitionRun(); run != nullptr) {
			*partition_run = std::move(run);
			ResetBindings();
			return;
		}
	}
	CpGaps::Phase(CpGaps::Cat::DispatchPipeline);
	if (WatchReadEnabled() && compute_program) {
		WatchReadOnDispatch(input_info.stage.program->shader_hash,
		                    std::span<const uint32_t>(cs_regs.cs_user_sgpr.value,
		                                              std::min<uint32_t>(cs_regs.cs_user_sgpr.count,
		                                                                 HW::UserSgprInfo::SGPRS_MAX)),
		                    static_cast<int>(m_context.GetGpu().GetFrameNum()));
	}
	if (!compute_program) {
		// Temporary until RT is implemented.
		DropProgramless(m_context.GetPipelineCache());
		return;
	}
	if (PipelineCache::TakeCallSplitRun() != nullptr) {
		// KYTY_SHADER_CALLS_SPLIT: an indirect dispatch's wave count is GPU data; not split yet.
		static std::atomic_flag logged = ATOMIC_FLAG_INIT;
		if (!logged.test_and_set()) {
			Log::WriteToConsoleAndLog("KYTY_SHADER_CALLS_SPLIT: indirect dispatches of split kernels are skipped.\n");
		}
		ResetBindings();
		return;
	}
	buffer.EndRendering();
	auto* pipeline_or_null =
	    m_context.GetPipelineCache().TryGetComputePipeline(input_info, compute_program);
	if (pipeline_or_null == nullptr) {
		ResetBindings();
		return;
	}
	auto& pipeline = *pipeline_or_null;
	const bool peek = DispatchPeekWanted(input_info);
	if (peek) DispatchPeek(m_context, buffer, input_info, 0);
	auto& bindings = m_compute_bindings;
	CpGaps::Phase(CpGaps::Cat::DispatchBind);
	PrepareBindings(input_info.stage, bindings);
	if (step != nullptr) {
		// KYTY_CALL_PARTITIONS: the first steps, to see that the parameters reach the program.
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_CALL_PARTITIONS: step of CS 0x{:016x}: call-state dword {}, report 0x{:x}, target 0x{:x}, bvh {}\n",
			    input_info.stage.program->shader_hash, input_info.stage.program->bindings.call_state_dword,
			    step->report, step->target, input_info.stage.program->info.uses_bvh ? 1 : 0));
		}
	}
	if (step != nullptr && input_info.stage.program->bindings.call_state_dword !=
	                         ShaderRecompiler::IR::PushData::NoStart) {
		const auto dword = input_info.stage.program->bindings.call_state_dword;
		const uint32_t params[ShaderRecompiler::IR::BindingLayout::CallStateDwords] = {
		    static_cast<uint32_t>(step->report), static_cast<uint32_t>(step->report >> 32u),
		    static_cast<uint32_t>(step->target), static_cast<uint32_t>(step->target >> 32u), 0u, 0u, 0u};
		std::copy(std::begin(params), std::end(params), bindings.shader_data.begin() + dword);
	}
	// GPU-produced dispatch arguments: workgroup ids stay unbounded for write-range proofs.
	bindings.has_dispatch_groups = false;
	if (input_info.stage.program->info.uses_dma) {
		PrefaultDmaPointers(m_context.GetBufferCache(), input_info.stage);
	}
	if (input_info.stage.program->info.bda_writes) {
		m_context.GetBufferCache().SnapshotBdaMergePages(input_info.stage.program->shader_hash); // KYTY_BDA_CPU_MERGE
	}
	// The writer of the bindings FindBuffers obtains is this dispatch (KYTY_WATCH_RANGE / WRITERS).
	WatchRange::SetCurrentOp("CS", input_info.stage.program->shader_hash);
	WatchRange::SetCurrentOpDetail(static_cast<uint32_t>(m_context.GetGpu().GetFrameNum()),
	                               input_info.stage.resources->user_data.data(),
	                               static_cast<uint32_t>(input_info.stage.resources->user_data.size()), 0xffffffffu,
	                               0xffffffffu, 0xffffffffu);
	BackAddressBases(m_context, input_info.stage);
	FindBuffers(bindings);
	CpGaps::Phase(CpGaps::Cat::DispatchBda);
	const auto& program = *input_info.stage.program;
	PassCensus::Exec('c', program.shader_hash, 1);
	WatchRange::SetCurrentOp("CS", program.shader_hash);
	WatchRange::NoteDispatch(program.shader_hash, async_compute, 0, 0, 0, true);
	if (program.info.uses_dma) {
		// KYTY_BDA_REACH: the dispatch's reach (workgroup ids bounded by a direct dispatch's size).
		const RenderContext::BdaReachStage reach {.runtime          = &input_info.stage,
		                                          .bindless_patches = bindings.bindless_patches,
		                                          .groups           = bindings.dispatch_groups,
		                                          .has_groups       = bindings.has_dispatch_groups};
		m_context.PrepareBda(std::span<const RenderContext::BdaReachStage>(&reach, 1));
	}
	// KYTY_RT_HW: the host acceleration structures of the program's guest TLAS.
	if (program.info.ray_query_sites != 0) {
		PrepareNativeRayTracingDispatch(m_context.GetNativeRayTracing(), buffer.Handle(), program.info,
		                                input_info.stage.resources->user_data, m_context.GetGpu().GetFrameNum(),
		                                [this](uint64_t address, uint64_t size) { return m_context.IsMapped(address, size); });
	}
	if (program.info.uses_bvh) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::RtStubDispatches);
	}
	if (program.info.bda_writes) {
		// KYTY_BDA_WRITES: settled synchronously right after the dispatch below. The settle relies
		// on PrepareBda's pass above having uploaded every CPU-dirty page first.
		EXIT_IF(!program.info.uses_dma);
		m_context.GetBufferCache().PrepareBdaWrites();
	}
	CpGaps::Phase(CpGaps::Cat::DispatchRebind);
	BindComputeLds(input_info, bindings, nullptr);
	RebindImages(bindings);
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	// KYTY_RAW_READ_IMAGE_SYNC_OBTAINS (BufferCache::PrepareReadRange), before the epoch is taken.
	m_context.GetBufferCache().PrepareReadRange(args_addr, sizeof(vk::DispatchIndirectCommand));
	m_rebind_epoch = m_context.GetBufferCache().UnregisterEpoch();
	auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
	DispatchGuard::Slot guard_slot {};
	const bool          guard_checks = DispatchGuard::Enabled();
	if (guard_checks || thread_counts) {
		if (m_dispatch_guard == nullptr) {
			m_dispatch_guard = std::make_unique<DispatchGuard>(m_context);
		}
		guard_slot = m_dispatch_guard->Reserve();
	}
	if (thread_counts && program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		// KYTY_INDIRECT_THREADS_GPU: the dispatch-thread dwords hold the address of the slot's copy
		// of the thread counts (written by the guard pass below), which bounds the lanes.
		const auto threads = m_dispatch_guard->SlotThreads(guard_slot);
		const auto dword   = program.bindings.dispatch_thread_dword;
		bindings.shader_data[dword]      = static_cast<uint32_t>(threads);
		bindings.shader_data[dword + 1u] = static_cast<uint32_t>(threads >> 32u);
		bindings.shader_data[dword + 2u] = 0u;
	}
	RebindBuffers(bindings);
	// KYTY_REBIND_AFTER_JOIN: the bindings' lookups joined the argument buffer (or one another's).
	if (RebindAfterJoinEnabled()) {
		auto& cache = m_context.GetBufferCache();
		for (uint32_t pass = 0; pass < 4 && cache.UnregisterEpoch() != m_rebind_epoch; pass++) {
			NoteRebindAfterJoin("indirect dispatch", nullptr);
			m_rebind_epoch               = cache.UnregisterEpoch();
			std::tie(args_buffer, args_offset) =
			    cache.ObtainBuffer(args_addr, sizeof(vk::DispatchIndirectCommand), false);
			EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
			RebindBuffers(bindings);
		}
	}
	CpGaps::Phase(CpGaps::Cat::DispatchEmit);
	PreparedBindings* descriptor_stage = &bindings;
	// Emission safe point (see DispatchDirect).
	buffer.BeginEmission();
	vk::Buffer dispatch_buffer = args_buffer->Handle();
	uint64_t   dispatch_offset = args_offset;
	if (guard_slot.buffer) {
		std::array<uint32_t, 3> local_size {0, 0, 0};
		if (thread_counts) {
			local_size = {std::max(cs_regs.cs_regs.num_thread_x, 1u),
			              std::max(cs_regs.cs_regs.num_thread_y, 1u),
			              std::max(cs_regs.cs_regs.num_thread_z, 1u)};
		}
		m_dispatch_guard->Record(buffer, guard_slot, args_buffer->Handle(), args_offset,
		                         program.shader_hash, local_size, guard_checks);
		dispatch_buffer = guard_slot.buffer;
		dispatch_offset = guard_slot.offset;
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	const bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage) || program.info.bda_writes ||
	    std::any_of(program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		    return image.written && image.resource_class ==
		                                ShaderRecompiler::IR::ImageResourceClass::Storage;
	    });
	// KYTY_PM4_BARRIERS=guest (render.h): the guest's own sync points order dispatches instead.
	const bool guest_barriers = GuestSyncBarriersEnabled() &&
	                            bindings.shared_memory.buffer == nullptr && !program.info.bda_writes;
	if (has_storage_writes && !guest_barriers) {
		ShaderWriteHazardBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	if (BarrierBatchEnabled()) {
		buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eAllGraphics |
		                                vk::PipelineStageFlagBits2::eComputeShader |
		                                vk::PipelineStageFlagBits2::eTransfer,
		                            vk::AccessFlagBits2::eShaderWrite |
		                                vk::AccessFlagBits2::eTransferWrite,
		                            vk::PipelineStageFlagBits2::eDrawIndirect,
		                            vk::AccessFlagBits2::eIndirectCommandRead,
		                            BarrierOrigin::IndirectArgs);
	} else {
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask =
		    vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		buffer.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
		                                    vk::PipelineStageFlagBits::eComputeShader |
		                                    vk::PipelineStageFlagBits::eTransfer,
		                                vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier,
		                                0, nullptr, 0, nullptr);
	}
	TraceDispatch(m_context, input_info, nullptr, args_addr);
	buffer.BindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	const bool dispatch_label = PassCensus::LabelsEnabled(); // --rd / KYTY_TRACE_VK_LABELS
	if (dispatch_label) {
		char text[256];
		PassCensus::LabelText(text, sizeof(text), 'c', program.shader_hash, 0);
		buffer.Sink().insertDebugLabel(text);
	}
	buffer.Sink().dispatchIndirect(dispatch_buffer, dispatch_offset);
	// KYTY_GPU_HAZARD: not noted as a chain dispatch (its argument read is in the draw-indirect stage,
	// which a later ShaderAccess flush does not order): the chain ends here.
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	if (auto& cache = m_context.GetTextureCache(); cache.HasStencilStorageShadows()) {
		cache.FlushStencilStorageShadows(); // KYTY_STENCIL_STORAGE_SHADOW
	}
	if (!guest_barriers) {
		ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	PassDumpNoteDispatch(input_info); // KYTY_PASS_DUMP
	if (const auto op = NanScanTake(); op != 0) {
		ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderTextureResource>(input_info.stage.resources->images[i]);
			(void)m_context.GetTextureCache().DebugNanScan(
			    r.Base40(), static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			    fmt::format("op {} CS 0x{:016x} indirect image {} {}", op, program.shader_hash, i,
			                program.info.images[i].written ? "written" : "read"));
		}
	}
	if (program.info.bda_writes) {
		m_context.GetBufferCache().SettleBdaWrites(program.shader_hash);
	}
	if (peek) DispatchPeek(m_context, buffer, input_info, 1);
	ResetBindings();
}

} // namespace Libs::Graphics
