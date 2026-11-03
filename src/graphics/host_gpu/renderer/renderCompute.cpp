#include "graphics/host_gpu/watchRange.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
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

#include <algorithm>
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
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {
// KYTY_DISPATCH_OVERLAP accounting, printed every 10 s once a direct dispatch ran:
//   DispatchOverlap 10s (on|off): <direct> direct dispatches, <pairs> unordered pairs (<pct>%),
//   <overlapped> overlapped, <indirect> indirect (<empty> empty skipped, KYTY_SKIP_EMPTY_INDIRECT)
// An unordered pair is a direct dispatch that followed another with nothing in between (it would
// overlap with the switch on); counted with the switch off too, for sizing.
static std::atomic<uint64_t> g_overlap_direct {0}, g_overlap_pairs {0}, g_overlap_taken {0},
    g_overlap_indirect {0}, g_overlap_empty_indirect {0};
static Live::Switch g_skip_empty_indirect("KYTY_SKIP_EMPTY_INDIRECT", Live::ParseDefaultOn);
static std::atomic<int64_t> g_overlap_last_print_ns {0};

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
	static const Request request = [] {
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
	}();
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

// KYTY_FRAME_TRACE=<folder>[:<count>] (diagnostic): while <folder>/trace.now exists, the next <count>
// (default 4000) compute dispatches and draws are logged as "FrameTrace" lines (programs, bound
// images and buffers, render targets; frameTrace.h), then the file is removed. One frame's producers
// and readers of any target, in submission order. Local logs only.
bool FrameTraceTake() {
	struct Request {
		std::filesystem::path trigger;
		uint32_t              count = 4000;
	};
	static const Request request = [] {
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
	}();
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
	std::string text = fmt::format("{}: hash=0x{:016x} #{} tick={} gpu_tick={}",
	                               frame ? "FrameTrace CS" : "DispatchTrace",
	                               program.shader_hash, ordinal,
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
		text += fmt::format(" | b{}{}=0x{:x}/s{}/n{}", i, program.info.buffers[i].written ? "w" : "",
		                    r.Base48(), r.Stride(), r.NumRecords());
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
					if (!base.empty() && base[0] == 's') {
						range.user_sgpr = static_cast<uint32_t>(std::strtoul(base.c_str() + 1, nullptr, 10));
						const auto at   = base.find('@');
						range.address =
						    at == std::string::npos ? 0 : std::strtoull(base.c_str() + at + 1, nullptr, 16);
					} else {
						range.address = std::strtoull(base.c_str(), nullptr, 16);
					}
					range.bytes   = std::min<uint32_t>(
                        static_cast<uint32_t>(std::strtoul(item.substr(plus + 1).c_str(), nullptr, 0)),
                        64u) & ~3u;
					if ((range.address != 0 || range.user_sgpr != UINT32_MAX) && range.bytes != 0) {
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

static bool DispatchPeekWanted(const ShaderComputeInputInfo& input_info) {
	const auto& request = DispatchTrace();
	if (DispatchPeekRanges().empty() || request.hashes.empty()) return false;
	const auto hash = input_info.stage.program->shader_hash;
	const bool all  = std::ranges::find(request.hashes, uint64_t {0}) != request.hashes.end();
	return all || std::ranges::find(request.hashes, hash) != request.hashes.end();
}

// phase 0 = before the dispatch, 1 = after it.
static void DispatchPeek(RenderContext& context, CommandBuffer& command,
                         const ShaderComputeInputInfo& input_info, uint32_t phase) {
	const auto hash = input_info.stage.program->shader_hash;
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
	for (auto& range: ranges) {
		if (range.user_sgpr == UINT32_MAX) continue;
		const auto low  = range.user_sgpr < user_data.size() ? user_data[range.user_sgpr] : 0u;
		const auto high = range.user_sgpr + 1 < user_data.size() ? user_data[range.user_sgpr + 1] : 0u;
		range.address += low | (static_cast<uint64_t>(high & 0xffffu) << 32u);
	}
	command.EndRendering();
	for (uint32_t i = 0; i < ranges.size(); i++) {
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
				const bool meta_cleared = resource.read ? cache.ClearMeta(descriptor.Base48())
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

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                CommandBuffer& command, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z, uint32_t mode) {
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
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_ORDERED_APPEND");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
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
		return;
	}
	if (auto split = PipelineCache::TakeCallSplitRun(); split != nullptr) {
		// A kernel with call loops runs as its chain of dispatches (callSplitDispatch.cpp).
		ResetBindings();
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
		}
		return;
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	WatchRange::SetCurrentOp("CS", program.shader_hash);
	WatchRange::NoteDispatch(program.shader_hash, async_compute, thread_group_x, thread_group_y, thread_group_z, false);
	const auto& resources = *input_info.stage.resources;
	// A recorded metadata clear consumes the dispatch unless KYTY_COMPUTE_META_CLEAR_DISPATCH keeps
	// its guest writes (upstream 01ce97388).
	bool keep_meta_dispatch = false;
	RecordHtileRewriteClear(buffer.GetContext().GetTextureCache(), input_info);
	if (UpdateComputeMetaClears(input_info, buffer, &keep_meta_dispatch) &&
	    !ComputeMetaClearDispatchEnabled() && !keep_meta_dispatch) {
		ResetBindings();
		return;
	}
	if (TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
	                                thread_group_z, mode)) {
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
		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = input_info.workgroup_counts[0];
		thread_group_y       = input_info.workgroup_counts[1];
		thread_group_z       = input_info.workgroup_counts[2];

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
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
	FindBuffers(bindings);
	CpGaps::Phase(CpGaps::Cat::DispatchBda);
	if (program.info.uses_dma) {
		m_context.PrepareBda();
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
	// KYTY_PM4_BARRIERS=guest (render.h): the guest's own sync points order dispatches instead.
	const bool guest_barriers = GuestSyncBarriersEnabled() && !no_overlap;
	if (has_storage_writes && !(overlap && overlap_enabled) && !guest_barriers) {
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
		buffer.Sink().dispatch(thread_group_x, thread_group_y, thread_group_z);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	if (!image_dump.items.empty()) {
		ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
		RunCsImageDump(m_context, image_dump);
	}
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
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0 ||
	        (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0);
	m_context.GetCommandScheduler().PopReadyOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	RenderLockGuard lock(m_context.GetMutex());
	// KYTY_DRAW_RUN: a dispatch ends the run of the draws before it (drawPrep/drawRun.h).
	BeginDrawRun();
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	CpGaps::Phase(CpGaps::Cat::DispatchProgram);
	ShaderProgram compute_program;
	// KYTY_DRAW_PREP_DISPATCH (drawPrep.h): as in DispatchDirect, from no workgroup counts.
	auto* const prepared = std::exchange(m_prepared_dispatch, nullptr);
	// KYTY_SKIP_EMPTY_INDIRECT (live, default on; upstream be5915c24): arguments no GPU write is
	// pending for, read from guest memory, with a zero group count: the dispatch does nothing.
	if (g_skip_empty_indirect.Get() != 0) {
		vk::DispatchIndirectCommand args {};
		if (!m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args)) &&
		    LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args)) &&
		    (args.x == 0 || args.y == 0 || args.z == 0)) {
			g_overlap_empty_indirect.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
	if (step != nullptr) {
		PipelineCache::SelectCallSplitCode(step->code);
	}
	if (step == nullptr && prepared != nullptr && !async_compute &&
	    DrawPrep::ValidateDispatch(*prepared, input_info)) {
		if (DrawPrep::VerifyMode() != 0) {
			auto serial_prep = std::make_unique<PipelineCache::StagePrep>();
			auto serial_info = input_info;
			const auto serial_program = m_context.GetPipelineCache().GetComputeProgram(
			    cs_regs, buffer.GetRegisters().GetShaderRegisters(), serial_info, *serial_prep);
			(void)DrawPrep::VerifyDispatch(*prepared, serial_program, serial_info, *serial_prep);
		}
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
			if (m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args)) ||
			    !LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args))) {
				return false;
			}
			input_info.workgroup_counts[0] = args.x;
			input_info.workgroup_counts[1] = args.y;
			input_info.workgroup_counts[2] = args.z;
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
	FindBuffers(bindings);
	CpGaps::Phase(CpGaps::Cat::DispatchBda);
	const auto& program = *input_info.stage.program;
	WatchRange::SetCurrentOp("CS", program.shader_hash);
	WatchRange::NoteDispatch(program.shader_hash, async_compute, 0, 0, 0, true);
	if (program.info.uses_dma) {
		m_context.PrepareBda();
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
	m_rebind_epoch = m_context.GetBufferCache().UnregisterEpoch();
	auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
	DispatchGuard::Slot guard_slot {};
	if (DispatchGuard::Enabled()) {
		if (m_dispatch_guard == nullptr) {
			m_dispatch_guard = std::make_unique<DispatchGuard>(m_context);
		}
		guard_slot = m_dispatch_guard->Reserve();
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
		m_dispatch_guard->Record(buffer, guard_slot, args_buffer->Handle(), args_offset,
		                         program.shader_hash);
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
	buffer.Sink().dispatchIndirect(dispatch_buffer, dispatch_offset);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	if (!guest_barriers) {
		ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
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
