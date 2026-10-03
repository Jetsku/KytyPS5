#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hangTrace.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/rendererBatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/deviceLostReport.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawPrep/readSet.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLayoutCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/host_gpu/renderer/pipeline/programDiskCache.h"
#include "graphics/host_gpu/renderer/pipeline/stagePrepWorker.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/CodegenFingerprint.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/ProgramCodec.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fmt/format.h>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <shared_mutex>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <stop_token>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

namespace {

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

// Entries from older revisions accumulate in the driver cache; start over beyond this size.
constexpr uint64_t MaxDriverCacheFileSize = 512ull * 1024 * 1024;

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// The driver validates its own cache header and keys every entry by the exact shader code and
	// pipeline state, so data written by another emulator revision can only miss, never match a
	// different pipeline. Tying the file to the git revision (and disabling it for uncommitted
	// builds) made every experimental build compile all pipelines from scratch at each boot.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n", properties.vendorID,
	                   properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

// Cache file layout: DriverCacheSignature text, XXH3-64 of the payload, payload
// (vkGetPipelineCacheData output). Fills `payload` and returns true only for a complete,
// matching file within MaxDriverCacheFileSize.
bool ReadDriverCacheFile(const std::filesystem::path& file_path, const std::string& signature,
                         std::vector<uint8_t>& payload, bool log_invalid) {
	payload.clear();
	const auto   path = Common::PathToString(file_path);
	Common::File file(file_path, Common::File::Mode::Read);
	const auto   file_size = file.IsInvalid() ? 0 : file.Size();
	if (file_size < signature.size() + sizeof(uint64_t) || file_size > MaxDriverCacheFileSize) {
		file.Close();
		if (log_invalid) {
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
		return false;
	}
	std::string cached_signature(signature.size(), '\0');
	uint64_t    payload_hash = 0;
	payload.resize(file_size - signature.size() - sizeof(payload_hash));
	uint32_t signature_read = 0;
	uint32_t hash_read      = 0;
	uint32_t payload_read   = 0;
	file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
	          &signature_read);
	file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
	file.Read(payload.data(), static_cast<uint32_t>(payload.size()), &payload_read);
	file.Close();
	if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
	    payload_read != payload.size() || cached_signature != signature ||
	    XXH3_64bits(payload.data(), payload.size()) != payload_hash) {
		payload.clear();
		if (log_invalid) {
			PipelineCacheLog(
			    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)", path);
		}
		return false;
	}
	return true;
}

uint64_t EnvU64(const char* name, uint64_t default_value) {
	const auto* value = std::getenv(name);
	if (value == nullptr || *value == '\0') {
		return default_value;
	}
	char*      end    = nullptr;
	const auto parsed = std::strtoull(value, &end, 10);
	return end != value ? parsed : default_value;
}

// When the background saver writes the driver cache (KYTY_PIPELINE_CACHE_SAVE=0 disables it and
// leaves only the save at exit). A save needs at least MIN_NEW (8) pipelines created since the
// last one and no new pipeline for SETTLE (2 s), so it does not compete with a compile burst; then
// it runs when INTERVAL_S (15 s) have passed since the last save, or when a burst has ended
// (QUIET_S = 3 s without new pipelines, at least 15 s after the last save: level loads and area
// transitions). A steady trickle is saved after 3 intervals regardless of calm, and fewer than
// MIN_NEW new pipelines after 5 intervals. The short defaults keep what a crash loses (ErrorDeviceLost,
// a guest fault) to seconds of compiles instead of minutes (docs/DIVISAO-TRAVAMENTOS.md, C1).
struct DriverCacheSaveSettings {
	bool     enabled     = true;
	uint64_t min_new     = 8;
	uint64_t interval_ns = 15'000'000'000ull;
	uint64_t quiet_ns    = 3'000'000'000ull;
	uint64_t settle_ns   = 2'000'000'000ull;

	static const DriverCacheSaveSettings& Get() {
		static const DriverCacheSaveSettings settings = [] {
			DriverCacheSaveSettings s;
			s.enabled     = EnvU64("KYTY_PIPELINE_CACHE_SAVE", 1) != 0;
			s.min_new     = std::max<uint64_t>(1, EnvU64("KYTY_PIPELINE_CACHE_SAVE_MIN_NEW", 8));
			s.interval_ns = EnvU64("KYTY_PIPELINE_CACHE_SAVE_INTERVAL_S", 15) * 1'000'000'000ull;
			s.quiet_ns    = EnvU64("KYTY_PIPELINE_CACHE_SAVE_QUIET_S", 3) * 1'000'000'000ull;
			return s;
		}();
		return settings;
	}
};

// Compile-time accounting (stutter attribution). Every new program permutation and pipeline is
// timed per phase and reported three ways: per compile in the hang trace (compiles.csv, and
// per-second summary.csv columns), as Tracy aggregate FrameWaits, and as process totals printed
// once when the cache is destroyed. Only compile (miss) paths read the clock.
uint64_t CompileClockNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

struct CompileTotals {
	std::atomic<uint64_t> programs {0};
	std::atomic<uint64_t> translate_ns {0};
	std::atomic<uint64_t> clone_ns {0};
	std::atomic<uint64_t> translation_reuses {0};
	std::atomic<uint64_t> emit_ns {0};
	std::atomic<uint64_t> validate_ns {0};
	std::atomic<uint64_t> validate_async {0};
	std::atomic<uint64_t> validate_async_ns {0};
	std::atomic<uint64_t> module_ns {0};
	std::atomic<uint64_t> gfx_pipelines {0};
	std::atomic<uint64_t> gfx_pipeline_ns {0};
	std::atomic<uint64_t> gfx_new {0};
	std::atomic<uint64_t> gfx_permutation {0};
	std::atomic<uint64_t> gfx_variant {0};
	std::atomic<uint64_t> cs_pipelines {0};
	std::atomic<uint64_t> cs_pipeline_ns {0};
	std::atomic<uint64_t> stalls {0};
	std::atomic<uint64_t> stall_ns {0};
	std::atomic<uint64_t> stall_max_ns {0};
	// Pipeline libraries (6/7).
	std::atomic<uint64_t> gpl_cache_hits {0};
	std::atomic<uint64_t> gpl_links {0};
	std::atomic<uint64_t> gpl_link_ns {0};
	std::atomic<uint64_t> gpl_monolithic {0};
	std::atomic<uint64_t> gpl_libraries {0};
	std::atomic<uint64_t> gpl_library_ns {0};
	std::atomic<uint64_t> gpl_optimized {0};
	std::atomic<uint64_t> gpl_optimize_ns {0};
	// Persistent program cache (KYTY_PROGRAM_CACHE): sources and permutations reloaded instead
	// of translated and emitted, time spent reloading, and verify-mode comparisons.
	std::atomic<uint64_t> translations {0};
	std::atomic<uint64_t> disk_source_hits {0};
	std::atomic<uint64_t> disk_permutation_hits {0};
	std::atomic<uint64_t> disk_load_ns {0};
	std::atomic<uint64_t> disk_verify_checks {0};
	std::atomic<uint64_t> disk_verify_mismatches {0};
};
CompileTotals g_compile_totals;

void AtomicMax(std::atomic<uint64_t>& target, uint64_t value) {
	auto current = target.load(std::memory_order_relaxed);
	while (value > current &&
	       !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
	}
}

// Compile time this thread spent for the draw or dispatch it is preparing. Program compiles add
// to it; the pipeline lookup that completes the draw adds its own compile and reports the sum as
// one stall, so compile_stall_max_us is the worst per-draw hitch rather than per compile.
thread_local uint64_t t_pending_compile_stall_ns = 0;

void AddCompileStall(uint64_t ns) {
	t_pending_compile_stall_ns += ns;
}

void FlushCompileStall() {
	const auto ns = t_pending_compile_stall_ns;
	if (ns == 0) {
		return;
	}
	t_pending_compile_stall_ns = 0;
	g_compile_totals.stalls.fetch_add(1, std::memory_order_relaxed);
	g_compile_totals.stall_ns.fetch_add(ns, std::memory_order_relaxed);
	AtomicMax(g_compile_totals.stall_max_ns, ns);
	HangTrace::RecordCompileStall(ns);
}

// Phase times of one new program permutation. clone_ns is the copy of a kept translation (reused)
// or the copy kept after translating (item 5, KYTY_TRANSLATION_CACHE).
struct ProgramCompileTimes {
	uint64_t translate_ns = 0;
	uint64_t clone_ns     = 0;
	uint64_t emit_ns      = 0;
	uint64_t validate_ns  = 0;
	uint64_t module_ns    = 0;
	uint64_t spirv_words  = 0;
	bool     reused       = false;
	// Persistent program cache: key, lookups and decoding (all zero with it off), and whether
	// the permutation was reloaded instead of emitted.
	uint64_t load_ns   = 0;
	bool     from_disk = false;
};

#ifdef _WIN32
extern "C" __declspec(dllimport) void* __stdcall GetCurrentThread();
extern "C" __declspec(dllimport) int __stdcall SetThreadPriority(void* thread, int priority);
#endif

// Background compile threads (KYTY_ASYNC_PIPELINES, KYTY_ASYNC_TRANSLATE) run below normal priority
// so that a burst of compiles does not take CPU time from the game's threads.
void LowerThreadPriorityBelowNormal() {
#ifdef _WIN32
	constexpr int below_normal = -1; // THREAD_PRIORITY_BELOW_NORMAL
	(void)SetThreadPriority(GetCurrentThread(), below_normal);
#endif
}

void RecordProgramCompile(const char* stage_name, uint64_t guest_hash, uint64_t id,
                          const ProgramCompileTimes& times, uint64_t total_ns,
                          std::string_view detail) {
	auto& totals = g_compile_totals;
	totals.programs.fetch_add(1, std::memory_order_relaxed);
	totals.translate_ns.fetch_add(times.translate_ns, std::memory_order_relaxed);
	totals.clone_ns.fetch_add(times.clone_ns, std::memory_order_relaxed);
	if (times.reused) totals.translation_reuses.fetch_add(1, std::memory_order_relaxed);
	totals.emit_ns.fetch_add(times.emit_ns, std::memory_order_relaxed);
	totals.validate_ns.fetch_add(times.validate_ns, std::memory_order_relaxed);
	totals.module_ns.fetch_add(times.module_ns, std::memory_order_relaxed);
	Profiler::AddFrameWait(Profiler::FrameWait::ShaderTranslate, 1, times.translate_ns);
	Profiler::AddFrameWait(Profiler::FrameWait::ShaderEmit, 1, times.emit_ns);
	if (times.validate_ns != 0) {
		Profiler::AddFrameWait(Profiler::FrameWait::ShaderValidate, 1, times.validate_ns);
	}
	Profiler::AddFrameWait(Profiler::FrameWait::ShaderModuleCreate, 1, times.module_ns);
	if (times.load_ns != 0) {
		totals.disk_load_ns.fetch_add(times.load_ns, std::memory_order_relaxed);
		Profiler::AddFrameWait(Profiler::FrameWait::ShaderDiskLoad, 1, times.load_ns);
	}
	if (HangTrace::Enabled()) {
		HangTrace::RecordCompile({.kind         = HangTrace::CompileKind::Program,
		                          .stage        = stage_name,
		                          .guest_hash   = guest_hash,
		                          .id           = id,
		                          .translate_ns = times.translate_ns,
		                          .emit_ns      = times.emit_ns,
		                          .validate_ns  = times.validate_ns,
		                          .module_ns    = times.module_ns,
		                          .total_ns     = total_ns,
		                          .spirv_words  = times.spirv_words,
		                          .detail       = detail,
		                          .clone_ns     = times.clone_ns,
		                          .reused       = times.reused,
		                          .load_ns      = times.load_ns,
		                          .from_disk    = times.from_disk});
	}
}

double LoadMs(const std::atomic<uint64_t>& ns) {
	return static_cast<double>(ns.load(std::memory_order_relaxed)) / 1.0e6;
}

void LogCompileTotals() {
	const auto& t        = g_compile_totals;
	const auto  ms       = LoadMs;
	const auto programs  = t.programs.load(std::memory_order_relaxed);
	const auto pipelines = t.gfx_pipelines.load(std::memory_order_relaxed);
	const auto compute   = t.cs_pipelines.load(std::memory_order_relaxed);
	if (programs == 0 && pipelines == 0 && compute == 0) {
		return;
	}
	PipelineCacheLog("Compile totals: {} programs (translate {:.1f} ms, {} reused translations, "
	                 "copies {:.1f} ms, emit {:.1f} ms, validate {:.1f} ms, background validation "
	                 "{} in {:.1f} ms, module {:.1f} ms); {} graphics pipelines in {:.1f} ms (new "
	                 "{}, permutation {}, variant {}); {} compute pipelines in {:.1f} ms; {} stalls "
	                 "totalling {:.1f} ms, worst {:.1f} ms",
	                 programs, ms(t.translate_ns),
	                 t.translation_reuses.load(std::memory_order_relaxed), ms(t.clone_ns),
	                 ms(t.emit_ns), ms(t.validate_ns),
	                 t.validate_async.load(std::memory_order_relaxed), ms(t.validate_async_ns),
	                 ms(t.module_ns), pipelines, ms(t.gfx_pipeline_ns),
	                 t.gfx_new.load(std::memory_order_relaxed),
	                 t.gfx_permutation.load(std::memory_order_relaxed),
	                 t.gfx_variant.load(std::memory_order_relaxed), compute, ms(t.cs_pipeline_ns),
	                 t.stalls.load(std::memory_order_relaxed), ms(t.stall_ns), ms(t.stall_max_ns));
	const auto links      = t.gpl_links.load(std::memory_order_relaxed);
	const auto cache_hits = t.gpl_cache_hits.load(std::memory_order_relaxed);
	const auto monolithic = t.gpl_monolithic.load(std::memory_order_relaxed);
	if (links != 0 || cache_hits != 0 || monolithic != 0) {
		PipelineCacheLog("Pipeline libraries: {} linked ({:.1f} ms), {} driver-cache hits, {} "
		                 "monolithic, {} libraries ({:.1f} ms), {} optimized in background "
		                 "({:.1f} ms)",
		                 links, ms(t.gpl_link_ns), cache_hits, monolithic,
		                 t.gpl_libraries.load(std::memory_order_relaxed), ms(t.gpl_library_ns),
		                 t.gpl_optimized.load(std::memory_order_relaxed), ms(t.gpl_optimize_ns));
	}
	const auto disk_sources      = t.disk_source_hits.load(std::memory_order_relaxed);
	const auto disk_permutations = t.disk_permutation_hits.load(std::memory_order_relaxed);
	const auto disk_checks       = t.disk_verify_checks.load(std::memory_order_relaxed);
	if (disk_sources != 0 || disk_permutations != 0 || disk_checks != 0) {
		PipelineCacheLog("Program cache: {} sources and {} permutations reloaded instead of "
		                 "translated (keys, lookups and decoding {:.1f} ms); verify: {} checks, {} "
		                 "mismatches",
		                 disk_sources, disk_permutations, ms(t.disk_load_ns), disk_checks,
		                 t.disk_verify_mismatches.load(std::memory_order_relaxed));
	}
}

struct ShaderReadAttempt {
	std::array<GuestRange, 64> missing {};
	size_t count = 0;
	bool materialization_failed = false;
	bool overflow = false;

	void Missing(uint64_t address, uint64_t size) {
		const GuestRange range {address, size};
		if (!range.Valid()) return;
		for (size_t i = 0; i < count; ++i) {
			if (missing[i] == range) return;
		}
		if (count < missing.size()) missing[count++] = range;
		else overflow = true;
	}

	bool Synchronize() const {
		KYTY_PROFILER_DETAIL_BLOCK("SRT::ReadinessWait");
		Profiler::ScopedFrameWait frame_wait(Profiler::FrameWait::ShaderReadiness);
		if (overflow) EXIT("resource readiness exceeded 64 missing ranges\n");
		bool ready = false;
		for (size_t i = 0; i < count; ++i) {
			ready |= LibKernel::Memory::SynchronizeGpuBackingForRead(missing[i].address,
			                                                          missing[i].size);
		}
		if (!ready) {
			// Diagnostic before the caller's fatal check: which reads could not be made ready
			// (the render context logs its own reason for ranges it handles).
			std::fprintf(stderr,
			             "Shader resource readiness: no progress on %zu missing range(s):\n",
			             count);
			for (size_t i = 0; i < count; ++i) {
				std::fprintf(stderr, "  [%zu] address=0x%016" PRIx64 " size=0x%" PRIx64 "\n", i,
				             missing[i].address, missing[i].size);
			}
			std::fflush(stderr);
		}
		return ready;
	}
};

bool NativeDccEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DCC_GPU");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

bool ReadShaderGuestMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
	const bool read = !values.empty() &&
	    LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(), values.size_bytes());
	if (!read && userdata != nullptr) {
		static_cast<ShaderReadAttempt*>(userdata)->Missing(address, values.size_bytes());
	}
	return read;
}

bool SrtReadRunsEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_SRT_READ_RUNS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

bool TryReadShaderCleanBacking(void*, uint64_t address, std::span<uint32_t> values) {
	// A failed probe must not request synchronization or alter the shader retry list.
	const bool read = !values.empty() &&
	    LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(), values.size_bytes());
	Profiler::CountFrameEvent(read ? Profiler::FrameEvent::SrtProbeHits
	                              : Profiler::FrameEvent::SrtProbeMisses);
	if (read) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::SrtProbeBytes, values.size_bytes());
		if (values.size() > 1) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::SrtProbeBatchHits);
		}
	}
	return read;
}

bool ResourceReuseEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_RESOURCE_REUSE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

bool ResourceDependencyCacheEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_RESOURCE_DEPENDENCY_CACHE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

bool SharedResourceEvaluationEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_SRT_SHARED_CLEAN_VALUES");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

// A successful materialization certificate. Every hit rereads all exact observed
// bytes through the current clean-backing predicate. Its output bundle travels
// with the certificate and is invalidated before a refresh can modify it.
class PreparedResourceReads {
public:
	static constexpr size_t MaxBytes = 32u * 1024u;
	static constexpr size_t MaxRanges = 256u;

	void Begin(std::span<const uint32_t> user_data, uint64_t shader_base,
	           std::span<uint8_t, MaxBytes> scratch) {
		m_valid = false;
		m_recordable = true;
		m_ranges.clear();
		m_storage.clear();
		m_capture_scratch = scratch;
		m_user_data.assign(user_data.begin(), user_data.end());
		m_shader_base = shader_base;
	}

	void Invalidate() { m_valid = false; }
	bool Valid() const { return m_valid; }
	size_t CapacityBytes() const {
		return m_ranges.capacity() * sizeof(Range) + m_storage.capacity() +
		       m_user_data.capacity() * sizeof(uint32_t);
	}

	void Finish(bool success) {
		m_valid = success && m_recordable;
		m_capture_scratch = {};
		if (!m_valid) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ResourceReuseRejectedCaptures);
		}
	}

	static void Observe(void* userdata, uint64_t address, std::span<const uint32_t> values,
	                    bool success) {
		static_cast<PreparedResourceReads*>(userdata)->Record(address, values, success);
	}

	bool Matches(std::span<const uint32_t> user_data, uint64_t shader_base,
	             std::span<const uint32_t> dependencies, bool projected) const {
		if (!m_valid || shader_base != m_shader_base || user_data.size() != m_user_data.size()) return false;
		if (!projected) return std::ranges::equal(user_data, m_user_data);
		for (const auto index: dependencies) {
			if (index >= user_data.size() || user_data[index] != m_user_data[index]) return false;
		}
		return true;
	}

	bool Validate(std::span<uint8_t, MaxBytes> scratch) const {
		if (!m_valid) return false;
		Profiler::ScopedFrameWait wait(Profiler::FrameWait::ResourceReuseValidation);
		uint64_t validation_bytes = 0;
		const auto finish = [&](bool valid) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ResourceReuseValidationBytes,
			                          validation_bytes);
			return valid;
		};
		for (const auto& range: m_ranges) {
			validation_bytes += range.size;
			// This is a silent validation. Failure must not synchronize, alter protection,
			// or append a missing range; the normal refresh below owns those operations.
			if (!LibKernel::Memory::TryReadGpuCleanBacking(range.address, scratch.data(),
			                                               range.size) ||
			    std::memcmp(scratch.data(), m_storage.data() + range.offset, range.size) != 0) {
				return finish(false);
			}
		}
		return finish(true);
	}

private:
	struct Range {
		uint64_t address;
		size_t offset;
		size_t size;
		uint64_t End() const { return address + size; }
	};

	void Record(uint64_t address, std::span<const uint32_t> values, bool success) {
		if (!m_recordable) return;
		constexpr uint64_t gpu_limit = uint64_t {1} << 40u;
		const auto bytes = values.size_bytes();
		if (!success || bytes == 0 || bytes > MaxBytes || address == 0 ||
		    address >= gpu_limit || bytes >= gpu_limit - address) {
			m_recordable = false;
			return;
		}
		const auto end = address + bytes;
		const auto* observed = reinterpret_cast<const uint8_t*>(values.data());
		auto first = std::lower_bound(m_ranges.begin(), m_ranges.end(), address,
		                             [](const Range& range, uint64_t value) {
			                             return range.address < value;
		                             });
		if (first != m_ranges.begin() && std::prev(first)->End() >= address) --first;
		auto last = first;
		uint64_t begin_union = address;
		uint64_t end_union = end;
		size_t replaced_bytes = 0;
		while (last != m_ranges.end() && last->address <= end_union) {
			// Reject differing snapshots of any overlapping bytes. No final reread is
			// substituted for what a branch, pointer or descriptor actually observed.
			const auto begin_overlap = std::max(address, last->address);
			const auto end_overlap = std::min(end, last->End());
			if (begin_overlap < end_overlap &&
			    std::memcmp(observed + (begin_overlap - address),
			                m_storage.data() + last->offset + (begin_overlap - last->address),
			                static_cast<size_t>(end_overlap - begin_overlap)) != 0) {
				m_recordable = false;
				return;
			}
			begin_union = std::min(begin_union, last->address);
			end_union = std::max(end_union, last->End());
			replaced_bytes += last->size;
			++last;
		}
		const auto merged_bytes = static_cast<size_t>(end_union - begin_union);
		const auto replaced_ranges = static_cast<size_t>(last - first);
		const auto stored_bytes = m_storage.size();
		const auto new_bytes = stored_bytes - replaced_bytes + merged_bytes;
		if (new_bytes > MaxBytes ||
		    m_ranges.size() - replaced_ranges + 1u > MaxRanges) {
			m_recordable = false;
			return;
		}
		if (replaced_ranges == 1u && begin_union == first->address &&
		    end_union == first->End()) return;
		// Allocate at most one bounded byte arena per source, then retain it across
		// misses. Range metadata likewise keeps its capacity when Begin clears it.
		if (m_storage.capacity() < MaxBytes) m_storage.reserve(MaxBytes);
		if (replaced_ranges == 1u && begin_union == first->address && last == m_ranges.end()) {
			// The usual ascending SRT sequence extends the final range in place.
			m_storage.resize(new_bytes);
			std::memcpy(m_storage.data() + first->offset + (address - begin_union), observed, bytes);
			first->size = merged_bytes;
			return;
		}
		// Existing ranges are packed in address order in the arena. Reconstruct the
		// changed union in shared scratch, shift its packed suffix, and keep no holes.
		const auto offset = first != m_ranges.end() ? first->offset : stored_bytes;
		const auto old_suffix = offset + replaced_bytes;
		for (auto it = first; it != last; ++it) {
			std::memcpy(m_capture_scratch.data() + (it->address - begin_union),
			            m_storage.data() + it->offset, it->size);
		}
		std::memcpy(m_capture_scratch.data() + (address - begin_union), observed, bytes);
		m_storage.resize(new_bytes);
		std::memmove(m_storage.data() + offset + merged_bytes,
		             m_storage.data() + old_suffix, stored_bytes - old_suffix);
		std::memcpy(m_storage.data() + offset, m_capture_scratch.data(), merged_bytes);
		const auto position = m_ranges.erase(first, last);
		for (auto it = position; it != m_ranges.end(); ++it) {
			it->offset += merged_bytes - replaced_bytes;
		}
		m_ranges.insert(position, Range {begin_union, offset, merged_bytes});
	}

	std::vector<Range> m_ranges;
	std::vector<uint8_t> m_storage;
	std::vector<uint32_t> m_user_data;
	std::span<uint8_t> m_capture_scratch;
	uint64_t m_shader_base = 0;
	bool m_valid = false;
	bool m_recordable = false;
};

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code, const std::string& decoded_dump) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto base = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(base.parent_path());
	for (const auto& [suffix, data, size]: {
	         std::tuple {".bin", static_cast<const void*>(code.data()), code.size_bytes()},
	         std::tuple {".rdna2", static_cast<const void*>(decoded_dump.data()),
	                     decoded_dump.size()},
	     }) {
		if (size == 0) {
			continue;
		}
		auto path = base;
		path += suffix;
		Common::File file(path);
		if (file.IsInvalid()) {
			const auto path_text = Common::PathToString(path);
			LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		} else {
			file.Write(data, size);
		}
	}
}

void DumpMatchedShaderInputs(const ShaderParams& params,
                             const ShaderRecompiler::CompileOptions& options,
                             const char* stage_name, std::span<const uint32_t> static_state,
                             uint32_t push_data_start_dword,
                             const std::vector<uint32_t>& spirv, std::string_view ir_dump) {
	if (!Config::GraphicsDebugDumpEnabled() ||
	    !((options.stage == ShaderType::Pixel && options.shader_hash == 0x3b809f9d156a95ddull) ||
	      (options.stage == ShaderType::Vertex && options.shader_hash == 0xe5398a1c6007f356ull))) {
		return;
	}

	// This is a compile-permutation snapshot. User data may change on later
	// draws that reuse the compiled module; no draw-time work is added here.
	using Json = nlohmann::ordered_json;
	const auto spirv_bytes = spirv.size() * sizeof(uint32_t);
	const auto spirv_hash  = XXH3_64bits(spirv.data(), spirv_bytes);
	const auto key_hash    = XXH3_64bits(static_state.data(), static_state.size_bytes());
	static std::atomic_uint32_t id = 0;
	const auto base = Config::GetShaderLogFolder() /
	                  fmt::format("{:04d}_matched_inputs_{}_{:016x}_{:016x}", id++, stage_name,
	                              options.shader_hash, spirv_hash);
	auto spirv_path = base;
	auto json_path  = base;
	auto ir_path    = base;
	spirv_path += ".spv";
	json_path += ".json";
	ir_path += ".ir.txt";
	constexpr size_t MaxIrDumpBytes = 16 * 1024 * 1024;
	const auto ir_bytes = std::min(ir_dump.size(), MaxIrDumpBytes);
	Json metadata = {
	    {"schema_version", 1},
	    {"snapshot_kind", "compile_permutation"},
	    {"stage", stage_name},
	    {"shader_hash", fmt::format("0x{:016x}", options.shader_hash)},
	    {"guest_code_base", fmt::format("0x{:016x}", params.Base())},
	    {"guest_code_words", params.code.size()},
	    {"guest_code_xxh3_64", fmt::format("0x{:016x}", XXH3_64bits(params.code.data(), params.code.size_bytes()))},
	    {"spirv_file", Common::PathToString(spirv_path.filename())},
	    {"spirv_bytes", spirv_bytes},
	    {"spirv_xxh3_64", fmt::format("0x{:016x}", spirv_hash)},
	    {"ir_file", Common::PathToString(ir_path.filename())},
	    {"ir_bytes", ir_bytes}, {"ir_original_bytes", ir_dump.size()},
	    {"ir_truncated", ir_bytes != ir_dump.size()},
	    {"static_state_xxh3_64", fmt::format("0x{:016x}", key_hash)},
	    {"static_state_words", std::vector<uint32_t>(static_state.begin(), static_state.end())},
	    {"user_data_count", params.user_data_count},
	    {"user_data_base", options.user_data_base},
	    {"user_data_words", std::vector<uint32_t>(options.user_data.begin(), options.user_data.end())},
	    {"captured_user_data_storage", params.user_data},
	    {"compile_wave_size", options.wave_size},
	    {"push_data_start_dword", push_data_start_dword},
	};
	metadata["program_key"] = {
	    {"stage", static_cast<uint32_t>(options.stage)},
	    {"hash", metadata["shader_hash"]},
	    {"user_data_count", params.user_data_count},
	    {"code_size_words", params.code.size()},
	    {"static_state_words", metadata["static_state_words"]},
	};
	if (options.stage == ShaderType::Pixel) {
		const auto& ps = *options.input_info.pixel;
		const auto input_count = std::min<uint32_t>(ps.input_num, std::size(ps.interpolator_settings));
		auto& pixel = metadata["pixel"];
		pixel = {
		    {"input_num", ps.input_num},
		    {"interpolator_settings", std::vector<uint32_t>(ps.interpolator_settings,
		                                                   ps.interpolator_settings + input_count)},
		    {"custom_interpolation_mask", ps.custom_interpolation_mask},
		    {"wave_size", ps.wave_size},
		    {"scratch_size_dwords", ps.scratch_size_dwords},
		    {"ps_system_input_base", ps.ps_system_input_base},
		    {"ps_perspective_center_vgpr", ps.ps_perspective_center_vgpr},
		    {"ps_perspective_centroid_vgpr", ps.ps_perspective_centroid_vgpr},
		    {"ps_pos_x", ps.ps_pos_x}, {"ps_pos_y", ps.ps_pos_y},
		    {"ps_pos_z", ps.ps_pos_z}, {"ps_pos_w", ps.ps_pos_w},
		    {"ps_front_face", ps.ps_front_face}, {"ps_ancillary", ps.ps_ancillary},
		    {"ps_no_perspective", ps.ps_no_perspective},
		    {"ps_sample_shading", ps.ps_sample_shading},
		    {"ps_pixel_kill_enable", ps.ps_pixel_kill_enable},
		    {"ps_depth_export_enable", ps.ps_depth_export_enable},
		    {"ps_sample_mask_export_enable", ps.ps_sample_mask_export_enable},
		    {"dual_source_blending", ps.dual_source_blending},
		    {"ps_early_z", ps.ps_early_z}, {"ps_execute_on_noop", ps.ps_execute_on_noop},
		    {"target_output_mode", std::vector<uint32_t>(std::begin(ps.target_output_mode),
		                                               std::end(ps.target_output_mode))},
		};
		pixel["interpolator_settings_hex"] = Json::array();
		for (uint32_t i = 0; i < input_count; i++) {
			pixel["interpolator_settings_hex"].push_back(fmt::format("0x{:08x}", ps.interpolator_settings[i]));
		}
		pixel["target_export_mapping_packed"] = Json::array();
		for (const auto& mapping: ps.target_export_mapping) {
			pixel["target_export_mapping_packed"].push_back(static_cast<uint32_t>(mapping.packed));
		}

		// This exact guest PS uses s28:s29 for its original SRT. Read only
		// clean backing: this diagnostic must not force GPU synchronization.
		constexpr uint32_t SrtRegister = 28;
		constexpr size_t SrtDwords = 384;
		constexpr size_t MaterialDescriptorByteOffset = 1328;
		const bool has_srt_registers = options.user_data_base <= SrtRegister &&
		                              options.user_data.size() >=
		                                  SrtRegister - options.user_data_base + 2;
		auto& srt = pixel["original_srt"];
		srt = {{"snapshot_kind", "compile_permutation"},
		       {"base_sgpr", SrtRegister}, {"registers_available", has_srt_registers},
		       {"requested_dwords", SrtDwords}, {"read_success", false},
		       {"memory_reader", "TryReadGpuCleanBacking"}, {"words", Json::array()}};
		if (has_srt_registers) {
			const auto register_index = SrtRegister - options.user_data_base;
			const uint64_t raw_address = options.user_data[register_index] |
			                             (static_cast<uint64_t>(options.user_data[register_index + 1]) << 32u);
			const uint64_t address = raw_address & 0x0000ffffffffffffull;
			srt["raw_register_address"] = fmt::format("0x{:016x}", raw_address);
			srt["address"] = fmt::format("0x{:016x}", address);
			std::array<uint32_t, SrtDwords> words {};
			const bool success = address != 0 && ReadShaderGuestMemory(nullptr, address, words);
			srt["read_success"] = success;
			if (success) {
				srt["words"] = words;
				ShaderBufferResource descriptor {};
				std::copy_n(words.data() + MaterialDescriptorByteOffset / sizeof(uint32_t),
				            std::size(descriptor.fields), descriptor.fields);
				std::array<uint32_t, 64> material_words {};
				const auto material_dwords = static_cast<size_t>(
				    std::min<uint64_t>(descriptor.GetSize(), sizeof(material_words)) / sizeof(uint32_t));
				const bool material_valid = descriptor.Type() == 0 && descriptor.Base48() != 0 &&
				                            material_dwords != 0;
				const bool material_success = material_valid && ReadShaderGuestMemory(
				    nullptr, descriptor.Base48(), std::span(material_words).first(material_dwords));
				srt["material"] = {
				    {"descriptor_table_byte_offset", MaterialDescriptorByteOffset},
				    {"descriptor_words", std::vector<uint32_t>(std::begin(descriptor.fields),
				                                                std::end(descriptor.fields))},
				    {"address", fmt::format("0x{:016x}", descriptor.Base48())},
				    {"descriptor_size_bytes", descriptor.GetSize()},
				    {"requested_bytes", material_dwords * sizeof(uint32_t)},
				    {"descriptor_valid", material_valid}, {"read_success", material_success},
				    {"words", material_success
				                  ? Json(std::vector<uint32_t>(material_words.begin(),
				                                               material_words.begin() + material_dwords))
				                  : Json::array()},
				};
			}
		}
	} else {
		const auto& vs = *options.input_info.vertex;
		auto& vertex = metadata["vertex"];
		vertex = {
		    {"logical_stage", static_cast<uint32_t>(vs.logical_stage)},
		    {"wave_size", vs.wave_size}, {"scratch_size_dwords", vs.scratch_size_dwords},
		    {"pa_cl_vs_out_cntl", vs.pa_cl_vs_out_cntl},
		    {"fetch_attrib_reg", vs.fetch_attrib_reg}, {"fetch_buffer_reg", vs.fetch_buffer_reg},
		    {"fetch_external", vs.fetch_external}, {"fetch_embedded", vs.fetch_embedded},
		    {"resources_num", vs.resources_num}, {"buffers_num", vs.buffers_num},
		    {"resources", Json::array()}, {"buffers", Json::array()},
		};
		for (int i = 0; i < std::clamp(vs.resources_num, 0, ShaderVertexInputInfo::RES_MAX); i++) {
			const auto& resource = vs.resources[i];
			const auto& destination = vs.resources_dst[i];
			vertex["resources"].push_back({
			    {"index", i}, {"fields", std::vector<uint32_t>(std::begin(resource.fields), std::end(resource.fields))},
			    {"base", fmt::format("0x{:016x}", resource.Base48())},
			    {"stride", resource.Stride()}, {"num_records", resource.NumRecords()},
			    {"format", resource.RawFormat()}, {"dst_sel_xyzw", resource.DstSelXYZW()},
			    {"swizzle_enabled", resource.SwizzleEnabled()},
			    {"out_of_bounds", resource.OutOfBounds()}, {"add_tid", resource.AddTid()},
			    {"destination", {{"register_start", destination.register_start},
			                     {"registers_num", destination.registers_num},
			                     {"attr_id", destination.attr_id}, {"fetch_index", destination.fetch_index}}},
			});
		}
		for (int i = 0; i < std::clamp(vs.buffers_num, 0, ShaderVertexInputInfo::RES_MAX); i++) {
			const auto& buffer = vs.buffers[i];
			const auto attr_count = std::clamp(buffer.attr_num, 0, ShaderVertexInputBuffer::ATTR_MAX);
			vertex["buffers"].push_back({
			    {"index", i}, {"address", fmt::format("0x{:016x}", buffer.addr)},
			    {"stride", buffer.stride}, {"num_records", buffer.num_records},
			    {"fetch_index", buffer.fetch_index}, {"attr_num", buffer.attr_num},
			    {"attr_indices", std::vector<int>(buffer.attr_indices, buffer.attr_indices + attr_count)},
			    {"attr_offsets", std::vector<uint32_t>(buffer.attr_offsets, buffer.attr_offsets + attr_count)},
			});
		}
	}
	Common::File::CreateDirectories(base.parent_path());
	const auto write_dump = [](const auto& path, const void* data, size_t size) {
		Common::File file(path);
		if (file.IsInvalid()) {
			const auto path_text = Common::PathToString(path);
			LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
			return;
		}
		file.Write(data, static_cast<uint32_t>(size));
	};
	write_dump(spirv_path, spirv.data(), spirv_bytes);
	if (ir_bytes != 0) {
		write_dump(ir_path, ir_dump.data(), ir_bytes);
	}
	const auto text = metadata.dump(2);
	write_dump(json_path, text.data(), text.size());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

// Background spirv-val (KYTY_SHADER_VALIDATION_ASYNC, default on; only when
// shader_validation_enabled is set). spirv-val cost 11.7 ms per shader on average (median 5.2 ms,
// up to 235 ms for a 1.45 MB module) over 248 SPIR-V modules dumped from Astro Bot, and grows
// with module size, which the mip-statistics instrumentation raised. It ran on the compiling
// draw's thread before vkCreateShaderModule. Here the compile path creates the module and hands
// the words to this thread, which validates them in order and stops the emulator on the first
// invalid module exactly as before (same log, dump and EXIT message). Validation still covers
// every module; the difference is that the driver may receive an invalid module before the
// report. KYTY_SHADER_VALIDATION_ASYNC=0 validates synchronously again.
class SpirvValidator {
public:
	SpirvValidator() {
		m_thread = std::jthread([this](std::stop_token stop) { Run(stop); });
	}
	~SpirvValidator() {
		// Pending modules are abandoned at shutdown.
		m_thread.request_stop();
		m_wake.notify_all();
	}
	SpirvValidator(const SpirvValidator&)            = delete;
	SpirvValidator& operator=(const SpirvValidator&) = delete;

	static bool AsyncEnabled() {
		static const bool enabled = EnvU64("KYTY_SHADER_VALIDATION_ASYNC", 1) != 0;
		return enabled;
	}

	void Submit(const char* label, const char* stage_name, uint64_t shader_hash,
	            std::vector<uint32_t> spirv) {
		{
			std::scoped_lock lock(m_mutex);
			m_jobs.push_back({label, stage_name, shader_hash, std::move(spirv)});
		}
		m_wake.notify_one();
	}

private:
	struct Job {
		const char*           label      = nullptr; // string literals
		const char*           stage_name = nullptr;
		uint64_t              shader_hash = 0;
		std::vector<uint32_t> spirv;
	};

	void Run(const std::stop_token& stop) {
		Profiler::SetThreadName("SpirvValidator");
		for (;;) {
			Job job;
			{
				std::unique_lock lock(m_mutex);
				if (!m_wake.wait(lock, stop, [this] { return !m_jobs.empty(); })) {
					return;
				}
				job = std::move(m_jobs.front());
				m_jobs.pop_front();
			}
			const auto begin = CompileClockNs();
			const bool valid = ValidateShaderSpirv(job.label, job.shader_hash, job.spirv);
			const auto ns    = CompileClockNs() - begin;
			g_compile_totals.validate_async.fetch_add(1, std::memory_order_relaxed);
			g_compile_totals.validate_async_ns.fetch_add(ns, std::memory_order_relaxed);
			Profiler::AddFrameWait(Profiler::FrameWait::ShaderValidate, 1, ns);
			HangTrace::RecordShaderValidation(ns);
			if (!valid) {
				DumpShaderSpirv(job.stage_name, job.shader_hash, job.spirv);
				EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", job.label,
				     job.shader_hash);
			}
		}
	}

	std::mutex                  m_mutex;
	std::condition_variable_any m_wake;
	std::deque<Job>             m_jobs;
	std::jthread                m_thread; // Last: joined before the members it uses go away.
};

} // namespace

struct PipelineCache::Permutation {
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	ShaderRecompiler::IR::CompiledShaderInfo     program;
	ShaderProgram                                handle;
	// KYTY_LOD_STATS_PLAIN_VARIANT: the same pixel program without GET_LOD_STATS feedback, for
	// draws in which no image has a mip-statistics counter (then the feedback code records nothing,
	// but its atomics alone make the driver run the depth/stencil tests after a shader that can
	// discard). Empty for programs without the instrumentation.
	ShaderProgram                                plain;
	// Position in the owning source's PermutationList, fixed when it is published.
	uint32_t                                     index = 0;
};

// Program preparation is split into a shared lookup (FindSource), a pure per-draw
// materialization into caller-owned StagePrep (MaterializeStage), a lock-free permutation
// lookup (FindPermutation) and a compile path (CompileAndPublish) that takes m_programs_mutex
// exclusively only around its lookups, its in-flight registration and its publication, and
// translates and compiles unlocked. Get composes them serially. Invariants:
// - `programs` nodes are never erased or moved, so a SourceEntry* stays valid for the cache
//   lifetime. Finds take m_programs_mutex shared; inserts take it exclusively.
// - A source or permutation being compiled is listed in `in_flight`; other threads needing the
//   same one wait for it (compile_done) instead of compiling a second copy.
// - A SourceEntry's plan, key and dependency list are immutable once inserted.
// - Permutations are append-only with stable addresses (PermutationList).
// - Only the O15 reuse state (KYTY_RESOURCE_REUSE) mutates an entry after insertion; it is
//   guarded by m_reuse_mutex, which is taken before m_programs_mutex when both are needed.
struct PipelineCache::ProgramCache {
	using Permutation = PipelineCache::Permutation;
	using StagePrep   = PipelineCache::StagePrep;

	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	// Append-only permutation storage whose entries never move. The first InlineCapacity
	// entries are published without a lock: the writer (holding m_programs_mutex exclusively)
	// fills the next slot and then release-stores the count, so a reader that acquire-loads the
	// count may use every slot below it. Beyond that a source keeps growing in `overflow`,
	// which, like the previous vector, is only accessed under m_programs_mutex.
	class PermutationList {
	public:
		static constexpr uint32_t InlineCapacity = 16;

		[[nodiscard]] uint32_t PublishedInline() const {
			return m_published.load(std::memory_order_acquire);
		}
		[[nodiscard]] const Permutation& Inline(uint32_t index) const { return *m_inline[index]; }
		// Requires m_programs_mutex (shared or exclusive).
		[[nodiscard]] const std::vector<std::unique_ptr<Permutation>>& Overflow() const {
			return m_overflow;
		}
		// Requires m_programs_mutex (shared or exclusive).
		[[nodiscard]] size_t Size() const { return PublishedInline() + m_overflow.size(); }

		// Requires m_programs_mutex exclusively.
		const Permutation& Append(Permutation permutation) {
			const auto published = m_published.load(std::memory_order_relaxed);
			auto       entry     = std::make_unique<Permutation>(std::move(permutation));
			entry->index         = static_cast<uint32_t>(published + m_overflow.size());
			if (published < InlineCapacity) {
				m_inline[published] = std::move(entry);
				m_published.store(published + 1u, std::memory_order_release);
				return *m_inline[published];
			}
			m_overflow.push_back(std::move(entry));
			return *m_overflow.back();
		}

		// Requires m_programs_mutex (shared or exclusive).
		template <typename Visit>
		void ForEach(Visit&& visit) const {
			for (uint32_t i = 0; i < PublishedInline(); ++i) {
				visit(*m_inline[i]);
			}
			for (const auto& permutation: m_overflow) {
				visit(*permutation);
			}
		}

	private:
		std::array<std::unique_ptr<Permutation>, InlineCapacity> m_inline;
		std::atomic<uint32_t>                                    m_published {0};
		std::vector<std::unique_ptr<Permutation>>                m_overflow;
	};

	// O15 (KYTY_RESOURCE_REUSE): a certified output kept per source, plus a small history.
	struct PreparedState {
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		PreparedResourceReads                       prepared_reads;
		size_t permutation_index = std::numeric_limits<size_t>::max();

		size_t CapacityBytes() const {
			const auto bytes = [](const auto& values) {
				return values.capacity() * sizeof(typename std::decay_t<decltype(values)>::value_type);
			};
			return prepared_reads.CapacityBytes() + bytes(resources.buffers) + bytes(resources.images) +
			       bytes(resources.samplers) + bytes(resources.flattened_srt) + bytes(resources.user_data) +
			       bytes(specialization.buffers) + bytes(specialization.images);
		}
	};

	struct ReuseState {
		PreparedState                current;
		std::array<PreparedState, 3> history;
		size_t                       next_victim = 0;
	};

	// Translation reuse (KYTY_TRANSLATION_CACHE, default on). TranslateProgram is a function of
	// the guest code and the ProgramKey fields (the static key covers every input-info field it
	// reads, which is also what makes reusing a source's permutations across draws exact), so a
	// source's first translation, kept unmodified, can stand in for translating it again: each
	// further permutation specializes a deep copy (IR::CloneProgram) instead of re-running decode,
	// CFG structurization, IR translation and the IR passes. Kept translations are bounded by
	// KYTY_TRANSLATION_CACHE_MB (default 256) of estimated IR, least recently used evicted first.
	struct KeptTranslation {
		ShaderRecompiler::TranslateResult translated;
		size_t                            bytes = 0;
	};

	static bool TranslationCacheEnabled() {
		static const bool enabled = EnvU64("KYTY_TRANSLATION_CACHE", 1) != 0;
		return enabled;
	}

	// KYTY_TRANSLATION_CACHE_VERIFY: unset/0 off; 1 compiles every reused permutation also from a
	// fresh translation and compares SPIR-V and shader metadata, using the fresh one and dropping
	// the source's kept translation on a difference; "exit" stops the emulator on the first.
	static int TranslationVerifyMode() {
		static const int mode = [] {
			const auto* value = std::getenv("KYTY_TRANSLATION_CACHE_VERIFY");
			if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) return 0;
			return std::strcmp(value, "exit") == 0 ? 2 : 1;
		}();
		return mode;
	}

	static size_t TranslationCacheBudget() {
		static const size_t bytes = EnvU64("KYTY_TRANSLATION_CACHE_MB", 256) * 1024u * 1024u;
		return bytes;
	}

	// Deep copy of a translation; false when the program cannot be copied (see CloneProgram).
	static bool CopyTranslation(const ShaderRecompiler::TranslateResult& from,
	                            ShaderRecompiler::TranslateResult&       to) {
		if (!ShaderRecompiler::IR::CloneProgram(from.program, to.program)) return false;
		to.decoded_dump  = from.decoded_dump;
		to.cfg_dump      = from.cfg_dump;
		to.skip_dispatch = from.skip_dispatch;
		return true;
	}

	// Approximate heap footprint of a kept translation, for the budget.
	static size_t EstimateTranslationBytes(const ShaderRecompiler::TranslateResult& translated) {
		using namespace ShaderRecompiler::IR;
		const auto& program = translated.program;
		const auto  inst_bytes = [](const Inst& inst) {
			return sizeof(Inst) + 2 * sizeof(void*) + inst.NumArgs() * sizeof(Value) +
			       inst.UseCount() * sizeof(Use) + inst.NumPhiBlocks() * sizeof(Block*);
		};
		size_t bytes = sizeof(KeptTranslation) + translated.decoded_dump.size() +
		               translated.cfg_dump.size();
		for (const auto& block: program.block_storage) {
			bytes += sizeof(Block);
			for (const auto& inst: *block) bytes += inst_bytes(inst);
		}
		for (const auto& inst: program.value_storage) bytes += inst_bytes(inst);
		bytes += program.descriptor_sources.size() * sizeof(DescriptorSource) +
		         program.block_info.size() * sizeof(BlockInfo) +
		         program.memory_info.size() * sizeof(MemoryInfo) +
		         program.evaluation_recipes.size() * sizeof(ResourcePlan::EvaluationRecipe);
		return bytes;
	}

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {
			if (!ResourceDependencyCacheEnabled()) return;
			using namespace ShaderRecompiler::IR;
			// ExtractResourcePlan clones every descriptor, flat read, branch condition,
			// indirect selector and uniform-fill root into this owned graph. Include all
			// retained register reads, even those on currently inactive paths.
			projected_key = true;
			for (const auto& inst: resource_plan.value_storage) {
				if (inst.GetOpcode() != ValueOpcode::GetUserData) continue;
				if (inst.NumArgs() != 1 || !inst.Arg(0).IsImmediate() ||
				    inst.Arg(0).GetType() != Type::ScalarReg) {
					projected_key = false;
					break;
				}
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < resource_plan.user_data_base ||
				    reg - resource_plan.user_data_base >= resource_plan.user_data_count) {
					projected_key = false;
					break;
				}
				user_dependencies.push_back(reg - resource_plan.user_data_base);
			}
			std::ranges::sort(user_dependencies);
			user_dependencies.erase(std::unique(user_dependencies.begin(), user_dependencies.end()),
			                        user_dependencies.end());
		}

		// Sealed at extraction; evaluation never writes to it.
		ShaderRecompiler::IR::ResourcePlan resource_plan;
		std::vector<uint32_t>              user_dependencies;
		bool                               projected_key = false;
		PermutationList                    permutations;
		// Set under the exclusive programs lock; may be set on an already published entry.
		std::atomic<bool>                  skip_dispatch {false};
		// Mutated in place by reuse-mode refreshes; only touched under m_reuse_mutex.
		mutable ReuseState                 reuse;
		// The source's unmodified translation for further permutations (KYTY_TRANSLATION_CACHE),
		// and its place in kept_lru. Guarded by m_programs_mutex; readers copy the pointer.
		std::shared_ptr<const KeptTranslation> kept_translation;
		std::list<SourceEntry*>::iterator      kept_position;
		// Set when a reused translation failed verification: always translate this source.
		bool                                   kept_disabled = false;
		// Persistent program cache: the record this source's plan was reloaded from (UINT32_MAX
		// when translated here) and its stored encoding. Set at insertion, then immutable.
		uint32_t                               disk_record = UINT32_MAX;
		std::span<const uint8_t>               disk_plan;
	};

	// The last source found per stage by this thread and the permutation it last matched.
	// Exact: programs entries are never erased or moved, so a remembered entry is the one
	// FindSource returns for an equal key; and permutations are unique per (specialization,
	// push-data start) because CompileAndPublish appends only after an exclusive failed search,
	// so a remembered permutation that matches is the one FindPermutation would return.
	struct LookupMemo {
		ProgramKey         key;
		const SourceEntry* source      = nullptr;
		const Permutation* permutation = nullptr;
	};
	static constexpr size_t LookupMemoStages = 16;

	// Per-caller lookup scratch (previously shared cache members). One instance must not be
	// used by two threads at once.
	struct ProgramScratch {
		ProgramScratch() {
			key.static_state.reserve(MaxStaticKeyWords);
			second_key.static_state.reserve(MaxStaticKeyWords);
		}

		std::span<uint8_t, PreparedResourceReads::MaxBytes> Validation() {
			// Only reuse mode reads certificates; allocate its bounded buffer on first use.
			if (validation.size() != PreparedResourceReads::MaxBytes) {
				validation.resize(PreparedResourceReads::MaxBytes);
			}
			return std::span<uint8_t, PreparedResourceReads::MaxBytes>(validation.data(),
			                                                           validation.size());
		}

		LookupMemo* Memo(ShaderType stage) {
			const auto index = static_cast<size_t>(stage);
			return index < memos.size() ? &memos[index] : nullptr;
		}

		ProgramKey           key;
		// The parallel path keeps both stages' keys alive at once.
		ProgramKey           second_key;
		std::vector<uint8_t> validation;
		std::array<LookupMemo, LookupMemoStages> memos;
	};

	static bool LookupMemoEnabled() {
		static const bool enabled = [] {
			const auto* value = std::getenv("KYTY_PROGRAM_LOOKUP_MEMO");
			return value == nullptr || std::strcmp(value, "0") != 0;
		}();
		return enabled;
	}

	// FindSource through the per-thread memo of the key's stage.
	const SourceEntry* FindSourceMemo(const ProgramKey& key, ProgramScratch& scratch) const {
		auto* memo = LookupMemoEnabled() ? scratch.Memo(key.stage) : nullptr;
		if (memo != nullptr && memo->source != nullptr && memo->key == key) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ProgramSourceMemoHits);
			return memo->source;
		}
		const auto* source = FindSource(key);
		if (memo != nullptr) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ProgramSourceMemoMisses);
			if (source != nullptr) {
				memo->key.stage           = key.stage;
				memo->key.hash            = key.hash;
				memo->key.user_data_count = key.user_data_count;
				memo->key.code_size       = key.code_size;
				memo->key.static_state.assign(key.static_state.begin(), key.static_state.end());
				memo->source      = source;
				memo->permutation = nullptr;
			} else {
				memo->source = nullptr;
			}
		}
		return source;
	}

	// FindPermutation, trying the permutation last matched for this source first.
	const Permutation* FindPermutationMemo(const SourceEntry& source, ShaderType stage,
	                                       const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                       uint32_t push_data_cursor, ProgramScratch& scratch) const {
		auto* memo = LookupMemoEnabled() ? scratch.Memo(stage) : nullptr;
		if (memo != nullptr && memo->source == &source && memo->permutation != nullptr &&
		    PermutationMatches(*memo->permutation, specialization, push_data_cursor)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::PermutationMemoHits);
			return memo->permutation;
		}
		const auto* permutation = FindPermutation(source, specialization, push_data_cursor);
		if (memo != nullptr) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::PermutationMemoMisses);
			if (memo->source == &source) memo->permutation = permutation;
		}
		return permutation;
	}

	static ProgramScratch& ThreadScratch() {
		thread_local ProgramScratch scratch;
		return scratch;
	}

	static constexpr size_t MaxHistoryBytes = 64u * 1024u * 1024u;
	static constexpr size_t MaxHistoryEntryBytes = 256u * 1024u;

	// Requires m_reuse_mutex.
	void ExchangeHistory(ReuseState& state, size_t index) {
		auto& current = state.current;
		auto& previous = state.history[index];
		const auto old_bytes = previous.CapacityBytes();
		auto new_bytes = current.CapacityBytes();
		// The current working output is mandatory; only the three extra states count
		// against this cache budget. Release an oversized outgoing state before a hit.
		if (new_bytes > MaxHistoryEntryBytes || history_bytes - old_bytes + new_bytes > MaxHistoryBytes) {
			current = PreparedState {};
			new_bytes = 0;
			Profiler::CountFrameEvent(Profiler::FrameEvent::ResourceCacheBudgetRejects);
		}
		std::swap(current, previous);
		history_bytes = history_bytes - old_bytes + new_bytes;
	}

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing up to 429 words first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 13 + ShaderVertexInputInfo::RES_MAX * 13;

	static const char* ProgramStageName(ShaderType stage) {
		switch (stage) {
			case ShaderType::Vertex: return "vs";
			case ShaderType::Mesh: return "ms";
			case ShaderType::Local: return "ls";
			case ShaderType::TessellationControl: return "hs";
			case ShaderType::TessellationEvaluation: return "ds";
			case ShaderType::Pixel: return "ps";
			case ShaderType::Compute: return "cs";
			default: EXIT("invalid pipeline shader stage\n");
		}
		return "";
	}

	// One emitted permutation: what the persistent program cache stores and reloads.
	struct EmittedProgram {
		std::vector<uint32_t>                    spirv;
		std::vector<uint32_t>                    spirv_plain;
		ShaderRecompiler::IR::CompiledShaderInfo info;
	};

	// Specialization and SPIR-V emission (CompileProgram) of a translation, and the debug dumps.
	static EmittedProgram EmitPermutation(const ShaderParams&                     params,
	                                      const ShaderRecompiler::CompileOptions& options,
	                                      ShaderRecompiler::TranslateResult       translated,
	                                      const ShaderRecompiler::IR::ResourceSpecialization&
	                                          specialization,
	                                      uint32_t                  push_data_start_dword,
	                                      std::span<const uint32_t> static_state,
	                                      ProgramCompileTimes&      times) {
		const char* stage_name = ProgramStageName(options.stage);
		const auto  emit_begin = CompileClockNs();
		ShaderRecompiler::CompileResult result;
		{
			KYTY_PROFILER_BLOCK("Shader::Emit");
			result = ShaderRecompiler::CompileProgram(std::move(translated), options,
			                                          specialization, push_data_start_dword);
		}
		times.emit_ns     = CompileClockNs() - emit_begin;
		times.spirv_words = result.spirv.size();
		DumpMatchedShaderInputs(params, options, stage_name, static_state,
		                        push_data_start_dword, result.spirv, result.ir_dump);
		DumpShaderOriginal(stage_name, options.shader_hash, params.code, result.decoded_dump);
		return {.spirv       = std::move(result.spirv),
		        .spirv_plain = std::move(result.spirv_plain),
		        .info        = std::move(result.program).TakeCompiledInfo()};
	}

	// Validation (here, or queued to the background validator), shader modules and ids of an
	// emitted or reloaded permutation. With `words`, the SPIR-V is copied there first (for the
	// persistent program cache).
	Permutation FinishPermutation(const ShaderRecompiler::CompileOptions&      options,
	                              EmittedProgram                               emitted,
	                              ShaderRecompiler::IR::ResourceSpecialization specialization,
	                              ProgramCompileTimes& times, EmittedProgram* words = nullptr) {
		const char* stage_name = ProgramStageName(options.stage);
		if (words != nullptr) {
			words->spirv       = emitted.spirv;
			words->spirv_plain = emitted.spirv_plain;
		}
		if (validator == nullptr) {
			const auto validate_begin = CompileClockNs();
			bool       valid          = false;
			{
				KYTY_PROFILER_BLOCK("Shader::Validate");
				valid = ValidateShaderSpirv(options.dump_label, options.shader_hash, emitted.spirv);
			}
			if (Config::ShaderValidationEnabled()) {
				times.validate_ns = CompileClockNs() - validate_begin;
			}
			if (!valid) {
				DumpShaderSpirv(stage_name, options.shader_hash, emitted.spirv);
				EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n",
				     options.dump_label, options.shader_hash);
			}
		}
		DumpShaderSpirv(stage_name, options.shader_hash, emitted.spirv);

		const auto       module_begin = CompileClockNs();
		vk::ShaderModule module       = nullptr;
		{
			KYTY_PROFILER_BLOCK("Shader::CreateModule");
			module = CompileSPV(emitted.spirv, device);
		}
		times.module_ns = CompileClockNs() - module_begin;
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(emitted.spirv.size()), options.wave_size);
		}
		if (validator != nullptr) {
			validator->Submit(options.dump_label, stage_name, options.shader_hash,
			                  std::move(emitted.spirv));
		}
		const auto id = next_shader_id.fetch_add(1, std::memory_order_relaxed) + 1u;
		GpuOpProfiler::RegisterShader(id, stage_name, options.shader_hash);
		ShaderProgram plain {};
		if (!emitted.spirv_plain.empty()) {
			if (validator == nullptr) {
				if (!ValidateShaderSpirv(options.dump_label, options.shader_hash,
				                         emitted.spirv_plain)) {
					DumpShaderSpirv(stage_name, options.shader_hash, emitted.spirv_plain);
					EXIT("%s failed hash=0x%016" PRIx64 ": plain variant SPIR-V validation failed\n",
					     options.dump_label, options.shader_hash);
				}
			}
			const auto plain_module = CompileSPV(emitted.spirv_plain, device);
			EXIT_IF(plain_module == nullptr);
			if (validator != nullptr) {
				validator->Submit(options.dump_label, stage_name, options.shader_hash,
				                  std::move(emitted.spirv_plain));
			}
			const auto plain_id = next_shader_id.fetch_add(1, std::memory_order_relaxed) + 1u;
			GpuOpProfiler::RegisterShader(plain_id, stage_name, options.shader_hash);
			plain = {.id = plain_id, .module = plain_module};
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(emitted.info),
		    .handle         = {.id = id, .module = module},
		    .plain          = plain,
		};
	}

	template <typename InputInfo>
	static ShaderType StageOf(const InputInfo& input_info) {
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			return input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			return ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			return ShaderType::Compute;
		}
	}

	template <typename InputInfo>
	static void BuildKey(const ShaderParams& params, const InputInfo& input_info, ProgramKey& key) {
		key.stage           = StageOf(input_info);
		key.hash            = params.hash;
		key.user_data_count = params.user_data_count;
		key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, key.static_state);
	}

	static ShaderRecompiler::IR::SrtRuntime MakeRuntime(const ShaderParams& params,
	                                                    ShaderReadAttempt&  read_attempt) {
		return {
		    .user_data                  = std::span(params.user_data).first(params.user_data_count),
		    .shader_base                = params.Base(),
		    .userdata                   = NativeDccEnabled() ? &read_attempt : nullptr,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .try_read_clean_backing = SrtReadRunsEnabled() ? TryReadShaderCleanBacking : nullptr,
		    .share_clean_values = SharedResourceEvaluationEnabled(),
		};
	}

	// Shared lookup. The returned entry stays valid for the cache lifetime.
	const SourceEntry* FindSource(const ProgramKey& key) const {
		std::shared_lock lock(m_programs_mutex);
		const auto entry = programs.find(key);
		return entry != programs.end() ? &entry->second : nullptr;
	}

	// Pure: reads the sealed plan and guest memory through `runtime` and writes only `scratch`
	// and `prep`, so any number of threads may prepare one source with their own scratch/prep.
	// A failed refresh leaves `prep` partially written; it must not be used.
	static bool MaterializeStage(const SourceEntry& source,
	                             const ShaderRecompiler::IR::SrtRuntime& runtime,
	                             ShaderRecompiler::IR::EvaluationScratch& scratch, StagePrep& prep) {
		prep.permutation = nullptr;
		return ShaderRecompiler::IR::MaterializeResources(source.resource_plan, runtime, scratch,
		                                                  prep.resources, prep.specialization);
	}

	// O15 reuse mode: refresh or reuse the certified output in the entry, then copy it into
	// `prep`. Requires m_reuse_mutex.
	bool MaterializeReusing(const SourceEntry& source, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                        ShaderRecompiler::IR::EvaluationScratch& evaluation,
	                        ProgramScratch& scratch, StagePrep& prep) {
		auto&      state       = source.reuse;
		const auto user_data   = runtime.user_data;
		const bool multi_state = ResourceDependencyCacheEnabled();
		const auto validate = [&](PreparedResourceReads& reads) {
			if (!reads.Matches(user_data, runtime.shader_base, source.user_dependencies,
			                   multi_state && source.projected_key)) return false;
			Profiler::CountFrameEvent(Profiler::FrameEvent::ResourceCacheKeyMatches);
			if (reads.Validate(scratch.Validation())) return true;
			// Matching registers with stale/dirty backing must never become a hit.
			reads.Invalidate();
			Profiler::CountFrameEvent(Profiler::FrameEvent::ResourceCacheBackingRejects);
			return false;
		};
		const auto publish = [&] {
			prep.resources      = state.current.resources;
			prep.specialization = state.current.specialization;
			prep.permutation    = nullptr;
			return true;
		};
		const auto hit = [&] {
			// These registers also supply shader constants and vertex/instance offsets.
			// Only descriptor evaluation uses the projected key; passthrough stays live.
			state.current.resources.user_data.assign(user_data.begin(), user_data.end());
			Profiler::CountFrameEvent(Profiler::FrameEvent::ResourceReuseHits);
			return publish();
		};
		if (validate(state.current.prepared_reads)) return hit();
		if (multi_state) {
			for (size_t index = 0; index < state.history.size(); ++index) {
				if (!validate(state.history[index].prepared_reads)) continue;
				ExchangeHistory(state, index);
				Profiler::CountFrameEvent(Profiler::FrameEvent::ResourceCacheHistoryHits);
				return hit();
			}
			if (state.current.prepared_reads.Valid()) {
				auto victim = state.next_victim;
				for (size_t index = 0; index < state.history.size(); ++index) {
					if (!state.history[index].prepared_reads.Valid()) { victim = index; break; }
				}
				ExchangeHistory(state, victim);
				state.next_victim = (victim + 1u) % state.history.size();
			}
		}
		// The entry owns the final successful outputs. Clear their certificate
		// before any refresh can partially overwrite either output object.
		state.current.prepared_reads.Invalidate();
		state.current.permutation_index = std::numeric_limits<size_t>::max();
		Profiler::CountFrameEvent(Profiler::FrameEvent::ResourceReuseMisses);
		state.current.prepared_reads.Begin(user_data, runtime.shader_base, scratch.Validation());
		auto observed_runtime              = runtime;
		observed_runtime.observe_read      = PreparedResourceReads::Observe;
		observed_runtime.observer_userdata = &state.current.prepared_reads;
		if (ShaderRecompiler::IR::MaterializeResources(source.resource_plan, observed_runtime,
		                                               evaluation, state.current.resources,
		                                               state.current.specialization)) {
			state.current.prepared_reads.Finish(true);
			return publish();
		}
		state.current.prepared_reads.Finish(false);
		return false;
	}

	// Materializes one stage and records a readiness failure for the retry loop.
	bool Materialize(const SourceEntry& source, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                 ShaderRecompiler::IR::EvaluationScratch& evaluation, ProgramScratch& scratch,
	                 StagePrep& prep, ShaderReadAttempt& read_attempt) {
		read_attempt.count = 0;
		read_attempt.materialization_failed = false;
		read_attempt.overflow = false;
		if (ResourceReuseEnabled() ? MaterializeReusing(source, runtime, evaluation, scratch, prep)
		                           : MaterializeStage(source, runtime, evaluation, prep)) {
			return true;
		}
		// An unsuccessful optional uniform-fill/active-source probe is harmless if the
		// complete refresh succeeded. Only a failed refresh requests a retry.
		if (!NativeDccEnabled() || read_attempt.count == 0) {
			// A failure no read can fix (an unsupported descriptor format, a specialization the
			// recompiler rejects): the stage gets no program and its draws/dispatches are dropped
			// instead of ending the emulator. The reason was already logged by the materializer.
			static std::atomic_uint64_t unmaterializable {0};
			const auto count = unmaterializable.fetch_add(1, std::memory_order_relaxed) + 1u;
			if (count <= 4u || (count & 2047u) == 0u) {
				// Missing reads here (KYTY_DCC_GPU=0 has no readiness retry) name guest memory the
				// renderer did not hold ready, not an unsupported shader.
				std::printf("Warning: stage materialization failed without a retryable read "
				            "(#%" PRIu64 ", %zu missing read(s), first 0x%016" PRIx64 "); its "
				            "draws/dispatches are dropped\n",
				            count, read_attempt.count,
				            read_attempt.count != 0 ? read_attempt.missing[0].address : 0);
			}
			return false;
		}
		read_attempt.materialization_failed = true;
		return false;
	}

	static bool PermutationMatches(const Permutation& candidate,
	                               const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                               uint32_t push_data_cursor) {
		const auto& layout = candidate.program.bindings;
		return layout.push_data_start_dword ==
		           ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
		                                                    layout.ShaderDataDwords()) &&
		       candidate.specialization == specialization;
	}

	// Searches in publication order. Inline entries need no lock; overflow entries need
	// m_programs_mutex, which is taken here unless the caller already holds it.
	const Permutation* FindPermutation(const SourceEntry& source,
	                                   const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                   uint32_t push_data_cursor, bool programs_locked = false) const {
		const auto& list      = source.permutations;
		const auto  published = list.PublishedInline();
		for (uint32_t i = 0; i < published; ++i) {
			if (PermutationMatches(list.Inline(i), specialization, push_data_cursor)) {
				return &list.Inline(i);
			}
		}
		if (published < PermutationList::InlineCapacity) return nullptr;
		std::shared_lock lock(m_programs_mutex, std::defer_lock);
		if (!programs_locked) lock.lock();
		for (const auto& candidate: list.Overflow()) {
			if (PermutationMatches(*candidate, specialization, push_data_cursor)) {
				return candidate.get();
			}
		}
		return nullptr;
	}

	const Permutation* PermutationAt(const SourceEntry& source, size_t index) const {
		const auto& list      = source.permutations;
		const auto  published = list.PublishedInline();
		if (index < published) return &list.Inline(static_cast<uint32_t>(index));
		if (published < PermutationList::InlineCapacity) return nullptr;
		std::shared_lock lock(m_programs_mutex);
		const auto overflow = index - PermutationList::InlineCapacity;
		return overflow < list.Overflow().size() ? list.Overflow()[overflow].get() : nullptr;
	}

	template <typename InputInfo>
	static ShaderProgram Bind(const Permutation& permutation, InputInfo& input_info,
	                          StagePrep& prep, uint32_t& push_data_cursor) {
		prep.permutation = &permutation;
		input_info.stage = {.program = &permutation.program, .resources = &prep.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);
		return permutation.handle;
	}

	// A compile running outside m_programs_mutex. While `source` is null the thread is translating
	// a source that is not in `programs` yet (covers every request with an equal key); once it is
	// set and `specialization_known`, it covers only that permutation request, and before that
	// every permutation of the source. Records live on the compiling thread's stack and are
	// listed in `in_flight`, both guarded by m_programs_mutex.
	struct InFlightCompile {
		ProgramKey                                   key;
		const SourceEntry*                           source               = nullptr;
		bool                                         specialization_known = false;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint32_t                                     push_data_cursor = 0;

		[[nodiscard]] bool CoversSource(const ProgramKey& other) const {
			return source == nullptr && key == other;
		}
		[[nodiscard]] bool CoversPermutation(
		    const SourceEntry* other, const ShaderRecompiler::IR::ResourceSpecialization& spec,
		    uint32_t cursor) const {
			return source == other &&
			       (!specialization_known ||
			        (push_data_cursor == cursor && specialization == spec));
		}
	};

	// Requires m_programs_mutex exclusively (held by `lock`). Waits while `covered` holds for an
	// in-flight compile and returns whether it waited (the caller then repeats its lookup).
	template <typename Covered>
	bool WaitForInFlight(std::unique_lock<std::shared_mutex>& lock, Covered&& covered) {
		const auto busy = [&] {
			return std::any_of(in_flight.begin(), in_flight.end(),
			                   [&](const InFlightCompile* record) { return covered(*record); });
		};
		if (!busy()) {
			return false;
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::ProgramCompileWaits);
		compile_done.wait(lock, [&] { return !busy(); });
		return true;
	}

	// Translations dropped under m_programs_mutex. The caller frees them after releasing the lock:
	// freeing a large IR program takes long enough to delay shared-lock readers (draw-prep
	// workers' FindSource) if done under it.
	using DroppedTranslations = std::vector<std::shared_ptr<const KeptTranslation>>;

	// Requires m_programs_mutex exclusively. Keeps `kept` for `source` (unless it already has one
	// or verification disabled it) and evicts least recently used translations over the budget.
	void KeepTranslation(SourceEntry& source, std::shared_ptr<const KeptTranslation> kept,
	                     DroppedTranslations& dropped) {
		if (kept == nullptr) return;
		if (source.kept_translation != nullptr || source.kept_disabled) {
			dropped.push_back(std::move(kept));
			return;
		}
		kept_bytes += kept->bytes;
		source.kept_translation = std::move(kept);
		source.kept_position    = kept_lru.insert(kept_lru.end(), &source);
		while (kept_bytes > TranslationCacheBudget() && !kept_lru.empty()) {
			ForgetTranslation(*kept_lru.front(), dropped);
		}
	}

	// Requires m_programs_mutex exclusively. Threads already copying it keep their reference.
	void ForgetTranslation(SourceEntry& source, DroppedTranslations& dropped) {
		if (source.kept_translation == nullptr) return;
		kept_bytes -= source.kept_translation->bytes;
		dropped.push_back(std::move(source.kept_translation)); // Leaves it empty.
		kept_lru.erase(source.kept_position);
	}

	// Requires m_programs_mutex exclusively.
	void TouchTranslation(SourceEntry& source) {
		kept_lru.splice(kept_lru.end(), kept_lru, source.kept_position);
	}

	// Persistent program cache (KYTY_PROGRAM_CACHE, programDiskCache.h); null when off.
	ProgramDiskCache* disk = nullptr;

	// KYTY_PROGRAM_CACHE_VERIFY:
	// - unset/0: off.
	// - 1: every reloaded source and permutation is translated and emitted anyway and compared
	//   with the stored one (plans by their encoding, permutations by SPIR-V, plain variant and
	//   every metadata member); a difference is logged and counted, the stored record dropped and
	//   the fresh result used. As slow as no cache; exact even if a key were incomplete.
	// - exit: 1, and the emulator stops on the first difference.
	// - background: the reloaded program is used at once and a background thread translates,
	//   emits and compares it (BackgroundChecks); a difference is logged and counted and drops
	//   the record for later runs, while this run keeps the reloaded program. Costs the command
	//   processor nothing.
	enum class DiskVerify : int { Off, Sync, SyncExit, Background };
	static DiskVerify DiskVerifyMode() {
		static const DiskVerify mode = [] {
			const auto* value = std::getenv("KYTY_PROGRAM_CACHE_VERIFY");
			if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
				return DiskVerify::Off;
			}
			if (std::strcmp(value, "exit") == 0) return DiskVerify::SyncExit;
			if (std::strcmp(value, "background") == 0) return DiskVerify::Background;
			return DiskVerify::Sync;
		}();
		return mode;
	}
	static bool SyncVerify() {
		return DiskVerifyMode() == DiskVerify::Sync || DiskVerifyMode() == DiskVerify::SyncExit;
	}

	// Shader logging and the debug dumps describe a translation while it runs; they bypass the
	// persistent cache.
	[[nodiscard]] bool DiskEnabled(const ShaderRecompiler::CompileOptions& options) const {
		return disk != nullptr && !options.dump_ir && !Config::GraphicsDebugDumpEnabled();
	}

#if defined(_MSC_VER) && defined(_WIN64) && defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL == 0
	// CompileOptions (11 members): stage, wave_size, user_data_base, shader_hash (the guest hash)
	// and plain_mip_stats_variant are key fields; user_data is read for its size only (the
	// user-data count); back_code is keyed word for word; input_info through the stage static key;
	// dump_ir, early_dump and dump_label only select logs and dumps (which bypass the cache). A
	// new member must join BuildDiskKey (or be shown not to change a translation) before this
	// size is updated.
	static_assert(sizeof(ShaderRecompiler::CompileOptions) == 88,
	              "CompileOptions changed: add the new member to the program cache key");
#endif

	// The exact key of a source: every input of TranslateProgram outside the file identity (the
	// codegen state). The code words are the ones translation reads (it reads params.code): from
	// the clean backing when that holds them, like HashShaderCode (no fault), else through the
	// guest mapping as translation would. `code_words`/`back_code_words` receive them.
	static void BuildDiskKey(const ShaderParams& params, const ShaderRecompiler::CompileOptions& options,
	                         const ProgramKey& key, ProgramDiskCache::SourceKey& disk_key,
	                         std::vector<uint32_t>& code_words, std::vector<uint32_t>& back_code_words) {
		const auto copy = [](std::span<const uint32_t> guest, std::vector<uint32_t>& words) {
			words.resize(guest.size());
			if (guest.empty()) return;
			if (!LibKernel::Memory::TryReadGpuCleanBacking(reinterpret_cast<uint64_t>(guest.data()),
			                                               words.data(), guest.size_bytes())) {
				std::memcpy(words.data(), guest.data(), guest.size_bytes());
			}
		};
		copy(params.code, code_words);
		copy(params.back_code, back_code_words);
		ProgramDiskCache::BuildSourceKey(
		    {.stage                   = static_cast<uint32_t>(key.stage),
		     .hash                    = key.hash,
		     .user_data_count         = key.user_data_count,
		     .code_size               = key.code_size,
		     .static_state            = key.static_state,
		     .wave_size               = options.wave_size,
		     .user_data_base          = options.user_data_base,
		     .plain_mip_stats_variant = options.plain_mip_stats_variant,
		     .code                    = code_words,
		     .back_code               = back_code_words},
		    disk_key);
	}

	// KYTY_PROGRAM_CACHE_VERIFY=background: one reloaded permutation (and, when its source was
	// reloaded in the same compile, the source's plan) with everything needed to translate and
	// emit it again. The stored spans stay valid for the disk cache's lifetime.
	struct BackgroundCheck {
		ShaderType                                   stage           = ShaderType::Unknown;
		uint64_t                                     shader_hash     = 0;
		uint32_t                                     wave_size       = 64;
		uint32_t                                     user_data_base  = 0;
		uint32_t                                     user_data_count = 0;
		bool                                         plain_mip_stats_variant = false;
		std::vector<uint32_t>                        code;
		std::vector<uint32_t>                        back_code;
		std::unique_ptr<ShaderVertexInputInfo>       vertex;
		std::unique_ptr<ShaderPixelInputInfo>        pixel;
		std::unique_ptr<ShaderComputeInputInfo>      compute;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint32_t                                     push_data_cursor   = 0;
		uint32_t                                     source_record      = UINT32_MAX;
		std::span<const uint8_t>                     stored_plan;
		uint32_t                                     permutation_record = UINT32_MAX;
		std::span<const uint8_t>                     stored_info;
		std::span<const uint8_t>                     stored_spirv;
		std::span<const uint8_t>                     stored_spirv_plain;
	};

	// The background thread of KYTY_PROGRAM_CACHE_VERIFY=background (started on first use).
	// Stop() drops what is still queued; Wait() returns once the queue is empty and no check runs.
	class BackgroundChecks {
	public:
		explicit BackgroundChecks(ProgramCache& owner): m_owner(owner) {
			m_thread = std::jthread([this](std::stop_token stop) { Run(stop); });
		}
		~BackgroundChecks() { Stop(); }
		BackgroundChecks(const BackgroundChecks&)            = delete;
		BackgroundChecks& operator=(const BackgroundChecks&) = delete;

		void Submit(BackgroundCheck check) {
			{
				std::scoped_lock lock(m_mutex);
				if (m_stopped) return;
				m_jobs.push_back(std::move(check));
			}
			m_wake.notify_all();
		}
		void Wait() {
			std::unique_lock lock(m_mutex);
			m_idle.wait(lock, [this] { return (m_jobs.empty() && !m_busy) || m_stopped; });
		}
		void Stop() {
			{
				std::scoped_lock lock(m_mutex);
				m_stopped = true;
				m_jobs.clear();
			}
			m_wake.notify_all();
			m_idle.notify_all();
			if (m_thread.joinable()) {
				m_thread.request_stop();
				m_thread.join();
			}
		}

	private:
		void Run(const std::stop_token& stop) {
			Profiler::SetThreadName("ProgramCacheCheck");
			for (;;) {
				BackgroundCheck job;
				{
					std::unique_lock lock(m_mutex);
					m_busy = false;
					m_idle.notify_all();
					if (!m_wake.wait(lock, stop, [this] { return !m_jobs.empty(); })) {
						return;
					}
					job = std::move(m_jobs.front());
					m_jobs.pop_front();
					m_busy = true;
				}
				m_owner.RunBackgroundCheck(job);
			}
		}

		ProgramCache&               m_owner;
		std::mutex                  m_mutex;
		std::condition_variable_any m_wake;
		std::condition_variable_any m_idle;
		std::deque<BackgroundCheck> m_jobs;
		bool                        m_busy    = false;
		bool                        m_stopped = false;
		std::jthread                m_thread; // Last: joined before the members it uses go away.
	};

	// Guarded by m_checks_mutex; created on first use.
	std::unique_ptr<BackgroundChecks> checks;
	std::mutex                        m_checks_mutex;

	void SubmitBackgroundCheck(BackgroundCheck check) {
		std::scoped_lock lock(m_checks_mutex);
		if (checks == nullptr) {
			checks = std::make_unique<BackgroundChecks>(*this);
		}
		checks->Submit(std::move(check));
	}
	void WaitBackgroundChecks() {
		std::scoped_lock lock(m_checks_mutex);
		if (checks != nullptr) checks->Wait();
	}
	// Before the disk cache goes (the checks invalidate its records).
	void StopBackgroundChecks() {
		std::scoped_lock lock(m_checks_mutex);
		checks.reset();
	}

	// KYTY_ASYNC_TRANSLATE=1 (default off): sources that draw-prep finds missing while it prepares
	// a draw ahead (TryPrepareSpeculative) are translated on background threads
	// (KYTY_ASYNC_TRANSLATE_THREADS, default 3, below normal priority). A job translates (or
	// reloads from the persistent cache), extracts the resource plan, inserts the source and keeps
	// its translation, exactly as the serial compile would, under an in-flight record that covers
	// the source: the recording thread then waits for it (WaitForInFlight) instead of translating
	// a second copy, materializes the source itself (guest reads stay on that thread) and emits
	// the permutation from the kept translation. Nothing is skipped, so rendering is unchanged;
	// what moves off the recording thread is translation and plan extraction, in parallel.
	static bool AsyncTranslateEnabled() {
		static const bool enabled = EnvU64("KYTY_ASYNC_TRANSLATE", 0) != 0;
		return enabled;
	}

	struct TranslateJob {
		ProgramKey                               key;
		uint64_t                                 hash            = 0;
		uint32_t                                 wave_size       = 64;
		uint32_t                                 user_data_base  = 0;
		bool                                     plain_mip_stats_variant = false;
		std::vector<uint32_t>                    code;
		std::vector<uint32_t>                    back_code;
		std::vector<uint32_t>                    user_data; // zeros; translation reads its size only
		std::unique_ptr<ShaderVertexInputInfo>   vertex;
		std::unique_ptr<ShaderPixelInputInfo>    pixel;
		std::unique_ptr<InFlightCompile>         record; // listed in in_flight until the job ends
	};

	class TranslateWorkers {
	public:
		TranslateWorkers(ProgramCache& owner, uint32_t count): m_owner(owner) {
			for (uint32_t i = 0; i < count; i++) {
				m_threads.emplace_back([this] { Run(); });
			}
		}
		~TranslateWorkers() { Stop(); }
		TranslateWorkers(const TranslateWorkers&)            = delete;
		TranslateWorkers& operator=(const TranslateWorkers&) = delete;

		// False after Stop(): the caller unregisters the job's record.
		[[nodiscard]] bool Enqueue(std::unique_ptr<TranslateJob>& job) {
			{
				std::scoped_lock lock(m_mutex);
				if (m_stopping) return false;
				m_jobs.push_back(std::move(job));
			}
			m_wake.notify_one();
			return true;
		}
		// Drops what is queued (unregistering each record, so no waiter blocks on it) and joins
		// the threads once their running jobs end.
		void Stop() {
			std::deque<std::unique_ptr<TranslateJob>> dropped;
			{
				std::scoped_lock lock(m_mutex);
				m_stopping = true;
				dropped.swap(m_jobs);
			}
			m_wake.notify_all();
			for (auto& job: dropped) {
				std::unique_lock lock(m_owner.m_programs_mutex);
				m_owner.FinishInFlight(*job->record);
			}
			for (auto& thread: m_threads) {
				if (thread.joinable()) thread.join();
			}
			m_threads.clear();
		}
		std::atomic<uint64_t> done {0};
		std::atomic<uint64_t> translate_ns {0};

	private:
		void Run() {
			Profiler::SetThreadName("ShaderTranslator");
			LowerThreadPriorityBelowNormal();
			for (;;) {
				std::unique_ptr<TranslateJob> job;
				{
					std::unique_lock lock(m_mutex);
					m_wake.wait(lock, [this] { return !m_jobs.empty() || m_stopping; });
					if (m_jobs.empty()) return;
					job = std::move(m_jobs.front());
					m_jobs.pop_front();
				}
				const auto begin = CompileClockNs();
				m_owner.RunTranslateJob(*job);
				translate_ns.fetch_add(CompileClockNs() - begin, std::memory_order_relaxed);
				done.fetch_add(1, std::memory_order_relaxed);
			}
		}

		ProgramCache&                             m_owner;
		std::mutex                                m_mutex;
		std::condition_variable                   m_wake;
		std::deque<std::unique_ptr<TranslateJob>> m_jobs;
		bool                                      m_stopping = false;
		std::vector<std::thread>                  m_threads; // Last: joined before the rest goes.
	};

	// Draw-prep worker (inside a preparation: guest reads go through the recorder) or recording
	// thread. Queues the translation of `key` unless it is published or already in flight. The
	// code is copied from the clean backing only; a dirty or unmapped range is left to the serial
	// compile.
	template <typename InputInfo>
	void EnqueueTranslate(const ShaderParams& params, const InputInfo& input_info,
	                      const ProgramKey& key) {
		if (!AsyncTranslateEnabled() || ResourceReuseEnabled() || SyncVerify() ||
		    Config::GetShaderLogDirection() != Config::LogDirection::Silent ||
		    Config::GraphicsDebugDumpEnabled()) {
			return;
		}
		{
			std::shared_lock lock(m_programs_mutex);
			if (programs.find(key) != programs.end()) return;
		}
		auto       job  = std::make_unique<TranslateJob>();
		const auto copy = [](std::span<const uint32_t> guest, std::vector<uint32_t>& words) {
			words.resize(guest.size());
			return guest.empty() || LibKernel::Memory::TryReadGpuCleanBacking(
			                            reinterpret_cast<uint64_t>(guest.data()), words.data(),
			                            guest.size_bytes());
		};
		if (!copy(params.code, job->code) || !copy(params.back_code, job->back_code)) {
			return;
		}
		job->key  = key;
		job->hash = params.hash;
		job->user_data.assign(key.user_data_count, 0u);
		ShaderRecompiler::CompileOptions options;
		ConfigureStageOptions(input_info, key.stage, options);
		job->wave_size      = options.wave_size;
		job->user_data_base = options.user_data_base;
		job->plain_mip_stats_variant =
		    key.stage == ShaderType::Pixel &&
		    LodStatsCounter::PlainVariant() != LodStatsCounter::Plain::Off;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			job->vertex = std::make_unique<ShaderVertexInputInfo>(input_info);
		} else {
			job->pixel = std::make_unique<ShaderPixelInputInfo>(input_info);
		}
		job->record      = std::make_unique<InFlightCompile>();
		job->record->key = key;
		{
			std::unique_lock lock(m_programs_mutex);
			if (programs.find(key) != programs.end() ||
			    std::any_of(in_flight.begin(), in_flight.end(),
			                [&](const InFlightCompile* record) { return record->CoversSource(key); })) {
				return;
			}
			in_flight.push_back(job->record.get());
		}
		InFlightCompile* record = job->record.get();
		{
			std::scoped_lock workers_lock(m_translate_mutex);
			if (!m_translate_stopped && m_translate_workers == nullptr) {
				const auto count = static_cast<uint32_t>(
				    std::clamp<uint64_t>(EnvU64("KYTY_ASYNC_TRANSLATE_THREADS", 3), 1, 8));
				m_translate_workers = std::make_unique<TranslateWorkers>(*this, count);
			}
			if (m_translate_workers != nullptr && m_translate_workers->Enqueue(job)) {
				return;
			}
		}
		std::unique_lock lock(m_programs_mutex);
		FinishInFlight(*record);
	}

	// A background translation: the source half of CompileAndPublish (no materialization, no
	// permutation), under the job's in-flight record.
	void RunTranslateJob(TranslateJob& job) {
		ShaderRecompiler::CompileOptions options;
		options.stage       = job.key.stage;
		options.shader_hash = job.hash;
		options.user_data   = job.user_data;
		options.back_code   = job.back_code;
		options.dump_ir     = false;
		options.early_dump  = false;
		options.dump_label  = "ShaderRecompiler (background)";
		// A union: set only the member of the job's stage.
		ShaderStageInputInfo stage_input {};
		if (job.vertex != nullptr) {
			stage_input.vertex = job.vertex.get();
		} else {
			stage_input.pixel = job.pixel.get();
		}
		options.input_info = stage_input;
		options.wave_size   = job.wave_size;
		options.user_data_base          = job.user_data_base;
		options.plain_mip_stats_variant = job.plain_mip_stats_variant;
		ShaderParams params;
		params.code            = job.code;
		params.back_code       = job.back_code;
		params.hash            = job.hash;
		params.user_data_count = job.key.user_data_count;

		DroppedTranslations dropped;
		// Persistent program cache: a stored source is inserted exactly as the serial compile
		// inserts it (the recording thread materializes it).
		const bool                  disk_on = DiskEnabled(options);
		ProgramDiskCache::SourceKey disk_key;
		if (disk_on) {
			std::vector<uint32_t> code_words;
			std::vector<uint32_t> back_code_words;
			BuildDiskKey(params, options, job.key, disk_key, code_words, back_code_words);
			if (auto stored = disk->FindSource(disk_key); stored.has_value()) {
				ShaderRecompiler::IR::ResourcePlan plan;
				if (stored->skip_dispatch ||
				    ShaderRecompiler::IR::DecodeResourcePlan(stored->plan, plan)) {
					if (stored->skip_dispatch) NoteDiskSkip(options);
					std::unique_lock lock(m_programs_mutex);
					auto& entry = programs.try_emplace(job.key, std::move(plan)).first->second;
					if (stored->skip_dispatch) {
						entry.skip_dispatch.store(true, std::memory_order_relaxed);
					} else {
						entry.disk_record = stored->id;
						entry.disk_plan   = stored->plan;
					}
					FinishInFlight(*job.record);
					g_compile_totals.disk_source_hits.fetch_add(1, std::memory_order_relaxed);
					return;
				}
				disk->Invalidate(stored->id);
			}
		}
		ShaderRecompiler::TranslateResult translated;
		{
			KYTY_PROFILER_BLOCK("Shader::Translate(background)");
			translated = ShaderRecompiler::TranslateProgram(params.code, options);
		}
		g_compile_totals.translations.fetch_add(1, std::memory_order_relaxed);
		if (translated.skip_dispatch) {
			if (disk_on) disk->AddSource(disk_key, true, {});
			std::unique_lock lock(m_programs_mutex);
			programs.try_emplace(job.key, ShaderRecompiler::IR::ResourcePlan {})
			    .first->second.skip_dispatch.store(true, std::memory_order_relaxed);
			FinishInFlight(*job.record);
			return;
		}
		std::shared_ptr<KeptTranslation> keep;
		if (TranslationCacheEnabled()) {
			auto copy = std::make_shared<KeptTranslation>();
			if (CopyTranslation(translated, copy->translated)) {
				copy->bytes = EstimateTranslationBytes(copy->translated);
				keep        = std::move(copy);
			}
		}
		auto          plan   = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		SourceEntry*  source = nullptr;
		{
			std::unique_lock lock(m_programs_mutex);
			source = &programs.try_emplace(job.key, std::move(plan)).first->second;
			KeepTranslation(*source, std::move(keep), dropped);
			FinishInFlight(*job.record);
		}
		dropped.clear();
		if (disk_on) {
			std::vector<uint8_t> encoded;
			if (ShaderRecompiler::IR::EncodeResourcePlan(source->resource_plan, encoded)) {
				disk->AddSource(disk_key, false, encoded);
			}
		}
	}

	// Before the disk cache goes and before the cache is destroyed: queued translations are
	// dropped, running ones finish.
	void StopAsyncTranslations() {
		std::unique_ptr<TranslateWorkers> workers;
		{
			std::scoped_lock lock(m_translate_mutex);
			m_translate_stopped = true;
			workers.swap(m_translate_workers);
		}
		if (workers != nullptr) {
			workers->Stop();
			PipelineCacheLog("Background shader translation: {} sources, {:.1f} ms of translation",
			                 workers->done.load(), static_cast<double>(workers->translate_ns.load()) / 1.0e6);
		}
	}

	// Translates and emits a reloaded program again and compares it with the stored bytes.
	void RunBackgroundCheck(const BackgroundCheck& job) {
		ShaderRecompiler::CompileOptions options;
		options.stage                   = job.stage;
		options.shader_hash             = job.shader_hash;
		options.wave_size               = job.wave_size;
		options.user_data_base          = job.user_data_base;
		options.plain_mip_stats_variant = job.plain_mip_stats_variant;
		options.dump_ir                 = false;
		options.early_dump              = false;
		options.dump_label              = "ProgramCache check";
		// Translation reads the user data's size only (the count is part of the key).
		const std::vector<uint32_t> user_data(job.user_data_count);
		options.user_data = user_data;
		options.back_code = job.back_code;
		if (job.vertex != nullptr) {
			options.input_info.vertex = job.vertex.get();
		} else if (job.pixel != nullptr) {
			options.input_info.pixel = job.pixel.get();
		} else {
			options.input_info.compute = job.compute.get();
		}
		auto translated = ShaderRecompiler::TranslateProgram(job.code, options);
		if (job.source_record != UINT32_MAX) {
			bool same = !translated.skip_dispatch;
			if (same) {
				const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
				std::vector<uint8_t> encoded;
				same = ShaderRecompiler::IR::EncodeResourcePlan(plan, encoded) &&
				       std::ranges::equal(encoded, job.stored_plan);
			}
			NoteDiskVerify(same, "resource plan (background check)", options, false);
			if (!same) disk->Invalidate(job.source_record);
		}
		if (translated.skip_dispatch) {
			if (job.source_record == UINT32_MAX) {
				NoteDiskVerify(false, "skip-dispatch verdict (background check)", options, false);
			}
			disk->Invalidate(job.permutation_record);
			return;
		}
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               job.specialization, job.push_data_cursor);
		const auto info = std::move(result.program).TakeCompiledInfo();
		ShaderRecompiler::IR::CompiledShaderInfo stored_info;
		const auto words = [](const std::vector<uint32_t>& spirv) {
			return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(spirv.data()),
			                                spirv.size() * sizeof(uint32_t));
		};
		const char* what =
		    !std::ranges::equal(words(result.spirv), job.stored_spirv) ? "SPIR-V (background check)"
		    : !std::ranges::equal(words(result.spirv_plain), job.stored_spirv_plain)
		        ? "plain-variant SPIR-V (background check)"
		    : !ShaderRecompiler::IR::DecodeCompiledShaderInfo(job.stored_info, stored_info) ||
		            !(stored_info == info)
		        ? "shader metadata (background check)"
		        : nullptr;
		NoteDiskVerify(what == nullptr, what != nullptr ? what : "", options, false);
		if (what != nullptr) disk->Invalidate(job.permutation_record);
	}

	static bool CopyStoredWords(std::span<const uint8_t> bytes, std::vector<uint32_t>& words) {
		if (bytes.size() % sizeof(uint32_t) != 0) return false;
		words.resize(bytes.size() / sizeof(uint32_t));
		if (!bytes.empty()) std::memcpy(words.data(), bytes.data(), bytes.size());
		return true;
	}

	// One comparison of a stored record with a fresh translation (`what` names the first
	// difference). `fresh_used`: this run uses the fresh result instead of the stored one (false
	// for a plan already in use when a later permutation found it different).
	static void NoteDiskVerify(bool same, const char* what,
	                           const ShaderRecompiler::CompileOptions& options,
	                           bool fresh_used = true) {
		g_compile_totals.disk_verify_checks.fetch_add(1, std::memory_order_relaxed);
		if (same) return;
		g_compile_totals.disk_verify_mismatches.fetch_add(1, std::memory_order_relaxed);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ProgramDiskVerifyMismatches);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
			PipelineCacheLog("Program cache: the stored {} of {} 0x{:016x} differs from a fresh "
			                 "translation; the stored record is dropped{}",
			                 what, ProgramStageName(options.stage), options.shader_hash,
			                 fresh_used ? " and the fresh result used"
			                            : " (this run keeps the stored one for this source)");
		}
		if (DiskVerifyMode() == DiskVerify::SyncExit) {
			EXIT("Program cache: the stored %s of 0x%016" PRIx64
			     " differs from a fresh translation\n",
			     what, options.shader_hash);
		}
	}

	static bool DiskSourceMatches(const ShaderRecompiler::TranslateResult&  translated,
	                              const ProgramDiskCache::SourceRecord&     stored,
	                              const ShaderRecompiler::CompileOptions&   options) {
		bool        same = translated.skip_dispatch == stored.skip_dispatch;
		const char* what = "skip-dispatch verdict";
		if (same && !stored.skip_dispatch) {
			const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			std::vector<uint8_t> encoded;
			same = ShaderRecompiler::IR::EncodeResourcePlan(plan, encoded) &&
			       std::ranges::equal(encoded, stored.plan);
			what = "resource plan";
		}
		NoteDiskVerify(same, what, options);
		return same;
	}

	static bool DiskPermutationMatches(const EmittedProgram& fresh, const EmittedProgram& stored,
	                                   const ShaderRecompiler::CompileOptions& options) {
		const char* what = fresh.spirv != stored.spirv               ? "SPIR-V"
		                   : fresh.spirv_plain != stored.spirv_plain ? "plain-variant SPIR-V"
		                   : !(fresh.info == stored.info)            ? "shader metadata"
		                                                             : nullptr;
		NoteDiskVerify(what == nullptr, what != nullptr ? what : "", options);
		return what == nullptr;
	}

	// A skip-dispatch verdict reloaded from disk: the translation that found it would have logged
	// why (ray tracing, KYTY_SRT_VARIANT_READS); say once per program that it is skipped.
	static void NoteDiskSkip(const ShaderRecompiler::CompileOptions& options) {
		static std::mutex                   mutex;
		static std::unordered_set<uint64_t> logged;
		{
			std::scoped_lock lock(mutex);
			if (!logged.insert(options.shader_hash ^ static_cast<uint64_t>(options.stage)).second) {
				return;
			}
		}
		PipelineCacheLog("Program cache: {} 0x{:016x} is not dispatched or drawn (the verdict of "
		                 "its translation, reloaded; KYTY_PROGRAM_CACHE=0 logs the reason)",
		                 ProgramStageName(options.stage), options.shader_hash);
	}

	// KYTY_TRANSLATION_CACHE_VERIFY: compiles the permutation from a copy of `kept` and from a
	// fresh translation and compares the outputs that reach the GPU and the renderer.
	bool VerifyKeptTranslation(const ShaderParams& params,
	                           const ShaderRecompiler::CompileOptions& options,
	                           const KeptTranslation& kept,
	                           const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                           uint32_t push_data_cursor) {
		ShaderRecompiler::TranslateResult copy;
		if (!CopyTranslation(kept.translated, copy)) return false;
		auto reused = ShaderRecompiler::CompileProgram(std::move(copy), options, specialization,
		                                               push_data_cursor);
		auto fresh_translation = ShaderRecompiler::TranslateProgram(params.code, options);
		auto fresh = ShaderRecompiler::CompileProgram(std::move(fresh_translation), options,
		                                              specialization, push_data_cursor);
		const auto a    = std::move(reused.program).TakeCompiledInfo();
		const auto b    = std::move(fresh.program).TakeCompiledInfo();
		const bool same = reused.spirv == fresh.spirv && a.stage == b.stage &&
		                  a.shader_hash == b.shader_hash && a.wave_size == b.wave_size &&
		                  a.user_data_base == b.user_data_base &&
		                  a.user_data_count == b.user_data_count &&
		                  a.scratch_dwords == b.scratch_dwords &&
		                  a.param_export_mask == b.param_export_mask &&
		                  a.has_address_writes == b.has_address_writes && a.info == b.info &&
		                  a.bindings == b.bindings && a.write_ranges == b.write_ranges;
		if (same) {
			return true;
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::TranslationVerifyMismatches);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
			PipelineCacheLog("Translation cache: reused translation of {} 0x{:016x} differs from a "
			                 "fresh one (SPIR-V {} vs {} words); using the fresh translation",
			                 ProgramStageName(options.stage), options.shader_hash,
			                 reused.spirv.size(), fresh.spirv.size());
		}
		if (TranslationVerifyMode() == 2) {
			EXIT("Translation cache: reused translation differs from a fresh one (0x%016" PRIx64
			     ")\n",
			     options.shader_hash);
		}
		return false;
	}

	// The stage's wave size and user-data base (TranslateProgram options), shared by the serial
	// compile and the background translation (KYTY_ASYNC_TRANSLATE).
	template <typename InputInfo>
	static void ConfigureStageOptions(const InputInfo& input_info, ShaderType stage,
	                                  ShaderRecompiler::CompileOptions& options) {
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size      = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size      = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
	}

	// Requires m_programs_mutex exclusively. Idempotent.
	void FinishInFlight(InFlightCompile& record) {
		if (std::erase(in_flight, &record) != 0) {
			compile_done.notify_all();
		}
	}

	// Unregisters `record` on every exit from the unlocked compile (including unwinding), so a
	// waiter can never observe a record whose owner is gone.
	struct InFlightScope {
		InFlightScope(ProgramCache& owner, InFlightCompile& compile,
		              std::unique_lock<std::shared_mutex>& programs_lock)
		    : cache(owner), record(compile), lock(programs_lock) {}
		ProgramCache&                        cache;
		InFlightCompile&                     record;
		std::unique_lock<std::shared_mutex>& lock;
		~InFlightScope() {
			if (!lock.owns_lock()) {
				lock.lock();
			}
			cache.FinishInFlight(record);
		}
		InFlightScope(const InFlightScope&)            = delete;
		InFlightScope& operator=(const InFlightScope&) = delete;
	};

	// The compile path: translates (or reloads), inserts a new source, materializes it into
	// `prep` when `prep` was not prepared from an existing entry, compiles (or reloads) and
	// publishes the permutation. Returns null for skip-dispatch shaders and for a failed
	// materialization.
	//
	// Persistent program cache (KYTY_PROGRAM_CACHE, programDiskCache.h): a source that is not in
	// `programs` is looked up on disk by its exact key (every guest code word included) before it
	// is translated; a stored plan is decoded and inserted exactly where the translation's own
	// would be, and a stored skip-dispatch verdict is taken as the translation's. A permutation
	// that is not published is looked up by (source, push-data cursor, specialization bytes)
	// before a translation is emitted; stored SPIR-V and metadata go through the same validation,
	// module creation and publication as emitted ones. Whatever this run translates and emits is
	// added. KYTY_PROGRAM_CACHE_VERIFY translates and emits anyway and compares (DiskVerifyMode).
	// Without it, a reloaded source that needs a translation for a new permutation after all has
	// its stored plan compared with the fresh one at no extra translation cost.
	//
	// Locking (draw-prep S2 noted that compiles held m_programs_mutex exclusively throughout,
	// blocking every FindSource): the exclusive lock is now held only to look up, to register the
	// compile in `in_flight`, and to insert the new source and publish the permutation.
	// Translation, resource-plan extraction, materialization, SPIR-V emission, validation and
	// module creation run unlocked, so FindSource on other threads (draw-prep workers) proceeds
	// and independent compiles run in parallel. A second thread that needs the same source or
	// permutation waits for the first (compile_done) instead of compiling it again. Publication
	// re-checks FindPermutation, so permutations stay unique per (specialization, push start),
	// which LookupMemo relies on. No mutex was added: the lock order stays m_reuse_mutex (reuse
	// mode only, held by the caller for the whole Get) before m_programs_mutex; the wait releases
	// m_programs_mutex. In reuse mode m_reuse_mutex serializes all preparation, so no other
	// thread can own an in-flight record there and nothing ever waits.
	template <typename InputInfo>
	const Permutation* CompileAndPublish(const ShaderParams& params, InputInfo& input_info,
	                                     uint32_t push_data_cursor, const ProgramKey& key,
	                                     const ShaderRecompiler::IR::SrtRuntime& runtime,
	                                     ShaderRecompiler::IR::EvaluationScratch& evaluation,
	                                     ProgramScratch& scratch, StagePrep& prep,
	                                     ShaderReadAttempt& read_attempt, bool prep_materialized) {
		// Cache hits returned before this. This covers translation through native shader-module
		// creation; readiness failures can retry, so count successful creations separately.
		Profiler::ScopedFrameWait shader_miss(Profiler::FrameWait::ShaderProgramMiss);
		// Everything from here on, lock and in-flight waits included, is compile time of the
		// waiting draw.
		struct StallScope {
			uint64_t begin = CompileClockNs();
			~StallScope() { AddCompileStall(CompileClockNs() - begin); }
		} stall;
		const auto publish_index = [&](const SourceEntry& source, const Permutation& permutation) {
			if (ResourceReuseEnabled()) source.reuse.current.permutation_index = permutation.index;
			return &permutation;
		};
		// Declared before `lock`, so destroyed after it is released on every return.
		DroppedTranslations dropped;
		std::unique_lock    lock(m_programs_mutex);
		SourceEntry*        source = nullptr;
		for (;;) {
			// Another preparer may have inserted or compiled this source since the shared lookup.
			const auto entry = programs.find(key);
			if (entry == programs.end()) {
				if (WaitForInFlight(lock, [&](const InFlightCompile& record) {
					    return record.CoversSource(key);
				    })) {
					continue;
				}
				break;
			}
			source = &entry->second;
			if (source->skip_dispatch.load(std::memory_order_relaxed)) return nullptr;
			if (!prep_materialized) {
				// The plan is sealed and never mutated here (reuse-mode state is guarded by
				// m_reuse_mutex, held by the caller), so this needs no programs lock.
				lock.unlock();
				const bool materialized =
				    Materialize(*source, runtime, evaluation, scratch, prep, read_attempt);
				lock.lock();
				if (!materialized) return nullptr;
				prep_materialized = true;
			}
			if (const auto* published =
			        FindPermutation(*source, prep.specialization, push_data_cursor, true)) {
				return publish_index(*source, *published);
			}
			if (WaitForInFlight(lock, [&](const InFlightCompile& record) {
				    return record.CoversPermutation(source, prep.specialization, push_data_cursor);
			    })) {
				continue;
			}
			break;
		}
		InFlightCompile record;
		record.key.stage           = key.stage;
		record.key.hash            = key.hash;
		record.key.user_data_count = key.user_data_count;
		record.key.code_size       = key.code_size;
		record.key.static_state    = key.static_state;
		record.source              = source;
		record.push_data_cursor    = push_data_cursor;
		if (source != nullptr) {
			record.specialization       = prep.specialization;
			record.specialization_known = true;
		}
		in_flight.push_back(&record);
		const InFlightScope in_flight_scope(*this, record, lock);
		lock.unlock();

		const auto stage = key.stage;
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = runtime.user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;
		options.plain_mip_stats_variant =
		    stage == ShaderType::Pixel && LodStatsCounter::PlainVariant() != LodStatsCounter::Plain::Off;

		ConfigureStageOptions(input_info, stage, options);
		ProgramCompileTimes               times;
		ShaderRecompiler::TranslateResult translated;
		// `translated` holds a translation of this source that nothing has consumed yet.
		bool translated_now = false;
		// A copy of a fresh translation to keep for this source's later permutations
		// (KYTY_TRANSLATION_CACHE), taken before anything reads or changes the program.
		std::shared_ptr<KeptTranslation> keep;
		bool                             had_kept = false;
		const auto translate_now = [&] {
			const auto translate_begin = CompileClockNs();
			{
				KYTY_PROFILER_BLOCK("Shader::Translate");
				translated = ShaderRecompiler::TranslateProgram(params.code, options);
			}
			times.translate_ns += CompileClockNs() - translate_begin;
			translated_now = true;
			g_compile_totals.translations.fetch_add(1, std::memory_order_relaxed);
			if (TranslationCacheEnabled() && !had_kept && !translated.skip_dispatch &&
			    keep == nullptr) {
				const auto copy_begin = CompileClockNs();
				auto       copy       = std::make_shared<KeptTranslation>();
				if (CopyTranslation(translated, copy->translated)) {
					copy->bytes = EstimateTranslationBytes(copy->translated);
					keep        = std::move(copy);
				}
				times.clone_ns += CompileClockNs() - copy_begin;
			}
		};
		// A translation to emit from: the unconsumed fresh one, a copy of the source's kept one,
		// or a new one.
		const auto obtain_translation = [&] {
			if (translated_now) return;
			// An existing source's kept translation replaces translating the guest code again.
			std::shared_ptr<const KeptTranslation> kept;
			if (source != nullptr && TranslationCacheEnabled()) {
				lock.lock();
				kept = source->kept_translation;
				if (kept != nullptr) TouchTranslation(*source);
				lock.unlock();
			}
			if (kept != nullptr) {
				const auto clone_begin = CompileClockNs();
				{
					KYTY_PROFILER_BLOCK("Shader::CopyTranslation");
					times.reused = CopyTranslation(kept->translated, translated);
				}
				if (times.reused && TranslationVerifyMode() != 0 &&
				    !VerifyKeptTranslation(params, options, *kept, prep.specialization,
				                           push_data_cursor)) {
					times.reused = false;
					lock.lock();
					ForgetTranslation(*source, dropped);
					source->kept_disabled = true;
					lock.unlock();
				}
				times.clone_ns += CompileClockNs() - clone_begin;
			}
			// Released unlocked: it may be the last reference to a translation evicted meanwhile.
			had_kept = kept != nullptr;
			kept.reset();
			if (!times.reused) {
				translate_now();
			}
		};

		// Persistent program cache: the source (programDiskCache.h).
		const bool                  disk_on = DiskEnabled(options);
		ProgramDiskCache::SourceKey disk_key;
		std::vector<uint32_t>       code_words;
		std::vector<uint32_t>       back_code_words;
		std::optional<ProgramDiskCache::SourceRecord> disk_source;
		if (disk_on) {
			const auto load_begin = CompileClockNs();
			BuildDiskKey(params, options, key, disk_key, code_words, back_code_words);
			if (source == nullptr) {
				disk_source = disk->FindSource(disk_key);
			}
			times.load_ns += CompileClockNs() - load_begin;
		}
		bool from_disk    = false;
		bool plan_checked = false; // verify mode compared the stored plan with a fresh one
		if (disk_source.has_value()) {
			const auto load_begin = CompileClockNs();
			ShaderRecompiler::IR::ResourcePlan stored_plan;
			bool usable = disk_source->skip_dispatch ||
			              ShaderRecompiler::IR::DecodeResourcePlan(disk_source->plan, stored_plan);
			times.load_ns += CompileClockNs() - load_begin;
			if (usable && SyncVerify()) {
				translate_now();
				usable       = DiskSourceMatches(translated, *disk_source, options);
				plan_checked = true;
			}
			if (!usable) {
				disk->Invalidate(disk_source->id);
			} else if (disk_source->skip_dispatch) {
				NoteDiskSkip(options);
				lock.lock();
				const auto entry =
				    programs.try_emplace(key, ShaderRecompiler::IR::ResourcePlan {}).first;
				entry->second.skip_dispatch.store(true, std::memory_order_relaxed);
				FinishInFlight(record);
				g_compile_totals.disk_source_hits.fetch_add(1, std::memory_order_relaxed);
				return nullptr;
			} else {
				// The stored plan takes the place of the translation's own.
				lock.lock();
				// Nobody else inserts this key while `record` covers it.
				source = &programs.try_emplace(key, std::move(stored_plan)).first->second;
				source->disk_record = disk_source->id;
				source->disk_plan   = disk_source->plan;
				record.source       = source; // Still covers every permutation of it.
				KeepTranslation(*source, std::move(keep), dropped);
				lock.unlock();
				dropped.clear();
				g_compile_totals.disk_source_hits.fetch_add(1, std::memory_order_relaxed);
				const bool materialized =
				    Materialize(*source, runtime, evaluation, scratch, prep, read_attempt);
				lock.lock();
				if (!materialized) {
					FinishInFlight(record);
					return nullptr;
				}
				record.specialization       = prep.specialization;
				record.specialization_known = true;
				compile_done.notify_all(); // Other permutations of the source may proceed.
				lock.unlock();
			}
		}
		if (source == nullptr) {
			if (!translated_now) {
				translate_now();
			}
			if (translated.skip_dispatch) {
				if (disk_on) {
					disk->AddSource(disk_key, true, {});
				}
				lock.lock();
				// An existing source never reaches here: it would have skip_dispatch set already.
				const auto entry =
				    programs.try_emplace(key, ShaderRecompiler::IR::ResourcePlan {}).first;
				entry->second.skip_dispatch.store(true, std::memory_order_relaxed);
				FinishInFlight(record);
				return nullptr;
			}
			// Pure function of the translated program; no lock needed.
			auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			lock.lock();
			// Nobody else inserts this key while `record` covers it.
			source        = &programs.try_emplace(key, std::move(plan)).first->second;
			record.source = source; // Still covers every permutation of it.
			KeepTranslation(*source, std::move(keep), dropped);
			lock.unlock();
			dropped.clear();
			if (disk_on) {
				// The plan is sealed; other threads only read it.
				std::vector<uint8_t> encoded;
				if (ShaderRecompiler::IR::EncodeResourcePlan(source->resource_plan, encoded)) {
					disk->AddSource(disk_key, false, encoded);
				}
			}
			const bool materialized =
			    Materialize(*source, runtime, evaluation, scratch, prep, read_attempt);
			lock.lock();
			if (!materialized) {
				FinishInFlight(record);
				return nullptr;
			}
			// No other thread can have compiled a permutation of this source: they all waited on
			// `record`, which covered the whole source until now.
			record.specialization       = prep.specialization;
			record.specialization_known = true;
			compile_done.notify_all(); // Other permutations of the source may proceed.
			lock.unlock();
		}

		// Only for a source whose plan was reloaded from a stale record (the stored verdict was
		// not skip-dispatch, a fresh translation is): nothing can be emitted, so the source is
		// skipped from here on, as a fresh run would skip it.
		const auto skip_after_all = [&] {
			NoteDiskVerify(false, "skip-dispatch verdict", options, false);
			if (source->disk_record != UINT32_MAX) {
				disk->Invalidate(source->disk_record);
			}
			lock.lock();
			source->skip_dispatch.store(true, std::memory_order_relaxed);
			FinishInFlight(record);
			return nullptr;
		};

		// The permutation: stored, or emitted from a translation.
		std::optional<Permutation>                         compiled;
		std::optional<ProgramDiskCache::PermutationRecord> reloaded; // from_disk
		std::vector<uint8_t>       specialization_bytes;
		EmittedProgram             words; // SPIR-V of an emitted permutation, for the disk cache
		bool                       store = false;
		if (disk_on) {
			ShaderRecompiler::IR::EncodeSpecialization(prep.specialization, specialization_bytes);
			const auto load_begin = CompileClockNs();
			const auto stored =
			    disk->FindPermutation(disk_key.digest, push_data_cursor, specialization_bytes);
			EmittedProgram loaded;
			bool usable = stored.has_value() &&
			              ShaderRecompiler::IR::DecodeCompiledShaderInfo(stored->info, loaded.info) &&
			              CopyStoredWords(stored->spirv, loaded.spirv) &&
			              CopyStoredWords(stored->spirv_plain, loaded.spirv_plain);
			times.load_ns += CompileClockNs() - load_begin;
			if (stored.has_value() && !usable) {
				disk->Invalidate(stored->id);
			}
			if (usable && SyncVerify()) {
				obtain_translation();
				if (translated.skip_dispatch) {
					return skip_after_all();
				}
				auto fresh = EmitPermutation(params, options, std::move(translated),
				                             prep.specialization, push_data_cursor,
				                             key.static_state, times);
				translated_now = false;
				if (!DiskPermutationMatches(fresh, loaded, options)) {
					disk->Invalidate(stored->id);
					compiled = FinishPermutation(options, std::move(fresh), prep.specialization,
					                             times, &words);
					store    = true;
					usable   = false;
				}
			}
			if (usable) {
				from_disk         = true;
				reloaded          = stored;
				times.spirv_words = loaded.spirv.size();
				compiled = FinishPermutation(options, std::move(loaded), prep.specialization, times);
				g_compile_totals.disk_permutation_hits.fetch_add(1, std::memory_order_relaxed);
				Profiler::CountFrameEvent(Profiler::FrameEvent::ProgramDiskHits);
			}
		}
		if (!compiled.has_value()) {
			obtain_translation();
			if (disk_on && translated_now && source->disk_record != UINT32_MAX && !plan_checked) {
				// A source reloaded from disk needed a translation after all (a permutation that
				// was not stored): compare the stored plan with the fresh one at no extra cost (a
				// fresh skip-dispatch verdict is handled below).
				if (!translated.skip_dispatch) {
					const auto fresh_plan =
					    ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
					std::vector<uint8_t> encoded;
					const bool same = ShaderRecompiler::IR::EncodeResourcePlan(fresh_plan, encoded) &&
					                  std::ranges::equal(encoded, source->disk_plan);
					NoteDiskVerify(same, "resource plan (checked at a later permutation)", options,
					               false);
					if (!same) {
						disk->Invalidate(source->disk_record);
					}
				}
			}
			if (translated.skip_dispatch) {
				return skip_after_all();
			}
			auto emitted = EmitPermutation(params, options, std::move(translated),
			                               prep.specialization, push_data_cursor, key.static_state,
			                               times);
			translated_now = false;
			compiled = FinishPermutation(options, std::move(emitted), prep.specialization, times,
			                             disk_on ? &words : nullptr);
			store    = disk_on;
			if (disk_on) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::ProgramDiskMisses);
			}
		}
		lock.lock();
		// An existing source that had none kept. Anything dropped is freed after the lock.
		KeepTranslation(*source, std::move(keep), dropped);
		if (const auto* published =
		        FindPermutation(*source, prep.specialization, push_data_cursor, true)) {
			// Only possible when another cursor mapped to the same push-data start (both
			// NoStart): keep permutations unique and drop this equal copy.
			device.destroyShaderModule(compiled->handle.module, nullptr);
			if (compiled->plain.module != nullptr) {
				device.destroyShaderModule(compiled->plain.module, nullptr);
			}
			Profiler::CountFrameEvent(Profiler::FrameEvent::ProgramCompileDuplicates);
			FinishInFlight(record);
			return publish_index(*source, *published);
		}
		// Why this permutation was needed (compiles.csv): the source's first, only another
		// push-data start of an existing specialization, or a new specialization.
		bool same_specialization = false;
		source->permutations.ForEach([&](const Permutation& existing) {
			same_specialization |= existing.specialization == prep.specialization;
		});
		const char* reason = source->permutations.Size() == 0 ? "first"
		                     : same_specialization            ? "push"
		                                                      : "spec";
		const auto& permutation = source->permutations.Append(std::move(*compiled));
		FinishInFlight(record);
		lock.unlock();
		Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderProgramsCreated);
		if (times.reused) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TranslationReuses);
		}
		times.from_disk   = from_disk;
		const auto detail = fmt::format("{}{}{}", reason, times.reused ? "+reused" : "",
		                                from_disk ? "+disk" : "");
		RecordProgramCompile(ProgramStageName(stage), params.hash, permutation.handle.id, times,
		                     CompileClockNs() - stall.begin, detail);
		CountCompiledPermutation(stage);
		if (from_disk && DiskVerifyMode() == DiskVerify::Background) {
			BackgroundCheck check;
			check.stage                   = stage;
			check.shader_hash             = options.shader_hash;
			check.wave_size               = options.wave_size;
			check.user_data_base          = options.user_data_base;
			check.user_data_count         = static_cast<uint32_t>(options.user_data.size());
			check.plain_mip_stats_variant = options.plain_mip_stats_variant;
			check.code                    = std::move(code_words);
			check.back_code               = std::move(back_code_words);
			if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
				check.vertex = std::make_unique<ShaderVertexInputInfo>(input_info);
			} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
				check.pixel = std::make_unique<ShaderPixelInputInfo>(input_info);
			} else {
				check.compute = std::make_unique<ShaderComputeInputInfo>(input_info);
			}
			check.specialization   = permutation.specialization;
			check.push_data_cursor = push_data_cursor;
			if (disk_source.has_value() && source->disk_record == disk_source->id) {
				// The source was reloaded by this compile: check its plan too.
				check.source_record = disk_source->id;
				check.stored_plan   = disk_source->plan;
			}
			check.permutation_record = reloaded->id;
			check.stored_info        = reloaded->info;
			check.stored_spirv       = reloaded->spirv;
			check.stored_spirv_plain = reloaded->spirv_plain;
			SubmitBackgroundCheck(std::move(check));
		}
		if (store) {
			// The published permutation is immutable; its metadata is encoded from it.
			std::vector<uint8_t> info;
			ShaderRecompiler::IR::EncodeCompiledShaderInfo(permutation.program, info);
			disk->AddPermutation(disk_key.digest, push_data_cursor, specialization_bytes, info,
			                     words.spirv, words.spirv_plain);
		}
		return publish_index(*source, permutation);
	}

	// Per-stage totals of compiled permutations (equal to the sum of every source's
	// permutations, which the console line used to recount under the exclusive lock).
	static constexpr size_t StageCountSlots =
	    static_cast<size_t>(ShaderType::TessellationEvaluation) + 1;
	std::array<std::atomic<size_t>, StageCountSlots> compiled_per_stage {};

	// The "Shaders: VS n | PS n | ..." console line after every compile. std::printf to a Windows
	// console costs milliseconds per line and the counts were recomputed over every source while
	// the exclusive programs lock was held, so it is off unless KYTY_SHADER_COUNT_LOG=1 or shader
	// logging is enabled (shader_log_direction). The compile totals line at shutdown and the hang
	// trace (compiles.csv) report the same information.
	static bool ShaderCountLogEnabled() {
		static const bool enabled = [] {
			const auto* value = std::getenv("KYTY_SHADER_COUNT_LOG");
			if (value != nullptr && *value != '\0') {
				return std::strcmp(value, "0") != 0;
			}
			return Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		}();
		return enabled;
	}

	void CountCompiledPermutation(ShaderType stage) {
		const auto slot = static_cast<size_t>(stage);
		if (slot < compiled_per_stage.size()) {
			compiled_per_stage[slot].fetch_add(1, std::memory_order_relaxed);
		}
		if (!ShaderCountLogEnabled()) {
			return;
		}
		const auto count = [&](ShaderType type) {
			return compiled_per_stage[static_cast<size_t>(type)].load(std::memory_order_relaxed);
		};
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            count(ShaderType::Vertex), count(ShaderType::Pixel), count(ShaderType::Compute),
		            count(ShaderType::Mesh), count(ShaderType::Local),
		            count(ShaderType::TessellationControl),
		            count(ShaderType::TessellationEvaluation));
	}

	// Serial composition of the pieces above for one stage. On success `input_info.stage`
	// points at the permutation and into `prep`.
	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info, StagePrep& prep,
	                  uint32_t& push_data_cursor, ShaderReadAttempt& read_attempt,
	                  ProgramScratch& scratch, ShaderRecompiler::IR::EvaluationScratch& evaluation) {
		KYTY_PROFILER_DETAIL_BLOCK("ProgramCache::Get");
		std::unique_lock<std::mutex> reuse_lock;
		if (ResourceReuseEnabled()) reuse_lock = std::unique_lock(m_reuse_mutex);

		auto& key = scratch.key;
		BuildKey(params, input_info, key);
		const auto* source = FindSourceMemo(key, scratch);
		if (source != nullptr && source->skip_dispatch.load(std::memory_order_relaxed)) {
			return {};
		}
		const auto runtime = MakeRuntime(params, read_attempt);
		if (source != nullptr) {
			if (!Materialize(*source, runtime, evaluation, scratch, prep, read_attempt)) return {};
			if (ResourceReuseEnabled() && ResourceDependencyCacheEnabled() &&
			    source->reuse.current.prepared_reads.Valid()) {
				const auto* cached = PermutationAt(*source, source->reuse.current.permutation_index);
				if (cached != nullptr &&
				    cached->program.bindings.push_data_start_dword ==
				        ShaderRecompiler::IR::PushData::StartFor(
				            push_data_cursor, cached->program.bindings.ShaderDataDwords())) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::ResourceCachePermutationHits);
					return Bind(*cached, input_info, prep, push_data_cursor);
				}
			}
			if (const auto* permutation = FindPermutationMemo(*source, key.stage, prep.specialization,
			                                                  push_data_cursor, scratch)) {
				if (ResourceReuseEnabled()) {
					source->reuse.current.permutation_index = permutation->index;
				}
				return Bind(*permutation, input_info, prep, push_data_cursor);
			}
		}
		const auto* permutation =
		    CompileAndPublish(params, input_info, push_data_cursor, key, runtime, evaluation,
		                      scratch, prep, read_attempt, source != nullptr);
		if (permutation == nullptr) return {};
		return Bind(*permutation, input_info, prep, push_data_cursor);
	}

	// Draw-prep S3: probe-only guest reads. A read that is not provably clean fails silently:
	// no fault, readback, synchronization or missing-range record. It is the same predicate and
	// the same bytes the serial path's strict reader and clean probe use, so a materialization
	// that succeeds with it equals the serial one; any failure falls back to the serial path.
	static bool SpeculativeRead(void*, uint64_t address, std::span<uint32_t> values) {
		return !values.empty() &&
		       LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(), values.size_bytes());
	}

	static ShaderRecompiler::IR::SrtRuntime MakeSpeculativeRuntime(const ShaderParams& params) {
		// read_memory == read_specialization_memory keeps batched flat-run probes enabled, and
		// every read of both walkers goes through the silent reader. Clean-value sharing needs
		// the ordinary fallback reader and stays off; it only avoids repeated evaluations.
		return {
		    .user_data                  = std::span(params.user_data).first(params.user_data_count),
		    .shader_base                = params.Base(),
		    .read_memory                = SpeculativeRead,
		    .userdata                   = nullptr,
		    .read_specialization_memory = SpeculativeRead,
		    .try_read_clean_backing     = TryReadShaderCleanBacking,
		    .share_clean_values         = false,
		};
	}

	struct StageJob {
		const SourceEntry*               source = nullptr;
		ShaderRecompiler::IR::SrtRuntime runtime;
		StagePrep*                       prep = nullptr;
		bool                             ok   = false;

		static void Run(void* context) {
			auto& job = *static_cast<StageJob*>(context);
			job.ok    = MaterializeStage(*job.source, job.runtime,
			                             ShaderRecompiler::IR::ThreadEvaluationScratch(), *job.prep);
		}
	};

	// Draw-prep S3: materializes the pixel stage on the DrawPrep helper while this (GPU) thread
	// materializes the vertex stage, both with probe-only reads, then looks up and binds the
	// permutations serially in the usual order (pixel first; push data follows that order).
	// Until the join this thread performs only those read-only probes (gpuReadDelegate.h).
	// Returns false, leaving outputs to be overwritten by the serial path, when the helper is
	// unavailable, a source or permutation is not published yet (the serial path compiles it),
	// or a read was not provably clean (the serial path reads it with fault/readback semantics).
	bool TryGetParallel(const ShaderParams& ps_params, ShaderPixelInputInfo& ps_info,
	                    StagePrep& ps_prep, const ShaderParams& vs_params,
	                    ShaderVertexInputInfo& vs_info, StagePrep& vs_prep,
	                    uint32_t push_data_cursor, ProgramScratch& scratch,
	                    ShaderRecompiler::IR::EvaluationScratch& evaluation,
	                    ShaderProgram& ps_program, ShaderProgram& vs_program) {
		KYTY_PROFILER_DETAIL_BLOCK("ProgramCache::GetParallel");
		auto* worker = StagePrepWorker::Get();
		if (worker == nullptr) return false;
		auto& ps_key = scratch.key;
		auto& vs_key = scratch.second_key;
		BuildKey(ps_params, ps_info, ps_key);
		BuildKey(vs_params, vs_info, vs_key);
		const auto* ps_source = FindSourceMemo(ps_key, scratch);
		const auto* vs_source = FindSourceMemo(vs_key, scratch);
		if (ps_source == nullptr || vs_source == nullptr ||
		    ps_source->skip_dispatch.load(std::memory_order_relaxed) ||
		    vs_source->skip_dispatch.load(std::memory_order_relaxed)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::StagePrepLookupFallbacks);
			return false;
		}
		StageJob job {.source = ps_source, .runtime = MakeSpeculativeRuntime(ps_params),
		              .prep = &ps_prep};
		if (!worker->TryFork(&StageJob::Run, &job)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::StagePrepForkDeclined);
			return false;
		}
		const bool vs_ok =
		    MaterializeStage(*vs_source, MakeSpeculativeRuntime(vs_params), evaluation, vs_prep);
		worker->Join();
		if (!job.ok || !vs_ok) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::StagePrepSpeculativeFailures);
			return false;
		}
		const auto* ps_permutation = FindPermutationMemo(*ps_source, ps_key.stage,
		                                                 ps_prep.specialization, push_data_cursor,
		                                                 scratch);
		if (ps_permutation == nullptr) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::StagePrepLookupFallbacks);
			return false;
		}
		ps_program = Bind(*ps_permutation, ps_info, ps_prep, push_data_cursor);
		const auto* vs_permutation = FindPermutationMemo(*vs_source, vs_key.stage,
		                                                 vs_prep.specialization, push_data_cursor,
		                                                 scratch);
		if (vs_permutation == nullptr) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::StagePrepLookupFallbacks);
			return false;
		}
		vs_program = Bind(*vs_permutation, vs_info, vs_prep, push_data_cursor);
		Profiler::CountFrameEvent(Profiler::FrameEvent::StagePrepParallelDraws);
		return true;
	}

	// Draw-prep: both stages materialized on the calling thread with probe-only reads (routed to
	// the active DrawPrep recorder), then the permutations looked up and bound in the serial
	// order (pixel first; push data follows that order). Never compiles or publishes.
	PipelineCache::SpeculativeResult
	TryPrepareSpeculative(bool pixel_active, const ShaderParams& ps_params,
	                      ShaderPixelInputInfo& ps_info, StagePrep& ps_prep,
	                      const ShaderParams& vs_params, ShaderVertexInputInfo& vs_info,
	                      StagePrep& vs_prep, uint32_t push_data_cursor,
	                      PipelineCache::GraphicsPrograms& programs) {
		using Result  = PipelineCache::SpeculativeResult;
		auto& scratch = ThreadScratch();
		auto& evaluation = ShaderRecompiler::IR::ThreadEvaluationScratch();
		auto& ps_key  = scratch.key;
		auto& vs_key  = scratch.second_key;
		const SourceEntry* ps_source = nullptr;
		bool missing = false;
		if (pixel_active) {
			BuildKey(ps_params, ps_info, ps_key);
			ps_source = FindSourceMemo(ps_key, scratch);
			if (ps_source == nullptr) {
				// KYTY_ASYNC_TRANSLATE: start translating it now, ahead of the draw.
				EnqueueTranslate(ps_params, ps_info, ps_key);
				missing = true;
			} else if (ps_source->skip_dispatch.load(std::memory_order_relaxed)) {
				return Result::NotPublished;
			}
		}
		BuildKey(vs_params, vs_info, vs_key);
		const auto* vs_source = FindSourceMemo(vs_key, scratch);
		if (vs_source == nullptr) {
			EnqueueTranslate(vs_params, vs_info, vs_key);
			return Result::NotPublished;
		}
		if (missing || vs_source->skip_dispatch.load(std::memory_order_relaxed)) {
			return Result::NotPublished;
		}
		if (pixel_active &&
		    !MaterializeStage(*ps_source, MakeSpeculativeRuntime(ps_params), evaluation, ps_prep)) {
			return Result::ReadFailed;
		}
		if (!MaterializeStage(*vs_source, MakeSpeculativeRuntime(vs_params), evaluation, vs_prep)) {
			return Result::ReadFailed;
		}
		if (pixel_active) {
			const auto* permutation = FindPermutationMemo(*ps_source, ps_key.stage,
			                                              ps_prep.specialization, push_data_cursor,
			                                              scratch);
			if (permutation == nullptr) {
				return Result::NotPublished;
			}
			programs.pixel = Bind(*permutation, ps_info, ps_prep, push_data_cursor);
		}
		const auto* permutation = FindPermutationMemo(*vs_source, vs_key.stage,
		                                              vs_prep.specialization, push_data_cursor,
		                                              scratch);
		if (permutation == nullptr) {
			return Result::NotPublished;
		}
		programs.vertex[0] = Bind(*permutation, vs_info, vs_prep, push_data_cursor);
		return Result::Ok;
	}

	explicit ProgramCache(vk::Device device): device(device) {
		if (Config::ShaderValidationEnabled() && SpirvValidator::AsyncEnabled()) {
			validator = std::make_unique<SpirvValidator>();
		}
	}
	~ProgramCache() {
		// Translation jobs insert sources: none may run past this point.
		StopAsyncTranslations();
		for (const auto& [key, entry]: programs) {
			(void)key;
			entry.permutations.ForEach([&](const Permutation& permutation) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
				if (permutation.plain.module != nullptr) {
					device.destroyShaderModule(permutation.plain.module, nullptr);
				}
			});
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	mutable std::shared_mutex                                   m_programs_mutex;
	// Serializes O15 reuse-mode preparation; ordered before m_programs_mutex.
	std::mutex                                                  m_reuse_mutex;
	size_t history_bytes = 0; // Guarded by m_reuse_mutex.
	vk::Device                                                  device;
	// Assigned by compiles running outside m_programs_mutex.
	std::atomic<uint64_t> next_shader_id {0};
	// Compiles in progress outside the lock and their completion signal (m_programs_mutex).
	std::vector<InFlightCompile*> in_flight;
	std::condition_variable_any   compile_done;
	// Kept translations, least recently used first, and their estimated bytes (m_programs_mutex).
	std::list<SourceEntry*> kept_lru;
	size_t                  kept_bytes = 0;
	// Background spirv-val; null when validation is off or synchronous.
	std::unique_ptr<SpirvValidator> validator;
	// KYTY_ASYNC_TRANSLATE workers, started by the first job; guarded by m_translate_mutex
	// (never taken while holding m_programs_mutex).
	std::mutex                        m_translate_mutex;
	std::unique_ptr<TranslateWorkers> m_translate_workers;
	bool                              m_translate_stopped = false;
};

// Classifies each new graphics pipeline for stutter attribution: whether a pipeline already existed
// for the same program ids (and which key groups differ from the closest one), only for the same
// guest shaders compiled as other program permutations, or for neither. Guarded by m_mutex.
struct PipelineCache::PipelineDiagnostics {
	using Ids = std::array<uint64_t, 4>; // vertex program ids (or guest hashes), pixel last
	struct IdsHash {
		std::size_t operator()(const Ids& ids) const {
			std::size_t hash = 0;
			for (const auto id: ids) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(id));
			}
			return hash;
		}
	};

	enum Group : uint32_t {
		Rendering   = 1u << 0u,
		VertexInput = 1u << 1u,
		Topology    = 1u << 2u,
		Raster      = 1u << 3u,
		Cull        = 1u << 4u,
		DepthBounds = 1u << 5u,
		ColorMask   = 1u << 6u,
		Blend       = 1u << 7u,
		Multisample = 1u << 8u,
	};

	static uint32_t Difference(const GraphicsPipelineKey& a, const GraphicsPipelineKey& b) {
		const auto& x    = a.static_params;
		const auto& y    = b.static_params;
		uint32_t    mask = 0;
		const auto  differ = [](const auto& left, const auto& right) {
			return std::memcmp(&left, &right, sizeof(left)) != 0;
		};
		if (!(a.rendering == b.rendering)) mask |= Rendering;
		if (!(a.vertex_input == b.vertex_input)) mask |= VertexInput;
		if (x.topology != y.topology || x.primitive_restart_enable != y.primitive_restart_enable) {
			mask |= Topology;
		}
		if (x.negative_one_to_one != y.negative_one_to_one ||
		    x.depth_clip_enable != y.depth_clip_enable || x.polygon_mode != y.polygon_mode ||
		    x.provoking_vtx_last != y.provoking_vtx_last) {
			mask |= Raster;
		}
		if (x.cull_front != y.cull_front || x.cull_back != y.cull_back || x.face != y.face) {
			mask |= Cull;
		}
		const float x_bounds[] {x.depth_min_bounds, x.depth_max_bounds};
		const float y_bounds[] {y.depth_min_bounds, y.depth_max_bounds};
		if (x.depth_bounds_test_enable != y.depth_bounds_test_enable ||
		    differ(x_bounds, y_bounds)) {
			mask |= DepthBounds;
		}
		if (differ(x.color_mask, y.color_mask)) mask |= ColorMask;
		if (differ(x.color_srcblend, y.color_srcblend) ||
		    differ(x.color_comb_fcn, y.color_comb_fcn) ||
		    differ(x.color_destblend, y.color_destblend) ||
		    differ(x.alpha_srcblend, y.alpha_srcblend) ||
		    differ(x.alpha_comb_fcn, y.alpha_comb_fcn) ||
		    differ(x.alpha_destblend, y.alpha_destblend) ||
		    differ(x.separate_alpha_blend, y.separate_alpha_blend) ||
		    differ(x.blend_enable, y.blend_enable) ||
		    x.blend_alpha_source_remap != y.blend_alpha_source_remap) {
			mask |= Blend;
		}
		if (x.samples != y.samples || x.sample_shading_enable != y.sample_shading_enable) {
			mask |= Multisample;
		}
		return mask;
	}

	static std::string Describe(uint32_t mask) {
		static constexpr std::array<const char*, 9> names {
		    "rt", "vi", "topo", "raster", "cull", "dbounds", "mask", "blend", "ms"};
		std::string text;
		for (uint32_t bit = 0; bit < names.size(); ++bit) {
			if ((mask & (1u << bit)) != 0) {
				if (!text.empty()) text += '+';
				text += names[bit];
			}
		}
		return text.empty() ? std::string("same") : text;
	}

	// Classifies `key` (not yet inserted) and records it; `key` must be the map's stored copy.
	HangTrace::PipelineOrigin Classify(const GraphicsPipelineKey& key, const Ids& guest,
	                                   std::string& detail) {
		const Ids programs {key.vertex_shader_ids[0], key.vertex_shader_ids[1],
		                    key.vertex_shader_ids[2], key.ps_shader_id};
		auto&     siblings = by_programs[programs];
		auto      origin   = HangTrace::PipelineOrigin::New;
		if (!siblings.empty()) {
			uint32_t best = UINT32_MAX;
			for (const auto* sibling: siblings) {
				const auto mask = Difference(*sibling, key);
				if (std::popcount(mask) < std::popcount(best)) best = mask;
			}
			origin = HangTrace::PipelineOrigin::Variant;
			detail = Describe(best);
		} else if (by_guest[guest] != 0) {
			origin = HangTrace::PipelineOrigin::Permutation;
		}
		siblings.push_back(&key);
		by_guest[guest]++;
		return origin;
	}

	std::unordered_map<Ids, std::vector<const GraphicsPipelineKey*>, IdsHash> by_programs;
	std::unordered_map<Ids, uint32_t, IdsHash>                                  by_guest;
};

// Graphics pipeline libraries (item 6/7, KYTY_PIPELINE_LIBRARY): the library cache and the
// background threads that compile the optimized monolithic pipeline for every fast-linked one.
struct PipelineCache::LibraryState {
	explicit LibraryState(PipelineCache& owner)
	    : cache(owner), library(owner.m_graphics),
	      optimize(EnvU64("KYTY_PIPELINE_LIBRARY_OPTIMIZE", 1) != 0) {
		const auto count = std::clamp<uint64_t>(EnvU64("KYTY_PIPELINE_LIBRARY_THREADS", 2), 1, 8);
		if (optimize) {
			for (uint64_t i = 0; i < count; ++i) {
				threads.emplace_back([this](std::stop_token stop) { Run(stop); });
			}
		}
	}
	~LibraryState() { Stop(); }
	LibraryState(const LibraryState&)            = delete;
	LibraryState& operator=(const LibraryState&) = delete;

	struct Job {
		const GraphicsPipelineKey*                key    = nullptr; // map node: never erased
		vk::Pipeline                              linked = nullptr;
		std::unique_ptr<GraphicsPipelineSnapshot> snapshot;
	};

	void Enqueue(Job job) {
		{
			std::scoped_lock lock(mutex);
			if (stopped) return; // after Save(): the linked pipeline stays
			jobs.push_back(std::move(job));
		}
		wake.notify_one();
	}

	// Joins the threads; queued compiles are dropped (their linked pipelines stay in use). Must
	// run before the driver cache they use is destroyed.
	void Stop() {
		{
			std::scoped_lock lock(mutex);
			stopped = true;
			jobs.clear();
		}
		for (auto& thread: threads) {
			thread.request_stop();
		}
		wake.notify_all();
		for (auto& thread: threads) {
			if (thread.joinable()) thread.join();
		}
		threads.clear();
	}

	void Run(const std::stop_token& stop) {
		Profiler::SetThreadName("PipelineOptimizer");
		for (;;) {
			Job job;
			{
				std::unique_lock lock(mutex);
				if (!wake.wait(lock, stop, [this] { return !jobs.empty(); })) {
					return;
				}
				job = std::move(jobs.front());
				jobs.pop_front();
			}
			// Exactly the create info the renderer builds without libraries (a verified deep copy),
			// so the result is the pipeline it would have created.
			const auto   begin     = CompileClockNs();
			vk::Pipeline optimized = nullptr;
			const auto   result    = cache.m_graphics.device.createGraphicsPipelines(
			    cache.m_driver_cache, 1, &job.snapshot->Info(), nullptr, &optimized);
			const auto ns = CompileClockNs() - begin;
			Profiler::AddFrameWait(Profiler::FrameWait::PipelineOptimize, 1, ns);
			HangTrace::RecordPipelineLibraryEvent(HangTrace::PipelineLibraryEvent::Optimized, ns);
			if (result != vk::Result::eSuccess || optimized == nullptr) {
				// The linked pipeline stays; it renders the same state.
				continue;
			}
			g_compile_totals.gpl_optimized.fetch_add(1, std::memory_order_relaxed);
			g_compile_totals.gpl_optimize_ns.fetch_add(ns, std::memory_order_relaxed);
			cache.ReplaceLinkedPipeline(job.key, job.linked, optimized);
		}
	}

	PipelineCache&          cache;
	GraphicsPipelineLibrary library; // guarded by PipelineCache::m_mutex
	const bool              optimize;
	// Linked pipelines replaced by their optimized builds; other threads may still hold a
	// reference, and recorded command buffers may use them, so they live until the cache goes
	// (guarded by PipelineCache::m_mutex).
	std::vector<std::unique_ptr<Pipeline>> retired;
	std::mutex                             mutex;
	std::condition_variable_any            wake;
	std::deque<Job>                        jobs;
	bool                                   stopped = false;
	std::vector<std::jthread>              threads; // Last: joined before the rest is destroyed.
};

void PipelineCache::ReplaceLinkedPipeline(const GraphicsPipelineKey* key, vk::Pipeline linked,
                                          vk::Pipeline optimized) {
	Common::LockGuard lock(m_mutex);
	auto              entry = m_graphics_pipelines.find(*key);
	if (entry == m_graphics_pipelines.end() || entry->second->pipeline != linked) {
		m_graphics.device.destroyPipeline(optimized, nullptr);
		return;
	}
	// A new object, not a changed handle: threads that already hold the old Pipeline& keep a
	// consistent (still valid) pipeline; the generation makes their memos look it up again.
	auto replacement      = std::make_unique<Pipeline>(*entry->second);
	replacement->pipeline = optimized;
	m_library->retired.push_back(std::move(entry->second));
	entry->second = std::move(replacement);
	GpuOpProfiler::RegisterGraphicsPipeline(optimized, key->vertex_shader_ids.data(),
	                                        static_cast<uint32_t>(key->vertex_shader_ids.size()),
	                                        key->ps_shader_id);
	m_pipeline_generation.fetch_add(1, std::memory_order_release);
}

bool PipelineCache::AsyncPipelinesEnabled() {
	static const bool enabled = EnvU64("KYTY_ASYNC_PIPELINES", 0) != 0;
	return enabled;
}

bool PipelineCache::AsyncPipelinesWait() {
	static const bool wait = EnvU64("KYTY_ASYNC_PIPELINES", 0) == 2;
	return wait;
}

// KYTY_ASYNC_PIPELINES (TryGetGraphicsPipeline): background compiles of new monolithic graphics
// pipelines; the draws that need them are skipped until they are published. A compile is exactly
// the create info the draw path builds (a verified deep copy, GraphicsPipelineSnapshot), so the
// published pipeline is the one the draw path would have created.
// KYTY_ASYNC_PIPELINE_THREADS (default 3, 1-8) compile threads, below normal priority so that a
// burst of compiles does not take CPU time from the game's threads.
struct PipelineCache::AsyncState {
	explicit AsyncState(PipelineCache& owner): cache(owner) {
		const auto count = std::clamp<uint64_t>(EnvU64("KYTY_ASYNC_PIPELINE_THREADS", 3), 1, 8);
		for (uint64_t i = 0; i < count; ++i) {
			threads.emplace_back([this] { Run(); });
		}
	}
	~AsyncState() { Stop(); }
	AsyncState(const AsyncState&)            = delete;
	AsyncState& operator=(const AsyncState&) = delete;

	struct Job {
		const GraphicsPipelineKey*                key    = nullptr; // map node: never erased
		Pipeline*                                 target = nullptr; // its pending entry
		std::unique_ptr<GraphicsPipelineSnapshot> snapshot;
		// compiles.csv row, recorded when the compile finishes (pipeline_us = setup + background).
		HangTrace::PipelineOrigin origin     = HangTrace::PipelineOrigin::None;
		uint64_t                  guest_hash = 0;
		uint64_t                  vs_id      = 0;
		uint64_t                  ps_id      = 0;
		uint64_t                  setup_ns   = 0;
		std::string               detail;
	};

	// False after Stop() (Save() ran): the caller creates the pipeline itself.
	[[nodiscard]] bool Enqueue(Job& job) {
		{
			std::scoped_lock lock(mutex);
			if (stopping) {
				return false;
			}
			jobs.push_back(std::move(job));
		}
		queued.fetch_add(1, std::memory_order_relaxed);
		wake.notify_one();
		return true;
	}

	// Compiles every queued job, then joins the threads: no entry is left pending. Must run before
	// the driver cache the compiles use is destroyed.
	void Stop() {
		{
			std::scoped_lock lock(mutex);
			stopping = true;
		}
		wake.notify_all();
		for (auto& thread: threads) {
			if (thread.joinable()) thread.join();
		}
		threads.clear();
	}

	void Run() {
		Profiler::SetThreadName("PipelineCompiler");
		LowerCurrentThreadPriority();
		for (;;) {
			Job job;
			{
				std::unique_lock lock(mutex);
				wake.wait(lock, [this] { return !jobs.empty() || stopping; });
				if (jobs.empty()) {
					return;
				}
				job = std::move(jobs.front());
				jobs.pop_front();
			}
			const auto   begin    = CompileClockNs();
			vk::Pipeline pipeline = nullptr;
			const auto   result   = cache.m_graphics.device.createGraphicsPipelines(
			    cache.m_driver_cache, 1, &job.snapshot->Info(), nullptr, &pipeline);
			const auto ns = CompileClockNs() - begin;
			if (result != vk::Result::eSuccess || pipeline == nullptr) {
				if (result == vk::Result::eErrorDeviceLost) {
					ReportDeviceFault(cache.m_graphics);
				}
				EXIT("vkCreateGraphicsPipelines failed (background compile): %s\n",
				     vk::to_string(result).c_str());
			}
			compiled.fetch_add(1, std::memory_order_relaxed);
			compile_ns.fetch_add(ns, std::memory_order_relaxed);
			cache.PublishAsyncPipeline(job.key, job.target, pipeline, ns);
			if (HangTrace::Enabled()) {
				// The real cost, off the recording thread: compile_stall_us does not include it.
				HangTrace::RecordCompile({.kind        = HangTrace::CompileKind::GraphicsPipeline,
				                          .origin      = job.origin,
				                          .stage       = "gfx",
				                          .guest_hash  = job.guest_hash,
				                          .id          = job.vs_id,
				                          .id2         = job.ps_id,
				                          .pipeline_ns = job.setup_ns + ns,
				                          .total_ns    = job.setup_ns + ns,
				                          .detail      = job.detail});
			}
		}
	}

	static void LowerCurrentThreadPriority();

	PipelineCache&           cache;
	std::mutex               mutex;
	std::condition_variable  wake;
	std::deque<Job>          jobs;
	bool                     stopping = false;
	std::atomic<uint64_t>    deferred_draws {0}; // draws skipped while their pipeline compiled
	std::atomic<uint64_t>    waited_draws {0};   // draws whose new pipeline was ready within the wait
	// Signalled after each publication, for a draw waiting for the pipeline it queued.
	std::mutex               publish_mutex;
	std::condition_variable  published;
	std::atomic<uint64_t>    queued {0};
	std::atomic<uint64_t>    compiled {0};
	std::atomic<uint64_t>    compile_ns {0};
	std::vector<std::thread> threads; // Last: joined before the rest is destroyed.
};

void PipelineCache::AsyncState::LowerCurrentThreadPriority() {
	LowerThreadPriorityBelowNormal();
}

void PipelineCache::PublishAsyncPipeline(const GraphicsPipelineKey* key, Pipeline* target,
                                         vk::Pipeline pipeline, uint64_t create_ns) {
	{
		Common::LockGuard lock(m_mutex);
		EXIT_IF(!target->pending);
		target->pipeline = pipeline;
		target->pending  = false;
		GpuOpProfiler::RegisterGraphicsPipeline(
		    pipeline, key->vertex_shader_ids.data(),
		    static_cast<uint32_t>(key->vertex_shader_ids.size()), key->ps_shader_id);
	}
	{
		// Not nested in m_mutex: WaitForQueuedPipeline takes m_mutex inside publish_mutex.
		std::scoped_lock lock(m_async->publish_mutex);
	}
	m_async->published.notify_all();
	g_compile_totals.gfx_pipeline_ns.fetch_add(create_ns, std::memory_order_relaxed);
	NotePipelineCreated(create_ns);
}

// KYTY_ASYNC_PIPELINE_WAIT_MS (default 20, 0: skip at once; Senaxx/KytyPS5 926add0b): how long the
// draw that queued a new pipeline waits for it before it is skipped. A pipeline the driver cache
// already holds compiles within that, so its draw is not skipped; draws that find the pipeline
// still pending afterwards are skipped without waiting.
bool PipelineCache::WaitForQueuedPipeline(const Pipeline& target) {
	static const uint64_t wait_ms = std::min<uint64_t>(EnvU64("KYTY_ASYNC_PIPELINE_WAIT_MS", 20), 1000);
	if (wait_ms == 0) {
		return false;
	}
	const auto begin    = CompileClockNs();
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
	bool       ready    = false;
	// Called with m_mutex held once by this thread: the publication needs it.
	m_mutex.Unlock();
	{
		std::unique_lock lock(m_async->publish_mutex);
		ready = m_async->published.wait_until(lock, deadline, [&] {
			Common::LockGuard pipelines(m_mutex);
			return !target.pending;
		});
	}
	m_mutex.Lock();
	AddCompileStall(CompileClockNs() - begin);
	if (ready) {
		m_async->waited_draws.fetch_add(1, std::memory_order_relaxed);
	}
	return ready;
}

namespace {
// Numbers PipelineCache instances for their generation ranges (see the constructor).
std::atomic<uint64_t> g_pipeline_cache_instances {0};
} // namespace

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device)),
      m_diagnostics(std::make_unique<PipelineDiagnostics>()) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	// Each cache counts its generations in its own range, so a per-thread lookup memo left by a
	// destroyed cache can never match a later cache allocated at the same address (tests create
	// several render contexts in one process).
	m_pipeline_generation.store(
	    (g_pipeline_cache_instances.fetch_add(1, std::memory_order_relaxed) + 1) << 32u,
	    std::memory_order_relaxed);
	InitializeDriverCache();
	if (m_driver_cache != nullptr && DriverCacheSaveSettings::Get().enabled) {
		m_saver = std::make_unique<DriverCacheSaver>(*this);
	}
	InitializeProgramDiskCache();
	if (GraphicsPipelineLibrary::Enabled(m_graphics)) {
		m_library = std::make_unique<LibraryState>(*this);
		PipelineCacheLog("Graphics pipeline libraries: enabled (fast link, {})",
		                 m_library->optimize ? "optimized in background" : "linked pipelines kept");
		if (ShaderRecompiler::GetCodegenOptions().mad_mode == ShaderRecompiler::MadMode::Fused) {
			// See pipelineLibrary.h: identical positions rely on the Invariant position.
			PipelineCacheLog("Graphics pipeline libraries: KYTY_MAD_MODE=fused drops the Invariant "
			                 "position; a linked pipeline and its optimized build may rasterize "
			                 "different depths");
		}
	} else if (PipelineLibraryRequested()) {
		PipelineCacheLog("Graphics pipeline libraries: requested but not supported by the device");
	}
	if (AsyncPipelinesEnabled()) {
		if (m_library != nullptr) {
			PipelineCacheLog("Asynchronous pipelines: off (pipeline libraries are on)");
		} else {
			m_async = std::make_unique<AsyncState>(*this);
			PipelineCacheLog("Asynchronous pipelines: on ({} compile threads)",
			                 m_async->threads.size());
		}
	}
}

PipelineCache::~PipelineCache() {
	m_program_cache->StopAsyncTranslations();
	m_program_cache->StopBackgroundChecks();
	LogCompileTotals();
	if (m_program_disk != nullptr) {
		const auto stats = m_program_disk->GetStats();
		PipelineCacheLog("Program cache: lookups hit {} sources and {} permutations, missed {} and "
		                 "{}; added {} sources and {} permutations; {} stored records dropped "
		                 "(verification or damage)",
		                 stats.source_hits, stats.permutation_hits, stats.source_misses,
		                 stats.permutation_misses, stats.added_sources, stats.added_permutations,
		                 stats.invalidated);
	}
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			// Interned layouts (pipelineLayoutCache.h) are shared and destroyed below.
			ReleasePipelineLayout(m_graphics, pipeline->pipeline_layout,
			                      pipeline->descriptor_set_layout);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_library != nullptr) {
		// Replaced linked pipelines: only their handles. Their layouts are the live entries'
		// (copied with the entry), released once above.
		for (const auto& retired: m_library->retired) {
			m_graphics.device.destroyPipeline(retired->pipeline, nullptr);
		}
		// Libraries after every pipeline linked from them, and before the interned layouts they
		// were created with.
		m_library.reset();
	}
	DestroyInternedPipelineLayouts(m_graphics);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	m_driver_cache_path    = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	auto path               = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	const auto           signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		ReadDriverCacheFile(m_driver_cache_path, signature, initial_data, true);
	}
	if (initial_data.empty()) {
		// A save writes "<file>.tmp" and then renames it over the file. Earlier builds deleted the
		// file before that rename, so a run killed in between left only the complete temporary
		// file. Its payload hash rejects a partially written one.
		auto temp_path = m_driver_cache_path;
		temp_path += ".tmp";
		if (Common::File::IsFileExisting(temp_path) &&
		    ReadDriverCacheFile(temp_path, signature, initial_data, false)) {
			path = Common::PathToString(temp_path);
			PipelineCacheLog("Vulkan pipeline cache: recovered the interrupted save {}", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

// KYTY_PROGRAM_CACHE=0|1: the persistent translated-program cache (programDiskCache.h), next to
// the driver cache as _PipelineCache/<title>.programs.bin (KYTY_PROGRAM_CACHE_PATH names another
// file; tests use it). Default: ProgramDiskCacheDefault. KYTY_PROGRAM_CACHE_SAVE=0: no periodic
// saves, only the one at exit. KYTY_PROGRAM_CACHE_VERIFY: see ProgramCache::DiskVerifyMode.
constexpr bool ProgramDiskCacheDefault = false;

void PipelineCache::InitializeProgramDiskCache() {
	const auto* setting = std::getenv("KYTY_PROGRAM_CACHE");
	const bool  enabled = setting == nullptr || *setting == '\0' ? ProgramDiskCacheDefault
	                                                              : std::strcmp(setting, "0") != 0;
	if (!enabled) {
		return;
	}
	std::filesystem::path path;
	if (const auto* custom = std::getenv("KYTY_PROGRAM_CACHE_PATH");
	    custom != nullptr && *custom != '\0') {
		path = std::filesystem::path(custom);
	} else {
		const auto title_id = PipelineCacheTitleId();
		if (title_id.empty()) {
			return;
		}
		if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
			PipelineCacheLog("Program cache: disabled (non-Release build)");
			return;
		}
		path = std::filesystem::path("_PipelineCache") / (title_id + ".programs.bin");
	}
	// The file's identity: the codegen version and runtime codegen state, and the device (its
	// state that the emitter reads is part of the fingerprint already; the device and driver are
	// named as well, so that no device-dependent input can be missed).
	ProgramDiskCache::Settings settings;
	settings.path           = path;
	settings.identity       = ShaderRecompiler::CodegenFingerprint();
	settings.periodic_saves = EnvU64("KYTY_PROGRAM_CACHE_SAVE", 1) != 0;
	// Validation workers of the boot load (microbench on a 115 MB file, 1/2/4/8 threads: 7.0 /
	// 5.0 / 4.3 / 4.6 ms median).
	settings.load_threads = static_cast<uint32_t>(
	    std::clamp<uint64_t>(EnvU64("KYTY_PROGRAM_CACHE_LOAD_THREADS", 4), 1, 64));
	const auto& properties  = m_graphics.GetPhysicalDeviceProperties();
	const auto  device      = fmt::format("device {:08x}:{:08x}:{:08x}", properties.vendorID,
	                                      properties.deviceID, properties.driverVersion);
	settings.identity.insert(settings.identity.end(), device.begin(), device.end());
	m_program_disk            = std::make_unique<ProgramDiskCache>(std::move(settings));
	m_program_cache->disk     = m_program_disk.get();
	PipelineCacheLog("Program cache: enabled ({}; codegen {}{})", Common::PathToString(path),
	                 ShaderRecompiler::CodegenVersion().substr(0, 16),
	                 [] {
		                 switch (ProgramCache::DiskVerifyMode()) {
			                 case ProgramCache::DiskVerify::Sync: return "; verify";
			                 case ProgramCache::DiskVerify::SyncExit:
				                 return "; verify, exit on a difference";
			                 case ProgramCache::DiskVerify::Background: return "; background verify";
			                 default: return "";
		                 }
	                 }());
}

// Background saver of the driver pipeline cache. The cache used to be written only by the clean
// exit (window close), so every killed, crashed or stalled run threw away all pipelines it had
// compiled and the next boot compiled them again. The thread wakes once a second and saves by the
// DriverCacheSaveSettings rules; the command processor only bumps two atomics per new pipeline.
struct PipelineCache::DriverCacheSaver {
	explicit DriverCacheSaver(PipelineCache& owner): cache(owner), last_save_ns(CompileClockNs()) {
		thread = std::jthread([this](std::stop_token stop) { Run(stop); });
	}
	~DriverCacheSaver() { Stop(); }
	DriverCacheSaver(const DriverCacheSaver&)            = delete;
	DriverCacheSaver& operator=(const DriverCacheSaver&) = delete;

	// Returns once no periodic save is running or can start.
	void Stop() {
		if (thread.joinable()) {
			thread.request_stop();
			thread.join();
		}
	}

	void NoteCreated() {
		last_created_ns.store(CompileClockNs(), std::memory_order_relaxed);
		created.fetch_add(1, std::memory_order_release);
	}

	[[nodiscard]] bool Due(uint64_t now) const {
		const auto& settings = DriverCacheSaveSettings::Get();
		const auto  pending  = created.load(std::memory_order_acquire) - saved_created;
		if (pending == 0 || oversized || now < retry_after_ns) {
			return false;
		}
		const auto since_save   = now - last_save_ns;
		const auto last_created = last_created_ns.load(std::memory_order_relaxed);
		const auto since_create = now > last_created ? now - last_created : 0;
		if (pending < settings.min_new) {
			return since_save >= 5 * settings.interval_ns && since_create >= settings.settle_ns;
		}
		return (since_save >= settings.interval_ns && since_create >= settings.settle_ns) ||
		       (since_create >= settings.quiet_ns && since_save >= 15'000'000'000ull) ||
		       since_save >= 3 * settings.interval_ns;
	}

	void Run(const std::stop_token& stop) {
		Profiler::SetThreadName("PipelineCacheSaver");
		std::mutex                  mutex;
		std::condition_variable_any wake;
		std::unique_lock            lock(mutex);
		while (!stop.stop_requested()) {
			wake.wait_for(lock, stop, std::chrono::seconds(1), [] { return false; });
			if (stop.stop_requested() || !Due(CompileClockNs())) {
				continue;
			}
			// Count only what the serialized data can contain: pipelines created before it.
			const auto covered = created.load(std::memory_order_acquire);
			const auto written = cache.WriteDriverCache(true);
			const auto now     = CompileClockNs();
			last_save_ns       = now;
			if (written == WriteOversized) {
				// It only grows: stop serializing it every interval. Exit still saves it.
				oversized = true;
			} else if (written == 0) {
				retry_after_ns = now + DriverCacheSaveSettings::Get().interval_ns;
			} else {
				saved_created = covered;
			}
		}
	}

	// WriteDriverCache result of a periodic save skipped for exceeding MaxDriverCacheFileSize.
	static constexpr uint64_t WriteOversized = UINT64_MAX;

	PipelineCache&        cache;
	std::atomic<uint64_t> created {0};
	std::atomic<uint64_t> last_created_ns {0};
	// Saver thread only, then Save() after Stop().
	uint64_t saved_created  = 0;
	uint64_t last_save_ns   = 0;
	uint64_t retry_after_ns = 0;
	bool     oversized      = false;
	// While a save runs vkGetPipelineCacheData; pipelines created meanwhile are counted as overlaps.
	std::atomic<bool> serializing {false};
	std::jthread      thread; // Last: stopped before the members it uses are destroyed.
};

void PipelineCache::NotePipelineCreated(uint64_t create_ns) {
	if (m_saver == nullptr) {
		return;
	}
	m_saver->NoteCreated();
	if (m_saver->serializing.load(std::memory_order_relaxed)) {
		HangTrace::RecordPipelineCacheSaveOverlap(create_ns);
	}
}

uint64_t PipelineCache::WriteDriverCache(bool periodic) {
	std::lock_guard write_lock(m_driver_cache_write);
	return WriteDriverCacheLocked(periodic ? "periodic" : "exit", periodic);
}

void PipelineCache::SaveEmergency() {
	if (m_program_disk != nullptr) {
		m_program_disk->Flush();
	}
	// Read without m_mutex: the handle changes only in InitializeDriverCache and Save(), and the
	// caller keeps the cache alive (no Save() runs) while this does.
	if (m_driver_cache == nullptr) {
		return;
	}
	std::unique_lock write_lock(m_driver_cache_write, std::chrono::seconds(1));
	if (!write_lock.owns_lock()) {
		PipelineCacheLog("Vulkan pipeline cache: emergency save skipped (another save is running)");
		return;
	}
	(void)WriteDriverCacheLocked("emergency", true);
}

uint64_t PipelineCache::WriteDriverCacheLocked(const char* kind, bool capped) {
	const auto begin = CompileClockNs();
	// Synchronization: vkGetPipelineCacheData has no externally synchronized parameter (vk.xml
	// declares none for pipelineCache), and m_driver_cache is created without
	// VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT, so vkCreate*Pipelines on the command
	// processor may use the cache concurrently: the implementation synchronizes internally. No
	// emulator lock is taken here, so a draw never waits for this serialization or the file I/O
	// below; at worst a pipeline created meanwhile waits inside the driver
	// (pcache_save_overlaps). vkDestroyPipelineCache, the only externally synchronized use, runs
	// in Save() after the saver has stopped. The handle itself is written only before the saver
	// starts and after it stops.
	std::vector<uint8_t> payload;
	size_t               size   = 0;
	vk::Result           result = vk::Result::eIncomplete;
	if (m_saver != nullptr) m_saver->serializing.store(true, std::memory_order_relaxed);
	for (uint32_t attempt = 0; attempt < 3 && result == vk::Result::eIncomplete; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		// Pipelines created between the two calls grow the data; leave room for them. Whatever
		// is returned is valid initial data, but only complete data replaces the file.
		payload.resize(size + size / 8u + 1024u * 1024u);
		size   = payload.size();
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
	}
	if (m_saver != nullptr) m_saver->serializing.store(false, std::memory_order_relaxed);
	const auto serialized = CompileClockNs();
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: {} save failed ({}, {} bytes)", kind,
		                 vk::to_string(result), size);
		return 0;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (capped && prefix.size() + payload.size() > MaxDriverCacheFileSize) {
		// The loader rejects such a file and starts over; keep the last file that it accepts.
		// The exit save still writes it, which is what resets an overgrown cache.
		PipelineCacheLog("Vulkan pipeline cache: {} bytes exceed the {} byte cap; periodic saves "
		                 "stop, the exit save still writes it",
		                 payload.size(), MaxDriverCacheFileSize);
		return DriverCacheSaver::WriteOversized;
	}
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return 0;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	// std::filesystem::rename replaces the destination in one step (MoveFileExW with
	// MOVEFILE_REPLACE_EXISTING on Windows, rename(2) elsewhere): a kill at any point leaves
	// either the previous or the new complete file. Common::File::RenameFile deleted the old
	// file first, so a kill between the delete and the move lost the cache.
	std::error_code error;
	if (prefix_written == prefix.size() && payload_written == payload.size() && flushed) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}{}",
		                 Common::PathToString(m_driver_cache_path),
		                 error ? " (" + error.message() + ")" : std::string());
		return 0;
	}
	const auto written = CompileClockNs();
	HangTrace::RecordPipelineCacheSave(payload.size(), serialized - begin, written - serialized);
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {} ({}, serialize {:.1f} ms, write "
	                 "{:.1f} ms)",
	                 payload.size(), Common::PathToString(m_driver_cache_path), kind,
	                 static_cast<double>(serialized - begin) / 1.0e6,
	                 static_cast<double>(written - serialized) / 1.0e6);
	return payload.size();
}

void PipelineCache::Save() {
	// Background translations insert sources and add them to the disk cache.
	m_program_cache->StopAsyncTranslations();
	if (m_program_disk != nullptr) {
		// Pending background checks are dropped; a check never outlives the disk cache.
		m_program_cache->StopBackgroundChecks();
		m_program_disk->Flush();
	}
	// From here on no periodic save runs, so destroying the driver cache below is safe.
	if (m_saver != nullptr) {
		m_saver->Stop();
	}
	// Background optimized compiles use the driver cache too; linked pipelines stay in use.
	if (m_library != nullptr) {
		m_library->Stop();
	}
	// Background compiles of pending pipelines finish first, so no entry stays pending.
	if (m_async != nullptr) {
		m_async->Stop();
	}
	Common::LockGuard lock(m_mutex);
	if (m_driver_cache == nullptr) {
		return;
	}
	if (m_saver != nullptr && m_saver->saved_created != 0 &&
	    m_saver->created.load(std::memory_order_acquire) == m_saver->saved_created) {
		PipelineCacheLog("Vulkan pipeline cache: no new pipelines since the last save");
	} else if (WriteDriverCache(false) == 0) {
		return;
	}
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
}

void PipelineCache::WaitProgramChecks() {
	m_program_cache->WaitBackgroundChecks();
}

PipelineCache::ProgramTotals PipelineCache::Totals() {
	const auto& t = g_compile_totals;
	return {.programs          = t.programs.load(std::memory_order_relaxed),
	        .translations      = t.translations.load(std::memory_order_relaxed),
	        .source_hits       = t.disk_source_hits.load(std::memory_order_relaxed),
	        .permutation_hits  = t.disk_permutation_hits.load(std::memory_order_relaxed),
	        .verify_checks     = t.disk_verify_checks.load(std::memory_order_relaxed),
	        .verify_mismatches = t.disk_verify_mismatches.load(std::memory_order_relaxed)};
}

bool PipelineDynamicRasterStateEnabled() {
	static const bool enabled = EnvU64("KYTY_PIPELINE_DYNAMIC_STATE", 1) != 0;
	return enabled;
}

namespace {

bool PipelineKeyNormalizationEnabled() {
	static const bool enabled = EnvU64("KYTY_PIPELINE_KEY_NORMALIZE", 1) != 0;
	return enabled;
}

// KYTY_BLEND_ALPHA_REMAP=1 (default off; upstream 73615c31f, PPSA02721): when target 0's export
// mapping moves alpha out of the fourth channel and the blend reads source alpha, the pixel
// program exports logical alpha through MRT1 and the pipeline blends with it as a second source;
// blending whose export mapping cannot be expressed that way is disabled (one warning). Off: the
// blend reads the swizzled output as before.
bool BlendAlphaRemapEnabled() {
	static const bool enabled = EnvU64("KYTY_BLEND_ALPHA_REMAP", 0) != 0;
	return enabled;
}

// Accounts one graphics pipeline created through the library path and names the path in the
// compiles.csv detail.
void RecordLibraryPipeline(const GraphicsPipelineLibrary::Result& result, std::string& detail) {
	auto& totals = g_compile_totals;
	if (result.libraries_created != 0) {
		totals.gpl_libraries.fetch_add(result.libraries_created, std::memory_order_relaxed);
		totals.gpl_library_ns.fetch_add(result.libraries_ns, std::memory_order_relaxed);
		Profiler::CountFrameEvent(Profiler::FrameEvent::PipelineLibrariesCreated,
		                          result.libraries_created);
		HangTrace::RecordPipelineLibraryEvent(HangTrace::PipelineLibraryEvent::Library,
		                                      result.libraries_ns, result.libraries_created);
	}
	const char* path = "gpl-monolithic";
	switch (result.path) {
		case GraphicsPipelineLibrary::Path::CacheHit:
			path = "gpl-cache-hit";
			totals.gpl_cache_hits.fetch_add(1, std::memory_order_relaxed);
			Profiler::CountFrameEvent(Profiler::FrameEvent::PipelineLibraryCacheHits);
			HangTrace::RecordPipelineLibraryEvent(HangTrace::PipelineLibraryEvent::CacheHit, 0);
			break;
		case GraphicsPipelineLibrary::Path::Linked:
			path = "gpl-linked";
			totals.gpl_links.fetch_add(1, std::memory_order_relaxed);
			totals.gpl_link_ns.fetch_add(result.link_ns, std::memory_order_relaxed);
			Profiler::CountFrameEvent(Profiler::FrameEvent::PipelineLibraryLinks);
			HangTrace::RecordPipelineLibraryEvent(HangTrace::PipelineLibraryEvent::Linked,
			                                      result.link_ns);
			break;
		default: totals.gpl_monolithic.fetch_add(1, std::memory_order_relaxed); break;
	}
	detail += detail.empty() ? "" : "|";
	detail += fmt::format("{} libs={}", path, result.libraries_created);
	if (result.fallback != nullptr) {
		detail += fmt::format(" ({})", result.fallback);
	}
}

// Zeroes pipeline-key fields that cannot change the pipeline CreatePipelineInternal creates, so
// draws that differ only in them share one pipeline instead of compiling identical copies.
// KYTY_PIPELINE_KEY_NORMALIZE=0 keeps them. Each rule and why the field is unused:
// - Blend factors and ops of an attachment with blending off: they only feed the blend equation
//   of VkPipelineColorBlendAttachmentState, which Vulkan ignores when blendEnable is VK_FALSE.
//   0 is BlendFactor::kZero / BlendOp::kAdd, which GetBlendFactor/GetBlendOp accept.
// - Alpha factors and op with separate alpha blending off: CreatePipelineInternal then copies the
//   color factors and op into the alpha ones and never reads these.
// - The depth-bounds test without a depth/stencil attachment: CreatePipelineInternal passes no
//   VkPipelineDepthStencilStateCreateInfo at all (pDepthStencilState is null).
// - Depth-bounds min/max while the test is disabled: they are only inputs of that test.
// Not normalized: the clip-space viewport transform in the vertex-program key
// (BuildStageStaticKey). The emitter bakes scale, offset and half extent into the SPIR-V as
// constants (ConvertPositionToClipSpace), so each value is a different shader; moving them to
// push constants would also turn the power-of-two division into a runtime division, which is not
// guaranteed to be bit-identical.
// With KYTY_PIPELINE_DYNAMIC_STATE, cull mode, front face and the depth-bounds state are dynamic
// (set per draw from the same registers) and leave the key entirely.
void NormalizePipelineKey(PipelineStaticParameters& params, bool with_depth) {
	if (PipelineKeyNormalizationEnabled()) {
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			if (!params.blend_enable[slot]) {
				params.color_srcblend[slot]       = 0;
				params.color_comb_fcn[slot]       = 0;
				params.color_destblend[slot]      = 0;
				params.separate_alpha_blend[slot] = false;
			}
			if (!params.blend_enable[slot] || !params.separate_alpha_blend[slot]) {
				params.alpha_srcblend[slot]  = 0;
				params.alpha_comb_fcn[slot]  = 0;
				params.alpha_destblend[slot] = 0;
			}
		}
		// The alpha-source remap only changes target 0's blend factors.
		if (!params.blend_enable[0]) {
			params.blend_alpha_source_remap = false;
		}
		if (!with_depth) {
			params.depth_bounds_test_enable = false;
		}
		if (!params.depth_bounds_test_enable) {
			params.depth_min_bounds = 0.0f;
			params.depth_max_bounds = 0.0f;
		}
	}
	if (PipelineDynamicRasterStateEnabled()) {
		params.cull_front               = false;
		params.cull_back                = false;
		params.face                     = false;
		params.depth_bounds_test_enable = false;
		params.depth_min_bounds         = 0.0f;
		params.depth_max_bounds         = 0.0f;
	}
}

// KYTY_STAGE_PREP_VERIFY: 0 (default) off; 1 reruns the serial preparation after every parallel
// one and logs and counts differences; "exit" also stops the emulator on the first difference.
int StagePrepVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_STAGE_PREP_VERIFY");
		if (value == nullptr || std::strcmp(value, "0") == 0 || *value == '\0') return 0;
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

bool SameSnapshot(const ShaderRecompiler::IR::ResourceSnapshot& a,
                  const ShaderRecompiler::IR::ResourceSnapshot& b) {
	return a.buffers == b.buffers && a.images == b.images && a.samplers == b.samplers &&
	       a.flattened_srt == b.flattened_srt && a.user_data == b.user_data &&
	       a.uniform_fill == b.uniform_fill;
}

// The oracle prepares copies of the stage inputs serially and compares every output that the
// draw consumes. It only reads clean memory (the parallel preparation proved it clean), so it
// cannot synchronize; a difference means a concurrent guest write or a bug.
template <typename Serial>
void VerifyParallelPrograms(const PipelineCache::GraphicsPrograms&      parallel,
                            const std::array<ShaderVertexInputInfo, 3>& vertex_info,
                            const ShaderPixelInputInfo&                 pixel_info,
                            const PipelineCache::GraphicsStagePreps&    preps,
                            const Serial&                               serial) {
	auto vertex_copy = std::make_unique<std::array<ShaderVertexInputInfo, 3>>(vertex_info);
	auto pixel_copy  = std::make_unique<ShaderPixelInputInfo>(pixel_info);
	auto prep_copy   = std::make_unique<PipelineCache::GraphicsStagePreps>();
	const auto expected = serial(*vertex_copy, *pixel_copy, *prep_copy);
	const bool programs_equal = expected.pixel.id == parallel.pixel.id &&
	                            expected.vertex[0].id == parallel.vertex[0].id;
	const bool pixel_equal  = SameSnapshot(prep_copy->pixel.resources, preps.pixel.resources) &&
	                          prep_copy->pixel.specialization == preps.pixel.specialization &&
	                          prep_copy->pixel.permutation == preps.pixel.permutation;
	const bool vertex_equal = SameSnapshot(prep_copy->vertex[0].resources,
	                                       preps.vertex[0].resources) &&
	                          prep_copy->vertex[0].specialization ==
	                              preps.vertex[0].specialization &&
	                          prep_copy->vertex[0].permutation == preps.vertex[0].permutation;
	if (programs_equal && pixel_equal && vertex_equal) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::StagePrepVerifyMismatches);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("StagePrepVerify: parallel preparation differs: programs=%d pixel=%d vertex=%d "
		     "ps=%" PRIu64 "/%" PRIu64 " vs=%" PRIu64 "/%" PRIu64 "\n",
		     programs_equal, pixel_equal, vertex_equal, parallel.pixel.id, expected.pixel.id,
		     parallel.vertex[0].id, expected.vertex[0].id);
	}
	if (StagePrepVerifyMode() == 2) {
		EXIT("StagePrepVerify: parallel preparation differs from the serial path\n");
	}
}

} // namespace

bool PipelineCache::TessellationActive(const HW::UserConfig& user_config) {
	return user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
}

namespace {

// Static stage information shared by GetGraphicsPrograms and the draw-prep speculative
// preparation: pure functions of the registers and constant device limits.
void FinishMeshStage(const GraphicContext& graphics, ShaderVertexInputInfo& vertex_info) {
	EXIT_NOT_IMPLEMENTED(!graphics.mesh_shader_enabled);
	auto& mesh              = vertex_info.mesh;
	mesh.host_subgroup_size = graphics.subgroup_size;
	const auto& limits      = graphics.mesh_shader_properties;
	const auto  logical_threads = mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
	const auto  host_threads    = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
	                          std::min(mesh.host_subgroup_size, mesh.wave_size);
	if (host_threads > limits.maxMeshWorkGroupInvocations ||
	    host_threads > limits.maxMeshWorkGroupSize[0] ||
	    mesh.max_vertices > limits.maxMeshOutputVertices ||
	    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
	    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
		EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
		     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
	}
}

void ApplyDualSourceBlending(const HW::Context& context, ShaderPixelInputInfo& pixel_info) {
	const auto& blend    = context.GetBlendControl(0);
	const bool  blending = blend.enable && !context.GetRenderTarget(0).info.blend_bypass;
	pixel_info.alpha_blend_source_remap = false;
	pixel_info.dual_source_blending =
	    blending && (BlendFactorIsDualSource(blend.color_srcblend) ||
	                 BlendFactorIsDualSource(blend.color_destblend) ||
	                 (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
	                                                 BlendFactorIsDualSource(blend.alpha_destblend))));
	if (pixel_info.dual_source_blending) {
		// MRT1 supplies a second blend source for the same render target as MRT0.
		pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
		pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
	} else if (BlendAlphaRemapEnabled() && blending && pixel_info.target_output_mode[0] != 0 &&
	           pixel_info.target_output_mode[0] != 7 &&
	           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
	                       std::end(pixel_info.target_output_mode),
	                       [](uint8_t mode) { return mode == 0; }) &&
	           ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0]) ==
	               BlendMappingSupport::SourceAlpha) {
		// The export mapping moves alpha out of the fourth channel: MRT1 carries logical alpha
		// as a second blend source (KYTY_BLEND_ALPHA_REMAP, upstream 73615c31f).
		pixel_info.alpha_blend_source_remap = true;
		pixel_info.dual_source_blending     = true;
		pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
		pixel_info.target_export_mapping[1] = {};
	}
}

void ApplyClipSpace(const GraphicContext& graphics, const HW::Context& context,
                    ShaderVertexInputInfo& last_vertex_stage) {
	if (!context.GetClipControl().clip_disable) {
		return;
	}
	const auto& viewport = context.GetScreenViewport().viewports[0];
	const auto& limits   = graphics.GetPhysicalDeviceProperties().limits;
	auto&       clip     = last_vertex_stage.clip_space;
	clip.scale[0]        = viewport.xscale;
	clip.scale[1]        = viewport.yscale;
	clip.offset[0]       = viewport.xoffset;
	clip.offset[1]       = viewport.yoffset;
	clip.half_extent[0] = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
	clip.half_extent[1] = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
	clip.enabled        = true;
}

} // namespace

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    GraphicsStagePreps& stage_preps) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	const bool tess_active = TessellationActive(user_config);
	std::array<ShaderParams, 3> vertex_params;
	// Static stage information (shader map, headers, vertex tables); ended before
	// materialization.
	static constexpr tracy::SourceLocationData prepare_static_location {
	    "Programs::PrepareStatic", TracyFunction, TracyFile, static_cast<uint32_t>(__LINE__), 0};
	Profiler::ScopedBlock static_zone(&prepare_static_location, Profiler::DetailedEnabled());
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		FinishMeshStage(m_graphics, vertex_info[0]);
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		ApplyDualSourceBlending(context, pixel_info);
	}
	ApplyClipSpace(m_graphics, context, vertex_info[tess_active ? 2u : 0u]);
	static_zone.End();
	// The program cache locks internally, so no pipeline lock is held across materialization.
	auto& scratch    = ProgramCache::ThreadScratch();
	auto& evaluation = ShaderRecompiler::IR::ThreadEvaluationScratch();
	const uint32_t push_data_start =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	const auto serial = [&](std::array<ShaderVertexInputInfo, 3>& vertex_inputs,
	                        ShaderPixelInputInfo& pixel_input,
	                        GraphicsStagePreps& preps) -> GraphicsPrograms {
		for (uint32_t attempt = 0; attempt < 64; ++attempt) {
			ShaderReadAttempt read_attempt;
			uint32_t          push_data_cursor = push_data_start;
			GraphicsPrograms  result;
			if (pixel_active) {
				result.pixel = m_program_cache->Get(pixel_params, pixel_input, preps.pixel,
				                                   push_data_cursor, read_attempt, scratch,
				                                   evaluation);
			}
			for (uint32_t i = 0; i < (tess_active ? 3u : 1u) &&
			                     !read_attempt.materialization_failed; ++i) {
				result.vertex[i] =
				    m_program_cache->Get(vertex_params[i], vertex_inputs[i], preps.vertex[i],
				                         push_data_cursor, read_attempt, scratch, evaluation);
			}
			if (!read_attempt.materialization_failed) return result;
			// No pipeline or texture-cache lock is held while the scheduler publishes bytes.
			// Restart all stages before final bindings/uploads, including their SRT refresh.
			EXIT_IF(!read_attempt.Synchronize());
		}
		EXIT("graphics resource readiness did not converge after 64 attempts\n");
	};
	// Draw-prep S3: PS and VS materialized concurrently. The O15 reuse mode keeps shared
	// per-entry state and tessellation has three vertex stages; both stay serial.
	if (pixel_active && !tess_active && !ResourceReuseEnabled()) {
		GraphicsPrograms result;
		if (m_program_cache->TryGetParallel(pixel_params, pixel_info, stage_preps.pixel,
		                                    vertex_params[0], vertex_info[0], stage_preps.vertex[0],
		                                    push_data_start, scratch, evaluation, result.pixel,
		                                    result.vertex[0])) {
			if (StagePrepVerifyMode() != 0) {
				VerifyParallelPrograms(result, vertex_info, pixel_info, stage_preps, serial);
			}
			return result;
		}
	}
	return serial(vertex_info, pixel_info, stage_preps);
}

PipelineCache::SpeculativeResult PipelineCache::PrepareGraphicsProgramsSpeculative(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    ShaderVertexInputInfo& vertex_info, ShaderPixelInputInfo& pixel_info, StagePrep& vertex_prep,
    StagePrep& pixel_prep, GraphicsPrograms& programs) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	EXIT_IF(!DrawPrep::Speculative());
	if (TessellationActive(user_config) || ResourceReuseEnabled()) {
		return SpeculativeResult::Ineligible;
	}
	const auto read_failed = [] { return DrawPrep::ActiveRecorder()->reads->Failed(); };
	// The same static stage preparation as GetGraphicsPrograms (non-tessellated).
	const auto vertex_params = PrepareProgram(vertex_regs, context, user_config, vertex_info);
	if (read_failed()) {
		return SpeculativeResult::ReadFailed;
	}
	if (vertex_info.logical_stage == ShaderType::Mesh) {
		FinishMeshStage(m_graphics, vertex_info);
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		ApplyDualSourceBlending(context, pixel_info);
	}
	ApplyClipSpace(m_graphics, context, vertex_info);
	if (read_failed()) {
		return SpeculativeResult::ReadFailed;
	}
	const uint32_t push_data_start = vertex_info.logical_stage == ShaderType::Mesh
	                                     ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount
	                                     : 0;
	const auto result = m_program_cache->TryPrepareSpeculative(
	    pixel_active, pixel_params, pixel_info, pixel_prep, vertex_params, vertex_info, vertex_prep,
	    push_data_start, programs);
	if (result == SpeculativeResult::Ok && read_failed()) {
		return SpeculativeResult::ReadFailed;
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info,
                                               StagePrep&                   stage_prep) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	auto& scratch    = ProgramCache::ThreadScratch();
	auto& evaluation = ShaderRecompiler::IR::ThreadEvaluationScratch();
	for (uint32_t attempt = 0; attempt < 64; ++attempt) {
		ShaderReadAttempt read_attempt;
		uint32_t push_data_cursor = 0;
		const auto result = m_program_cache->Get(params, input_info, stage_prep, push_data_cursor,
		                                         read_attempt, scratch, evaluation);
		if (!read_attempt.materialization_failed) return result;
		EXIT_IF(!read_attempt.Synchronize());
	}
	EXIT("compute resource readiness did not converge after 64 attempts\n");
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

void PipelineCache::PipelineKeyHash::MixStaticParams(std::size_t& hash,
                                                    const PipelineStaticParameters& params) {
	if (Common::RendererBatchEnabled()) {
		// Equality already compares this exact byte representation, including padding.
		Mix(hash, static_cast<std::size_t>(XXH3_64bits(&params, sizeof(params))));
		return;
	}
	const auto* bytes = reinterpret_cast<const uint8_t*>(&params);
	for (std::size_t i = 0; i < sizeof(params); ++i) Mix(hash, bytes[i]);
}

ShaderProgram PipelineCache::PlainPixelProgram(const StagePrep& prep) {
	return prep.permutation != nullptr ? prep.permutation->plain : ShaderProgram {};
}

// Pipeline: temporary acceptance of EXEC_ON_NOOP with the depth test enabled (first 16 draws).
static void NoteExecOnNoop(const RenderDepthInfo& depth, const ShaderPixelInputInfo* ps_input_info) {
	if (ps_input_info != nullptr && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}
}

bool PipelineCache::SamePipelineTargets(const PipelineTargets&           targets,
                                        std::span<const RenderColorInfo> colors,
                                        const RenderDepthInfo&           depth) {
	if (colors.size() != targets.color_count) {
		return false;
	}
	for (uint32_t i = 0; i < targets.color_count; i++) {
		const auto& planned = targets.colors[i];
		const auto& color   = colors[i];
		if (color.target_slot != planned.slot || !color.image_id ||
		    color.desc.view_info.format != planned.format ||
		    color.desc.info.samples != planned.samples ||
		    color.export_mapping.packed != planned.export_mapping.packed) {
			return false;
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth != targets.with_depth ||
	    (with_depth && (depth.desc.view_info.format != targets.depth_format ||
	                    depth.desc.info.samples != targets.depth_samples))) {
		return false;
	}
	return depth.depth_bounds_test_enable == targets.depth_bounds_test_enable &&
	       std::bit_cast<uint32_t>(depth.depth_min_bounds) ==
	           std::bit_cast<uint32_t>(targets.depth_min_bounds) &&
	       std::bit_cast<uint32_t>(depth.depth_max_bounds) ==
	           std::bit_cast<uint32_t>(targets.depth_max_bounds);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool PipelineCache::BuildGraphicsPipelineKey(const PipelineTargets& targets, const HW::Context& ctx,
                                             const HW::UserConfig&        user_config,
                                             const ShaderVertexInputInfo& vs_input_info,
                                             const ShaderPixelInputInfo*  ps_input_info,
                                             vk::PrimitiveTopology        topology,
                                             bool                         primitive_restart_enable,
                                             const GraphicsPrograms& programs, bool fatal,
                                             GraphicsPipelineKey& key) const {
	// Where the serial lookup stops the emulator, a plan's lookup gives up.
	const auto refuse = [fatal](bool condition, const char* what) {
		if (condition && fatal) {
			EXIT("Pipeline: unsupported graphics pipeline key input: %s\n", what);
		}
		return condition;
	};
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	const bool  ps_active      = ps_input_info != nullptr;
	if (refuse(!vertex_program, "no vertex program") ||
	    refuse(ps_active && !pixel_program, "no pixel program")) {
		return false;
	}

	const HW::ModeControl& mc = ctx.GetModeControl();

	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_active ? pixel_program.id : 0;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < targets.color_count; i++) {
		const auto& color = targets.colors[i];
		const auto  slot  = color.slot;
		if (refuse(slot >= RENDER_COLOR_ATTACHMENTS_MAX, "colour slot") ||
		    refuse(color.format == vk::Format::eUndefined, "colour format")) {
			return false;
		}
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		static_params.color_mask[slot] =
		    color.export_mapping.ApplyMask(render_target_mask_slot(ctx.GetRenderTargetMask(), slot));
		rendering.color_formats[slot] = color.format;
		if (attachment_samples == 0) {
			attachment_samples = color.samples;
		} else if (attachment_samples != color.samples) {
			if (fatal) {
				EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
				     attachment_samples, color.samples);
			}
			return false;
		}
		const auto& rt                        = ctx.GetRenderTarget(slot);
		const auto& bc                        = ctx.GetBlendControl(slot);
		static_params.color_srcblend[slot]       = bc.color_srcblend;
		static_params.color_comb_fcn[slot]       = bc.color_comb_fcn;
		static_params.color_destblend[slot]      = bc.color_destblend;
		static_params.alpha_srcblend[slot]       = bc.alpha_srcblend;
		static_params.alpha_comb_fcn[slot]       = bc.alpha_comb_fcn;
		static_params.alpha_destblend[slot]      = bc.alpha_destblend;
		static_params.separate_alpha_blend[slot] = bc.separate_alpha_blend;
		static_params.blend_enable[slot]         = bc.enable && !rt.info.blend_bypass;
		if (BlendAlphaRemapEnabled()) {
			// Upstream 73615c31f: logical alpha through MRT1 (ApplyDualSourceBlending), or no
			// blending where the export mapping cannot be blended correctly.
			const bool alpha_remap =
			    slot == 0 && ps_input_info != nullptr && ps_input_info->alpha_blend_source_remap;
			if (static_params.blend_enable[slot] && !alpha_remap &&
			    ClassifyBlendMapping(bc, color.export_mapping) != BlendMappingSupport::Direct) {
				static_params.blend_enable[slot] = false;
				static std::atomic_bool warned = false;
				if (!warned.exchange(true, std::memory_order_relaxed)) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Warning: blending disabled for unsupported color mapping "
					    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
					    slot, color.export_mapping.packed, bc.color_srcblend, bc.color_destblend,
					    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
				}
			}
			if (alpha_remap) {
				static_params.blend_alpha_source_remap = true;
			}
		}
	}
	const bool with_depth = targets.with_depth;
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(targets.depth_format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth ? targets.depth_format
		                                                                     : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? targets.depth_format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = targets.depth_samples;
		} else if (attachment_samples != targets.depth_samples) {
			if (fatal) {
				EXIT("mixed color/depth sample counts are unsupported: %u and %u\n",
				     attachment_samples, targets.depth_samples);
			}
			return false;
		}
	}
	if (targets.color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		if (refuse(!static_cast<bool>(m_graphics.GetPhysicalDeviceProperties()
		                                  .limits.framebufferNoAttachmentsSampleCounts &
		                              vulkan_sample_count(attachment_samples)),
		           "sample count without attachments")) {
			return false;
		}
	}
	if (refuse(attachment_samples == 0 ||
	               vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {},
	           "sample count")) {
		return false;
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		if (fatal) {
			EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
		}
		return false;
	}
	static_params.depth_bounds_test_enable = targets.depth_bounds_test_enable;
	static_params.depth_min_bounds         = targets.depth_min_bounds;
	static_params.depth_max_bounds         = targets.depth_max_bounds;
	const bool rect_list = Prospero::IsRectList(user_config.GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);
	NormalizePipelineKey(static_params, with_depth);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		if (refuse(vs_input_info.buffers_num < 0 ||
		               vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		               vs_input_info.resources_num < 0 ||
		               vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX,
		           "vertex input counts")) {
			return false;
		}
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		uint32_t attributes_num          = 0;
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			if (refuse(buffer.attr_num < 0 || buffer.attr_num > ShaderVertexInputBuffer::ATTR_MAX,
			           "vertex attribute count")) {
				return false;
			}
			attributes_num += static_cast<uint32_t>(buffer.attr_num);
			if (refuse(attributes_num > static_cast<uint32_t>(vs_input_info.resources_num),
			           "vertex attributes")) {
				return false;
			}
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
			for (int attribute = 0; attribute < buffer.attr_num; attribute++) {
				const auto index = buffer.attr_indices[attribute];
				if (refuse(index < 0 || index >= vs_input_info.resources_num, "vertex attribute")) {
					return false;
				}
				key.vertex_input.attributes[index] = {
				    .offset  = buffer.attr_offsets[attribute],
				    .binding = static_cast<uint8_t>(binding),
				};
			}
		}
		if (refuse(attributes_num != static_cast<uint32_t>(vs_input_info.resources_num),
		           "vertex attribute total")) {
			return false;
		}
	}
	return true;
}

PipelineCache::PlanLookup PipelineCache::FindGraphicsPipelineForPlan(
    const PipelineTargets& targets, const HW::Context& ctx, const HW::UserConfig& user_config,
    const ShaderVertexInputInfo& vs_input_info, const ShaderPixelInputInfo* ps_input_info,
    vk::PrimitiveTopology topology, bool primitive_restart_enable, const GraphicsPrograms& programs,
    const Pipeline*& pipeline, uint64_t& generation) {
	pipeline = nullptr;
	GraphicsPipelineKey key {};
	if (!BuildGraphicsPipelineKey(targets, ctx, user_config, vs_input_info, ps_input_info, topology,
	                              primitive_restart_enable, programs, false, key)) {
		return PlanLookup::Unsupported;
	}
	// A per-thread memo of the keys this thread found. An entry keeps the generation its object
	// was found under: while it is unchanged the map still holds that object for the key (objects
	// are never freed, a replacement bumps the generation, and every cache instance counts its
	// generations in a range of its own). It points at the map's own key: map nodes are never
	// erased and their keys never change, so the key is read only while the generation proves the
	// entry is this cache's.
	struct Remembered {
		const PipelineCache*       cache      = nullptr;
		const GraphicsPipelineKey* key        = nullptr;
		const Pipeline*            pipeline   = nullptr;
		uint64_t                   generation = 0;
		std::size_t                hash       = 0;
	};
	static thread_local std::array<Remembered, 1024> memo {};
	const auto hash    = GraphicsPipelineKeyHash {}(key);
	auto&      entry   = memo[hash % memo.size()];
	const auto current = m_pipeline_generation.load(std::memory_order_acquire);
	if (entry.cache == this && entry.generation == current && entry.hash == hash &&
	    entry.key != nullptr && *entry.key == key) {
		pipeline   = entry.pipeline;
		generation = current;
		return PlanLookup::Found;
	}
	// The lock is held briefly by lookups (the command processor's, other threads'), for tens of
	// milliseconds by a creation: a short bounded retry covers the former only.
	bool locked = m_mutex.TryLock();
	for (uint32_t attempt = 0; !locked && attempt < 8; attempt++) {
		for (uint32_t pause = 0; pause < 16; pause++) {
#if defined(_M_X64) || defined(__x86_64__)
			_mm_pause();
#endif
		}
		locked = m_mutex.TryLock();
	}
	if (!locked) {
		return PlanLookup::Busy;
	}
	const auto iter = m_graphics_pipelines.find(key);
	// A pending entry (KYTY_ASYNC_PIPELINES) has no pipeline yet: absent for the plan.
	const auto found = iter != m_graphics_pipelines.end() && !iter->second->pending
	                       ? iter->second.get()
	                       : nullptr;
	const auto* found_key = iter != m_graphics_pipelines.end() ? &iter->first : nullptr;
	// Read under the lock, so it matches the object found (replacements bump it under the lock).
	const auto found_generation = m_pipeline_generation.load(std::memory_order_relaxed);
	m_mutex.Unlock();
	if (found == nullptr) {
		return PlanLookup::Absent;
	}
	entry.cache      = this;
	entry.key        = found_key;
	entry.pipeline   = found;
	entry.generation = found_generation;
	entry.hash       = hash;
	pipeline         = found;
	generation       = found_generation;
	return PlanLookup::Found;
}

void PipelineCache::NotePlannedPipeline(const RenderDepthInfo&      depth,
                                        const ShaderPixelInputInfo* ps_input_info) {
	NoteExecOnNoop(depth, ps_input_info);
	FlushCompileStall();
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	return *TryGetGraphicsPipeline(colors, depth, vertex_info, command, ps_input_info, topology,
	                               primitive_restart_enable, programs, false);
}

PipelineCache::Pipeline* PipelineCache::TryGetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs, bool may_defer,
    const char* sync_reason) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);

	// The key is built from the draw's registers and inputs only; m_mutex guards the map below.
	PipelineTargets targets;
	targets.color_count = static_cast<uint32_t>(colors.size());
	for (uint32_t i = 0; i < targets.color_count; i++) {
		EXIT_IF(colors[i].target_slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		targets.colors[i] = {.slot           = colors[i].target_slot,
		                     .format         = colors[i].desc.view_info.format,
		                     .samples        = colors[i].desc.info.samples,
		                     .export_mapping = colors[i].export_mapping};
	}
	targets.with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	targets.depth_format             = depth.desc.view_info.format;
	targets.depth_samples            = depth.desc.info.samples;
	targets.depth_bounds_test_enable = depth.depth_bounds_test_enable;
	targets.depth_min_bounds         = depth.depth_min_bounds;
	targets.depth_max_bounds         = depth.depth_max_bounds;

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	(void)BuildGraphicsPipelineKey(targets, command.GetRegisters(), command.GetUserConfig(),
	                               vs_input_info, ps_input_info, topology, primitive_restart_enable,
	                               programs, true, key);
	NoteExecOnNoop(depth, ps_input_info);

	// Last-key memo (KYTY_PIPELINE_MEMO): most draws use their predecessor's pipeline. Exact:
	// the comparison is the map's own key equality, and pipeline objects are never destroyed
	// before the cache is; an object replaced by an optimized build (pipeline libraries) bumps
	// m_pipeline_generation, which invalidates every memo.
	static const bool memo_enabled = [] {
		const auto* value = std::getenv("KYTY_PIPELINE_MEMO");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	struct LastPipeline {
		const PipelineCache* cache      = nullptr;
		Pipeline*            pipeline   = nullptr;
		uint64_t             generation = 0;
		GraphicsPipelineKey  key {};
	};
	static thread_local LastPipeline last;
	if (memo_enabled) {
		if (last.cache == this && last.pipeline != nullptr &&
		    last.generation == m_pipeline_generation.load(std::memory_order_acquire) &&
		    last.key == key) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::PipelineMemoHits);
			FlushCompileStall();
			return last.pipeline;
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::PipelineMemoMisses);
	}
	// Called with m_mutex held, so the generation read matches the object found.
	const auto remember = [&](Pipeline& pipeline) -> Pipeline& {
		if (memo_enabled) {
			last.cache      = this;
			last.pipeline   = &pipeline;
			last.generation = m_pipeline_generation.load(std::memory_order_relaxed);
			last.key        = key;
		}
		FlushCompileStall();
		return pipeline;
	};

	if (m_async != nullptr) {
		// A pending entry: a draw that may be skipped is; any other waits, outside the lock, for
		// the background compile, which publishes under the lock. The wait is a compile stall of
		// this thread (compile_stall_us), like an inline compile.
		uint64_t wait_begin = 0;
		for (;;) {
			{
				Common::LockGuard lock(m_mutex);
				const auto iter = m_graphics_pipelines.find(key);
				if (iter == m_graphics_pipelines.end() || !iter->second->pending) {
					break;
				}
				if (may_defer) {
					m_async->deferred_draws.fetch_add(1, std::memory_order_relaxed);
					FlushCompileStall();
					return nullptr;
				}
			}
			if (wait_begin == 0) {
				wait_begin = CompileClockNs();
			}
			std::this_thread::sleep_for(std::chrono::microseconds(200));
		}
		if (wait_begin != 0) {
			AddCompileStall(CompileClockNs() - wait_begin);
		}
	}

	Common::LockGuard lock(m_mutex);
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		// Only this (the recording) thread creates entries, so one found here was published.
		EXIT_IF(iter->second->pending);
		return &remember(*iter->second);
	}
	const auto create_begin = CompileClockNs();

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	// Pipeline libraries (6/7): the monolithic create info goes to the library path, which may
	// fast-link it and hand back the create info for the background optimized compile.
	GraphicsPipelineLibrary::Result library_result;
	GraphicsPipelineCreateHook      library_hook;
	if (m_library != nullptr) {
		library_hook = [&](const vk::GraphicsPipelineCreateInfo& info,
		                   std::span<const uint32_t> layout_signature, vk::Pipeline* pipeline) {
			library_result = m_library->library.Create(info, layout_signature, m_driver_cache);
			*pipeline      = library_result.pipeline;
			return library_result.result;
		};
	}
	// KYTY_ASYNC_PIPELINES: a draw that may be skipped hands the create info to a background
	// compile, unless the snapshot cannot copy it (mesh, tessellation), which is created here.
	std::unique_ptr<GraphicsPipelineSnapshot> async_snapshot;
	GraphicsPipelineCreateHook                async_hook;
	const bool                                async = may_defer && m_async != nullptr;
	if (async) {
		async_hook = [&](const vk::GraphicsPipelineCreateInfo& info, std::span<const uint32_t>,
		                 vk::Pipeline* pipeline) {
			async_snapshot = GraphicsPipelineSnapshot::Capture(info);
			if (async_snapshot != nullptr) {
				return vk::Result::eOperationDeferredKHR;
			}
			return m_graphics.device.createGraphicsPipelines(m_driver_cache, 1, &info, nullptr,
			                                                 pipeline);
		};
	}
	{
		Profiler::ScopedFrameWait pipeline_create(Profiler::FrameWait::GraphicsPipelineCreate);
		CreatePipelineInternal(m_graphics, *cached, key.rendering, key.vertex_input, vertex_info,
		                       ps_input_info, programs, key.static_params, m_driver_cache,
		                       async                 ? &async_hook
		                       : m_library != nullptr ? &library_hook
		                                              : nullptr);
	}
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);
	if (async_snapshot != nullptr) {
		EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
		cached->pending       = true;
		auto [iter, inserted] = m_graphics_pipelines.emplace(key, std::move(cached));
		EXIT_IF(!inserted);
		AsyncState::Job job {.key = &iter->first, .target = iter->second.get(),
		                     .snapshot = std::move(async_snapshot)};
		const auto  setup_ns = CompileClockNs() - create_begin;
		PipelineDiagnostics::Ids guest {};
		for (size_t i = 0; i < vertex_info.size() && i < 3; ++i) {
			guest[i] = vertex_info[i].stage.program->shader_hash;
		}
		guest[3] = ps_active ? ps_input_info->stage.program->shader_hash : 0;
		std::string detail;
		const auto  origin = m_diagnostics->Classify(iter->first, guest, detail);
		auto&       totals = g_compile_totals;
		totals.gfx_pipelines.fetch_add(1, std::memory_order_relaxed);
		auto* origin_total = origin == HangTrace::PipelineOrigin::New ? &totals.gfx_new
		                     : origin == HangTrace::PipelineOrigin::Permutation
		                         ? &totals.gfx_permutation
		                         : &totals.gfx_variant;
		origin_total->fetch_add(1, std::memory_order_relaxed);
		job.origin     = origin;
		job.guest_hash = guest[0];
		job.vs_id      = vs_id;
		job.ps_id      = ps_id;
		job.setup_ns   = setup_ns;
		job.detail     = detail.empty() ? "async" : "async " + detail;
		AddCompileStall(setup_ns);
		auto& target = *iter->second;
		if (m_async->Enqueue(job)) {
			if (WaitForQueuedPipeline(target)) {
				return &remember(target);
			}
			m_async->deferred_draws.fetch_add(1, std::memory_order_relaxed);
			FlushCompileStall();
			return nullptr;
		}
		// The queue stopped (Save() ran meanwhile; it waits for m_mutex before destroying the
		// driver cache): create it here, exactly as queued.
		vk::Pipeline created = nullptr;
		const auto   result  = m_graphics.device.createGraphicsPipelines(
		    m_driver_cache, 1, &job.snapshot->Info(), nullptr, &created);
		if (result != vk::Result::eSuccess || created == nullptr) {
			EXIT("vkCreateGraphicsPipelines failed: %s\n", vk::to_string(result).c_str());
		}
		iter->second->pipeline = created;
		iter->second->pending  = false;
		return &remember(*iter->second);
	}

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GraphicsPipelinesCreated);
	GpuOpProfiler::RegisterGraphicsPipeline(cached->pipeline, key.vertex_shader_ids.data(),
	                                        static_cast<uint32_t>(key.vertex_shader_ids.size()),
	                                        ps_id);

	const auto created_pipeline = cached->pipeline;
	auto [iter, inserted]       = m_graphics_pipelines.emplace(key, std::move(cached));
	EXIT_IF(!inserted);
	if (library_result.path == GraphicsPipelineLibrary::Path::Linked && m_library->optimize) {
		m_library->Enqueue({.key      = &iter->first,
		                    .linked   = created_pipeline,
		                    .snapshot = std::move(library_result.optimize)});
	}

	const auto create_ns = CompileClockNs() - create_begin;
	PipelineDiagnostics::Ids guest {};
	for (size_t i = 0; i < vertex_info.size() && i < 3; ++i) {
		guest[i] = vertex_info[i].stage.program->shader_hash;
	}
	guest[3] = ps_active ? ps_input_info->stage.program->shader_hash : 0;
	std::string detail;
	const auto  origin = m_diagnostics->Classify(iter->first, guest, detail);
	if (m_library != nullptr) {
		RecordLibraryPipeline(library_result, detail);
	}
	if (m_async != nullptr) {
		// Why this pipeline was created on the recording thread (KYTY_ASYNC_PIPELINES).
		const char* reason = may_defer ? "snapshot" : sync_reason != nullptr ? sync_reason : "draw";
		detail = std::string("sync:") + reason + (detail.empty() ? "" : " " + detail);
	}
	auto&       totals = g_compile_totals;
	totals.gfx_pipelines.fetch_add(1, std::memory_order_relaxed);
	totals.gfx_pipeline_ns.fetch_add(create_ns, std::memory_order_relaxed);
	auto* origin_total = origin == HangTrace::PipelineOrigin::New           ? &totals.gfx_new
	                     : origin == HangTrace::PipelineOrigin::Permutation ? &totals.gfx_permutation
	                                                                         : &totals.gfx_variant;
	origin_total->fetch_add(1, std::memory_order_relaxed);
	if (HangTrace::Enabled()) {
		HangTrace::RecordCompile({.kind        = HangTrace::CompileKind::GraphicsPipeline,
		                          .origin      = origin,
		                          .stage       = "gfx",
		                          .guest_hash  = guest[0],
		                          .id          = vs_id,
		                          .id2         = ps_id,
		                          .pipeline_ns = create_ns,
		                          .total_ns    = create_ns,
		                          .detail      = detail});
	}
	AddCompileStall(create_ns);
	NotePipelineCreated(create_ns);
	return &remember(*iter->second);
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	Common::LockGuard lock(m_mutex);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		FlushCompileStall();
		return *iter->second;
	}
	const auto create_begin = CompileClockNs();

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);
	GpuOpProfiler::RegisterComputePipeline(cached->pipeline, compute_program.id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	const auto create_ns = CompileClockNs() - create_begin;
	g_compile_totals.cs_pipelines.fetch_add(1, std::memory_order_relaxed);
	g_compile_totals.cs_pipeline_ns.fetch_add(create_ns, std::memory_order_relaxed);
	Profiler::CountFrameEvent(Profiler::FrameEvent::ComputePipelinesCreated);
	Profiler::AddFrameWait(Profiler::FrameWait::ComputePipelineCreate, 1, create_ns);
	if (HangTrace::Enabled()) {
		HangTrace::RecordCompile({.kind        = HangTrace::CompileKind::ComputePipeline,
		                          .stage       = "cs",
		                          .guest_hash  = input_info.stage.program->shader_hash,
		                          .id          = compute_program.id,
		                          .pipeline_ns = create_ns,
		                          .total_ns    = create_ns});
	}
	AddCompileStall(create_ns);
	FlushCompileStall();
	NotePipelineCreated(create_ns);
	return *iter->second;
}
} // namespace Libs::Graphics
