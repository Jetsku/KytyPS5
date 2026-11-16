#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "common/staticOnce.h"
#include "graphics/host_gpu/renderer/pipeline/pixelRateVariant.h"
#include "graphics/host_gpu/renderer/pipeline/aaMaskCoverage.h"
#include "common/ramStats.h"
#include "common/paceLog.h"

#include "common/assert.h"
#include "common/compileWatch.h"
#include "common/cpuPlacement.h"
#include "common/prepPhases.h"
#include "common/crashContext.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hangTrace.h"
#include "common/hangWatchdog.h"
#include "common/kytyEvents.h"
#include "common/liveSwitch.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/rendererBatch.h"
#include "common/sehGuard.h"
#include "common/skipInvalid.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/renderer/drawPrep/readSet.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLayoutCache.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineFeedback.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineFastFirst.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStats.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineListGameFile.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineListInt21.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCacheMappedFile.h"
#include "graphics/host_gpu/renderer/pipeline/programDiskCache.h"
#include "graphics/host_gpu/renderer/pipeline/rtSession.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"
#include "graphics/host_gpu/renderer/pipeline/stagePrepWorker.h"
#include "graphics/host_gpu/renderer/pipeline/shaderCodeSnapshot.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/guest_gpu/command_processor/passCensus.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/CodegenFingerprint.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/CallSplitPlan.h"
#include "graphics/shader/recompiler/ShaderCalls.h"
#include "graphics/shader/recompiler/ir/passes/NullTrace.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/ProgramCodec.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/fileSystem.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <map>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <fmt/format.h>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
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

Live::Switch g_pipeline_prefetch("KYTY_PIPELINE_PREFETCH", Live::ParseDefaultOff);
Live::Switch g_program_prefetch("KYTY_PIPELINE_PREFETCH_PROGRAMS", Live::ParseDefaultOff);
// KYTY_SRT_GPU_FILL (live, default 0 in this line, 2 in the Insomniac title presets; =1 without store offsets; =0 off; port of senaxx 0845beadf): a compute refresh on
// the command processor leaves a flat SRT slot that only the shader uses (the plan's
// gpu_fill_slots) to the GPU when its guest dword is GPU-written buffer memory: the walk writes
// a placeholder instead of faulting and draining the GPU, and RebindBuffers copies the dword
// from its buffer into the uploaded flat buffer on the GPU, between barriers, before the
// dispatch (the value the shader's own load would read when it runs).
// =2 also fills slots only buffer-store address operands use; the dispatch then takes a written
// binding of up to 16 MiB whose write range reads such a slot as written whole (a larger one reads
// the slot on the host for an exact range).
Live::Switch g_srt_gpu_fill("KYTY_SRT_GPU_FILL", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0') {
		return 0;
	}
	return std::clamp<int64_t>(std::strtoll(value, nullptr, 10), 0, 2);
});

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// PA_SU_SC_MODE_CNTL.POLY_MODE disables both per-face modes when it is zero.
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

// Ordinary VS keys must be linked after the compiled PS interface is known.
bool MayRelocatePixelParameters(const ShaderPixelInputInfo& pixel) {
	uint32_t flat_locations   = 0;
	uint32_t smooth_locations = 0;
	for (uint32_t input = 0; input < pixel.input_num; ++input) {
		const auto bit = 1u << (ShaderPixelParameterMappedLocation(pixel, input) & 31u);
		(ShaderPixelParameterIsFlat(pixel, input) ? flat_locations : smooth_locations) |= bit;
	}
	return (flat_locations & smooth_locations) != 0;
}

// Ordinary VS outputs follow the bound PS program's active inputs (PR #27); every other draw
// clears the aliases, so a reused input info never carries those of an earlier draw.
void LinkPixelParameters(bool link, ShaderVertexInputInfo& vertex, const ShaderPixelInputInfo& pixel) {
	if (link) {
		ShaderLinkVertexPixelParameters(vertex, pixel);
	} else {
		vertex.param_alias_mask = 0;
	}
}

// Entries from older revisions accumulate in the driver cache; start over beyond this size.
// KYTY_DRIVER_CACHE_MAX_MB (default 512, at most 3584: vkGetPipelineCacheData sizes past 4 GiB
// are refused) raises it per title. GT7's pipelines are huge (usa-ratt/KytyPS5-GT7 rounds 65 and
// 92 measured 0.8 to 3.75 GB): at 512 MB the cache was discarded at every start, so each session
// compiled everything again.
uint64_t DriverCacheFileCap() {
	static const uint64_t cap = [] {
		uint64_t mb = 512;
		if (const char* value = std::getenv("KYTY_DRIVER_CACHE_MAX_MB");
		    value != nullptr && value[0] != '\0') {
			mb = std::clamp<uint64_t>(std::strtoull(value, nullptr, 10), 64u, 3584u);
		}
		return mb * 1024ull * 1024ull;
	}();
	return cap;
}

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

// The stem of the cache files written under _PipelineCache. Under --rd, RenderDoc reports another
// Vulkan version (1.4.0) and wraps the device, so the program and driver caches are rejected and
// would be overwritten: a RenderDoc session keeps its own set ("<title>.renderdoc.*") and the
// normal play cache stays untouched.
static std::string PipelineCacheFileStem() {
	auto stem = PipelineCacheTitleId();
	if (!stem.empty() && Config::RenderDocEnabled()) {
		stem += ".renderdoc";
	}
	return stem;
}

// Under --rd: a cache file of the RenderDoc set that does not exist yet starts as a copy of the normal one (the
// journals and the driver cache hold nothing RenderDoc changes; a rejected copy is simply rebuilt), so a first
// RenderDoc session precompiles what the normal sessions found. The normal file is only read.
static void SeedRenderDocCacheFile(const std::filesystem::path& path, const char* suffix) {
	if (!Config::RenderDocEnabled()) {
		return;
	}
	std::error_code error;
	if (std::filesystem::exists(path, error)) {
		return;
	}
	const auto normal = path.parent_path() / (PipelineCacheTitleId() + suffix);
	if (!std::filesystem::exists(normal, error)) {
		return;
	}
	std::filesystem::copy_file(normal, path, error);
	std::printf("RenderDoc cache: %s %s from %s\n", error ? "could not seed" : "seeded",
	            Common::PathToString(path).c_str(), Common::PathToString(normal).c_str());
}

// A hint, not a decoder: whether a code word is shaped like IMAGE_BVH_INTERSECT_RAY or
// IMAGE_BVH64_INTERSECT_RAY (MIMG family, opcode 0xe6/0xe7 with the opcode's top bit in bit 0). A
// literal can look like one; the only use is ordering the shader precompile replay.
bool MayContainBvhIntersect(std::span<const uint32_t> code) {
	for (const auto word: code) {
		const auto opcode = (word >> 18u) & 0x7fu;
		if ((word >> 26u) == 0x3cu && (word & 1u) != 0 && (opcode == 0x66u || opcode == 0x67u)) {
			return true;
		}
	}
	return false;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

// Cache file layout: DriverCacheSignature text, XXH3-64 of the payload, payload
// (vkGetPipelineCacheData output). Fills `payload` and returns true only for a complete,
// matching file within DriverCacheFileCap().
bool ReadDriverCacheFile(const std::filesystem::path& file_path, const std::string& signature,
                         std::vector<uint8_t>& payload, bool log_invalid) {
	payload.clear();
	const auto   path = Common::PathToString(file_path);
	Common::File file(file_path, Common::File::Mode::Read);
	const auto   file_size = file.IsInvalid() ? 0 : file.Size();
	if (file_size < signature.size() + sizeof(uint64_t) || file_size > DriverCacheFileCap()) {
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
// leaves only the save at exit). A save needs at least MIN_NEW (32) pipelines created since the
// last one and no new pipeline for SETTLE (2 s), so it does not compete with a compile burst; then
// it runs when INTERVAL_S (60 s) have passed since the last save, or when a burst has ended
// (QUIET_S = 10 s without new pipelines, at least 15 s after the last save: level loads and area
// transitions). A steady trickle is saved after 3 intervals regardless of calm, and fewer than
// MIN_NEW new pipelines after 5 intervals.
struct DriverCacheSaveSettings {
	bool     enabled     = true;
	uint64_t min_new     = 32;
	uint64_t interval_ns = 60'000'000'000ull;
	uint64_t quiet_ns    = 10'000'000'000ull;
	uint64_t settle_ns   = 2'000'000'000ull;

	static const DriverCacheSaveSettings& Get() {
		static const DriverCacheSaveSettings settings = [] {
			DriverCacheSaveSettings s;
			s.enabled     = EnvU64("KYTY_PIPELINE_CACHE_SAVE", 1) != 0;
			s.min_new     = std::max<uint64_t>(1, EnvU64("KYTY_PIPELINE_CACHE_SAVE_MIN_NEW", 32));
			s.interval_ns = EnvU64("KYTY_PIPELINE_CACHE_SAVE_INTERVAL_S", 60) * 1'000'000'000ull;
			s.quiet_ns    = EnvU64("KYTY_PIPELINE_CACHE_SAVE_QUIET_S", 10) * 1'000'000'000ull;
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
	// Permutations published by the shader precompile replay (KYTY_SHADER_PRECOMPILE).
	std::atomic<uint64_t> replayed {0};
	// How pipelines were made, for the KYTY_PIPELINE_STATS line (pipelineStats.h).
	std::atomic<uint64_t> gfx_hit {0}, gfx_fast {0}, gfx_full {0}, gfx_other {0}, gfx_cp_ns {0};
	std::atomic<uint64_t> cs_hit {0}, cs_fast {0}, cs_full {0}, cs_cp_ns {0}, cs_prewarmed {0};
	std::atomic<uint64_t> known_misses {0};
	std::atomic<uint64_t> prewarm_built {0}, prewarm_ns {0};
	std::atomic<uint64_t> journal_built {0}, journal_skipped {0}, journal_ns {0}, journal_unpredicted {0};
	std::atomic<uint64_t> window_stall_max_ns {0}; // reset by each stats line
};
CompileTotals g_compile_totals;

// KYTY_SHADER_PRECOMPILE_STATS=1: permutations the precompile replay published, distinct
// permutations draws and dispatches bound (ProgramCache::Bind) and how many of those the replay had
// published, and the replay threads' CompileSPV phase times.
struct PermutationUse {
	std::atomic<uint64_t> replayed {0};
	std::atomic<uint64_t> bound {0};
	std::atomic<uint64_t> replay_bound {0};
	std::atomic<uint64_t> spv_calls {0}, spv_shrink_ns {0}, spv_validated {0}, spv_validate_ns {0};
	std::atomic<uint64_t> spv_validate_skipped {0}, spv_create_ns {0};
};
PermutationUse g_permutation_use;

// KYTY_SHADER_PRECOMPILE_USED=1: the permutation identities (PermutationIdentity, the
// translator-independent inputs) that draws and dispatches bound, in any run, kept in
// <journal>.used (a header, then 8-byte identities appended as permutations are first bound; a
// torn tail is dropped by the next load, which rewrites the file). The precompile replay takes these
// entries first (ShaderPrecompiler::Settings::used_entry).
class UsedPermutations {
public:
	static constexpr char Magic[16] = "KYTY PERMUSED 1";

	uint64_t Open(const std::filesystem::path& path) {
		std::scoped_lock lock(m_mutex);
		if (m_file != nullptr) return m_known.size();
		std::vector<uint8_t> bytes;
		if (std::FILE* in = std::fopen(Common::PathToString(path).c_str(), "rb"); in != nullptr) {
			std::array<uint8_t, 65536> chunk {};
			for (size_t got; (got = std::fread(chunk.data(), 1, chunk.size(), in)) != 0;) {
				bytes.insert(bytes.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(got));
			}
			std::fclose(in);
		}
		if (bytes.size() >= sizeof(Magic) && std::memcmp(bytes.data(), Magic, sizeof(Magic)) == 0) {
			for (size_t at = sizeof(Magic); at + sizeof(uint64_t) <= bytes.size(); at += sizeof(uint64_t)) {
				uint64_t identity = 0;
				std::memcpy(&identity, bytes.data() + at, sizeof(identity));
				m_known.insert(identity);
			}
		}
		m_loaded = m_known.size();
		m_file   = std::fopen(Common::PathToString(path).c_str(), "wb");
		if (m_file == nullptr) return m_loaded;
		std::fwrite(Magic, 1, sizeof(Magic), m_file);
		for (const auto identity: m_known) std::fwrite(&identity, sizeof(identity), 1, m_file);
		std::fflush(m_file);
		m_enabled.store(true, std::memory_order_release);
		return m_loaded;
	}
	[[nodiscard]] bool Enabled() const { return m_enabled.load(std::memory_order_relaxed); }
	// Loaded from the file (before this run added any).
	[[nodiscard]] bool WasUsed(uint64_t identity) const {
		std::scoped_lock lock(m_mutex);
		return m_known.contains(identity);
	}
	void Add(uint64_t identity) {
		std::scoped_lock lock(m_mutex);
		if (!m_known.insert(identity).second || m_file == nullptr) return;
		m_added++;
		std::fwrite(&identity, sizeof(identity), 1, m_file);
		std::fflush(m_file);
	}
	[[nodiscard]] uint64_t Added() const {
		std::scoped_lock lock(m_mutex);
		return m_added;
	}

private:
	mutable std::mutex           m_mutex;
	std::unordered_set<uint64_t> m_known;
	uint64_t                     m_loaded = 0;
	uint64_t                     m_added  = 0;
	std::FILE*                   m_file   = nullptr;
	std::atomic<bool>            m_enabled {false};
};
UsedPermutations g_used_permutations;

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
	AtomicMax(g_compile_totals.window_stall_max_ns, ns);
	HangTrace::RecordCompileStall(ns);
}

// Content hashes of the program cache's shader modules: XXH3-64 of the SPIR-V the program cache
// emitted (before CompileSPV's function-array shrink and cold salt). They key pipelines across
// processes (KYTY_PIPELINE_KNOWN) and let journaled pipelines find their modules again
// (KYTY_PIPELINE_JOURNAL). Published modules also carry their permutation's identity (the
// translator-independent inputs: guest code hash, static state, push-data cursor, specialization)
// and a hash of their descriptor bindings, so a journaled pipeline can find the module another
// translator revision made from the same inputs. A module handed to a background build is
// pinned; destroying a pinned module waits for its last unpin, which then destroys it.
class ShaderModuleRegistry {
public:
	void Add(vk::ShaderModule module, std::span<const uint32_t> spirv) {
		const auto hash = XXH3_64bits(spirv.data(), spirv.size_bytes());
		std::scoped_lock lock(m_mutex);
		m_modules[static_cast<VkShaderModule>(module)] = {.hash = hash, .words = spirv.size()};
		m_by_hash.try_emplace(hash, static_cast<VkShaderModule>(module));
	}
	// The module was published for the permutation `identity` with these descriptor bindings.
	void SetIdentity(vk::ShaderModule module, uint64_t identity, uint64_t bindings) {
		std::scoped_lock lock(m_mutex);
		const auto       found = m_modules.find(static_cast<VkShaderModule>(module));
		if (found == m_modules.end() || found->second.doomed) return;
		found->second.identity = identity;
		found->second.bindings = bindings;
		m_by_identity.try_emplace(identity, found->first);
	}
	// {identity, bindings hash} of a published module ({0, 0} if unknown).
	[[nodiscard]] std::pair<uint64_t, uint64_t> IdentityOf(vk::ShaderModule module) const {
		std::scoped_lock lock(m_mutex);
		const auto       found = m_modules.find(static_cast<VkShaderModule>(module));
		if (found == m_modules.end() || found->second.doomed) return {0, 0};
		return {found->second.identity, found->second.bindings};
	}
	// A live module published for this permutation identity with these bindings, pinned; null if none.
	[[nodiscard]] vk::ShaderModule PinIdentity(uint64_t identity, uint64_t bindings) {
		std::scoped_lock lock(m_mutex);
		const auto       found = m_by_identity.find(identity);
		if (found == m_by_identity.end()) return nullptr;
		auto& info = m_modules[found->second];
		if (info.bindings != bindings) return nullptr;
		info.pins++;
		return found->second;
	}
	[[nodiscard]] bool Contains(uint64_t hash) const {
		std::scoped_lock lock(m_mutex);
		return m_by_hash.contains(hash);
	}
	[[nodiscard]] uint64_t HashOf(vk::ShaderModule module) const {
		std::scoped_lock lock(m_mutex);
		const auto       found = m_modules.find(static_cast<VkShaderModule>(module));
		return found != m_modules.end() && !found->second.doomed ? found->second.hash : 0;
	}
	// The SPIR-V words the module was made from (0 if unknown); KYTY_COMPUTE_BACKGROUND_WORDS.
	[[nodiscard]] uint64_t WordsOf(vk::ShaderModule module) const {
		std::scoped_lock lock(m_mutex);
		const auto       found = m_modules.find(static_cast<VkShaderModule>(module));
		return found != m_modules.end() && !found->second.doomed ? found->second.words : 0;
	}
	// Pins this live module itself (false if unknown or doomed); Unpin releases it.
	[[nodiscard]] bool PinModule(vk::ShaderModule module) {
		std::scoped_lock lock(m_mutex);
		const auto       found = m_modules.find(static_cast<VkShaderModule>(module));
		if (found == m_modules.end() || found->second.doomed) return false;
		found->second.pins++;
		return true;
	}
	// A live module created from code with this hash, pinned until Unpin; null if none.
	[[nodiscard]] vk::ShaderModule Pin(uint64_t hash) {
		std::scoped_lock lock(m_mutex);
		const auto       found = m_by_hash.find(hash);
		if (found == m_by_hash.end()) return nullptr;
		m_modules[found->second].pins++;
		return found->second;
	}
	void Unpin(vk::Device device, vk::ShaderModule module) {
		{
			std::scoped_lock lock(m_mutex);
			const auto       found = m_modules.find(static_cast<VkShaderModule>(module));
			if (found == m_modules.end() || --found->second.pins != 0 || !found->second.doomed) return;
			m_modules.erase(found);
		}
		device.destroyShaderModule(module, nullptr);
	}
	void Destroy(vk::Device device, vk::ShaderModule module) {
		{
			std::scoped_lock lock(m_mutex);
			const auto       found = m_modules.find(static_cast<VkShaderModule>(module));
			if (found != m_modules.end()) {
				if (const auto indexed = m_by_identity.find(found->second.identity);
				    indexed != m_by_identity.end() && indexed->second == found->first) {
					m_by_identity.erase(indexed);
				}
				if (const auto indexed = m_by_hash.find(found->second.hash);
				    indexed != m_by_hash.end() && indexed->second == found->first) {
					m_by_hash.erase(indexed);
					// Another live module of the same code (a duplicate compile) takes over (rare).
					for (const auto& [other, info]: m_modules) {
						if (other != found->first && info.hash == found->second.hash && !info.doomed) {
							m_by_hash.emplace(info.hash, other);
							break;
						}
					}
				}
				if (found->second.pins != 0) {
					found->second.doomed = true;
					return;
				}
				m_modules.erase(found);
			}
		}
		device.destroyShaderModule(module, nullptr);
	}

private:
	struct Info {
		uint64_t hash     = 0;
		uint64_t words    = 0;
		uint64_t identity = 0;
		uint64_t bindings = 0;
		uint32_t pins     = 0;
		bool     doomed   = false;
	};
	mutable std::mutex                           m_mutex;
	std::unordered_map<VkShaderModule, Info>     m_modules;
	std::unordered_map<uint64_t, VkShaderModule> m_by_hash;
	std::unordered_map<uint64_t, VkShaderModule> m_by_identity;
};
ShaderModuleRegistry g_shader_modules;

// Cross-process identity of a graphics pipeline: its create info serialized with module content
// hashes (GraphicsPipelineSnapshot::Serialize), the layout signature appended, and the XXH3-64 of
// those words. key 0: no identity (an unhashed module, or a create info the snapshot cannot copy).
struct GraphicsContent {
	std::shared_ptr<GraphicsPipelineSnapshot> snapshot;
	std::vector<uint32_t>                     words; // snapshot words, then the layout signature
	uint32_t                                  snapshot_words = 0;
	uint64_t                                  key            = 0;
	// Per stage (snapshot order): the module's permutation identity and bindings hash (0 if unknown).
	std::vector<std::pair<uint64_t, uint64_t>> identities;
};

// Why a graphics pipeline gets no journal key (logged once per reason): which part of its create
// info GraphicsPipelineSnapshot::Capture or Serialize does not copy.
void NoteUnkeyedPipeline(const vk::GraphicsPipelineCreateInfo& info, bool serialize_failed) {
	std::string why;
	const auto  chain = [](const void* next) {
		std::string types;
		for (const auto* base = static_cast<const vk::BaseInStructure*>(next); base != nullptr; base = base->pNext) {
			types += fmt::format(" {}", static_cast<uint32_t>(base->sType));
		}
		return types;
	};
	if (serialize_failed) why += " module-without-hash";
	if (info.pMultisampleState != nullptr && info.pMultisampleState->pNext != nullptr)
		why += " multisample-pNext" + chain(info.pMultisampleState->pNext);
	if (info.pMultisampleState != nullptr && info.pMultisampleState->pSampleMask != nullptr) why += " sample-mask";
	if (info.pRasterizationState != nullptr && info.pRasterizationState->pNext != nullptr)
		why += " raster-pNext" + chain(info.pRasterizationState->pNext);
	if (info.pDepthStencilState != nullptr && info.pDepthStencilState->pNext != nullptr) why += " depth-pNext";
	if (info.pColorBlendState != nullptr && info.pColorBlendState->pNext != nullptr)
		why += " blend-pNext" + chain(info.pColorBlendState->pNext);
	if (info.pViewportState != nullptr && info.pViewportState->pNext != nullptr)
		why += " viewport-pNext" + chain(info.pViewportState->pNext);
	if (info.pTessellationState != nullptr) why += " tessellation";
	for (uint32_t i = 0; i < info.stageCount; i++) {
		if (info.pStages[i].pSpecializationInfo != nullptr) why += " specialization";
		if (info.pStages[i].pNext != nullptr) why += " stage-pNext" + chain(info.pStages[i].pNext);
	}
	why += " create-pNext" + chain(info.pNext);
	if (info.flags != vk::PipelineCreateFlags {}) why += fmt::format(" flags=0x{:x}", static_cast<uint32_t>(info.flags));
	static std::mutex                      mutex;
	static std::unordered_set<std::string> noted;
	{
		std::scoped_lock lock(mutex);
		if (noted.size() >= 32 || !noted.insert(why).second) return;
	}
	Log::WriteToConsoleAndLog(fmt::format("Pipeline journal: a pipeline without a key:{}\n", why));
}

GraphicsContent DescribeGraphicsPipeline(const vk::GraphicsPipelineCreateInfo& info,
                                         std::span<const uint32_t>             layout_signature) {
	GraphicsContent content;
	content.snapshot = GraphicsPipelineSnapshot::Capture(info, true);
	if (content.snapshot == nullptr) {
		NoteUnkeyedPipeline(info, false);
		return content;
	}
	if (!content.snapshot->Serialize(content.words,
	                                 [](vk::ShaderModule module) { return g_shader_modules.HashOf(module); })) {
		content.words.clear();
		NoteUnkeyedPipeline(info, true);
		return content;
	}
	content.snapshot_words = static_cast<uint32_t>(content.words.size());
	content.words.push_back(static_cast<uint32_t>(layout_signature.size()));
	content.words.insert(content.words.end(), layout_signature.begin(), layout_signature.end());
	content.key = XXH3_64bits(content.words.data(), content.words.size() * sizeof(uint32_t));
	if (content.key == 0) content.key = 1;
	const auto& created = content.snapshot->Info();
	for (uint32_t i = 0; i < created.stageCount; i++) {
		content.identities.push_back(g_shader_modules.IdentityOf(created.pStages[i].module));
	}
	return content;
}

// The same for a compute pipeline (module, entry point, required subgroup size); the layout follows
// from the module's bindings. 0: no identity.
uint64_t ComputePipelineKey(const vk::ComputePipelineCreateInfo& info) {
	const auto module_hash = g_shader_modules.HashOf(info.stage.module);
	if (module_hash == 0 || info.stage.pName == nullptr) return 0;
	uint32_t subgroup = 0;
	if (info.stage.pNext != nullptr) {
		const auto* base = static_cast<const vk::BaseInStructure*>(info.stage.pNext);
		if (base->sType != vk::StructureType::ePipelineShaderStageRequiredSubgroupSizeCreateInfo) return 0;
		subgroup = static_cast<const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo*>(info.stage.pNext)
		               ->requiredSubgroupSize;
	}
	std::vector<uint32_t> words {0x31534350u /* "PCS1" */, static_cast<uint32_t>(module_hash),
	                             static_cast<uint32_t>(module_hash >> 32u), subgroup,
	                             static_cast<uint32_t>(static_cast<VkPipelineCreateFlags>(info.flags)),
	                             static_cast<uint32_t>(static_cast<VkPipelineShaderStageCreateFlags>(info.stage.flags))};
	for (const char* c = info.stage.pName; *c != '\0'; ++c) words.push_back(static_cast<uint8_t>(*c));
	const auto key = XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
	return key == 0 ? 1 : key;
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

// Permutation churn report (stderr): once per second, the shaders that compiled new
// specializations of an already compiled source in that second (not first compiles, prefetches or
// journal replays), and once per shader when it reaches 8 such permutations, with the number of
// V#s KYTY_SPEC_UNMAPPED_NULL has nulled so far. KYTY_SPEC_CHURN_LOG=0 turns it off.
void NotePermutationChurn(uint64_t hash, size_t permutations) {
	static const bool enabled = [] {
		const auto* env = std::getenv("KYTY_SPEC_CHURN_LOG");
		return env == nullptr || std::strcmp(env, "0") != 0;
	}();
	if (!enabled) return;
	static std::mutex                                mutex;
	static std::vector<std::pair<uint64_t, uint32_t>> window;
	static std::unordered_set<uint64_t>              reported;
	static auto                                      window_start = std::chrono::steady_clock::now();
	std::string lines;
	{
		const std::lock_guard lock(mutex);
		const auto now = std::chrono::steady_clock::now();
		if (now - window_start >= std::chrono::seconds(1) && !window.empty()) {
			lines += "Kyty new permutations (last 1 s):";
			size_t shown = 0;
			for (const auto& [h, n]: window) {
				if (shown++ == 8) {
					lines += fmt::format(" +{} more", window.size() - 8);
					break;
				}
				lines += fmt::format(" {:016x}x{}", h, n);
			}
			lines += fmt::format(" (null V#s so far {})\n", ShaderRecompiler::IR::UnmappedBufferNulls());
			window.clear();
		}
		if (window.empty()) window_start = now;
		auto found = std::find_if(window.begin(), window.end(), [&](const auto& e) { return e.first == hash; });
		if (found == window.end()) {
			window.emplace_back(hash, 1u);
		} else {
			found->second++;
		}
		if (permutations >= 8 && reported.insert(hash).second) {
			lines += fmt::format("Kyty spec churn: shader {:016x} has {} permutations (null V#s so far {})\n",
			                     hash, permutations, ShaderRecompiler::IR::UnmappedBufferNulls());
		}
	}
	if (!lines.empty()) {
		std::fputs(lines.c_str(), stderr);
	}
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
		return ready;
	}
};

// KYTY_DROP_CS_HASHES=<hash>,... (live diagnostic, default unset): the dispatches of these compute
// shaders are dropped like programs without a translation (same-boot A/B of a shader's effect).
static std::array<std::atomic<uint64_t>, 8> g_drop_cs_list {};
static Live::Switch g_drop_cs_hashes("KYTY_DROP_CS_HASHES", [](const char* value) -> int64_t {
	size_t count = 0;
	for (const char* text = value; text != nullptr && *text != '\0' && count < g_drop_cs_list.size();) {
		char*      end  = nullptr;
		const auto hash = std::strtoull(text, &end, 16);
		if (end == text) break;
		g_drop_cs_list[count++].store(hash, std::memory_order_relaxed);
		text = *end == ',' ? end + 1 : end;
		if (*end != ',') break;
	}
	for (size_t i = count; i < g_drop_cs_list.size(); ++i) g_drop_cs_list[i].store(0, std::memory_order_relaxed);
	return static_cast<int64_t>(count);
});
static bool DropCsHashListed(uint64_t hash) {
	for (const auto& entry: g_drop_cs_list) {
		if (entry.load(std::memory_order_relaxed) == hash && hash != 0) return true;
	}
	return false;
}

// KYTY_READINESS_DIAG=1 (diagnostic, default off): a program whose resource tables read memory
// that cannot be synchronized is logged with every missing range and why it is not clean
// (LibKernel::Memory::DescribeGpuBackingForRead), at most 4 times per shader and 400 lines.
static void NoteReadinessFailure(const char* stage, uint64_t hash, const ShaderReadAttempt& attempt) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_READINESS_DIAG");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	if (!enabled) return;
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, uint32_t> per_hash;
	static uint32_t                               lines = 0;
	std::scoped_lock                              lock {mutex};
	if (lines >= 400u || per_hash[hash]++ >= 4u) return;
	for (size_t i = 0; i < attempt.count && lines < 400u; ++i, ++lines) {
		const auto& range = attempt.missing[i];
		Log::WriteToConsoleAndLog(fmt::format(
		    "ReadinessDiag: {} 0x{:016x} pass {} missing 0x{:x}+0x{:x} ({}/{}): {}\n", stage, hash,
		    PassCensus::CurrentKey(), range.address, range.size, i + 1, attempt.count,
		    LibKernel::Memory::DescribeGpuBackingForRead(range.address, range.size)));
	}
}

bool NativeDccEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DCC_GPU");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

// A strict SRT read that failed because no guest page can back the address: never-mapped
// addresses, and with KYTY_SRT_VARIANT_READS any address outside the GPU-mapped guest ranges. No
// synchronization could make such a read succeed, so the materialization would stop the emulator;
// it reads 0 instead, as the GPU reads an unmapped page (SrtWalker::InPlaceReadable does the same
// for the in-place reads).
bool ReadsUnmappedGuestMemory(uint64_t address, std::span<uint32_t> values) {
	const auto size = values.size_bytes();
	if (!ShaderRecompiler::IR::NeverMappedAddress(address, size) &&
	    (!ShaderRecompiler::GetCodegenOptions().srt_variant_reads ||
	     LibKernel::Memory::IsGpuMapped(address, size))) {
		return false;
	}
	std::fill(values.begin(), values.end(), 0u);
	Profiler::CountFrameEvent(Profiler::FrameEvent::SrtUnmappedReads);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "SRT: a resource read of unmapped address 0x{:x} (0x{:x} bytes) before a dispatch or "
		    "draw returns 0, as the GPU reads an unmapped page.\n",
		    address, size));
	}
	return true;
}

bool ReadShaderGuestMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
	const bool read = !values.empty() &&
	    LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(), values.size_bytes());
	if (!read && !values.empty() && ReadsUnmappedGuestMemory(address, values)) {
		return true;
	}
	// KYTY_SRT_STALE_IMAGE_READS: refused only for GPU-modified images a CPU write superseded.
	if (!read && !values.empty() &&
	    LibKernel::Memory::TryReadGpuBackingOverStaleImages(address, values.data(), values.size_bytes())) {
		return true;
	}
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
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_RESOURCE_REUSE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
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

// The code of each function a skipped S_SWAPPC_B64 program calls (ShaderRecompiler CallTarget), next
// to the program's own dump: original/callee_<stage>_<program hash>_<callee address>.bin. It holds
// the callee from its first instruction up to and including the first S_SETPC_B64 (or null-
// destination S_SWAPPC_B64) through the return pair, at most 4 KiB; a literal equal to that
// encoding would end it early. Only guest memory the GPU can read is copied: through the clean
// backing, else in place inside the GPU-mapped ranges; an unreadable callee logs a line instead.
void DumpCallTargets(const char* stage_name, uint64_t shader_hash,
                     std::span<const ShaderRecompiler::CallTarget> targets) {
	if (!Config::GraphicsDebugDumpEnabled() || targets.empty()) {
		return;
	}
	constexpr size_t MaxWords   = 1024; // 4 KiB
	constexpr size_t ChunkWords = 64;
	for (const auto& target: targets) {
		std::vector<uint32_t> words(MaxWords);
		size_t                read = 0;
		if ((target.address & 3u) == 0u) {
			while (read < MaxWords) {
				const auto chunk   = std::span(words).subspan(read, std::min(ChunkWords, MaxWords - read));
				const auto address = target.address + read * sizeof(uint32_t);
				if (!LibKernel::Memory::TryReadGpuCleanBacking(address, chunk.data(),
				                                              chunk.size_bytes())) {
					if (ShaderRecompiler::IR::NeverMappedAddress(address, chunk.size_bytes()) ||
					    !LibKernel::Memory::IsGpuMapped(address, chunk.size_bytes())) {
						break;
					}
					std::memcpy(chunk.data(), reinterpret_cast<const void*>(address),
					            chunk.size_bytes());
				}
				read += chunk.size();
			}
		}
		size_t end      = read;
		bool   returned = false;
		for (size_t i = 0; i < read; i++) {
			const auto word = words[i];
			const bool sop1 = (word >> 23u) == 0x17du;
			const auto op   = (word >> 8u) & 0xffu;
			const auto sdst = (word >> 16u) & 0x7fu;
			if (sop1 && (word & 0xffu) == target.return_sgpr &&
			    (op == 0x20u || (op == 0x21u && sdst == 125u))) {
				end      = i + 1;
				returned = true;
				break;
			}
		}
		if (end == 0) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_SRT_VARIANT_READS: callee 0x{:x} of {} shader 0x{:016x} is not readable "
			    "guest memory; not dumped.\n",
			    target.address, stage_name, shader_hash));
			continue;
		}
		const auto path = Config::GetShaderLogFolder() / "original" /
		                  fmt::format("callee_{}_{:016x}_{:012x}.bin", stage_name, shader_hash,
		                              target.address);
		Common::File::CreateDirectories(path.parent_path());
		Common::File file(path);
		if (file.IsInvalid()) {
			const auto path_text = Common::PathToString(path);
			LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
			continue;
		}
		file.Write(words.data(), end * sizeof(uint32_t));
		Log::WriteToConsoleAndLog(fmt::format(
		    "KYTY_SRT_VARIANT_READS: dumped callee 0x{:x} (s[{}:{}]) of {} shader 0x{:016x}: {} "
		    "bytes, {}.\n",
		    target.address, target.user_sgpr, target.user_sgpr + 1u, stage_name, shader_hash,
		    end * sizeof(uint32_t), returned ? "up to its return" : "no return within the dump"));
	}
}

// Reads guest memory for the call-table diagnostics: the clean backing, else in place inside the
// GPU-mapped ranges (as DumpCallTargets).
static bool ReadCallTableMemory(uint64_t address, void* data, size_t size) {
	if (size == 0 || LibKernel::Memory::TryReadGpuCleanBacking(address, data, size)) {
		return true;
	}
	if (ShaderRecompiler::IR::NeverMappedAddress(address, size) ||
	    !LibKernel::Memory::IsGpuMapped(address, size)) {
		return false;
	}
	std::memcpy(data, reinterpret_cast<const void*>(address), size);
	return true;
}

// The functions an S_SWAPPC_B64 program reaches through call tables (ShaderRecompiler CallTable),
// evaluated with the user data of the dispatch that compiles it: one log line per call with the
// table's size and its distinct non-null targets. Returns those targets for DumpCallTargets.
std::vector<ShaderRecompiler::CallTarget>
DescribeCallTables(const char* stage_name, const ShaderRecompiler::CompileOptions& options,
                   std::span<const ShaderRecompiler::CallTable> tables) {
	std::vector<ShaderRecompiler::CallTarget> targets;
	for (const auto& table: tables) {
		std::string           failure;
		std::vector<uint32_t> values;
		if (table.user_sgpr >= options.user_data_base &&
		    table.user_sgpr - options.user_data_base < options.user_data.size()) {
			const auto first = options.user_data.subspan(table.user_sgpr - options.user_data_base);
			values.assign(first.begin(), first.end());
		}
		for (const auto& step: table.steps) {
			if (step.base_index + 1u >= values.size()) {
				failure = "a base outside the loaded value";
				break;
			}
			const auto low     = static_cast<uint64_t>(values[step.base_index]);
			const auto high    = static_cast<uint64_t>(values[step.base_index + 1u]);
			auto       address = (step.buffer ? low | ((high & 0xffffu) << 32u)
			                                  : low | (high << 32u)) &
			                     0x0000ffffffffffffull;
			address += static_cast<uint64_t>(static_cast<int64_t>(step.offset));
			values.assign(step.dwords, 0u);
			if (!ReadCallTableMemory(address, values.data(), values.size() * sizeof(uint32_t))) {
				failure = fmt::format("unreadable 0x{:x}", address);
				break;
			}
		}
		if (failure.empty() && table.table_index + 3u >= values.size()) {
			failure = "no table descriptor";
		}
		uint64_t                     base = 0, bytes = 0, records = 0;
		std::map<uint64_t, uint32_t> callees;
		if (failure.empty()) {
			const auto* vsharp = &values[table.table_index];
			base               = (static_cast<uint64_t>(vsharp[0]) |
                    (static_cast<uint64_t>(vsharp[1] & 0xffffu) << 32u)) &
			       0x0000ffffffffffffull;
			const auto stride = (vsharp[1] >> 16u) & 0x3fffu;
			bytes = stride != 0 ? static_cast<uint64_t>(vsharp[2]) * stride : vsharp[2];
			const uint64_t record_bytes = uint64_t {1} << table.record_shift;
			const auto     words        = record_bytes / sizeof(uint32_t);
			records = record_bytes >= 8u ? std::min<uint64_t>(bytes, 16ull << 20u) / record_bytes : 0;
			std::vector<uint32_t> data(records * words);
			if (record_bytes < 8u || table.target_word + 1u >= words ||
			    !ReadCallTableMemory(base, data.data(), data.size() * sizeof(uint32_t))) {
				failure = fmt::format("unreadable table 0x{:x} (0x{:x} bytes)", base, bytes);
			} else {
				for (uint64_t r = 0; r < records; r++) {
					const auto low  = static_cast<uint64_t>(data[r * words + table.target_word]);
					const auto high = static_cast<uint64_t>(data[r * words + table.target_word + 1u]);
					const auto callee = (low | (high << 32u)) & 0x0000ffffffffffffull;
					if (callee != 0) {
						callees[callee]++;
					}
				}
			}
		}
		std::string list;
		for (const auto& [callee, count]: callees) {
			if (targets.size() < 64u) {
				targets.push_back({callee, 0u, table.return_sgpr});
			}
			if (list.size() < 600u) {
				list += fmt::format("{}0x{:x} x{}", list.empty() ? "" : ", ", callee, count);
			}
		}
		Log::WriteToConsoleAndLog(fmt::format(
		    "KYTY_SRT_VARIANT_READS: call table of {} shader 0x{:016x} at pc 0x{:x} (from s{}, {} "
		    "load step(s), record {} bytes, target dword {}): {}\n",
		    stage_name, options.shader_hash, table.call_pc, table.user_sgpr, table.steps.size(),
		    1u << table.record_shift, table.target_word,
		    failure.empty()
		        ? fmt::format("table 0x{:x}, 0x{:x} bytes, {} records, {} distinct callees: {}",
		                      base, bytes, records, callees.size(), list)
		        : failure));
	}
	return targets;
}

// Whether a compute program may call through S_SWAPPC_B64 (any word looks like a call): memoized
// per thread by program address and hash, so programs without calls pay one lookup per dispatch.
static bool ComputeProgramMayCall(const ShaderParams& params) {
	thread_local std::unordered_map<uint64_t, bool> known;
	const auto key = params.Base() ^ (params.hash * 0x9e3779b97f4a7c15ull) ^ params.code.size();
	const auto [entry, inserted] = known.try_emplace(key, false);
	if (inserted) {
		entry->second = ShaderRecompiler::Decoder::MayCallFunctions(params.code);
	}
	return entry->second;
}

// KYTY_SHADER_CALLS (default on): a compute program that calls functions through S_SWAPPC_B64 is
// replaced by its expansion (ShaderRecompiler::ShaderCallExpander), which then goes through the
// program cache like any program: `code` holds the expanded words and the key the expansion's
// digest. A program that does not expand keeps its code (and is skipped as before).
// KYTY_DUMP_PROGRAMS=<hash>,...;<folder> (diagnostics): the guest code and disassembly of those
// programs, once per hash, stage and user-data count, as their compile starts. Local files only.
static void DumpRequestedProgram(const ShaderParams& params, ShaderType stage) {
	static const auto request = [] {
		std::pair<std::set<uint64_t>, std::string> parsed;
		const char* value = std::getenv("KYTY_DUMP_PROGRAMS");
		if (value == nullptr) return parsed;
		std::string_view text(value);
		const auto       split = text.find(';');
		if (split == std::string_view::npos) return parsed;
		parsed.second = std::string(text.substr(split + 1));
		for (auto list = text.substr(0, split); !list.empty();) {
			const auto comma = list.find(',');
			parsed.first.insert(std::strtoull(std::string(list.substr(0, comma)).c_str(), nullptr, 16));
			if (comma == std::string_view::npos) break;
			list.remove_prefix(comma + 1);
		}
		return parsed;
	}();
	if (request.first.empty()) return;
	// A listed 1 logs the guest address and hash of every program translated.
	if (request.first.contains(1)) {
		const auto line = fmt::format("ProgramAddr: stage={} base=0x{:016x} hash=0x{:016x} words={}\n",
		                               static_cast<uint32_t>(stage), params.Base(), params.hash,
		                               params.code.size());
		LOGF("%s", line.c_str());
	}
	// A listed value also matches the guest address of the program (as diagnostics print it).
	if (request.first.empty() ||
	    (!request.first.contains(params.hash) && !request.first.contains(params.Base()))) {
		return;
	}
	static std::mutex                 mutex;
	static std::set<std::string>      done;
	const auto name = fmt::format("{:016x}_stage{}_ud{}", params.hash, static_cast<uint32_t>(stage),
	                              params.user_data_count);
	{
		std::scoped_lock lock(mutex);
		if (!done.insert(name).second) return;
	}
	std::error_code error;
	std::filesystem::create_directories(request.second, error);
	const auto base = std::filesystem::path(request.second) / name;
	if (std::FILE* file = std::fopen((base.string() + ".bin").c_str(), "wb"); file != nullptr) {
		std::fwrite(params.code.data(), sizeof(uint32_t), params.code.size(), file);
		std::fclose(file);
	}
	ShaderRecompiler::Decoder::Program decoded;
	ShaderRecompiler::Decoder::DecodeProgram(params.code, decoded, true);
	std::string text = fmt::format("user data ({}):", params.user_data_count);
	for (uint32_t i = 0; i < params.user_data_count; i++) text += fmt::format(" {:08x}", params.user_data[i]);
	text += "\n" + ShaderRecompiler::Decoder::ProgramToString(decoded);
	if (std::FILE* file = std::fopen((base.string() + ".txt").c_str(), "wb"); file != nullptr) {
		std::fwrite(text.data(), 1, text.size(), file);
		std::fclose(file);
	}
}
static ShaderRecompiler::ShaderCallExpander& CallExpander() {
	static ShaderRecompiler::ShaderCallExpander expander;
	// KYTY_CALL_PARTITION_MEMORY: _PipelineCache/<title>.partitions.
	static const bool memory = [] {
		if (ShaderRecompiler::CallPartitionMemoryEnabled()) {
			std::string title_id;
			if ((Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) && !title_id.empty()) ||
			    (Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) && !title_id.empty())) {
				std::error_code error;
				std::filesystem::create_directories("_PipelineCache", error);
				expander.SetPartitionMemory(
				    (std::filesystem::path("_PipelineCache") / (title_id + ".partitions")).string());
			}
		}
		return true;
	}();
	(void)memory;
	return expander;
}

// Where the callee bodies of each expansion handed out are (by its digest), for the background
// translation to name a callee that keeps an expansion from translating.
static std::mutex g_call_bodies_mutex;
static std::unordered_map<uint64_t, std::shared_ptr<const std::vector<ShaderRecompiler::Decoder::ShaderCallBody>>>
    g_call_bodies;

// KYTY_CALL_EXCLUDE_FAILED=1 (default off; adapted from usa-ratt/KytyPS5-GT7 c17e3ccdd): a function
// expansion whose translation gives up inside a placed callee body leaves that callee out of every
// later expansion on the compile path too (the background translation, KYTY_SHADER_CALLS_ASYNC,
// always did), and such a refusal is neither stored in the program cache nor taken from it: a
// stored one kept the expansion skipped in every later session, as nothing learned the callee.
static bool CallExcludeFailedEnabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_CALL_EXCLUDE_FAILED");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return on;
}

static std::shared_ptr<const std::vector<ShaderRecompiler::Decoder::ShaderCallBody>> ExpansionBodies(
    uint64_t call_expansion) {
	if (call_expansion == 0) return nullptr;
	std::scoped_lock lock(g_call_bodies_mutex);
	const auto       found = g_call_bodies.find(call_expansion);
	return found != g_call_bodies.end() ? found->second : nullptr;
}

// A function expansion with placed callee bodies (KYTY_CALL_EXCLUDE_FAILED).
static bool ExpansionHasBodies(uint64_t call_expansion) {
	if (!CallExcludeFailedEnabled()) return false;
	const auto bodies = ExpansionBodies(call_expansion);
	return bodies != nullptr && !bodies->empty();
}

// A refused translation of a function expansion: the callee whose body holds the failing pc is
// dropped. True when the refusal belongs to such an expansion (it is not stored then).
static bool ExcludeFailedCallee(uint64_t call_expansion, const ShaderRecompiler::TranslateResult& translated,
                                const ShaderRecompiler::CompileOptions& options) {
	if (!CallExcludeFailedEnabled()) return false;
	const auto bodies = ExpansionBodies(call_expansion);
	if (bodies == nullptr || bodies->empty()) return false;
	if (translated.skip_pc != UINT32_MAX) {
		for (const auto& body: *bodies) {
			if (translated.skip_pc >= body.begin && translated.skip_pc < body.end) {
				CallExpander().DropTarget(body.address,
				                          fmt::format("{} at +0x{:x} in CS 0x{:016x}", translated.skip_reason,
				                                      translated.skip_pc - body.begin, options.shader_hash));
				break;
			}
		}
	}
	return true;
}

// KYTY_SHADER_CALLS_SPLIT: the split program a renderer step selected, and the run a kernel's
// lookup produced (taken by the renderer).
thread_local const ShaderRecompiler::CallSplit::Code*          t_split_code = nullptr;
thread_local std::shared_ptr<ShaderRecompiler::CallSplit::Run> t_split_run;
thread_local std::shared_ptr<ShaderRecompiler::CallPartitionRun> t_partition_run;

// Why a compute lookup returned no program (PipelineCache::TakeEmptyProgramReason): the reason of
// the last empty result on this thread, and the translation's skip reason per shader hash (a skip
// verdict reloaded from the program cache has none unless the pass census retranslates it).
static thread_local uint64_t    t_empty_hash = 0;
static thread_local std::string t_empty_why;
static std::mutex                                g_skip_reason_mutex;
static std::unordered_map<uint64_t, std::string> g_skip_reasons;

static void NoteEmptyProgram(uint64_t hash, std::string why) {
	t_empty_hash = hash;
	t_empty_why  = std::move(why);
}

static void NoteSkipReason(uint64_t hash, const ShaderRecompiler::TranslateResult& translated) {
	std::string text = translated.skip_reason.empty() ? std::string("translation skip verdict")
	                                                  : translated.skip_reason;
	if (translated.skip_pc != UINT32_MAX) text += fmt::format(" (pc=0x{:x})", translated.skip_pc);
	std::scoped_lock lock(g_skip_reason_mutex);
	g_skip_reasons[hash] = std::move(text);
}

static std::string SkipReasonText(uint64_t hash) {
	std::scoped_lock lock(g_skip_reason_mutex);
	const auto       it = g_skip_reasons.find(hash);
	return it != g_skip_reasons.end() ? it->second : std::string("verdict reloaded from the program cache");
}

static std::string TakeEmptyProgramReasonImpl(uint64_t* hash) {
	if (hash != nullptr) *hash = std::exchange(t_empty_hash, 0);
	return std::exchange(t_empty_why, std::string());
}

// Programs with calls go through ExpandComputeCalls: KYTY_SHADER_CALLS (expansion, and splitting
// with KYTY_SHADER_CALLS_SPLIT) or KYTY_SHADER_CALLS_SPLIT alone (splitting only).
static bool CallProgramsHandled() {
	return ShaderRecompiler::GetCodegenOptions().shader_calls || ShaderRecompiler::CallSplit::Planner::Enabled();
}

// False: the dispatch is skipped (a split plan is being built).
static bool ExpandComputeCalls(ShaderParams& params, const ShaderComputeInputInfo& input_info) {
	const auto wave_size = input_info.wave_size;
	t_split_run.reset();
	t_partition_run.reset();
	if (const auto* split = std::exchange(t_split_code, nullptr); split != nullptr) {
		params.guest_base     = params.Base();
		params.code           = split->words;
		params.call_expansion = split->digest;
		return true;
	}
	if (!CallProgramsHandled() || !params.back_code.empty() || params.code.empty() ||
	    !ComputeProgramMayCall(params)) {
		return true;
	}
	const auto read_memory = [](uint64_t address, std::span<uint32_t> words) {
		return ReadCallTableMemory(address, words.data(), words.size_bytes());
	};
	if (ShaderRecompiler::CallSplit::Planner::Enabled()) {
		ShaderRecompiler::CompileOptions options;
		ShaderStageInputInfo             stage_input {};
		auto                             info = input_info;
		stage_input.compute = &info;
		options.stage       = ShaderType::Compute;
		options.wave_size   = wave_size;
		options.shader_hash = params.hash;
		options.user_data   = std::span(params.user_data).first(params.user_data_count);
		options.input_info  = stage_input;
		auto run = std::make_shared<ShaderRecompiler::CallSplit::Run>();
		switch (ShaderRecompiler::CallSplit::Planner::Instance().Plan(params.code, params.Base(), options,
		                                                              info, read_memory, *run)) {
			case ShaderRecompiler::CallSplit::Planner::Result::NotSplit: break;
			case ShaderRecompiler::CallSplit::Planner::Result::Pending: return false;
			case ShaderRecompiler::CallSplit::Planner::Result::Ready:
				params.guest_base     = params.Base();
				params.code           = run->start->words;
				params.call_expansion = run->start->digest;
				t_split_run           = std::move(run);
				return true;
		}
	}
	if (!ShaderRecompiler::GetCodegenOptions().shader_calls) {
		// KYTY_SHADER_CALLS_SPLIT alone: a program that is not split keeps its code (skipped as before).
		return true;
	}
	auto&                            expander = CallExpander();
	// KYTY_CALL_PARTITIONS: one dispatch per known target of a large table call (ShaderCalls.h);
	// KYTY_CALL_LEARN: one dispatch of the learned targets' expansion, through the same report set.
	if (ShaderRecompiler::CallPartitionsEnabled() || expander.learn) {
		ShaderRecompiler::CompileOptions part_options;
		part_options.stage       = ShaderType::Compute;
		part_options.wave_size   = wave_size;
		part_options.shader_hash = params.hash;
		part_options.user_data   = std::span(params.user_data).first(params.user_data_count);
		auto run = std::make_shared<ShaderRecompiler::CallPartitionRun>();
		if (expander.Partition(params.code, params.Base(), part_options, read_memory, *run)) {
			params.guest_base     = params.Base();
			params.code           = *run->parts.front().code;
			params.call_expansion = run->parts.front().digest;
			t_partition_run       = std::move(run);
			return true;
		}
	}
	ShaderRecompiler::CompileOptions options;
	options.stage       = ShaderType::Compute;
	options.wave_size   = wave_size;
	options.shader_hash = params.hash;
	options.user_data   = std::span(params.user_data).first(params.user_data_count);
	ShaderRecompiler::ShaderCallExpander::Expanded expanded;
	if (!expander.Expand(params.code, params.Base(), options, read_memory, expanded)) {
		return true;
	}
	params.guest_base     = params.Base();
	params.code           = *expanded.code;
	params.call_expansion = expanded.digest | 1u; // never 0
	{
		std::scoped_lock lock(g_call_bodies_mutex);
		g_call_bodies.try_emplace(params.call_expansion, expanded.bodies);
	}
	return true;
}

// The background translations of CallTranslations (PipelineCache::GetCallTranslationTotals).
struct CallTranslationCounters {
	std::atomic<uint64_t> queued {0}, finished {0}, words {0}, ms {0}, logged {0};
};
static CallTranslationCounters g_call_translation_totals;

// Permutations whose specialization the program cannot take (IR::UnsupportedResourceSpecialization),
// by source and specialization: skipped without translating again.
struct UnsupportedPermutations {
	static bool Contains(uint64_t key) {
		auto&            state = Get();
		std::scoped_lock lock(state.mutex);
		return state.keys.contains(key);
	}
	static void Add(uint64_t key, const ShaderRecompiler::CompileOptions& options, const char* why) {
		auto& state = Get();
		{
			std::scoped_lock lock(state.mutex);
			if (state.keys.size() >= 65536u || !state.keys.insert(key).second) return;
			if (state.logged.size() >= 256u || !state.logged.insert(options.shader_hash).second) return;
		}
		Log::WriteToConsoleAndLog(fmt::format("{} gave up hash=0x{:016x} for one specialization: {} (it is "
		                                      "skipped with it)\n",
		                                      options.dump_label != nullptr ? options.dump_label : "Shader",
		                                      options.shader_hash, why));
	}

private:
	struct State {
		std::mutex                   mutex;
		std::unordered_set<uint64_t> keys;
		std::unordered_set<uint64_t> logged;
	};
	static State& Get() {
		static State state;
		return state;
	}
};

// KYTY_SHADER_CALLS_ASYNC (default on): an expanded call program can be large (Miles Morales'
// ray-tracing kernels: about 160K words) and its translation takes seconds, so it is translated on
// a background thread and its dispatches are skipped until the translation is ready; the compile
// path then takes that translation instead of translating (CompileAndPublish). The translation is a
// function of the code, the static stage state and the user-data count, which the id covers.
class CallTranslations {
public:
	enum class State { Missing, Pending, Ready };

	static bool Enabled() {
		static const bool enabled = [] {
			const auto* value = std::getenv("KYTY_SHADER_CALLS_ASYNC");
			return value == nullptr || std::strcmp(value, "0") != 0;
		}();
		return enabled;
	}
	static CallTranslations& Instance() {
		static CallTranslations instance;
		return instance;
	}
	static uint64_t Id(uint64_t hash, uint32_t user_data_count, std::span<const uint32_t> static_state) {
		return XXH3_64bits_withSeed(static_state.data(), static_state.size_bytes(),
		                            hash ^ (uint64_t {user_data_count} << 56u));
	}

	State Query(uint64_t id) {
		std::scoped_lock lock(m_mutex);
		const auto entry = m_entries.find(id);
		if (entry == m_entries.end()) return State::Missing;
		return entry->second.done ? State::Ready : State::Pending;
	}
	void Queue(uint64_t id, const ShaderParams& params, const ShaderComputeInputInfo& info) {
		auto job        = std::make_unique<Job>();
		job->id         = id;
		job->hash       = params.hash;
		job->code.assign(params.code.begin(), params.code.end());
		job->user_data.assign(params.user_data.begin(), params.user_data.begin() + params.user_data_count);
		job->info       = info;
		job->info.stage = {};
		{
			std::scoped_lock lock(g_call_bodies_mutex);
			if (const auto bodies = g_call_bodies.find(params.call_expansion); bodies != g_call_bodies.end()) {
				job->bodies = bodies->second;
			}
		}
		{
			std::scoped_lock lock(m_mutex);
			if (!m_entries.try_emplace(id).second) return;
			m_jobs.push_back(std::move(job));
			g_call_translation_totals.queued.fetch_add(1, std::memory_order_relaxed);
			// KYTY_SHADER_CALLS_THREADS (default 3): call-split programs come in hundreds.
			static const size_t threads = [] {
				const auto* value = std::getenv("KYTY_SHADER_CALLS_THREADS");
				return std::clamp<size_t>(value != nullptr ? std::strtoull(value, nullptr, 0) : 3u, 1u, 16u);
			}();
			if (m_threads.size() < std::min(threads, m_jobs.size() + m_threads.size())) {
				m_threads.emplace_back([this] { Run(); });
			}
		}
		m_wake.notify_one();
	}
	// The ready translation of `id` (once; a later compile of the source translates again).
	std::optional<ShaderRecompiler::TranslateResult> Take(uint64_t id) {
		std::scoped_lock lock(m_mutex);
		const auto entry = m_entries.find(id);
		if (entry == m_entries.end() || !entry->second.done || !entry->second.result) return {};
		auto result = std::move(entry->second.result);
		entry->second.result.reset();
		return result;
	}
	~CallTranslations() {
		{
			std::scoped_lock lock(m_mutex);
			m_stop = true;
		}
		m_wake.notify_all();
		for (auto& thread: m_threads) {
			if (thread.joinable()) thread.join();
		}
	}

private:
	struct Job {
		uint64_t               id   = 0;
		uint64_t               hash = 0;
		std::vector<uint32_t>  code;
		std::vector<uint32_t>  user_data;
		ShaderComputeInputInfo info {};
		std::shared_ptr<const std::vector<ShaderRecompiler::Decoder::ShaderCallBody>> bodies;
	};
	struct Entry {
		bool                                             done = false;
		std::optional<ShaderRecompiler::TranslateResult> result;
	};

	void Run() {
		for (;;) {
			std::unique_ptr<Job> job;
			{
				std::unique_lock lock(m_mutex);
				m_wake.wait(lock, [&] { return m_stop || !m_jobs.empty(); });
				if (m_stop) return;
				job = std::move(m_jobs.front());
				m_jobs.pop_front();
			}
			ShaderRecompiler::CompileOptions options;
			ShaderStageInputInfo             stage_input {};
			stage_input.compute = &job->info;
			options.stage       = ShaderType::Compute;
			options.shader_hash = job->hash;
			options.user_data   = job->user_data;
			options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
			options.early_dump  = options.dump_ir;
			options.dump_label  = "ShaderRecompiler CS";
			options.input_info  = stage_input;
			options.wave_size   = job->info.wave_size;
			const auto begin    = std::chrono::steady_clock::now();
			auto       result   = ShaderRecompiler::TranslateProgram(job->code, options);
			const auto ms       = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - begin)
			                    .count();
			// The first few go to the console; the rest (call-split programs come in hundreds) only to
			// the log file, counted in the 10 s "RT:" status line (callSplitDispatch.cpp).
			// KYTY_SHADER_CALLS_LOG=1 prints every one to the console.
			{
				static const bool every = [] {
					const auto* value = std::getenv("KYTY_SHADER_CALLS_LOG");
					return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
				}();
				constexpr uint64_t ConsoleLines = 4;
				const auto         line         = fmt::format(
                    "KYTY_SHADER_CALLS: CS shader 0x{:016x}: expansion of {} words translated in the "
                    "background in {} ms{}.\n",
                    job->hash, job->code.size(), ms, result.skip_dispatch ? " (its dispatches are skipped)" : "");
				const auto n = g_call_translation_totals.logged.fetch_add(1, std::memory_order_relaxed);
				if (every || result.skip_dispatch || n < ConsoleLines) {
					Log::WriteToConsoleAndLog(line);
				} else {
					Log::Write(line);
				}
				if (!every && n + 1u == ConsoleLines) {
					Log::WriteToConsoleAndLog(
					    "KYTY_SHADER_CALLS: further background translations go to the log file only; the "
					    "\"RT:\" status line counts them every 10 s (KYTY_SHADER_CALLS_LOG=1 prints each)\n");
				}
				g_call_translation_totals.words.fetch_add(job->code.size(), std::memory_order_relaxed);
				g_call_translation_totals.ms.fetch_add(static_cast<uint64_t>(ms), std::memory_order_relaxed);
				g_call_translation_totals.finished.fetch_add(1, std::memory_order_relaxed);
			}
			// A callee that keeps it from translating is left out of the next expansions.
			if (result.skip_dispatch && result.skip_pc != UINT32_MAX && job->bodies != nullptr) {
				for (const auto& body: *job->bodies) {
					if (result.skip_pc >= body.begin && result.skip_pc < body.end) {
						CallExpander().DropTarget(
						    body.address, fmt::format("{} at +0x{:x} in CS 0x{:016x}", result.skip_reason,
						                              result.skip_pc - body.begin, job->hash));
						break;
					}
				}
			}
			std::scoped_lock lock(m_mutex);
			auto& entry  = m_entries[job->id];
			entry.done   = true;
			entry.result = std::move(result);
		}
	}

	std::mutex                          m_mutex;
	std::condition_variable             m_wake;
	std::deque<std::unique_ptr<Job>>    m_jobs;
	std::unordered_map<uint64_t, Entry> m_entries;
	std::vector<std::thread>            m_threads;
	bool                                m_stop = false;
};

// KYTY_ASYNC_TRANSLATION=1 (default off, live; adapted from Senaxx's GT7 fork a3d1eba5c): a new
// vertex or pixel shader of a draw that may be skipped (RenderExecutor's register-level version of
// the KYTY_PIPELINE_SKIP refusals, plus the fork's rule that the depth test is on) is translated on
// a worker thread while the draw is skipped, as a draw is skipped while its pipeline builds. The
// translation is a function of the guest code and the ProgramKey (KYTY_TRANSLATION_CACHE relies on
// the same), so the worker gets copies of the code, the user data and the stage input; the
// materialization and the SPIR-V emission, which read guest memory, stay on the draw's thread at
// the next draw of the shader, which takes the finished translation (CompileAndPublish).
// KYTY_ASYNC_TRANSLATION_THREADS=<n> (default 2, 1..8, startup): the worker count.
static Live::Switch g_async_translation("KYTY_ASYNC_TRANSLATION", Live::ParseDefaultOff); // live

// KYTY_PIPELINE_SKIP_ORDER=1 (default off, startup): see the skip builder in the PipelineCache
// constructor; the translation workers take the newest job first as well.
static bool SkipOrderEnabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_PIPELINE_SKIP_ORDER");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return on;
}

// Set by GetGraphicsPrograms around one draw's program lookups: whether its new VS/PS translations
// may go to the workers (the draw is skipped until they are ready).
thread_local bool t_defer_translation = false;
// Set by CompileAndPublish when it returned because a translation is on the workers, or when it
// took a finished one.
thread_local bool t_translation_pending = false;
thread_local bool t_translation_taken   = false;
// PipelineCache::AllowDeferredTranslation: the request for the next GetGraphicsPrograms.
thread_local bool t_defer_requested = false;

class GraphicsTranslations {
public:
	enum class Poll { Ready, Pending };

	static GraphicsTranslations& Instance() {
		static GraphicsTranslations instance;
		return instance;
	}

	struct Job {
		uint64_t                          id   = 0;
		uint64_t                          hash = 0;
		ShaderType                        stage = ShaderType::Vertex;
		std::vector<uint32_t>             code;
		std::vector<uint32_t>             back_code;
		std::vector<uint32_t>             user_data;
		ShaderVertexInputInfo             vertex {};
		ShaderPixelInputInfo              pixel {};
		ShaderRecompiler::CompileOptions  options;
		uint64_t                          queued_ns = 0;
	};

	[[nodiscard]] bool Empty() const { return m_count.load(std::memory_order_relaxed) == 0; }

	// The finished translation of `id` (taken once), or nothing: `pending` tells whether it is
	// still on the workers. With `make` (a draw that may wait by being skipped), a missing one is
	// queued and reported pending.
	std::optional<ShaderRecompiler::TranslateResult> TakeOrQueue(uint64_t id, bool& pending,
	                                                            const std::function<std::unique_ptr<Job>()>* make) {
		pending = false;
		std::unique_lock lock(m_mutex);
		if (const auto entry = m_entries.find(id); entry != m_entries.end()) {
			if (!entry->second.done) {
				if (make != nullptr) entry->second.discard = false; // wanted again
				pending = true;
				return std::nullopt;
			}
			auto result = std::move(entry->second.result);
			const auto waited_ms = static_cast<double>(NowNs() - entry->second.queued_ns) / 1e6;
			const auto hash      = entry->second.hash;
			const auto stage     = entry->second.stage;
			m_entries.erase(entry);
			m_count.store(m_entries.size(), std::memory_order_relaxed);
			const auto taken = ++m_taken;
			const auto skips = m_skips.load(std::memory_order_relaxed);
			lock.unlock();
			if (taken <= 32) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Async translation: {} 0x{:016x} ready after {:.0f} ms ({} draws skipped so far "
				    "while translating)\n",
				    stage == ShaderType::Pixel ? "PS" : "VS", hash, waited_ms, skips));
			}
			return result;
		}
		if (make == nullptr) return std::nullopt;
		auto job = (*make)();
		if (job == nullptr) return std::nullopt;
		job->id        = id;
		job->queued_ns = NowNs();
		auto& entry     = m_entries[id];
		entry.hash      = job->hash;
		entry.stage     = job->stage;
		entry.queued_ns = job->queued_ns;
		m_count.store(m_entries.size(), std::memory_order_relaxed);
		m_jobs.push_back(std::move(job));
		static const size_t threads = std::clamp<size_t>(EnvU64Local("KYTY_ASYNC_TRANSLATION_THREADS", 2), 1u, 8u);
		if (m_threads.size() < threads) {
			if (m_threads.empty()) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Async translation: new VS/PS of skippable draws translate on up to {} worker "
				    "threads (KYTY_ASYNC_TRANSLATION_THREADS)\n", threads));
			}
			m_threads.emplace_back([this] { Run(); });
		}
		lock.unlock();
		m_wake.notify_one();
		pending = true;
		return std::nullopt;
	}

	// A draw that may not wait translated the source itself: the worker's copy is not needed.
	void Drop(uint64_t id) {
		std::scoped_lock lock(m_mutex);
		if (const auto entry = m_entries.find(id); entry != m_entries.end()) {
			if (entry->second.done) {
				m_entries.erase(entry);
				m_count.store(m_entries.size(), std::memory_order_relaxed);
			} else {
				entry->second.discard = true;
			}
		}
	}

	void NoteSkip() { m_skips.fetch_add(1, std::memory_order_relaxed); }
	[[nodiscard]] uint64_t Skips() const { return m_skips.load(std::memory_order_relaxed); }

	// Shaders whose programs refuse skipping (PipelineCache::NoteAsyncTranslationRefused).
	void Refuse(uint64_t hash) {
		std::scoped_lock lock(m_mutex);
		if (m_refused.size() < 65536u && m_refused.insert(hash).second && m_refused.size() <= 8u) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Async translation: shader 0x{:016x} refuses skipping; its draws translate at once\n", hash));
		}
	}
	[[nodiscard]] bool Refused(uint64_t hash) {
		std::scoped_lock lock(m_mutex);
		return m_refused.contains(hash);
	}

	~GraphicsTranslations() {
		{
			std::scoped_lock lock(m_mutex);
			m_stop = true;
			m_jobs.clear();
		}
		m_wake.notify_all();
		for (auto& thread: m_threads) {
			if (thread.joinable()) thread.join();
		}
	}

private:
	struct Entry {
		bool                                             done = false;
		bool                                             discard = false;
		uint64_t                                         hash = 0;
		ShaderType                                       stage = ShaderType::Vertex;
		uint64_t                                         queued_ns = 0;
		std::optional<ShaderRecompiler::TranslateResult> result;
	};

	static uint64_t NowNs() {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                 std::chrono::steady_clock::now().time_since_epoch())
		                                 .count());
	}
	static uint64_t EnvU64Local(const char* name, uint64_t fallback) {
		const char* value = std::getenv(name);
		return value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 0) : fallback;
	}

	void Run() {
		if (SkipOrderEnabled()) Common::LowerCurrentThreadPriority();
		for (;;) {
			std::unique_ptr<Job> job;
			{
				std::unique_lock lock(m_mutex);
				m_wake.wait(lock, [&] { return m_stop || !m_jobs.empty(); });
				if (m_stop) return;
				// KYTY_PIPELINE_SKIP_ORDER: newest first, the scene on screen before the backlog.
				if (SkipOrderEnabled()) {
					job = std::move(m_jobs.back());
					m_jobs.pop_back();
				} else {
					job = std::move(m_jobs.front());
					m_jobs.pop_front();
				}
			}
			auto result = ShaderRecompiler::TranslateProgram(job->code, job->options);
			std::scoped_lock lock(m_mutex);
			if (const auto entry = m_entries.find(job->id); entry != m_entries.end()) {
				if (entry->second.discard) {
					m_entries.erase(entry);
					m_count.store(m_entries.size(), std::memory_order_relaxed);
				} else {
					entry->second.done   = true;
					entry->second.result = std::move(result);
				}
			}
		}
	}

	std::mutex                          m_mutex;
	std::condition_variable             m_wake;
	std::deque<std::unique_ptr<Job>>    m_jobs;
	std::unordered_map<uint64_t, Entry> m_entries;
	std::atomic<size_t>                 m_count {0};
	std::atomic<uint64_t>               m_skips {0};
	uint64_t                            m_taken = 0;
	std::unordered_set<uint64_t>        m_refused;
	std::vector<std::thread>            m_threads;
	bool                                m_stop = false;
};

// KYTY_DUMP_SPIRV_MIN_WORDS=<words>;<folder> (diagnostics, local files only): the emitted SPIR-V of
// every permutation of at least <words> words, as <folder>/<stage>_<hash>_<words>.spv (what makes the
// slow driver compiles large).
void DumpLargeSpirv(const char* stage_name, uint64_t hash, const std::vector<uint32_t>& spirv) {
	static const auto request = [] {
		std::pair<uint64_t, std::string> parsed {0, {}};
		const char* value = std::getenv("KYTY_DUMP_SPIRV_MIN_WORDS");
		if (value == nullptr) return parsed;
		const std::string_view text(value);
		const auto             split = text.find(';');
		if (split == std::string_view::npos) return parsed;
		parsed.first  = std::strtoull(std::string(text.substr(0, split)).c_str(), nullptr, 10);
		parsed.second = std::string(text.substr(split + 1));
		return parsed;
	}();
	if (request.first == 0 || spirv.size() < request.first) return;
	std::error_code error;
	std::filesystem::create_directories(request.second, error);
	const auto path = std::filesystem::path(request.second) /
	                  fmt::format("{}_{:016x}_{}.spv", stage_name, hash, spirv.size());
	if (std::FILE* file = std::fopen(path.string().c_str(), "wb"); file != nullptr) {
		std::fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), file);
		std::fclose(file);
	}
}

// KYTY_SHADER_DUMP=<folder>;<hash>[,<hash>...] or <folder>;all, ';' or '@' (diagnostics, local files only): the
// selected guest shaders bypass the persistent program cache (dump_ir) and every emitted
// permutation is written as <folder>/<stage>_<hash>_<spirv hash>.rdna2 (decoded guest code),
// .ir.txt (native IR and bindings) and .spv. A 32-bit hash matches either half of the 64-bit one.
// Default off.
struct ShaderDumpRequest {
	std::string           folder;
	std::vector<uint64_t> hashes;
	bool                  all = false;
};

const ShaderDumpRequest& ShaderDumpSettings() {
	static const ShaderDumpRequest request = [] {
		ShaderDumpRequest parsed;
		const char*       value = std::getenv("KYTY_SHADER_DUMP");
		if (value == nullptr) return parsed;
		const std::string text(value);
		const auto        split = text.find_first_of(";@");
		if (split == std::string::npos) return parsed;
		parsed.folder = text.substr(0, split);
		std::string list = text.substr(split + 1);
		if (list == "all") {
			parsed.all = true;
			return parsed;
		}
		size_t at = 0;
		while (at < list.size()) {
			const auto next = list.find(',', at);
			const auto item = list.substr(at, next == std::string::npos ? std::string::npos : next - at);
			if (!item.empty()) parsed.hashes.push_back(std::strtoull(item.c_str(), nullptr, 16));
			if (next == std::string::npos) break;
			at = next + 1;
		}
		return parsed;
	}();
	return request;
}

bool ShaderDumpSelected(uint64_t hash) {
	const auto& request = ShaderDumpSettings();
	if (request.folder.empty()) return false;
	return request.all || std::ranges::find(request.hashes, hash) != request.hashes.end() ||
	       std::ranges::find(request.hashes, hash & 0xffffffffull) != request.hashes.end() ||
	       std::ranges::find(request.hashes, hash >> 32u) != request.hashes.end();
}

void DumpSelectedShader(const char* stage_name, uint64_t hash, const std::string& decoded,
                        const std::string& ir, const std::vector<uint32_t>& spirv) {
	if (!ShaderDumpSelected(hash)) return;
	const auto& request = ShaderDumpSettings();
	std::error_code error;
	std::filesystem::create_directories(request.folder, error);
	const auto spirv_hash = XXH3_64bits(spirv.data(), spirv.size() * sizeof(uint32_t));
	const auto base       = std::filesystem::path(request.folder) /
	                  fmt::format("{}_{:016x}_{:016x}", stage_name, hash, spirv_hash);
	const auto write = [&](const char* suffix, const void* data, size_t size) {
		if (size == 0) return;
		auto path = base;
		path += suffix;
		if (std::FILE* file = std::fopen(path.string().c_str(), "wb"); file != nullptr) {
			std::fwrite(data, 1, size, file);
			std::fclose(file);
		}
	};
	write(".rdna2", decoded.data(), decoded.size());
	write(".ir.txt", ir.data(), ir.size());
	write(".spv", spirv.data(), spirv.size() * sizeof(uint32_t));
}

// KYTY_SPIRV_TRY=<flags>@<hash>[,<hash>...][+<flags>@<hashes>...] (diagnostics, default off): the
// modules given to the driver for the listed guest shaders are edited (the program cache keeps the
// emitted words). Flags: z drops the SignedZeroInfNanPreserve execution mode and capability; c drops
// the Coherent decoration of the LDS device buffer (lds_dwords / lds_qwords).
struct SpirvTryGroup {
	std::string           flags;
	std::vector<uint64_t> hashes;
};

const std::vector<SpirvTryGroup>& SpirvTrySettings() {
	static const std::vector<SpirvTryGroup> groups = [] {
		std::vector<SpirvTryGroup> parsed;
		const char*                value = std::getenv("KYTY_SPIRV_TRY");
		if (value == nullptr) return parsed;
		std::string text(value);
		size_t      at = 0;
		while (at < text.size()) {
			auto next = text.find('+', at);
			auto item = text.substr(at, next == std::string::npos ? std::string::npos : next - at);
			const auto split = item.find('@');
			if (split != std::string::npos) {
				SpirvTryGroup group;
				group.flags = item.substr(0, split);
				std::string list = item.substr(split + 1);
				size_t      p    = 0;
				while (p < list.size()) {
					const auto comma = list.find(',', p);
					const auto h = list.substr(p, comma == std::string::npos ? std::string::npos : comma - p);
					if (!h.empty()) group.hashes.push_back(std::strtoull(h.c_str(), nullptr, 16));
					if (comma == std::string::npos) break;
					p = comma + 1;
				}
				parsed.push_back(std::move(group));
			}
			if (next == std::string::npos) break;
			at = next + 1;
		}
		return parsed;
	}();
	return groups;
}

std::string SpirvTryFlags(uint64_t hash) {
	std::string flags;
	for (const auto& group: SpirvTrySettings()) {
		if (std::ranges::find(group.hashes, hash) != group.hashes.end()) flags += group.flags;
	}
	return flags;
}

// Edits a copy of `code` per `flags`; empty when nothing changed.
std::vector<uint32_t> ApplySpirvTry(std::span<const uint32_t> code, const std::string& flags,
                                    uint64_t hash) {
	if (flags.empty() || code.size() < 5) return {};
	const bool drop_szinp    = flags.find('z') != std::string::npos;
	const bool drop_coherent = flags.find('c') != std::string::npos;
	std::unordered_set<uint32_t> lds_ids;
	for (size_t i = 5; i < code.size();) {
		const uint32_t count = code[i] >> 16u;
		const uint32_t op    = code[i] & 0xffffu;
		if (count == 0 || i + count > code.size()) return {};
		if (op == 5 /* OpName */ && count >= 3) {
			const char* name = reinterpret_cast<const char*>(&code[i + 2]);
			const auto  max  = (count - 2) * 4;
			const std::string_view text(name, strnlen(name, max));
			if (text == "lds_dwords" || text == "lds_qwords") lds_ids.insert(code[i + 1]);
		}
		i += count;
	}
	std::vector<uint32_t> out(code.begin(), code.begin() + 5);
	out.reserve(code.size());
	uint32_t removed = 0;
	for (size_t i = 5; i < code.size();) {
		const uint32_t count = code[i] >> 16u;
		const uint32_t op    = code[i] & 0xffffu;
		bool           drop  = false;
		if (drop_szinp && op == 16 /* OpExecutionMode */ && count >= 3 && code[i + 2] == 4461) drop = true;
		if (drop_szinp && op == 17 /* OpCapability */ && count == 2 && code[i + 1] == 4466) drop = true;
		if (drop_coherent && op == 71 /* OpDecorate */ && count == 3 && code[i + 2] == 23 &&
		    lds_ids.contains(code[i + 1])) {
			drop = true;
		}
		if (drop) {
			removed++;
		} else {
			out.insert(out.end(), code.begin() + static_cast<std::ptrdiff_t>(i),
			           code.begin() + static_cast<std::ptrdiff_t>(i + count));
		}
		i += count;
	}
	if (removed == 0) return {};
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
		LOGF("KYTY_SPIRV_TRY: shader 0x%016" PRIx64 " flags %s: %u instructions removed\n", hash,
		     flags.c_str(), removed);
	}
	return out;
}

// KYTY_SPIRV_VARIANT=<flags>@<hash>[,<hash>...][+<flags>@<hashes>...] or <flags>@all (diagnostic,
// default off): each '+' group gives its listed guest shaders an edited module under an id of its
// own; draws use group n's copies while the live switch KYTY_SPIRV_VARIANT_LIVE is n (1-based,
// renderDraw.cpp; 0 = the programs themselves), so one boot can alternate them (same-boot A/B of a
// translation change without a program-cache change). Flags:
// those of KYTY_SPIRV_TRY (z, c) and
//   e  adds EarlyFragmentTests to a fragment shader that has neither it nor DepthReplacing;
//   t  replaces the result of every image sample, fetch, gather and LOD query with zeros (texture
//      cost probe: pictures are wrong);
//   k  removes every OpKill / OpTerminateInvocation / OpDemoteToHelperInvocation (pictures may be wrong);
//   o  reads every buffer byte offset (EmitMemoryOffsets' 8-bit shader_data fields) as 0: constant
//      buffer indices (pictures are right only where every bound buffer's offset is 0);
//   p  drops every NoContraction decoration ("precise": KYTY_MAD_MODE's unfused position math);
//   x  removes the body of a module with one function (cost probe: outputs undefined);
//   d  drops the f32 denormal-flush selects (results differ only for denormal inputs);
//   F  limits the group to fragment shaders.
struct SpirvVariantGroup {
	std::string           flags;
	std::vector<uint64_t> hashes;
	bool                  all = false;
};

const std::vector<SpirvVariantGroup>& SpirvVariantSettings() {
	static const std::vector<SpirvVariantGroup> groups = [] {
		std::vector<SpirvVariantGroup> parsed;
		const char*                    value = std::getenv("KYTY_SPIRV_VARIANT");
		if (value == nullptr) return parsed;
		std::string text(value);
		size_t      at = 0;
		while (at < text.size()) {
			auto       next  = text.find('+', at);
			auto       item  = text.substr(at, next == std::string::npos ? std::string::npos : next - at);
			const auto split = item.find('@');
			if (split != std::string::npos) {
				SpirvVariantGroup group;
				group.flags      = item.substr(0, split);
				std::string list = item.substr(split + 1);
				if (list == "all") {
					group.all = true;
				} else {
					size_t p = 0;
					while (p < list.size()) {
						const auto comma = list.find(',', p);
						const auto h = list.substr(p, comma == std::string::npos ? std::string::npos : comma - p);
						if (!h.empty()) group.hashes.push_back(std::strtoull(h.c_str(), nullptr, 16));
						if (comma == std::string::npos) break;
						p = comma + 1;
					}
				}
				parsed.push_back(std::move(group));
			}
			if (next == std::string::npos) break;
			at = next + 1;
		}
		if (!parsed.empty()) {
			std::printf("Kyty SPIR-V variants: %zu group(s) (KYTY_SPIRV_VARIANT=%s), live switch KYTY_SPIRV_VARIANT_LIVE\n",
			            parsed.size(), value);
		}
		return parsed;
	}();
	return groups;
}

// The flags of group `index` (0-based) for `hash`, empty when the group does not list it.
std::string SpirvVariantFlags(uint64_t hash, size_t index) {
	const auto& groups = SpirvVariantSettings();
	if (index >= groups.size()) return {};
	const auto& group = groups[index];
	if (group.all || std::ranges::find(group.hashes, hash) != group.hashes.end() ||
	    std::ranges::find(group.hashes, hash & 0xffffffffull) != group.hashes.end() ||
	    std::ranges::find(group.hashes, hash >> 32u) != group.hashes.end()) {
		return group.flags;
	}
	return {};
}

// KYTY_SPIRV_VARIANT's edits e, t, k (after ApplySpirvTry's z, c); empty when nothing changed.
std::vector<uint32_t> ApplySpirvVariantEdits(std::span<const uint32_t> code, const std::string& flags,
                                             uint64_t hash) {
	const bool early   = flags.find('e') != std::string::npos;
	const bool notex   = flags.find('t') != std::string::npos;
	const bool nokill  = flags.find('k') != std::string::npos;
	const bool nooffs  = flags.find('o') != std::string::npos;
	const bool noprec  = flags.find('p') != std::string::npos;
	const bool nobody  = flags.find('x') != std::string::npos;
	const bool nodnz   = flags.find('d') != std::string::npos;
	if ((!early && !notex && !nokill && !nooffs && !noprec && !nobody && !nodnz) || code.size() < 5) return {};
	// d: Select(ULessThan(x & 0x7fffffff, 0x800000), x & 0x80000000, x) (the f32 denormal flush the
	// translator emits before transcendental and compare inputs) becomes x.
	std::unordered_map<uint32_t, uint32_t> dnz_source; // select result -> x
	if (nodnz) {
		std::unordered_map<uint32_t, uint32_t>                 constants;
		std::unordered_map<uint32_t, std::array<uint32_t, 3>> defs; // id -> {opcode, a, b}
		for (size_t i = 5; i < code.size();) {
			const uint32_t count = code[i] >> 16u;
			const uint32_t op    = code[i] & 0xffffu;
			if (count == 0 || i + count > code.size()) return {};
			if (op == 43 && count == 4) constants[code[i + 2]] = code[i + 3];
			if ((op == 199 /* OpBitwiseAnd */ || op == 176 /* OpULessThan */) && count == 5) {
				defs[code[i + 2]] = {op, code[i + 3], code[i + 4]};
			}
			if (op == 169 /* OpSelect */ && count == 6) {
				const auto is_const = [&](uint32_t id, uint32_t value) {
					const auto c = constants.find(id);
					return c != constants.end() && c->second == value;
				};
				const auto masked = [&](uint32_t id, uint32_t mask, uint32_t& x) {
					const auto d = defs.find(id);
					if (d == defs.end() || d->second[0] != 199) return false;
					if (is_const(d->second[2], mask)) { x = d->second[1]; return true; }
					if (is_const(d->second[1], mask)) { x = d->second[2]; return true; }
					return false;
				};
				const auto cond = defs.find(code[i + 3]);
				uint32_t   x1 = 0, x2 = 0;
				if (cond != defs.end() && cond->second[0] == 176 && is_const(cond->second[2], 0x00800000u) &&
				    masked(cond->second[1], 0x7fffffffu, x1) && masked(code[i + 4], 0x80000000u, x2) &&
				    x1 == x2 && code[i + 5] == x1) {
					dnz_source[code[i + 2]] = x1;
				}
			}
			i += count;
		}
	}
	// x: a module with one function whose body becomes its first block's variables and OpReturn
	// (the stage's cost probe: outputs are undefined).
	if (nobody) {
		uint32_t functions = 0;
		for (size_t i = 5; i < code.size();) {
			const uint32_t count = code[i] >> 16u;
			if (count == 0 || i + count > code.size()) return {};
			if ((code[i] & 0xffffu) == 54 /* OpFunction */) functions++;
			i += count;
		}
		if (functions != 1) return {};
		std::vector<uint32_t> out(code.begin(), code.begin() + 5);
		bool in_function = false, first_label = false, done = false;
		for (size_t i = 5; i < code.size();) {
			const uint32_t count = code[i] >> 16u;
			const uint32_t op    = code[i] & 0xffffu;
			const auto     copy  = [&] {
				out.insert(out.end(), code.begin() + static_cast<std::ptrdiff_t>(i),
				           code.begin() + static_cast<std::ptrdiff_t>(i + count));
			};
			if (op == 54) in_function = true;
			if (!in_function || op == 54 || op == 55 /* OpFunctionParameter */ || op == 56 /* OpFunctionEnd */) {
				if (op == 56 && !done) {
					out.push_back((1u << 16u) | 253u); // OpReturn
					done = true;
				}
				copy();
			} else if (op == 248 /* OpLabel */ && !first_label) {
				first_label = true;
				copy();
			} else if (op == 59 /* OpVariable */ && first_label && !done) {
				copy();
			}
			i += count;
		}
		std::printf("KYTY_SPIRV_VARIANT: shader 0x%016" PRIx64 " flags x: body removed\n", hash);
		return out;
	}
	// o: the byte offsets EmitMemoryOffsets unpacks (8-bit fields of shader_data words) read as 0.
	std::unordered_set<uint32_t> shader_data_vars, shader_data_ptrs, shader_data_words, offset_fields;
	std::unordered_map<uint32_t, uint32_t> u32_constants; // id -> value
	if (nooffs) {
		for (size_t i = 5; i < code.size();) {
			const uint32_t count = code[i] >> 16u;
			const uint32_t op    = code[i] & 0xffffu;
			if (count == 0 || i + count > code.size()) return {};
			if (op == 5 /* OpName */ && count >= 3) {
				const char* name = reinterpret_cast<const char*>(&code[i + 2]);
				if (std::string_view(name, strnlen(name, (count - 2) * 4)) == "shader_data") {
					shader_data_vars.insert(code[i + 1]);
				}
			}
			if (op == 43 /* OpConstant */ && count == 4) u32_constants[code[i + 2]] = code[i + 3];
			if ((op == 65 || op == 66) /* OpAccessChain, OpInBoundsAccessChain */ && count >= 4 &&
			    (shader_data_vars.contains(code[i + 3]) || shader_data_ptrs.contains(code[i + 3]))) {
				shader_data_ptrs.insert(code[i + 2]);
			}
			if (op == 61 /* OpLoad */ && count >= 4 && shader_data_ptrs.contains(code[i + 3])) {
				shader_data_words.insert(code[i + 2]);
			}
			if (op == 202 /* OpBitFieldUExtract */ && count == 6 && shader_data_words.contains(code[i + 3])) {
				const auto width = u32_constants.find(code[i + 5]);
				if (width != u32_constants.end() && width->second == 8u) offset_fields.insert(code[i + 2]);
			}
			i += count;
		}
	}
	uint32_t main_id = 0;
	bool     fragment = false, has_early = false, depth_replacing = false;
	size_t   first_function = 0;
	std::vector<uint32_t> null_types; // result types of replaced image results, in first-seen order
	for (size_t i = 5; i < code.size();) {
		const uint32_t count = code[i] >> 16u;
		const uint32_t op    = code[i] & 0xffffu;
		if (count == 0 || i + count > code.size()) return {};
		if (op == 15 /* OpEntryPoint */ && count >= 3) {
			main_id  = code[i + 2];
			fragment = code[i + 1] == 4; // Fragment
		}
		if (op == 16 /* OpExecutionMode */ && count >= 3) {
			has_early |= code[i + 2] == 9;        // EarlyFragmentTests
			depth_replacing |= code[i + 2] == 12; // DepthReplacing
		}
		if (op == 54 /* OpFunction */ && first_function == 0) first_function = i;
		i += count;
	}
	const auto is_image_result = [](uint32_t op) {
		return (op >= 87 && op <= 97) /* samples, fetch, gathers */ || op == 105 /* OpImageQueryLod */ ||
		       (op >= 305 && op <= 315) /* sparse samples, fetch, gathers */;
	};
	std::unordered_map<uint32_t, uint32_t> null_of_type;
	uint32_t                               bound = code[3];
	const auto replaced = [&](uint32_t op, size_t i, uint32_t count) {
		return count >= 3 && ((notex && is_image_result(op)) ||
		                      (nooffs && op == 202 && offset_fields.contains(code[i + 2])));
	};
	if (notex || nooffs) {
		for (size_t i = first_function; first_function != 0 && i < code.size();) {
			const uint32_t count = code[i] >> 16u;
			const uint32_t op    = code[i] & 0xffffu;
			if (replaced(op, i, count) && !null_of_type.contains(code[i + 1])) {
				null_of_type[code[i + 1]] = bound++;
				null_types.push_back(code[i + 1]);
			}
			i += count;
		}
	}
	std::vector<uint32_t> out(code.begin(), code.begin() + 5);
	out.reserve(code.size() + 64);
	uint32_t changed = 0;
	for (size_t i = 5; i < code.size();) {
		const uint32_t count = code[i] >> 16u;
		const uint32_t op    = code[i] & 0xffffu;
		if (i == first_function) {
			for (const auto type: null_types) {
				out.insert(out.end(), {(3u << 16u) | 46u /* OpConstantNull */, type, null_of_type[type]});
			}
		}
		out.insert(out.end(), code.begin() + static_cast<std::ptrdiff_t>(i),
		           code.begin() + static_cast<std::ptrdiff_t>(i + count));
		if (early && fragment && !has_early && !depth_replacing && op == 16 && count >= 3 &&
		    code[i + 1] == main_id && code[i + 2] == 7 /* OriginUpperLeft */) {
			out.insert(out.end(), {(3u << 16u) | 16u, main_id, 9u});
			changed++;
		}
		if (first_function != 0 && i > first_function && replaced(op, i, count)) {
			out.resize(out.size() - count);
			out.insert(out.end(), {(4u << 16u) | 83u /* OpCopyObject */, code[i + 1], code[i + 2],
			                       null_of_type[code[i + 1]]});
			changed++;
		}
		if (nokill && (op == 252 /* OpKill */ || op == 4416 /* OpTerminateInvocation */) && count == 1) {
			// A block must end in a terminator: return instead (the shader's outputs stay as written).
			out.back() = (1u << 16u) | 253u; // OpReturn
			changed++;
		}
		if (nodnz && op == 169 && count == 6 && dnz_source.contains(code[i + 2])) {
			out.resize(out.size() - count);
			out.insert(out.end(), {(4u << 16u) | 83u /* OpCopyObject */, code[i + 1], code[i + 2],
			                       dnz_source[code[i + 2]]});
			changed++;
		}
		if (noprec && op == 71 /* OpDecorate */ && count == 3 && code[i + 2] == 42 /* NoContraction */) {
			out.resize(out.size() - count);
			changed++;
		}
		if (nokill && op == 5380 /* OpDemoteToHelperInvocation */ && count == 1) {
			out.pop_back();
			changed++;
		}
		i += count;
	}
	if (changed == 0) return {};
	out[3] = bound;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 256) {
		std::printf("KYTY_SPIRV_VARIANT: shader 0x%016" PRIx64 " flags %s: %u edits\n", hash, flags.c_str(), changed);
	}
	return out;
}

} // namespace
// KYTY_SPIRV_VARIANT: the registry of edited copies (defined with PipelineCache::SpirvVariantProgram).
static void RegisterSpirvVariant(uint64_t id, size_t group, ShaderProgram variant);
namespace {

// The variant module's words of group `index` for `hash` (KYTY_SPIRV_VARIANT), or empty without a
// change.
std::vector<uint32_t> SpirvVariantWords(std::span<const uint32_t> code, uint64_t hash, size_t index) {
	const auto flags = SpirvVariantFlags(hash, index);
	if (flags.empty()) return {};
	// F: fragment shaders only (a group "xF@all" removes every pixel shader body, no other stage's).
	if (flags.find('F') != std::string::npos) {
		bool fragment = false;
		for (size_t i = 5; i + 1 < code.size();) {
			const uint32_t count = code[i] >> 16u;
			if (count == 0) break;
			if ((code[i] & 0xffffu) == 15 /* OpEntryPoint */) {
				fragment = code[i + 1] == 4;
				break;
			}
			i += count;
		}
		if (!fragment) return {};
	}
	auto       first  = ApplySpirvTry(code, flags, hash);
	const auto source = first.empty() ? code : std::span<const uint32_t>(first);
	auto       second = ApplySpirvVariantEdits(source, flags, hash);
	return !second.empty() ? second : first;
}

void DumpMatchedShaderInputs(const ShaderParams& params,
                             const ShaderRecompiler::CompileOptions& options,
                             const char* stage_name, std::span<const uint32_t> static_state,
                             uint32_t push_data_start_dword,
                             const std::vector<uint32_t>& spirv, std::string_view ir_dump) {
	if (!Config::GraphicsDebugDumpEnabled() ||
	    !((options.stage == ShaderType::Pixel && options.shader_hash == 0x3b809f9d156a95ddull) ||
	      (options.stage == ShaderType::Vertex && (options.shader_hash == 0xe5398a1c6007f356ull ||
	                                               options.shader_hash == 0xcf1834bb2d5ac83dull ||
	                                               options.shader_hash == 0xd9cc5c62178518faull)) ||
	      (options.stage == ShaderType::Compute && options.shader_hash == 0x305afd0aa0f66b9aull))) {
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
	Json             metadata       = {
	    {"schema_version", 1},
	    {"snapshot_kind", "compile_permutation"},
	    {"cp_queue", HangWatchdog::CurrentCpQueue()},
	    {"cp_submission", HangWatchdog::CurrentCpSubmission()},
	    {"stage", stage_name},
	    {"shader_hash", fmt::format("0x{:016x}", options.shader_hash)},
	    {"guest_code_base", fmt::format("0x{:016x}", params.Base())},
	    {"guest_code_words", params.code.size()},
	    {"guest_code_xxh3_64",
	     fmt::format("0x{:016x}", XXH3_64bits(params.code.data(), params.code.size_bytes()))},
	    {"spirv_file", Common::PathToString(spirv_path.filename())},
	    {"spirv_bytes", spirv_bytes},
	    {"spirv_xxh3_64", fmt::format("0x{:016x}", spirv_hash)},
	    {"ir_file", Common::PathToString(ir_path.filename())},
	    {"ir_bytes", ir_bytes},
	    {"ir_original_bytes", ir_dump.size()},
	    {"ir_truncated", ir_bytes != ir_dump.size()},
	    {"static_state_xxh3_64", fmt::format("0x{:016x}", key_hash)},
	    {"static_state_words", std::vector<uint32_t>(static_state.begin(), static_state.end())},
	    {"user_data_count", params.user_data_count},
	    {"user_data_base", options.user_data_base},
	    {"user_data_words",
	     std::vector<uint32_t>(options.user_data.begin(), options.user_data.end())},
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
	} else if (options.stage == ShaderType::Compute) {
		const auto& cs      = *options.input_info.compute;
		metadata["compute"] = {
		    {"dispatch_threads_num",
		     {cs.dispatch_threads_num[0], cs.dispatch_threads_num[1], cs.dispatch_threads_num[2]}},
		    {"wave_size", cs.wave_size},
		    {"host_subgroup_size", cs.host_subgroup_size},
		    {"thread_ids_num", cs.thread_ids_num},
		    {"workgroup_register", cs.workgroup_register},
		};
		// This short argument-preparation shader carries four direct buffer descriptors.
		// Copy descriptor words only; diagnosis must not read or synchronize GPU memory.
		auto& buffers = metadata["compute"]["direct_buffers"];
		buffers       = Json::array();
		for (size_t i = 0; i + 4 <= std::min<size_t>(16, options.user_data.size()); i += 4) {
			ShaderBufferResource resource {};
			std::copy_n(options.user_data.data() + i, 4, resource.fields);
			buffers.push_back(
			    {{"sgpr", i},
			     {"fields", std::vector<uint32_t>(resource.fields, resource.fields + 4)},
			     {"base", fmt::format("0x{:016x}", resource.Base48())},
			     {"size", resource.GetSize()},
			     {"stride", resource.Stride()}});
		}
	} else {
		const auto& vs = *options.input_info.vertex;
		auto& vertex = metadata["vertex"];
		vertex = {
		    {"logical_stage", static_cast<uint32_t>(vs.logical_stage)},
		    {"wave_size", vs.wave_size}, {"scratch_size_dwords", vs.scratch_size_dwords},
		    {"pa_cl_vs_out_cntl", vs.pa_cl_vs_out_cntl},
		    {"start_instance_sgpr", vs.start_instance_sgpr},
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
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled = false) {
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
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	// KYTY_UNIFORM_BUFFERS emits std430-style dword arrays in uniform blocks; the device enables
	// uniformBufferStandardLayout whenever it allows that path, so validate with the same rules.
	validator_options.SetUniformBufferStandardLayout(
	    ShaderRecompiler::Spirv::GetHostBufferRobustness().uniform_dword_loads_return_zero);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
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

// The validator's verdict on a per-pixel copy (pixelRateVariant.h), whatever --shader-validation
// says; failures are logged.
bool ValidatePixelRateSpirv(uint64_t shader_hash, const std::vector<uint32_t>& spirv) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&,
	                                     const char* message) {
		messages += message;
		messages += '\n';
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetUniformBufferStandardLayout(
	    ShaderRecompiler::Spirv::GetHostBufferRobustness().uniform_dword_loads_return_zero);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return true;
	}
	LOGF_COLOR(Log::Color::BrightRed,
	           "per-pixel copy SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s", shader_hash,
	           messages.c_str());
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
	            std::vector<uint32_t> spirv, bool maintenance8_enabled) {
		{
			std::scoped_lock lock(m_mutex);
			m_jobs.push_back({label, stage_name, shader_hash, std::move(spirv),
			                  maintenance8_enabled});
		}
		m_wake.notify_one();
	}

private:
	struct Job {
		const char*           label      = nullptr; // string literals
		const char*           stage_name = nullptr;
		uint64_t              shader_hash = 0;
		std::vector<uint32_t> spirv;
		bool                  maintenance8_enabled = false; // CompileOptions::maintenance8_enabled
	};

	void Run(const std::stop_token& stop) {
		Profiler::SetThreadName("SpirvValidator");
		Common::PlaceCurrentThreadCold();
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
			HangWatchdog::Scope validate("program-background-validation", job.shader_hash);
			const bool valid = ValidateShaderSpirv(job.label, job.shader_hash, job.spirv,
			                                       job.maintenance8_enabled);
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
	// Bit 0: published by the precompile replay; bit 1: bound by a draw or dispatch (atomic_ref;
	// KYTY_SHADER_PRECOMPILE_STATS counts, g_permutation_use).
	mutable uint8_t                              use = 0;
	// PermutationIdentity, set before publication when KYTY_SHADER_PRECOMPILE_USED records first
	// binds (0 otherwise).
	uint64_t                                     identity = 0;
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
		    : resource_plan(std::move(plan)),
		      workgroup_counts(ShaderRecompiler::GetCodegenOptions().workgroup_images &&
		                       ShaderRecompiler::IR::PlanReadsWorkgroupCounts(resource_plan)) {
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
		// KYTY_WORKGROUP_IMAGES: the plan enumerates images over the dispatch's workgroup counts,
		// which reuse's read certificates do not cover: such a source always refreshes.
		bool                               workgroup_counts = false;
		std::vector<uint32_t>              user_dependencies;
		bool                               projected_key = false;
		PermutationList                    permutations;
		// Set under the exclusive programs lock; may be set on an already published entry. Also
		// set without it by Materialize for a plan that can never materialize (SkipVariantPlan).
		mutable std::atomic<bool>          skip_dispatch {false};
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
		// CompileAndPublish sets it when it published a new permutation (the caller clears it).
		bool                 published_new = false;
		// ... and whether that permutation was reloaded from the persistent program cache.
		bool                 published_from_disk = false;
	};

	static bool LookupMemoEnabled() {
		static constexpr auto enabled_init = [] {
			const auto* value = std::getenv("KYTY_PROGRAM_LOOKUP_MEMO");
			return value == nullptr || std::strcmp(value, "0") != 0;
		};
		static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
		const bool& enabled = enabled_once.Get(enabled_init);
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
		DumpLargeSpirv(stage_name, options.shader_hash, result.spirv);
		DumpSelectedShader(stage_name, options.shader_hash, result.decoded_dump, result.ir_dump,
		                   result.spirv);
		if (!result.spirv_plain.empty()) {
			DumpSelectedShader("psplain", options.shader_hash, {}, {}, result.spirv_plain);
		}
		DumpMatchedShaderInputs(params, options, stage_name, static_state,
		                        push_data_start_dword, result.spirv, result.ir_dump);
		DumpShaderOriginal(stage_name, options.shader_hash, params.code, result.decoded_dump);
		DumpSelectedShader(stage_name, options.shader_hash, result.decoded_dump, result.ir_dump,
		                   result.spirv);
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
		HangWatchdog::Scope finish("program-finish", options.shader_hash,
		                           static_cast<uint64_t>(options.stage));
		if (words != nullptr) {
			words->spirv       = emitted.spirv;
			words->spirv_plain = emitted.spirv_plain;
		}
		if (validator == nullptr) {
			const auto validate_begin = CompileClockNs();
			bool       valid          = false;
			{
				KYTY_PROFILER_BLOCK("Shader::Validate");
				valid = ValidateShaderSpirv(options.dump_label, options.shader_hash, emitted.spirv,
				                            options.maintenance8_enabled);
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
		// KYTY_SPIRV_TRY: the driver gets an edited copy of the listed shaders' modules.
		const auto try_flags = SpirvTryFlags(options.shader_hash);
		auto       tried     = ApplySpirvTry(emitted.spirv, try_flags, options.shader_hash);
		const auto& driver_spirv = tried.empty() ? emitted.spirv : tried;
		{
			KYTY_PROFILER_BLOCK("Shader::CreateModule");
			module = CompileSPV(driver_spirv, device);
		}
		g_shader_modules.Add(module, driver_spirv);
		times.module_ns = CompileClockNs() - module_begin;
		EXIT_IF(module == nullptr);
		PipelineStats::RegisterModule(module, stage_name, options.shader_hash); // KYTY_PIPELINE_STATS
		// KYTY_SPIRV_VARIANT: each group's edited words, taken before the emitted words move on.
		const auto variant_groups = SpirvVariantSettings().size();
		std::vector<std::vector<uint32_t>> variant_words(variant_groups);
		for (size_t g = 0; g < variant_groups; g++) {
			variant_words[g] = SpirvVariantWords(driver_spirv, options.shader_hash, g);
		}
		// KYTY_EQAA_MIXED on NV mixed samples: a pixel shader that shades per sample gets a
		// per-pixel copy for EQAA draws (pixelRateVariant.h); registered under its id below.
		bool             shades_per_sample = false;
		vk::ShaderModule pixel_rate_module = nullptr;
		if (pixel_rate_variants && options.stage == ShaderType::Pixel &&
		    SpirvShadesPerSample(emitted.spirv)) {
			shades_per_sample = true;
			pixel_rate_module = CompilePixelRateVariant(options, emitted.spirv);
		}
		// KYTY_UNNORMALIZED_FIX: a hull shader that samples gets a copy whose explicit-LOD samples
		// take normalized coordinates (pixelRateVariant.h), registered under its id below with an
		// id of its own; GetGraphicsPrograms runs it where the draw's HS samplers are all
		// unnormalized.
		vk::ShaderModule normalized_module = nullptr;
		if (options.stage == ShaderType::TessellationControl && UnnormalizedSamplingFix() &&
		    !emitted.info.info.sampled_pairs.empty()) {
			normalized_module = CompileNormalizedVariant(options, emitted.spirv);
		}
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(emitted.spirv.size()), options.wave_size);
		}
		const auto id = next_shader_id.fetch_add(1, std::memory_order_relaxed) + 1u;
		// KYTY_AA_MASK_COVERAGE: the pixel SPIR-V kept (compressed) for the AA mask copies.
		if (options.stage == ShaderType::Pixel) {
			AaMaskCoverage::Register(id, options.shader_hash, emitted.spirv);
		}
		if (validator != nullptr) {
			validator->Submit(options.dump_label, stage_name, options.shader_hash,
			                  std::move(emitted.spirv), options.maintenance8_enabled);
		}
		GpuOpProfiler::RegisterShader(id, stage_name, options.shader_hash);
		if (shades_per_sample) {
			PixelRate::Register(id, true, pixel_rate_module);
		}
		// KYTY_SPIRV_VARIANT: the edited copy under an id of its own (draws pick it while live).
		const auto make_variants = [&](uint64_t base_id, std::vector<std::vector<uint32_t>>& groups) {
			for (size_t g = 0; g < groups.size(); g++) {
				auto& edited = groups[g];
				if (edited.empty()) continue;
				const auto variant_module = CompileSPV(edited, device);
				if (variant_module == nullptr) continue;
				g_shader_modules.Add(variant_module, edited);
				PipelineStats::RegisterModule(variant_module, "variant", options.shader_hash);
				const auto variant_id = next_shader_id.fetch_add(1, std::memory_order_relaxed) + 1u;
				GpuOpProfiler::RegisterShader(variant_id, stage_name, options.shader_hash);
				RegisterSpirvVariant(base_id, g, {.id = variant_id, .module = variant_module});
			}
		};
		make_variants(id, variant_words);
		if (normalized_module != nullptr) {
			const auto variant_id = next_shader_id.fetch_add(1, std::memory_order_relaxed) + 1u;
			GpuOpProfiler::RegisterShader(variant_id, stage_name, options.shader_hash);
			NormalizedLod::Register(id, normalized_module, variant_id);
		}
		ShaderProgram plain {};
		if (!emitted.spirv_plain.empty()) {
			if (validator == nullptr) {
				if (!ValidateShaderSpirv(options.dump_label, options.shader_hash,
				                         emitted.spirv_plain, options.maintenance8_enabled)) {
					DumpShaderSpirv(stage_name, options.shader_hash, emitted.spirv_plain);
					EXIT("%s failed hash=0x%016" PRIx64 ": plain variant SPIR-V validation failed\n",
					     options.dump_label, options.shader_hash);
				}
			}
			auto        plain_tried  = ApplySpirvTry(emitted.spirv_plain, try_flags, options.shader_hash);
			const auto& plain_driver = plain_tried.empty() ? emitted.spirv_plain : plain_tried;
			const auto  plain_module = CompileSPV(plain_driver, device);
			EXIT_IF(plain_module == nullptr);
			g_shader_modules.Add(plain_module, plain_driver);
			PipelineStats::RegisterModule(plain_module, "psplain", options.shader_hash); // KYTY_PIPELINE_STATS
			std::vector<std::vector<uint32_t>> plain_variant_words(variant_groups);
			for (size_t g = 0; g < variant_groups; g++) {
				plain_variant_words[g] = SpirvVariantWords(plain_driver, options.shader_hash, g);
			}
			const auto plain_id = next_shader_id.fetch_add(1, std::memory_order_relaxed) + 1u;
			make_variants(plain_id, plain_variant_words);
			if (options.stage == ShaderType::Pixel) {
				AaMaskCoverage::Register(plain_id, options.shader_hash, emitted.spirv_plain);
			}
			// The plain variant shades per sample as the program does (KYTY_EQAA_MIXED).
			vk::ShaderModule plain_rate_module = nullptr;
			if (shades_per_sample) {
				plain_rate_module = CompilePixelRateVariant(options, emitted.spirv_plain);
			}
			if (validator != nullptr) {
				validator->Submit(options.dump_label, stage_name, options.shader_hash,
				                  std::move(emitted.spirv_plain), options.maintenance8_enabled);
			}
			GpuOpProfiler::RegisterShader(plain_id, stage_name, options.shader_hash);
			if (shades_per_sample) {
				PixelRate::Register(plain_id, true, plain_rate_module);
			}
			plain = {.id = plain_id, .module = plain_module};
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(emitted.info),
		    .handle         = {.id = id, .module = module},
		    .plain          = plain,
		};
	}

	// The module NV mixed-sample draws use for a pixel shader that reads its sample index
	// (SpirvWithoutSampleId), or null. The first copies are validated whatever
	// --shader-validation says: the rewrite is mechanical, but a broken copy is a broken pipeline.
	vk::ShaderModule CompilePixelRateVariant(const ShaderRecompiler::CompileOptions& options,
	                                         const std::vector<uint32_t>&            spirv) {
		auto variant = SpirvWithoutSampleId(spirv);
		if (variant.empty()) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("Pipeline: pixel shader 0x%016" PRIx64
				     " shades per sample in a form the per-pixel copy does not rewrite; its EQAA "
				     "draws are skipped (KYTY_EQAA_MIXED)\n",
				     options.shader_hash);
			}
			return nullptr;
		}
		constexpr uint32_t CheckedCopies = 32;
		if (pixel_rate_checked.load(std::memory_order_relaxed) < CheckedCopies ||
		    pixel_rate_failed.load(std::memory_order_relaxed)) {
			if (!ValidatePixelRateSpirv(options.shader_hash, variant)) {
				pixel_rate_failed.store(true, std::memory_order_relaxed);
				LOGF("Pipeline: the per-pixel copy of pixel shader 0x%016" PRIx64
				     " is not valid SPIR-V; its EQAA draws are skipped\n",
				     options.shader_hash);
				return nullptr;
			}
			pixel_rate_checked.fetch_add(1, std::memory_order_relaxed);
		}
		const auto                   variant_module = CompileSPV(variant, device);
		// Registered by content, so pipelines that use the copy keep a journal key (they had none:
		// GT7 logged over 1024 "other pipelines created without a journal key").
		if (variant_module != nullptr) g_shader_modules.Add(variant_module, variant);
		static std::atomic<uint32_t> logged {0};
		if (variant_module != nullptr && logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("Pipeline: pixel shader 0x%016" PRIx64
			     " reads its sample index: EQAA draws run a per-pixel copy (index 0)\n",
			     options.shader_hash);
		}
		return variant_module;
	}

	// The copy of a hull shader whose explicit-LOD samples take normalized coordinates
	// (SpirvNormalizedExplicitLod), or null. The first copies and every one after a failure are
	// validated; one that fails is not used (its draws sample as before).
	vk::ShaderModule CompileNormalizedVariant(const ShaderRecompiler::CompileOptions& options,
	                                          const std::vector<uint32_t>&            spirv) {
		constexpr uint32_t MaxFailures = 8;
		if (normalized_failures.load(std::memory_order_relaxed) >= MaxFailures) {
			return nullptr;
		}
		const auto variant = SpirvNormalizedExplicitLod(spirv);
		if (variant.empty()) {
			return nullptr;
		}
		constexpr uint32_t CheckedCopies = 32;
		if (normalized_copies.load(std::memory_order_relaxed) < CheckedCopies ||
		    normalized_failures.load(std::memory_order_relaxed) > 0 ||
		    Config::ShaderValidationEnabled()) {
			if (!ValidatePixelRateSpirv(options.shader_hash, variant)) {
				const auto failures =
				    normalized_failures.fetch_add(1, std::memory_order_relaxed) + 1u;
				LOGF("Pipeline: the normalized-LOD copy of hull shader 0x%016" PRIx64
				     " is not valid SPIR-V; its draws sample as before (%u such so far)\n",
				     options.shader_hash, failures);
				return nullptr;
			}
		}
		const auto module = CompileSPV(variant, device);
		if (module != nullptr) g_shader_modules.Add(module, variant); // a journal key for its pipelines
		if (module != nullptr && normalized_copies.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("Pipeline: hull shader 0x%016" PRIx64
			     " samples at an explicit LOD: a copy with normalized coordinates is used where "
			     "its samplers are unnormalized (KYTY_UNNORMALIZED_FIX)\n",
			     options.shader_hash);
		}
		return module;
	}

	std::atomic<uint32_t> normalized_copies {0};
	std::atomic<uint32_t> normalized_failures {0};

	// KYTY_EQAA_MIXED with NV mixed samples (and not KYTY_EQAA_PER_SAMPLE=1): per-pixel copies.
	bool                  pixel_rate_variants = false;
	std::atomic<uint32_t> pixel_rate_checked {0};
	std::atomic<bool>     pixel_rate_failed {false};

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
		if (params.call_expansion != 0) [[unlikely]] {
			// An expanded S_SWAPPC_B64 program (ExpandComputeCalls): its code depends on the callees.
			key.static_state.push_back(static_cast<uint32_t>(params.call_expansion));
			key.static_state.push_back(static_cast<uint32_t>(params.call_expansion >> 32u));
		}
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
		    // KYTY_SRT_VARIANT_READS: an in-place flat read outside the GPU-mapped guest ranges
		    // reads 0 (a garbage or null pointer the shader only follows on some paths). Without
		    // the switch only never-mapped addresses do (SrtWalker::InPlaceReadable).
		    .is_guest_mapped = ShaderRecompiler::GetCodegenOptions().srt_variant_reads
		                           ? LibKernel::Memory::IsGpuMapped
		                           : nullptr,
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

	// A plan that failed to materialize with no guest range left to synchronize. When one of its
	// flat SRT reads has a loop-carried address (a BVH traversal's instance record) and
	// KYTY_SRT_VARIANT_READS is off, no evaluation before the dispatch can ever produce it: skip
	// the source's dispatches and draws with one console line per shader instead of stopping the
	// emulator. Every other such failure still exits. A concurrent Get of the same source may
	// still materialize once more; it fails the same way.
	static bool SkipVariantPlan(const SourceEntry& source) {
		const auto& plan = source.resource_plan;
		uint32_t    pc   = 0;
		if (ShaderRecompiler::GetCodegenOptions().srt_variant_reads ||
		    !ShaderRecompiler::IR::FindVariantFlatRead(plan, pc)) {
			return false;
		}
		if (source.skip_dispatch.exchange(true, std::memory_order_relaxed)) {
			return true;
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::VariantPlanSkips);
		static std::mutex                   mutex;
		static std::unordered_set<uint64_t> logged;
		{
			std::scoped_lock lock(mutex);
			if (!logged.insert(plan.shader_hash ^ static_cast<uint64_t>(plan.stage)).second) {
				return true;
			}
		}
		std::string stage = ProgramStageName(plan.stage);
		std::ranges::transform(stage, stage.begin(), [](unsigned char c) {
			return static_cast<char>(std::toupper(c));
		});
		PipelineCacheLog("{} shader 0x{:016x} reads SRT data through a loop-carried address "
		                 "(pc=0x{:08x}), which no evaluation before the dispatch can produce; its "
		                 "dispatches and draws are skipped (KYTY_SRT_VARIANT_READS=1 compiles such "
		                 "reads).",
		                 stage, plan.shader_hash, pc);
		return true;
	}

	// Materializes one stage and records a readiness failure for the retry loop.
	bool Materialize(const SourceEntry& source, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                 ShaderRecompiler::IR::EvaluationScratch& evaluation, ProgramScratch& scratch,
	                 StagePrep& prep, ShaderReadAttempt& read_attempt) {
		read_attempt.count = 0;
		read_attempt.materialization_failed = false;
		PassCensus::SetShader(source.resource_plan.shader_hash); // KYTY_TRACE_PASS_CENSUS
		read_attempt.overflow = false;
		if (ResourceReuseEnabled() && !source.workgroup_counts
		        ? MaterializeReusing(source, runtime, evaluation, scratch, prep)
		        : MaterializeStage(source, runtime, evaluation, prep)) {
			return true;
		}
		// A failed clean probe only declines speculative preparation; the ordered draw retries.
		const bool clean_compiles = g_program_prefetch.On();
		if (clean_compiles && DrawPrep::Speculative()) return false;
		// An unsuccessful optional uniform-fill/active-source probe is harmless if the
		// complete refresh succeeded. Only a failed refresh requests a retry.
		if (read_attempt.count == 0 && SkipVariantPlan(source)) {
			return false; // not a readiness failure: the caller skips the dispatch or draw
		}
		if (!NativeDccEnabled() || read_attempt.count == 0) {
			// A failure no read can fix (an unsupported descriptor format, a specialization the
			// recompiler rejects): the stage gets no program and its draws/dispatches are dropped
			// instead of ending the emulator. The reason was already logged by the materializer.
			PassCensus::Null("stage materialization failed without a retryable read (stage dropped)");
			static std::atomic_uint64_t unmaterializable {0};
			const auto count = unmaterializable.fetch_add(1, std::memory_order_relaxed) + 1u;
			if (count <= 4u || (count & 2047u) == 0u) {
				std::printf("Warning: stage materialization failed without a retryable read "
				            "(#%" PRIu64 ", %s shader 0x%016" PRIx64 "); its draws/dispatches are "
				            "dropped\n",
				            count, ProgramStageName(source.resource_plan.stage),
				            source.resource_plan.shader_hash);
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

	// The first bind of a permutation by a draw or dispatch (g_permutation_use): one relaxed byte load
	// per bind after that.
	static void NoteBound(const Permutation& permutation) {
		std::atomic_ref<uint8_t> use(permutation.use);
		if ((use.load(std::memory_order_relaxed) & 2u) != 0) return;
		const auto before = use.fetch_or(2u, std::memory_order_relaxed);
		if ((before & 2u) != 0) return;
		if (permutation.identity != 0 && g_used_permutations.Enabled()) {
			g_used_permutations.Add(permutation.identity);
		}
		g_permutation_use.bound.fetch_add(1, std::memory_order_relaxed);
		if ((before & 1u) != 0) g_permutation_use.replay_bound.fetch_add(1, std::memory_order_relaxed);
	}

	// `note_use`: false for the replay's own bind (rt_prewarm), which is no use by the game.
	template <typename InputInfo>
	static ShaderProgram Bind(const Permutation& permutation, InputInfo& input_info,
	                          StagePrep& prep, uint32_t& push_data_cursor, bool note_use = true) {
		if (note_use) NoteBound(permutation);
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
	bool WaitForInFlight(std::unique_lock<std::shared_mutex>& lock, const ProgramKey& key,
	                     Covered&& covered) {
		const auto busy = [&] {
			return std::any_of(in_flight.begin(), in_flight.end(),
			                   [&](const InFlightCompile* record) { return covered(*record); });
		};
		if (!busy()) {
			return false;
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::ProgramCompileWaits);
		HangWatchdog::Scope wait("program-in-flight", key.hash, static_cast<uint64_t>(key.stage),
		                         in_flight.size(), 0, reinterpret_cast<uint64_t>(this));
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
	                         std::vector<uint32_t>& code_words, std::vector<uint32_t>& back_code_words,
	                         bool owned_code = false) {
		const auto copy = [owned_code](std::span<const uint32_t> guest, std::vector<uint32_t>& words) {
			words.resize(guest.size());
			if (guest.empty()) return;
			if (owned_code || !LibKernel::Memory::TryReadGpuCleanBacking(reinterpret_cast<uint64_t>(guest.data()),
			                                               words.data(), guest.size_bytes())) {
				std::memcpy(words.data(), guest.data(), guest.size_bytes());
			}
		};
		copy(params.code, code_words);
		copy(params.back_code, back_code_words);
		// KYTY_MAD_EXACT_SHADERS: a selected program gets its own records (stage bit 31).
		const bool mad_exact = ShaderRecompiler::MadExactProgram(static_cast<uint32_t>(options.stage), options.shader_hash);
		ProgramDiskCache::BuildSourceKey(
		    {.stage                   = static_cast<uint32_t>(key.stage) | (mad_exact ? 0x80000000u : 0u),
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
		// A packed record's expanded body (the stored spans point into it).
		std::shared_ptr<const std::vector<uint8_t>>  stored_body;
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
		Common::PlaceCurrentThreadCold();
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

	// Translates and emits a reloaded program again and compares it with the stored bytes.
	void RunBackgroundCheck(const BackgroundCheck& job) {
		HangWatchdog::Scope              compile("program-background-check", job.shader_hash,
		                                         static_cast<uint64_t>(job.stage));
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
	                                     ShaderReadAttempt& read_attempt, bool prep_materialized,
	                                     bool speculative = false, bool replay = false) {
		// Cache hits returned before this. This covers translation through native shader-module
		// creation; readiness failures can retry, so count successful creations separately.
		// A precompile replay (ReplayEntry) runs on a background thread with `prep` already
		// holding the permutation's stored specialization: nothing is read from guest memory, no
		// draw waits for it, and the journal already has it.
		std::optional<Profiler::ScopedFrameWait> shader_miss;
		if (!replay) shader_miss.emplace(Profiler::FrameWait::ShaderProgramMiss);
		// Everything from here on, lock and in-flight waits included, is compile time of the
		// waiting draw.
		struct StallScope {
			uint64_t begin = CompileClockNs();
			bool account;
			~StallScope() { if (account) AddCompileStall(CompileClockNs() - begin); }
		} stall {.account = !speculative && !replay};
		// A draw or dispatch waits for its own translation; replays and look-ahead translations do not.
		CompileWatch::Build watch_build(CompileWatch::Kind::Shader, key.hash);
		std::optional<CompileWatch::Wait> watch_wait;
		if (!speculative && !replay) watch_wait.emplace("draw/dispatch preparation", CompileWatch::Kind::Shader, key.hash);
		std::vector<uint32_t> owned_code;
		auto translation_code = params.code;
		DumpRequestedProgram(params, key.stage);
		if (speculative) {
			// A certificate checked at draw consumption cannot repair a cache entry published
			// from changed code. Keep an immutable copy and verify its identity before publication.
			// Non-content hashes and merged back-code need a richer identity proof; decline them.
			if (params.code.empty() || !params.back_code.empty() || graphics_debug_dump_enabled() ||
			    Config::GetShaderLogDirection() != Config::LogDirection::Silent) return nullptr;
			if (!CopyVerifiedShaderCode(params.code, key.hash, owned_code,
			    [](std::span<const uint32_t> guest, std::span<uint32_t> copy) {
				    return LibKernel::Memory::TryReadGpuCleanBacking(reinterpret_cast<uint64_t>(guest.data()),
				        copy.data(), guest.size_bytes());
			    })) return nullptr;
			translation_code = owned_code;
		}
		const auto publish_index = [&](const SourceEntry& source, const Permutation& permutation) {
			if (ResourceReuseEnabled() && !replay) {
				source.reuse.current.permutation_index = permutation.index;
			}
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
				if (WaitForInFlight(lock, key, [&](const InFlightCompile& record) {
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
			if (WaitForInFlight(lock, key, [&](const InFlightCompile& record) {
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
		HangWatchdog::Scope compiling("program-compile", params.hash,
		                              static_cast<uint64_t>(key.stage));
		const InFlightScope in_flight_scope(*this, record, lock);
		lock.unlock();
		HangWatchdog::DebugDelay("compile", params.hash);

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
		// VK_KHR_maintenance8 (upstream 460a196fc): SPIR-V validation accepts dynamic texel offsets.
		options.maintenance8_enabled = maintenance8_enabled;
		options.shader_hash = params.hash;
		options.user_data   = runtime.user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		if (ShaderDumpSelected(params.hash)) options.dump_ir = true; // KYTY_SHADER_DUMP
		options.dump_label  = label;
		options.input_info  = stage_input;
		options.plain_mip_stats_variant =
		    stage == ShaderType::Pixel && LodStatsCounter::PlainVariant() != LodStatsCounter::Plain::Off;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
			}
			// A hull stage compiles with its wave size (64 unless KYTY_TESS_WAVE32 accepts wave32
			// stages, shader.cpp; upstream 27f290e95).
			if (stage == ShaderType::Mesh) {
				options.wave_size = input_info.mesh.wave_size;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		ProgramCompileTimes               times;
		ShaderRecompiler::TranslateResult translated;
		// `translated` holds a translation of this source that nothing has consumed yet.
		bool translated_now = false;
		// A copy of a fresh translation to keep for this source's later permutations
		// (KYTY_TRANSLATION_CACHE), taken before anything reads or changes the program.
		std::shared_ptr<KeptTranslation> keep;
		bool                             had_kept = false;
		// KYTY_ASYNC_TRANSLATION: a worker's finished translation of this source (defer_translation).
		std::optional<ShaderRecompiler::TranslateResult> async_ready;
		const auto translate_now = [&] {
			const auto translate_begin = CompileClockNs();
			{
				KYTY_PROFILER_BLOCK("Shader::Translate");
				// An expanded call program translated in the background (CallTranslations).
				auto ready = params.call_expansion != 0
				                 ? CallTranslations::Instance().Take(CallTranslations::Id(
				                       key.hash, key.user_data_count, key.static_state))
				                 : std::move(async_ready);
				async_ready.reset();
				translated = ready ? std::move(*ready)
				                   : ShaderRecompiler::TranslateProgram(translation_code, options);
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
		// KYTY_ASYNC_TRANSLATION: true when the translation this compile needs is on the workers and
		// the draw is skipped (t_translation_pending); a finished one is left in async_ready for
		// translate_now. A draw that may not be skipped translates here and drops a worker's copy.
		const auto defer_translation = [&]() -> bool {
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				return false;
			} else {
				if (speculative || replay || params.call_expansion != 0 || stage == ShaderType::Mesh) {
					return false;
				}
				auto&      workers = GraphicsTranslations::Instance();
				const bool allow   = t_defer_translation && g_async_translation.On() && !SyncVerify() &&
				                   Config::GetShaderLogDirection() == Config::LogDirection::Silent &&
				                   !Config::GraphicsDebugDumpEnabled() && !workers.Refused(params.hash);
				if (!allow && workers.Empty()) return false;
				const uint64_t id = XXH3_64bits_withSeed(
				    key.static_state.data(), key.static_state.size() * sizeof(uint32_t),
				    key.hash ^ (uint64_t {key.user_data_count} << 56u) ^ (uint64_t {key.code_size} << 24u) ^
				        (static_cast<uint64_t>(stage) << 8u));
				const std::function<std::unique_ptr<GraphicsTranslations::Job>()> make = [&] {
					auto job   = std::make_unique<GraphicsTranslations::Job>();
					job->hash  = params.hash;
					job->stage = stage;
					job->code.assign(translation_code.begin(), translation_code.end());
					job->back_code.assign(params.back_code.begin(), params.back_code.end());
					job->user_data.assign(options.user_data.begin(), options.user_data.end());
					job->options            = options;
					job->options.user_data  = job->user_data;
					job->options.back_code  = job->back_code;
					job->options.dump_ir    = ShaderDumpSelected(params.hash);
					job->options.early_dump = false;
					ShaderStageInputInfo job_input {};
					if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
						job->vertex       = input_info;
						job->vertex.stage = {};
						job_input.vertex  = &job->vertex;
					} else {
						job->pixel       = input_info;
						job->pixel.stage = {};
						job_input.pixel  = &job->pixel;
					}
					job->options.input_info = job_input;
					return job;
				};
				bool pending = false;
				auto ready   = workers.TakeOrQueue(id, pending, allow ? &make : nullptr);
				if (ready.has_value()) {
					async_ready         = std::move(ready);
					t_translation_taken = true;
					return false;
				}
				if (pending && allow) {
					workers.NoteSkip();
					t_translation_pending = true;
					return true;
				}
				if (pending) workers.Drop(id);
				return false;
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
			auto disk_params = params;
			disk_params.code = translation_code;
			BuildDiskKey(disk_params, options, key, disk_key, code_words, back_code_words,
			             speculative || params.call_expansion != 0);
			if (source == nullptr) {
				disk_source = disk->FindSource(disk_key, !replay);
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
			// KYTY_CALL_EXCLUDE_FAILED: a stored refusal of a function expansion (an older cache) is
			// translated again, so the callee behind it is learned and left out.
			if (usable && disk_source->skip_dispatch && ExpansionHasBodies(params.call_expansion)) {
				usable = false;
			}
			if (usable && SyncVerify()) {
				translate_now();
				usable       = DiskSourceMatches(translated, *disk_source, options);
				plan_checked = true;
			}
			if (!usable) {
				disk->Invalidate(disk_source->id);
			} else if (disk_source->skip_dispatch) {
				NoteDiskSkip(options);
				// KYTY_TRACE_PASS_CENSUS: the reason the disk record does not keep (one translation).
				if (PassCensus::Enabled() && !speculative) {
					if (!translated_now) translate_now();
					if (translated.skip_dispatch) NoteSkipReason(options.shader_hash, translated);
				}
				NoteEmptyProgram(options.shader_hash, "skip verdict: " + SkipReasonText(options.shader_hash));
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
				    replay || Materialize(*source, runtime, evaluation, scratch, prep, read_attempt);
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
				if (defer_translation()) {
					NoteEmptyProgram(options.shader_hash, "translation deferred (KYTY_ASYNC_TRANSLATION)");
					return nullptr; // KYTY_ASYNC_TRANSLATION: skipped meanwhile
				}
				translate_now();
			}
			if (translated.skip_dispatch) {
				NoteSkipReason(options.shader_hash, translated);
				NoteEmptyProgram(options.shader_hash, "skip verdict: " + SkipReasonText(options.shader_hash));
				// Skipped ray-tracing shaders are never compiled; dump their guest code here.
				DumpShaderOriginal(ProgramStageName(options.stage), options.shader_hash,
				                   params.code, translated.decoded_dump);
				auto call_targets = translated.call_targets;
				// The call-table description reads guest memory: only for dumps or
				// KYTY_CALL_TABLE_LOG=1 (startup, default off).
				static const bool call_table_log = [] {
					const char* value = std::getenv("KYTY_CALL_TABLE_LOG");
					return value != nullptr && value[0] == '1';
				}();
				if (call_table_log || Config::GraphicsDebugDumpEnabled()) {
					for (const auto& callee: DescribeCallTables(ProgramStageName(options.stage), options,
					                                            translated.call_tables)) {
						call_targets.push_back(callee);
					}
				}
				DumpCallTargets(ProgramStageName(options.stage), options.shader_hash, call_targets);
				// KYTY_CALL_EXCLUDE_FAILED: the callee whose placed body keeps an expansion from
				// translating is left out of later expansions (the background translation does so
				// already), and the refusal is not stored: a later session learns its exclusions again.
				const bool expansion_refused = ExcludeFailedCallee(params.call_expansion, translated, options);
				if (disk_on && !expansion_refused) {
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
			    replay || Materialize(*source, runtime, evaluation, scratch, prep, read_attempt);
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

		// A precompile replay brings the specialization the journal stored, recorded by an earlier
		// run that may have used other codegen switches (another preset): when its resource counts
		// do not fit this translation's plan, the entry is skipped (ApplyResourceSpecialization
		// would stop the emulator; Ghost of Yotei's CS 0x8a7c27138dda831d after a preset change).
		if (replay && (source->resource_plan.info.buffers.size() != prep.specialization.buffers.size() ||
		               source->resource_plan.info.images.size() > prep.specialization.images.size())) {
			lock.lock();
			FinishInFlight(record);
			return nullptr;
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

		// A specialization the program cannot take (an enumerated material image read other than by a
		// sample, IR::UnsupportedResourceSpecialization): that permutation is skipped, as a shader
		// that gives up is, instead of ending the emulator; remembered so later draws skip it at once.
		std::vector<uint8_t> unsupported_bytes;
		ShaderRecompiler::IR::EncodeSpecialization(prep.specialization, unsupported_bytes);
		const uint64_t unsupported_key =
		    XXH3_64bits_withSeed(unsupported_bytes.data(), unsupported_bytes.size(),
		                         reinterpret_cast<uintptr_t>(source) ^ (uint64_t {push_data_cursor} << 48u));
		if (UnsupportedPermutations::Contains(unsupported_key)) {
			lock.lock();
			FinishInFlight(record);
			return nullptr;
		}
		const auto unsupported_specialization = [&](const ShaderRecompiler::TranslateResult& from) -> bool {
			const char* why =
			    ShaderRecompiler::IR::UnsupportedResourceSpecialization(from.program, prep.specialization);
			if (why == nullptr) return false;
			UnsupportedPermutations::Add(unsupported_key, options, why);
			return true;
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
			    disk->FindPermutation(disk_key.digest, push_data_cursor, specialization_bytes, !replay);
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
				if (unsupported_specialization(translated)) {
					translated_now = false;
					lock.lock();
					FinishInFlight(record);
					return nullptr;
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
			// KYTY_ASYNC_TRANSLATION: a new permutation of a source without a kept translation.
			if (!translated_now && (t_defer_translation || !GraphicsTranslations::Instance().Empty())) {
				bool has_kept = false;
				if (TranslationCacheEnabled()) {
					lock.lock();
					has_kept = source->kept_translation != nullptr;
					lock.unlock();
				}
				if (!has_kept && defer_translation()) return nullptr;
			}
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
			if (unsupported_specialization(translated)) {
				lock.lock();
				FinishInFlight(record);
				return nullptr;
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
			PixelRate::Forget(device, compiled->handle.id);
			PixelRate::Forget(device, compiled->plain.id);
			AaMaskCoverage::Forget(device, compiled->handle.id);
			AaMaskCoverage::Forget(device, compiled->plain.id);
			NormalizedLod::Forget(device, compiled->handle.id);
			g_shader_modules.Destroy(device, compiled->handle.module);
			if (compiled->plain.module != nullptr) {
				g_shader_modules.Destroy(device, compiled->plain.module);
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
		// KYTY_SPEC_DIFF_LOG=1 (diagnostics, default off): a new specialization of a compiled
		// source logs the buffers and images in which it differs from the source's first one.
		static const bool spec_diff_log = [] {
			const auto* env = std::getenv("KYTY_SPEC_DIFF_LOG");
			return env != nullptr && env[0] == '1';
		}();
		if (spec_diff_log && !same_specialization && source->permutations.Size() != 0) {
			const Permutation* first = nullptr;
			source->permutations.ForEach([&](const Permutation& existing) {
				if (first == nullptr) first = &existing;
			});
			const auto& a = first->specialization;
			const auto& b = prep.specialization;
			std::string diff;
			for (size_t i = 0; i < std::max(a.buffers.size(), b.buffers.size()); ++i) {
				if (i < a.buffers.size() && i < b.buffers.size() && a.buffers[i] == b.buffers[i]) continue;
				const auto f = [](const ShaderRecompiler::IR::ResourceSpecialization& s, size_t n) {
					if (n >= s.buffers.size()) return std::string("-");
					const auto& x = s.buffers[n];
					return fmt::format("stride {:#x} fmt {} sel {:#x} zoob {}", x.packed_stride,
					                   static_cast<uint32_t>(x.descriptor_format), x.descriptor_swizzle,
					                   x.zero_stride_oob ? 1 : 0);
				};
				diff += fmt::format(" b{}[{} -> {}]", i, f(a, i), f(b, i));
			}
			for (size_t i = 0; i < std::max(a.images.size(), b.images.size()); ++i) {
				if (i < a.images.size() && i < b.images.size() && a.images[i] == b.images[i]) continue;
				const auto g = [](const ShaderRecompiler::IR::ResourceSpecialization& s, size_t n) {
					if (n >= s.images.size()) return std::string("-");
					const auto& x = s.images[n];
					return fmt::format("cls {} dim {} mips {} conv {} sw {:#x} cube {} bindless {} rt {}",
					                   static_cast<uint32_t>(x.numeric_class), static_cast<uint32_t>(x.dimension),
					                   x.mip_count, static_cast<uint32_t>(x.conversion_format), x.shader_swizzle,
					                   x.cube ? 1 : 0, x.bindless ? 1 : 0, x.runtime ? 1 : 0);
				};
				diff += fmt::format(" i{}[{} -> {}]", i, g(a, i), g(b, i));
			}
			PipelineCacheLog("Spec diff: shader {:016x} permutation {}:{}", params.hash,
			                 source->permutations.Size(), diff);
		}
		if (g_used_permutations.Enabled()) {
			if (specialization_bytes.empty()) {
				ShaderRecompiler::IR::EncodeSpecialization(prep.specialization, specialization_bytes);
			}
			compiled->identity = PermutationIdentity(
			    static_cast<uint32_t>(key.stage), key.hash, key.user_data_count, key.code_size, options.wave_size,
			    options.user_data_base, options.plain_mip_stats_variant, push_data_cursor, key.static_state,
			    specialization_bytes);
		}
		const auto& permutation = source->permutations.Append(std::move(*compiled));
		FinishInFlight(record);
		lock.unlock();
		Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderProgramsCreated);
		if (times.reused) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TranslationReuses);
		}
		times.from_disk   = from_disk;
		const auto detail = fmt::format("{}{}{}{}{}", reason, times.reused ? "+reused" : "",
		                                from_disk ? "+disk" : "", speculative ? "+prefetch" : "",
		                                replay ? "+replay" : "");
		if (speculative) speculative_programs.fetch_add(1, std::memory_order_relaxed);
		RecordProgramCompile(ProgramStageName(stage), params.hash, permutation.handle.id, times,
		                     CompileClockNs() - stall.begin, detail);
		CountCompiledPermutation(stage);
		if (!replay && !speculative && !same_specialization && source->permutations.Size() > 1) {
			NotePermutationChurn(params.hash, source->permutations.Size());
		}
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
			check.stored_body        = reloaded->body;
			SubmitBackgroundCheck(std::move(check));
		}
		scratch.published_new       = true;
		scratch.published_from_disk = from_disk;
		if (store) {
			// The published permutation is immutable; its metadata is encoded from it.
			std::vector<uint8_t> info;
			ShaderRecompiler::IR::EncodeCompiledShaderInfo(permutation.program, info);
			disk->AddPermutation(disk_key.digest, push_data_cursor, specialization_bytes, info,
			                     words.spirv, words.spirv_plain);
		}
		// An expanded call program (host-owned code) is not journaled: its key names its callees.
		if (journal != nullptr && !replay && params.call_expansion == 0) {
			JournalPermutation(params, translation_code, key, options, input_info,
			                   push_data_cursor, permutation.specialization, specialization_bytes,
			                   code_words);
		}
		if (identify_modules) IdentifyModules(key, options, push_data_cursor, permutation, specialization_bytes);
		return publish_index(*source, permutation);
	}

	// Shader precompile (shaderPrecompile.h; KYTY_SHADER_PRECOMPILE): null when off.
	ShaderJournal* journal = nullptr;
	// Published modules get their permutation identity (KYTY_PIPELINE_JOURNAL; ShaderModuleRegistry).
	bool identify_modules = false;

	// The translator-independent identity of a permutation (the shader journal's source identity,
	// push-data cursor and encoded specialization) and its layout bindings, for its modules.
	void IdentifyModules(const ProgramKey& key, const ShaderRecompiler::CompileOptions& options,
	                     uint32_t push_data_cursor, const Permutation& permutation,
	                     std::vector<uint8_t>& specialization_bytes) {
		if (specialization_bytes.empty()) {
			ShaderRecompiler::IR::EncodeSpecialization(permutation.specialization, specialization_bytes);
		}
		const auto identity =
		    PermutationIdentity(static_cast<uint32_t>(key.stage), key.hash, key.user_data_count, key.code_size,
		                        options.wave_size, options.user_data_base, options.plain_mip_stats_variant,
		                        push_data_cursor, key.static_state, specialization_bytes);
		std::vector<uint32_t> layout;
		const auto&           program = permutation.program;
		for (const auto& binding: program.bindings.descriptors) {
			layout.push_back(ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind));
			layout.push_back(static_cast<uint32_t>(NativeDescriptorType(binding.kind)));
			layout.push_back(NativeDescriptorCount(binding));
		}
		const auto bindings = XXH3_64bits(layout.data(), layout.size() * sizeof(uint32_t)) | 1u;
		g_shader_modules.SetIdentity(permutation.handle.module, identity, bindings);
		if (permutation.plain.module != nullptr) {
			g_shader_modules.SetIdentity(permutation.plain.module, identity ^ PlainModuleIdentitySalt, bindings);
		}
	}
	// Called by the precompile replay for each software RT compute program it publishes: starts the
	// background build of the kernel's pipeline, so that the game finds it built. May be empty.
	std::function<void(const ShaderComputeInputInfo&, const ShaderProgram&)> rt_prewarm;
	// rt_prewarm gets every replayed compute program, not only ray-tracing kernels
	// (KYTY_PIPELINE_PREWARM_COMPUTE).
	bool prewarm_all_compute = false;

	template <typename InputInfo>
	static constexpr ShaderJournal::Kind JournalKind() {
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			return ShaderJournal::Kind::Vertex;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			return ShaderJournal::Kind::Pixel;
		} else {
			return ShaderJournal::Kind::Compute;
		}
	}

	// Journals a published permutation's inputs. Only programs whose guest hash is the content
	// hash of their code are journaled (every headerless shader: Astro Bot's all), so that a
	// replayed source can only stand for the code it names; a merged stage (an NGG mesh program with
	// its GS back half) when its hash is that of the two halves' content hashes.
	// `specialization_bytes` and `code_words` are filled when the persistent program cache already
	// encoded and copied them. Cost: once per new permutation, on its compile.
	template <typename InputInfo>
	void JournalPermutation(const ShaderParams& params, std::span<const uint32_t> code,
	                        const ProgramKey& key, const ShaderRecompiler::CompileOptions& options,
	                        const InputInfo& input_info, uint32_t push_data_cursor,
	                        const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                        std::vector<uint8_t>& specialization_bytes,
	                        const std::vector<uint32_t>& code_words) {
		static_assert(std::is_trivially_copyable_v<InputInfo>);
		if (code.empty() || key.code_size != code.size()) return;
		ShaderJournal::Source source;
		source.stage                   = static_cast<uint32_t>(key.stage);
		source.kind                    = JournalKind<InputInfo>();
		source.hash                    = key.hash;
		source.user_data_count         = key.user_data_count;
		source.code_size               = key.code_size;
		source.wave_size               = options.wave_size;
		source.user_data_base          = options.user_data_base;
		source.plain_mip_stats_variant = options.plain_mip_stats_variant;
		source.static_state            = key.static_state;
		if (!journal->HasSource(source)) {
			if (!code_words.empty()) {
				source.code = code_words;
			} else {
				source.code.resize(code.size());
				if (!LibKernel::Memory::TryReadGpuCleanBacking(reinterpret_cast<uint64_t>(code.data()),
				                                               source.code.data(), code.size_bytes())) {
					std::memcpy(source.code.data(), code.data(), code.size_bytes());
				}
			}
			if (!params.back_code.empty()) {
				source.back_code.resize(params.back_code.size());
				if (!LibKernel::Memory::TryReadGpuCleanBacking(reinterpret_cast<uint64_t>(params.back_code.data()),
				                                               source.back_code.data(), params.back_code.size_bytes())) {
					std::memcpy(source.back_code.data(), params.back_code.data(), params.back_code.size_bytes());
				}
			}
			if (MergedSourceHash(source.code, source.back_code) != key.hash) {
				return;
			}
			InputInfo copy = input_info;
			copy.stage     = {};
			source.input_info.assign(reinterpret_cast<const uint8_t*>(&copy),
			                         reinterpret_cast<const uint8_t*>(&copy) +
			                             ShaderStoredInputInfoSize<InputInfo>());
		}
		if (specialization_bytes.empty()) {
			ShaderRecompiler::IR::EncodeSpecialization(specialization, specialization_bytes);
		}
		journal->Record(std::move(source), push_data_cursor, specialization_bytes);
	}

	template <typename InputInfo>
	ShaderPrecompiler::Outcome ReplayWith(InputInfo& input_info, ShaderParams& params,
	                                      const ShaderJournal::Source& source,
	                                      const ShaderJournal::Entry& entry,
	                                      ShaderRecompiler::IR::ResourceSpecialization specialization) {
		using Outcome = ShaderPrecompiler::Outcome;
		constexpr auto stored = ShaderStoredInputInfoSize<InputInfo>();
		if (source.input_info.size() != stored) return Outcome::Skipped;
		std::memcpy(static_cast<void*>(&input_info), source.input_info.data(), stored);
		input_info.stage = {};
		ShaderSanitizeStoredInputInfo(input_info);
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			input_info.param_alias_mask = 0;
		}
		auto& scratch = ThreadScratch();
		ProgramKey key;
		BuildKey(params, input_info, key);
		// The journal's key must be what this build derives from the stored input info.
		if (key.stage != static_cast<ShaderType>(source.stage) || key.static_state != source.static_state ||
		    key.code_size != source.code_size) {
			return Outcome::Skipped;
		}
		if (const auto* existing = FindSource(key)) {
			if (existing->skip_dispatch.load(std::memory_order_relaxed)) return Outcome::Skipped;
			if (FindPermutation(*existing, specialization, entry.push_data_cursor)) {
				return Outcome::Present;
			}
		}
		StagePrep prep;
		prep.specialization = std::move(specialization);
		const ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data   = std::span(params.user_data).first(params.user_data_count),
		    .shader_base = params.Base(),
		};
		ShaderReadAttempt attempt;
		scratch.published_new       = false;
		scratch.published_from_disk = false;
		const auto* published =
		    CompileAndPublish(params, input_info, entry.push_data_cursor, key, runtime,
		                      ShaderRecompiler::IR::ThreadEvaluationScratch(), scratch, prep,
		                      attempt, true, false, true);
		if (!scratch.published_new) return Outcome::Skipped;
		g_compile_totals.replayed.fetch_add(1, std::memory_order_relaxed);
		if (published != nullptr) {
			const auto before = std::atomic_ref<uint8_t>(published->use).fetch_or(1u, std::memory_order_relaxed);
			g_permutation_use.replayed.fetch_add(1, std::memory_order_relaxed);
			// A draw bound it between its publication and here.
			if ((before & 2u) != 0) g_permutation_use.replay_bound.fetch_add(1, std::memory_order_relaxed);
		}
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			if (rt_prewarm && published != nullptr && (published->program.info.uses_bvh || prewarm_all_compute)) {
				uint32_t cursor = entry.push_data_cursor;
				const auto program = Bind(*published, input_info, prep, cursor, false);
				rt_prewarm(input_info, program);
			}
		}
		return scratch.published_from_disk ? Outcome::Reloaded : Outcome::Compiled;
	}

	// One journal entry compiled and published as the draw that first needed it would have: the
	// same CompileAndPublish (disk cache, translation reuse, in-flight sharing with the command
	// processor's own compiles), from the stored specialization instead of guest memory.
	ShaderPrecompiler::Outcome ReplayEntry(const ShaderJournal::Source& source,
	                                       const ShaderJournal::Entry&  entry) {
		using Outcome = ShaderPrecompiler::Outcome;
		ShaderParams params;
		if (source.code.empty() || source.code.size() != source.code_size ||
		    source.user_data_count > params.user_data.size() ||
		    MergedSourceHash(source.code, source.back_code) != source.hash) {
			return Outcome::Skipped;
		}
		params.code            = source.code;
		params.back_code       = source.back_code;
		params.user_data_count = source.user_data_count;
		params.hash            = source.hash;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		if (!ShaderRecompiler::IR::DecodeSpecialization(entry.specialization, specialization)) {
			return Outcome::Skipped;
		}
		switch (source.kind) {
			case ShaderJournal::Kind::Vertex: {
				auto info = std::make_unique<ShaderVertexInputInfo>();
				return ReplayWith(*info, params, source, entry, std::move(specialization));
			}
			case ShaderJournal::Kind::Pixel: {
				auto info = std::make_unique<ShaderPixelInputInfo>();
				return ReplayWith(*info, params, source, entry, std::move(specialization));
			}
			case ShaderJournal::Kind::Compute: {
				auto info = std::make_unique<ShaderComputeInputInfo>();
				return ReplayWith(*info, params, source, entry, std::move(specialization));
			}
		}
		return Outcome::Skipped;
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
	                  ProgramScratch& scratch, ShaderRecompiler::IR::EvaluationScratch& evaluation,
	                  bool speculative = false, uint64_t* compile_ns = nullptr) {
		KYTY_PROFILER_DETAIL_BLOCK("ProgramCache::Get");
		std::unique_lock<std::mutex> reuse_lock;
		if (ResourceReuseEnabled()) reuse_lock = std::unique_lock(m_reuse_mutex);

		auto& key = scratch.key;
		const SourceEntry* source = nullptr;
		{
			// KYTY_CP_GAP_STATS (drawPrep/cpGaps.h): the program lookup's parts on the GPU thread.
			const CpGaps::Scope gap(CpGaps::Cat::ProgKey);
			BuildKey(params, input_info, key);
			source = FindSourceMemo(key, scratch);
		}
		// KYTY_DUMP_PROGRAMS: also programs the program cache already holds (once per name).
		if (source != nullptr) DumpRequestedProgram(params, key.stage);
		if (source != nullptr && source->skip_dispatch.load(std::memory_order_relaxed)) {
			ShaderRecompiler::IR::NullTraceNote(ShaderRecompiler::IR::NullTraceKind::SkippedProgram, params.hash);
			if (!speculative) NoteEmptyProgram(params.hash, "skip verdict: " + SkipReasonText(params.hash));
			return {};
		}
		auto runtime = speculative ? MakeSpeculativeRuntime(params) : MakeRuntime(params, read_attempt);
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			// Image descriptor tables indexed by the workgroup id (upstream 0ee880090).
			runtime.workgroup_counts = input_info.workgroup_counts;
			if (!speculative && g_srt_gpu_fill.On() && GuestGpu::IsGpuThread()) {
				runtime.gpu_written            = LibKernel::Memory::IsGpuFillableWord;
				runtime.gpu_fill_store_offsets = g_srt_gpu_fill.Get() >= 2;
			}
		}
		if (source != nullptr) {
			bool materialized = false;
			{
				const CpGaps::Scope gap(CpGaps::Cat::ProgMaterialize);
				materialized = Materialize(*source, runtime, evaluation, scratch, prep, read_attempt);
			}
			if (!materialized) {
				ShaderRecompiler::IR::NullTraceNote(ShaderRecompiler::IR::NullTraceKind::SkippedResources, params.hash);
				if (!speculative) NoteEmptyProgram(params.hash, "resource materialization failed");
				return {};
			}
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
			const Permutation* permutation = nullptr;
			{
				const CpGaps::Scope gap(CpGaps::Cat::ProgPermutation);
				permutation = FindPermutationMemo(*source, key.stage, prep.specialization,
				                                  push_data_cursor, scratch);
			}
			if (permutation != nullptr) {
				if (ResourceReuseEnabled()) {
					source->reuse.current.permutation_index = permutation->index;
				}
				return Bind(*permutation, input_info, prep, push_data_cursor);
			}
		}
		const auto begin = CompileClockNs();
		const auto* permutation =
		    CompileAndPublish(params, input_info, push_data_cursor, key, runtime, evaluation,
		                      scratch, prep, read_attempt, source != nullptr, speculative);
		if (compile_ns != nullptr) *compile_ns += CompileClockNs() - begin;
		if (permutation == nullptr) {
			ShaderRecompiler::IR::NullTraceNote(ShaderRecompiler::IR::NullTraceKind::SkippedProgram, params.hash);
			if (!speculative && t_empty_hash != params.hash) {
				// No more specific reason was noted by CompileAndPublish.
				NoteEmptyProgram(params.hash, "no permutation (materialization or compile failed)");
			}
			return {};
		}
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
		// The vertex stage is materialized before the pixel program is known, so a draw whose pixel
		// inputs may be relocated takes the serial path; every other draw has no aliases.
		if (vs_info.logical_stage == ShaderType::Vertex &&
		    MayRelocatePixelParameters(ps_info)) return false;
		vs_info.param_alias_mask = 0;
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
	// order (pixel first; push data follows that order). Optional clean compiles use Get's
	// in-flight guard; otherwise this only looks up published programs.
	PipelineCache::SpeculativeResult
	TryPrepareSpeculative(bool pixel_active, const ShaderParams& ps_params,
	                      ShaderPixelInputInfo& ps_info, StagePrep& ps_prep,
	                      const ShaderParams& vs_params, ShaderVertexInputInfo& vs_info,
	                      StagePrep& vs_prep, uint32_t push_data_cursor,
	                      PipelineCache::GraphicsPrograms& programs, bool link_parameters,
	                      uint64_t* compile_ns = nullptr) {
		using Result  = PipelineCache::SpeculativeResult;
		namespace Phases = Common::PrepPhases; // KYTY_DRAW_PREP_PHASES
		auto& scratch = ThreadScratch();
		auto& evaluation = ShaderRecompiler::IR::ThreadEvaluationScratch();
		const auto initial_cursor = push_data_cursor;
		const auto compile_missing = [&]() -> Result {
			if (compile_ns == nullptr) return Result::NotPublished;
			// Pure cache publication from clean snapshots. Failed probes decline the preparation;
			// Validate still checks the complete read certificate before the actual ordered draw.
			ShaderReadAttempt attempt;
			auto cursor = initial_cursor; // PS may already have been bound by the read-only probe.
			if (pixel_active) {
				programs.pixel = Get(ps_params, ps_info, ps_prep, cursor, attempt,
				    scratch, evaluation, true, compile_ns);
				if (!programs.pixel) return Result::NotPublished;
			}
			LinkPixelParameters(link_parameters, vs_info, ps_info);
			programs.vertex[0] = Get(vs_params, vs_info, vs_prep, cursor, attempt,
				    scratch, evaluation, true, compile_ns);
			return programs.vertex[0] ? Result::Ok : Result::NotPublished;
		};
		auto& ps_key  = scratch.key;
		auto& vs_key  = scratch.second_key;
		const SourceEntry* ps_source = nullptr;
		// The pixel stage is bound first: the vertex key carries the parameter aliases of the bound
		// pixel program (LinkPixelParameters), as in the serial path.
		if (pixel_active) {
			BuildKey(ps_params, ps_info, ps_key);
			ps_source = FindSourceMemo(ps_key, scratch);
			if (ps_source == nullptr || ps_source->skip_dispatch.load(std::memory_order_relaxed)) {
				return compile_missing();
			}
			Phases::Mark(Phases::Phase::PsLookup);
			if (!MaterializeStage(*ps_source, MakeSpeculativeRuntime(ps_params), evaluation,
			                      ps_prep)) {
				return Result::ReadFailed;
			}
			Phases::Mark(Phases::Phase::PsMaterialize);
			const auto* permutation = FindPermutationMemo(*ps_source, ps_key.stage,
			                                              ps_prep.specialization, push_data_cursor,
			                                              scratch);
			if (permutation == nullptr) {
				return compile_missing();
			}
			programs.pixel = Bind(*permutation, ps_info, ps_prep, push_data_cursor);
			Phases::Mark(Phases::Phase::PsBind);
		}
		LinkPixelParameters(link_parameters, vs_info, ps_info);
		BuildKey(vs_params, vs_info, vs_key);
		const auto* vs_source = FindSourceMemo(vs_key, scratch);
		if (vs_source == nullptr || vs_source->skip_dispatch.load(std::memory_order_relaxed)) {
			return compile_missing();
		}
		Phases::Mark(Phases::Phase::VsLookup);
		if (!MaterializeStage(*vs_source, MakeSpeculativeRuntime(vs_params), evaluation, vs_prep)) {
			return Result::ReadFailed;
		}
		Phases::Mark(Phases::Phase::VsMaterialize);
		const auto* permutation = FindPermutationMemo(*vs_source, vs_key.stage,
		                                              vs_prep.specialization, push_data_cursor,
		                                              scratch);
		if (permutation == nullptr) {
			return compile_missing();
		}
		programs.vertex[0] = Bind(*permutation, vs_info, vs_prep, push_data_cursor);
		Phases::Mark(Phases::Phase::VsBind);
		return Result::Ok;
	}

	// KYTY_DRAW_PREP_TESS: TryPrepareSpeculative for a tessellated draw, in the serial path's
	// order: the pixel stage, then LS, HS and TES (no parameter linking, as GetGraphicsPrograms).
	PipelineCache::SpeculativeResult
	TryPrepareSpeculativeTess(bool pixel_active, const ShaderParams& ps_params,
	                          ShaderPixelInputInfo& ps_info, StagePrep& ps_prep,
	                          const std::array<ShaderParams, 3>&      vs_params,
	                          std::array<ShaderVertexInputInfo*, 3> vs_info,
	                          std::array<StagePrep*, 3> vs_prep, uint32_t push_data_cursor,
	                          PipelineCache::GraphicsPrograms& programs,
	                          uint64_t* compile_ns = nullptr) {
		using Result     = PipelineCache::SpeculativeResult;
		auto& scratch    = ThreadScratch();
		auto& evaluation = ShaderRecompiler::IR::ThreadEvaluationScratch();
		const auto initial_cursor = push_data_cursor;
		const auto compile_missing = [&]() -> Result {
			if (compile_ns == nullptr) return Result::NotPublished;
			ShaderReadAttempt attempt;
			auto cursor = initial_cursor;
			if (pixel_active) {
				programs.pixel = Get(ps_params, ps_info, ps_prep, cursor, attempt, scratch,
				                     evaluation, true, compile_ns);
				if (!programs.pixel) return Result::NotPublished;
			}
			for (uint32_t i = 0; i < 3; i++) {
				LinkPixelParameters(false, *vs_info[i], ps_info);
				programs.vertex[i] = Get(vs_params[i], *vs_info[i], *vs_prep[i], cursor, attempt,
				                         scratch, evaluation, true, compile_ns);
				if (!programs.vertex[i]) return Result::NotPublished;
			}
			return Result::Ok;
		};
		auto& key = scratch.key;
		if (pixel_active) {
			BuildKey(ps_params, ps_info, key);
			const auto* source = FindSourceMemo(key, scratch);
			if (source == nullptr || source->skip_dispatch.load(std::memory_order_relaxed)) {
				return compile_missing();
			}
			if (!MaterializeStage(*source, MakeSpeculativeRuntime(ps_params), evaluation, ps_prep)) {
				return Result::ReadFailed;
			}
			const auto* permutation = FindPermutationMemo(*source, key.stage, ps_prep.specialization,
			                                              push_data_cursor, scratch);
			if (permutation == nullptr) {
				return compile_missing();
			}
			programs.pixel = Bind(*permutation, ps_info, ps_prep, push_data_cursor);
		}
		for (uint32_t i = 0; i < 3; i++) {
			auto& info = *vs_info[i];
			auto& prep = *vs_prep[i];
			LinkPixelParameters(false, info, ps_info);
			BuildKey(vs_params[i], info, key);
			const auto* source = FindSourceMemo(key, scratch);
			if (source == nullptr || source->skip_dispatch.load(std::memory_order_relaxed)) {
				return compile_missing();
			}
			if (!MaterializeStage(*source, MakeSpeculativeRuntime(vs_params[i]), evaluation, prep)) {
				return Result::ReadFailed;
			}
			const auto* permutation = FindPermutationMemo(*source, key.stage, prep.specialization,
			                                              push_data_cursor, scratch);
			if (permutation == nullptr) {
				return compile_missing();
			}
			programs.vertex[i] = Bind(*permutation, info, prep, push_data_cursor);
		}
		return Result::Ok;
	}

	// KYTY_DRAW_PREP_DISPATCH: GetComputeProgram's lookup as TryPrepareSpeculative does it for a
	// draw: the stage materialized on the calling thread with probe-only reads (the workgroup counts
	// of the dispatch, as Get passes them), then the permutation looked up and bound.
	PipelineCache::SpeculativeResult TryPrepareSpeculativeCompute(const ShaderParams&      params,
	                                                              ShaderComputeInputInfo& info,
	                                                              StagePrep&               prep,
	                                                              ShaderProgram&           program,
	                                                              uint64_t* compile_ns = nullptr) {
		using Result     = PipelineCache::SpeculativeResult;
		auto& scratch    = ThreadScratch();
		auto& evaluation = ShaderRecompiler::IR::ThreadEvaluationScratch();
		const auto compile_missing = [&]() -> Result {
			if (compile_ns == nullptr) return Result::NotPublished;
			ShaderReadAttempt attempt;
			uint32_t          cursor = 0;
			program = Get(params, info, prep, cursor, attempt, scratch, evaluation, true, compile_ns);
			return program ? Result::Ok : Result::NotPublished;
		};
		auto& key = scratch.key;
		BuildKey(params, info, key);
		const auto* source = FindSourceMemo(key, scratch);
		if (source == nullptr || source->skip_dispatch.load(std::memory_order_relaxed)) {
			// A skipped source makes the serial lookup return no program: it decides.
			return source == nullptr ? compile_missing() : Result::NotPublished;
		}
		auto runtime             = MakeSpeculativeRuntime(params);
		runtime.workgroup_counts = info.workgroup_counts;
		if (!MaterializeStage(*source, runtime, evaluation, prep)) {
			return Result::ReadFailed;
		}
		uint32_t cursor = 0;
		const auto* permutation =
		    FindPermutationMemo(*source, key.stage, prep.specialization, cursor, scratch);
		if (permutation == nullptr) {
			return compile_missing();
		}
		program = Bind(*permutation, info, prep, cursor);
		return Result::Ok;
	}

	explicit ProgramCache(vk::Device device, bool maintenance8_enabled)
	    : device(device), maintenance8_enabled(maintenance8_enabled) {
		if (Config::ShaderValidationEnabled() && SpirvValidator::AsyncEnabled()) {
			validator = std::make_unique<SpirvValidator>();
		}
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			entry.permutations.ForEach([&](const Permutation& permutation) {
				PixelRate::Forget(device, permutation.handle.id);
				PixelRate::Forget(device, permutation.plain.id);
				AaMaskCoverage::Forget(device, permutation.handle.id);
				AaMaskCoverage::Forget(device, permutation.plain.id);
				NormalizedLod::Forget(device, permutation.handle.id);
				g_shader_modules.Destroy(device, permutation.handle.module);
				if (permutation.plain.module != nullptr) {
					g_shader_modules.Destroy(device, permutation.plain.module);
				}
			});
		}
	}

	void ReportRamStats() {
		std::shared_lock lock(m_programs_mutex, std::try_to_lock);
		if (lock.owns_lock()) {
			const auto sources = programs.size();
			const auto retained = kept_lru.size();
			const auto bytes = kept_bytes;
			lock.unlock();
			std::printf("RAM cache: sources=%zu retained_translations=%zu "
			            "translation_estimate_bytes=%zu\n", sources, retained, bytes);
		}
		std::unique_lock reuse_lock(m_reuse_mutex, std::try_to_lock);
		if (reuse_lock.owns_lock()) {
			const auto bytes = history_bytes;
			reuse_lock.unlock();
			std::printf("RAM cache: reuse_history_estimate_bytes=%zu\n", bytes);
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	mutable std::shared_mutex                                   m_programs_mutex;
	// Serializes O15 reuse-mode preparation; ordered before m_programs_mutex.
	std::mutex                                                  m_reuse_mutex;
	size_t history_bytes = 0; // Guarded by m_reuse_mutex.
	vk::Device                                                  device;
	bool                                                        maintenance8_enabled = false;
	// Assigned by compiles running outside m_programs_mutex.
	std::atomic<uint64_t> next_shader_id {0};
	std::atomic<uint64_t> speculative_programs {0};
	// Compiles in progress outside the lock and their completion signal (m_programs_mutex).
	std::vector<InFlightCompile*> in_flight;
	std::condition_variable_any   compile_done;
	// Kept translations, least recently used first, and their estimated bytes (m_programs_mutex).
	std::list<SourceEntry*> kept_lru;
	size_t                  kept_bytes = 0;
	// Background spirv-val; null when validation is off or synchronous.
	std::unique_ptr<SpirvValidator> validator;
};

void PipelineCache::ReportRamStats() {
	if (!Common::RamStats::Enabled()) return;
	if (m_mutex.TryLock()) {
		const auto graphics = m_graphics_pipelines.size();
		const auto compute = m_compute_pipelines.size();
		m_mutex.Unlock();
		std::printf("RAM cache: graphics_pipelines=%zu compute_pipelines=%zu "
		            "driver_heap_bytes=unknown\n", graphics, compute);
	}
	m_program_cache->ReportRamStats();
}

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
		Common::PlaceCurrentThreadCold();
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
			HangWatchdog::Scope compile(
			    "graphics-pipeline-optimize",
			    reinterpret_cast<uint64_t>(static_cast<VkPipeline>(job.linked)),
			    job.key->vertex_shader_ids[0], job.key->ps_shader_id);
			const auto   result    = PipelineFeedback::CreateGraphics(cache.m_graphics.device, cache.m_driver_cache,
			                                                          job.snapshot->Info(), &optimized,
			                                                          PipelineFeedback::Path::Builder);
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

// Fast-first pipeline creation (KYTY_PIPELINE_FAST_FIRST, pipelineFastFirst.h): the unoptimized
// first build, the background optimized compiles and the retirement of replaced pipelines.
struct PipelineCache::FastFirstState {
	// Deep copy of a compute create info (with its optional required-subgroup-size structure)
	// for the background compile. The module is owned by the program cache, the layout is interned
	// or lives in the cached Pipeline object (which is copied, not freed, on replacement).
	struct ComputeSnapshot {
		vk::ComputePipelineCreateInfo                         info {};
		vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup {};

		static std::shared_ptr<ComputeSnapshot> Capture(const vk::ComputePipelineCreateInfo& in) {
			if (in.flags != vk::PipelineCreateFlags {} || in.pNext != nullptr ||
			    in.stage.flags != vk::PipelineShaderStageCreateFlags {} ||
			    in.stage.pSpecializationInfo != nullptr || in.basePipelineHandle != nullptr) {
				return nullptr;
			}
			auto snapshot  = std::make_shared<ComputeSnapshot>();
			snapshot->info = in;
			if (in.stage.pNext != nullptr) {
				const auto* base = static_cast<const vk::BaseInStructure*>(in.stage.pNext);
				if (base->sType !=
				        vk::StructureType::ePipelineShaderStageRequiredSubgroupSizeCreateInfo ||
				    base->pNext != nullptr) {
					return nullptr;
				}
				snapshot->subgroup = *static_cast<
				    const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo*>(in.stage.pNext);
				snapshot->info.stage.pNext = &snapshot->subgroup;
			}
			return snapshot;
		}
	};

	// What a fast build hands to the code that publishes the pipeline.
	struct GraphicsFast {
		std::shared_ptr<GraphicsPipelineSnapshot> snapshot;
		uint64_t                                  fast_ns   = 0;
		uint64_t                                  key       = 0;     // GraphicsContent::key
		bool                                      cache_hit = false; // a probe found it cached
	};
	struct ComputeFast {
		std::shared_ptr<ComputeSnapshot> snapshot;
		uint64_t                         fast_ns   = 0;
		uint64_t                         key       = 0;
		bool                             cache_hit = false;
	};

	explicit FastFirstState(PipelineCache& owner)
	    : cache(owner),
	      scheduler(static_cast<size_t>(std::clamp<uint64_t>(
	                    FastFirstEnvU64("KYTY_PIPELINE_FAST_FIRST_THREADS", 2), 1, 8)),
	                static_cast<size_t>(std::clamp<uint64_t>(
	                    FastFirstEnvU64("KYTY_PIPELINE_FAST_FIRST_MAX_PENDING", 1024), 1, 1u << 20)),
	                Common::PlaceCurrentThreadCold),
	      retire_age_ns(FastFirstEnvU64("KYTY_PIPELINE_FAST_FIRST_RETIRE_S", 60) * 1'000'000'000ull),
	      drain_ms(FastFirstEnvU64("KYTY_PIPELINE_FAST_FIRST_DRAIN_S", 5) * 1000ull),
	      probe_supported(owner.m_graphics.pipeline_creation_cache_control_enabled),
	      probe_all(FastFirstEnvU64("KYTY_PIPELINE_FAST_FIRST_PROBE", 0) != 0),
	      reporter([this](std::stop_token stop) { Report(stop); }) {}
	FastFirstState(const FastFirstState&)            = delete;
	FastFirstState& operator=(const FastFirstState&) = delete;

	void Log(const char* when) {
		PipelineCacheLog("{} ({})", FormatFastFirst(SnapshotFastFirst(counters, scheduler.Pending())),
		                 when);
	}
	// Every 60 s, active or not, so a killed process still leaves its counts in the log.
	void Report(const std::stop_token& stop) {
		std::unique_lock lock(report_mutex);
		while (!report_cv.wait_for(lock, stop, std::chrono::seconds(60), [] { return false; }) &&
		       !stop.stop_requested()) {
			Log("60 s report");
		}
	}
	// One early line after the first 100 new pipelines.
	void NoteSeen() {
		if (counters.seen.fetch_add(1, std::memory_order_relaxed) + 1 == 100) Log("first 100 pipelines");
	}

	// Exit: queued optimized compiles run until the drain budget is spent (the driver cache saved
	// after this holds them), the rest is dropped and their fast pipelines stay.
	void Stop() {
		if (scheduler.Stopped()) return;
		reporter.request_stop();
		if (reporter.joinable()) reporter.join();
		scheduler.Stop(drain_ms);
		Log("exit");
	}

	// A pipeline whose key the driver cache is known to hold (KYTY_PIPELINE_KNOWN: built optimized
	// into it by this or an earlier process on this driver, or by the pipeline journal replay) is
	// probed first with VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT: a hit is the
	// optimized pipeline at the cost of a cache lookup, without the unoptimized rebuild and its
	// background compile. Unknown keys go straight to the fast build (no probe: on a cold cache it
	// only adds time). KYTY_PIPELINE_FAST_FIRST_PROBE=1 probes every pipeline.
	vk::Result CreateGraphics(const vk::GraphicsPipelineCreateInfo& info, GraphicsContent& content,
	                          vk::Pipeline* pipeline, GraphicsFast& out) {
		auto&      device = cache.m_graphics.device;
		const auto plain  = [&] {
			const auto result = PipelineFeedback::CreateGraphics(device, cache.m_driver_cache, info, pipeline,
			                                                     PipelineFeedback::Path::Sync);
			if (result == vk::Result::eSuccess) cache.NoteKnownPipeline(content.key);
			return result;
		};
		if (auto adopted = cache.TakeAdoptedPipeline(content.key); adopted != nullptr) {
			PipelineFeedback::NoteAdopted(PipelineFeedback::Path::Sync);
			out.cache_hit = true;
			*pipeline     = adopted;
			return vk::Result::eSuccess;
		}
		NoteSeen();
		auto snapshot = content.snapshot != nullptr
		                    ? content.snapshot
		                    : std::shared_ptr<GraphicsPipelineSnapshot>(GraphicsPipelineSnapshot::Capture(info));
		if (snapshot == nullptr) {
			counters.ineligible.fetch_add(1, std::memory_order_relaxed);
			return plain();
		}
		const bool known = cache.IsKnownPipeline(content.key);
		if (probe_supported && (probe_all || known)) {
			auto probe_info  = info;
			probe_info.flags = vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
			vk::Pipeline cached      = nullptr;
			const auto   probe_begin = CompileClockNs();
			const auto   probed = PipelineFeedback::CreateGraphics(device, cache.m_driver_cache, probe_info, &cached,
			                                                       PipelineFeedback::Path::Probe);
			counters.probe_ns.fetch_add(CompileClockNs() - probe_begin, std::memory_order_relaxed);
			PaceLog::NoteStall(probe_begin, CompileClockNs(), PaceLog::StallProbe, 0, 0, "driver cache probe");
			if (probed == vk::Result::eSuccess && cached != nullptr) {
				counters.cache_hits.fetch_add(1, std::memory_order_relaxed);
				out.cache_hit = true;
				*pipeline     = cached;
				cache.NoteKnownPipeline(content.key);
				return probed;
			}
			if (cached != nullptr) device.destroyPipeline(cached, nullptr);
			if (known) g_compile_totals.known_misses.fetch_add(1, std::memory_order_relaxed);
		}
		if (!scheduler.TryReserve()) {
			counters.cap_fallbacks.fetch_add(1, std::memory_order_relaxed);
			return plain();
		}
		// No driver cache: the unoptimized pipeline would otherwise be stored in it. The
		// optimized build fills the cache.
		auto fast_info = info;
		fast_info.flags |= vk::PipelineCreateFlagBits::eDisableOptimization;
		const auto                  outer = Common::CrashContext::Current();
		Common::CrashContext::Scope note("graphics pipeline unoptimized build (VS, PS)", outer.a,
		                                 outer.b);
		LOGF("PipelineTrace: graphics unoptimized build VS=0x%016" PRIx64 " PS=0x%016" PRIx64 "\n",
		     outer.a, outer.b);
		const auto begin  = CompileClockNs();
		EmptyFragment::Storage empty_fs; // KYTY_EMPTY_FS: the driver's copy only
		const auto result = device.createGraphicsPipelines(nullptr, 1, &EmptyFragment::Apply(fast_info, empty_fs),
		                                                   nullptr, pipeline);
		if (result != vk::Result::eSuccess || *pipeline == nullptr) {
			scheduler.Release();
			counters.fast_failed.fetch_add(1, std::memory_order_relaxed);
			if (*pipeline != nullptr) device.destroyPipeline(*pipeline, nullptr);
			*pipeline = nullptr;
			return plain();
		}
		out.snapshot = std::move(snapshot);
		out.fast_ns  = CompileClockNs() - begin;
		if (PaceLog::StallTraceEnabled()) {
			uint64_t words = 0;
			for (uint32_t i = 0; i < info.stageCount; i++) words += g_shader_modules.WordsOf(info.pStages[i].module);
			PaceLog::NoteStall(begin, begin + out.fast_ns, PaceLog::StallFastBuild, outer.a,
			                   static_cast<int>(words), "unoptimized graphics pipeline build");
		}
		out.key      = content.key;
		counters.graphics_fast.fetch_add(1, std::memory_order_relaxed);
		counters.fast_ns.fetch_add(out.fast_ns, std::memory_order_relaxed);
		return result;
	}

	vk::Result CreateCompute(const vk::ComputePipelineCreateInfo& info, uint64_t key, vk::Pipeline* pipeline,
	                         ComputeFast& out) {
		auto&      device = cache.m_graphics.device;
		const auto plain  = [&] {
			const auto result = device.createComputePipelines(cache.m_driver_cache, 1, &info, nullptr, pipeline);
			if (result == vk::Result::eSuccess) cache.NoteKnownPipeline(key);
			return result;
		};
		NoteSeen();
		auto snapshot = ComputeSnapshot::Capture(info);
		if (snapshot == nullptr) {
			counters.ineligible.fetch_add(1, std::memory_order_relaxed);
			return plain();
		}
		const bool known = cache.IsKnownPipeline(key);
		if (probe_supported && (probe_all || known)) {
			auto probe_info  = info;
			probe_info.flags = vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
			vk::Pipeline cached      = nullptr;
			const auto   probe_begin = CompileClockNs();
			const auto   probed =
			    device.createComputePipelines(cache.m_driver_cache, 1, &probe_info, nullptr, &cached);
			counters.probe_ns.fetch_add(CompileClockNs() - probe_begin, std::memory_order_relaxed);
			PaceLog::NoteStall(probe_begin, CompileClockNs(), PaceLog::StallProbe, 0, 0, "driver cache probe");
			if (probed == vk::Result::eSuccess && cached != nullptr) {
				counters.cache_hits.fetch_add(1, std::memory_order_relaxed);
				out.cache_hit = true;
				*pipeline     = cached;
				cache.NoteKnownPipeline(key);
				return probed;
			}
			if (cached != nullptr) device.destroyPipeline(cached, nullptr);
			if (known) g_compile_totals.known_misses.fetch_add(1, std::memory_order_relaxed);
		}
		if (!scheduler.TryReserve()) {
			counters.cap_fallbacks.fetch_add(1, std::memory_order_relaxed);
			return plain();
		}
		auto fast_info = info;
		fast_info.flags |= vk::PipelineCreateFlagBits::eDisableOptimization;
		const auto                  outer = Common::CrashContext::Current();
		Common::CrashContext::Scope note("compute pipeline unoptimized build (CS)", outer.a);
		const auto begin  = CompileClockNs();
		const auto result = device.createComputePipelines(nullptr, 1, &fast_info, nullptr, pipeline);
		if (result != vk::Result::eSuccess || *pipeline == nullptr) {
			scheduler.Release();
			counters.fast_failed.fetch_add(1, std::memory_order_relaxed);
			if (*pipeline != nullptr) device.destroyPipeline(*pipeline, nullptr);
			*pipeline = nullptr;
			return plain();
		}
		out.snapshot = std::move(snapshot);
		out.fast_ns  = CompileClockNs() - begin;
		PaceLog::NoteStall(begin, begin + out.fast_ns, PaceLog::StallFastBuild, outer.a,
		                   static_cast<int>(g_shader_modules.WordsOf(info.stage.module)),
		                   "unoptimized compute pipeline build");
		out.key      = key;
		counters.compute_fast.fetch_add(1, std::memory_order_relaxed);
		counters.fast_ns.fetch_add(out.fast_ns, std::memory_order_relaxed);
		return result;
	}

	// The pipeline is cached under the key: hand its optimized compile to a worker. The slot was
	// reserved by the fast build.
	void EnqueueGraphics(const GraphicsPipelineKey* key, vk::Pipeline fast, GraphicsFast& in,
	                     uint64_t vs_hash, uint64_t ps_hash) {
		const auto fast_ns = in.fast_ns;
		const auto known   = in.key;
		scheduler.Submit(
		    [this, key, fast, fast_ns, known, vs_hash, ps_hash, snapshot = std::move(in.snapshot)] {
			    Common::CrashContext::Scope note("graphics pipeline optimized build (VS, PS)", vs_hash,
			                                     ps_hash);
			    LOGF("PipelineTrace: graphics optimized build begin VS=0x%016" PRIx64
			         " PS=0x%016" PRIx64 "\n",
			         vs_hash, ps_hash);
			    OptimizeGraphics(key, fast, fast_ns, known, *snapshot);
			    LOGF("PipelineTrace: graphics optimized build done VS=0x%016" PRIx64
			         " PS=0x%016" PRIx64 "\n",
			         vs_hash, ps_hash);
		    },
		    [this] { counters.optimize_skipped.fetch_add(1, std::memory_order_relaxed); });
	}
	void EnqueueCompute(uint64_t id, vk::Pipeline fast, ComputeFast& in, uint64_t cs_hash) {
		const auto fast_ns = in.fast_ns;
		const auto known   = in.key;
		scheduler.Submit(
		    [this, id, fast, fast_ns, known, cs_hash, snapshot = std::move(in.snapshot)] {
			    Common::CrashContext::Scope note("compute pipeline optimized build (CS)", cs_hash);
			    OptimizeCompute(id, fast, fast_ns, known, *snapshot);
		    },
		    [this] { counters.optimize_skipped.fetch_add(1, std::memory_order_relaxed); });
	}

	static void NameThread() {
		static thread_local bool named = false;
		if (!named) {
			named = true;
			Profiler::SetThreadName("PipelineOptimizer");
		}
	}

	void OptimizeGraphics(const GraphicsPipelineKey* key, vk::Pipeline fast, uint64_t fast_ns,
	                      uint64_t known_key, const GraphicsPipelineSnapshot& snapshot) {
		NameThread();
		vk::Pipeline        optimized = nullptr;
		const auto          begin     = CompileClockNs();
		HangWatchdog::Scope compile("graphics-pipeline-optimize",
		                            reinterpret_cast<uint64_t>(static_cast<VkPipeline>(fast)),
		                            key->vertex_shader_ids[0], key->ps_shader_id);
		const auto          result = PipelineFeedback::CreateGraphics(cache.m_graphics.device, cache.m_driver_cache,
		                                                              snapshot.Info(), &optimized,
		                                                              PipelineFeedback::Path::Builder);
		Finish(result, optimized, begin, [&](uint64_t ns) {
			cache.NoteKnownPipeline(known_key);
			cache.ReplaceFastPipeline(key, 0, fast, optimized, fast_ns, ns);
		});
	}
	void OptimizeCompute(uint64_t id, vk::Pipeline fast, uint64_t fast_ns, uint64_t known_key,
	                     const ComputeSnapshot& snapshot) {
		NameThread();
		vk::Pipeline        optimized = nullptr;
		const auto          begin     = CompileClockNs();
		HangWatchdog::Scope compile("compute-pipeline-optimize",
		                            reinterpret_cast<uint64_t>(static_cast<VkPipeline>(fast)), id);
		const auto          result = cache.m_graphics.device.createComputePipelines(
            cache.m_driver_cache, 1, &snapshot.info, nullptr, &optimized);
		Finish(result, optimized, begin, [&](uint64_t ns) {
			cache.NoteKnownPipeline(known_key);
			cache.ReplaceFastPipeline(nullptr, id, fast, optimized, fast_ns, ns);
		});
	}
	template <class Replace>
	void Finish(vk::Result result, vk::Pipeline optimized, uint64_t begin, Replace&& replace) {
		const auto ns = CompileClockNs() - begin;
		PaceLog::NoteStall(begin, begin + ns, PaceLog::StallOptimize, 0, 0, "optimized pipeline compile");
		Profiler::AddFrameWait(Profiler::FrameWait::PipelineOptimize, 1, ns);
		if (result != vk::Result::eSuccess || optimized == nullptr) {
			// The unoptimized pipeline stays; it renders the same state.
			counters.optimize_failed.fetch_add(1, std::memory_order_relaxed);
			if (optimized != nullptr) cache.m_graphics.device.destroyPipeline(optimized, nullptr);
			return;
		}
		counters.optimize_ns.fetch_add(ns, std::memory_order_relaxed);
		// The driver cache just received the optimized pipeline: schedule the periodic save.
		cache.NotePipelineCreated(ns);
		replace(ns);
	}

	PipelineCache&     cache;
	FastFirstCounters  counters;
	FastFirstScheduler scheduler;
	const uint64_t     retire_age_ns;
	const uint64_t     drain_ms;
	const bool         probe_supported; // VK_EXT_pipeline_creation_cache_control
	const bool         probe_all;       // KYTY_PIPELINE_FAST_FIRST_PROBE=1
	bool               compute_only = false; // KYTY_PIPELINE_FAST_FIRST_COMPUTE=1 without KYTY_PIPELINE_FAST_FIRST
	std::mutex                  report_mutex;
	std::condition_variable_any report_cv;
	std::jthread               reporter;
	// Guarded by PipelineCache::m_mutex. Objects stay allocated until the cache goes (other
	// threads may hold a reference to one; it is a few words); only the Vulkan handle of a
	// replaced pipeline is destroyed, after retire_age_ns, because recorded command buffers may
	// still use it.
	FastFirstRetireList<vk::Pipeline>      retired_handles;
	std::vector<std::unique_ptr<Pipeline>> retired_objects;
};

void PipelineCache::ReplaceFastPipeline(const GraphicsPipelineKey* graphics_key, uint64_t compute_id,
                                        vk::Pipeline fast, vk::Pipeline optimized,
                                        uint64_t fast_ns, uint64_t optimize_ns) {
	auto&                      state = *m_fast_first;
	Common::LockGuard          lock(m_mutex);
	std::unique_ptr<Pipeline>* slot = nullptr;
	if (graphics_key != nullptr) {
		if (auto entry = m_graphics_pipelines.find(*graphics_key);
		    entry != m_graphics_pipelines.end()) {
			slot = &entry->second;
		}
	} else if (auto entry = m_compute_pipelines.find(compute_id);
	           entry != m_compute_pipelines.end()) {
		slot = &entry->second;
	}
	if (slot == nullptr || (*slot)->pipeline != fast) {
		m_graphics.device.destroyPipeline(optimized, nullptr);
		state.counters.swaps_dropped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	// A new object, not a changed handle: threads that already hold the old Pipeline& keep a
	// consistent (still valid) pipeline; the generation makes their memos look it up again.
	auto replacement      = std::make_unique<Pipeline>(**slot);
	replacement->pipeline = optimized;
	state.retired_objects.push_back(std::move(*slot));
	*slot = std::move(replacement);
	if (graphics_key != nullptr) {
		GpuOpProfiler::RegisterGraphicsPipeline(
		    optimized, graphics_key->vertex_shader_ids.data(),
		    static_cast<uint32_t>(graphics_key->vertex_shader_ids.size()),
		    graphics_key->ps_shader_id);
	} else {
		GpuOpProfiler::RegisterComputePipeline(optimized, compute_id);
	}
	m_pipeline_generation.fetch_add(1, std::memory_order_release);

	const auto now = FastFirstNowNs();
	state.retired_handles.Add(fast, now);
	state.counters.swaps.fetch_add(1, std::memory_order_relaxed);
	state.counters.saved_ns.fetch_add(optimize_ns > fast_ns ? optimize_ns - fast_ns : 0,
	                                  std::memory_order_relaxed);
	for (const auto handle: state.retired_handles.TakeExpired(now, state.retire_age_ns)) {
		m_graphics.device.destroyPipeline(handle, nullptr);
		state.counters.handles_destroyed.fetch_add(1, std::memory_order_relaxed);
	}
}

namespace {
// Numbers PipelineCache instances for their generation ranges (see the constructor).
std::atomic<uint64_t> g_pipeline_cache_instances {0};
} // namespace

struct PipelineCache::PrefetchState {
	struct Result {
		std::unique_ptr<Pipeline> pipeline;
		uint64_t compile_ns = 0;
		// Keyed builds (the skip builder): the content identity, journaled when it is published.
		GraphicsContent content;
	};
	struct Pending {
		std::future<Result> future;
		uint64_t ticket;
		// Set by the worker before it runs the build: a build DropPending removed never started,
		// and its future only holds a broken promise (no exceptions on Linux, so check first).
		std::shared_ptr<std::atomic<bool>> started;
	};
	// keyed: builds describe their content and note it in the known-pipeline list and the pipeline
	// journal when they are published, like the CP's own builds (the skip builder, KYTY_PIPELINE_SKIP,
	// and the prefetch pool: with prefetch on nearly every pipeline is built here, and without the
	// key the journal stayed empty, so a cold start replayed nothing). drop_on_stop: Stop drops the
	// queued builds no draw waits for (the skip builder).
	// ordering (KYTY_PIPELINE_SKIP_ORDER, the skip builder): the worker order of
	// PipelineCompileQueue::Ordering, at below normal priority.
	explicit PrefetchState(PipelineCache& owner, size_t threads, size_t limit = 128, bool keyed_builds = false,
	                       const char* name = "Pipeline prefetch", bool drop_queued_on_stop = false,
	                       PipelineCompileQueue::Ordering ordering = {})
	    : cache(owner), max_pending(limit), keyed(keyed_builds), drop_on_stop(drop_queued_on_stop), log_name(name),
	      ordered(ordering.recent_first),
	      queue(threads, limit,
	            ordering.recent_first ? std::function<void()>([] {
		            Common::PlaceCurrentThreadCold();
		            Common::LowerCurrentThreadPriority();
	            })
	                                  : std::function<void()>(Common::PlaceCurrentThreadCold),
	            ordering) {}
	~PrefetchState() {
		Stop();
		for (auto& [key, job]: pending) {
			(void)key;
			if (!job.future.valid()) continue;
			if (!job.started->load(std::memory_order_acquire)) continue; // dropped by Stop
			Result result = job.future.get();
			if (result.pipeline == nullptr) continue;
			cache.m_graphics.device.destroyPipeline(result.pipeline->pipeline, nullptr);
			ReleasePipelineLayout(cache.m_graphics, result.pipeline->pipeline_layout,
			                      result.pipeline->descriptor_set_layout);
		}
		PipelineCacheLog("{}: submitted {}, used {}, unused {}; worker compile {} ms, "
		                 "CP wait {} ms (worst {} ms)", log_name, submitted.load(), used.load(), pending.size(),
		                 compile_ns.load() / 1000000, wait_ns.load() / 1000000, max_wait_ns.load() / 1000000);
	}
	void Stop() {
		{
			std::lock_guard lock(mutex);
			stopped = true;
		}
		// The skip builder's queue can hold hundreds of builds no draw waits for: drop them.
		if (drop_on_stop) (void)queue.DropPending();
		queue.Stop();
	}
	// True when this call queued the build. `vertices` holds the vertex stage, or LS/HS/TES of a
	// tessellated draw (KYTY_PIPELINE_SKIP_TESS).
	bool Request(const GraphicsPipelineKey& key, std::span<const ShaderVertexInputInfo> vertices,
	             const ShaderPixelInputInfo* pixel, const GraphicsPrograms& programs) {
		std::lock_guard lock(mutex);
		if (stopped || pending.size() >= max_pending || pending.contains(key)) return false;
		if (vertices.empty() || vertices.size() > 3) return false;
		// Interface/resource arrays are copied. Compiled program metadata and shader modules
		// are immutable and owned by ProgramCache until after Stop has joined every task.
		std::array<ShaderVertexInputInfo, 3> stages {};
		std::copy(vertices.begin(), vertices.end(), stages.begin());
		const size_t stage_count = vertices.size();
		auto task = std::make_shared<std::packaged_task<Result()>>(
		    [this, key, stages, stage_count,
		     pixel_copy = pixel != nullptr ? std::optional(*pixel) : std::nullopt, programs] {
			    const auto& vertex = stages[0];
			    Profiler::SetThreadName("PipelineCompiler");
			    queued_builds.Sub();
			    CompileWatch::Build watch(CompileWatch::Kind::GraphicsPipeline,
			                              vertex.stage.program != nullptr ? vertex.stage.program->shader_hash : 0,
			                              pixel_copy && pixel_copy->stage.program != nullptr
			                                  ? pixel_copy->stage.program->shader_hash
			                                  : 0);
			    Result result {std::make_unique<Pipeline>()};
			    const auto begin = CompileClockNs();
			    GraphicsPipelineCreateHook keyed_hook;
			    if (keyed && (cache.m_known != nullptr || cache.m_journal != nullptr)) {
				    keyed_hook = [&](const vk::GraphicsPipelineCreateInfo& info,
				                     std::span<const uint32_t> layout_signature, vk::Pipeline* pipeline) {
					    result.content = DescribeGraphicsPipeline(info, layout_signature);
					    if (auto adopted = cache.TakeAdoptedPipeline(result.content.key); adopted != nullptr) {
						    PipelineFeedback::NoteAdopted(PipelineFeedback::Path::Builder);
						    *pipeline = adopted;
						    return vk::Result::eSuccess;
					    }
					    const auto made = PipelineFeedback::CreateGraphics(cache.m_graphics.device,
					                                                       cache.m_driver_cache, info, pipeline,
					                                                       PipelineFeedback::Path::Builder);
					    if (made == vk::Result::eSuccess) cache.NoteKnownPipeline(result.content.key);
					    return made;
				    };
			    }
			    CreatePipelineInternal(cache.m_graphics, *result.pipeline, key.rendering,
			                           key.vertex_input,
			                           std::span<const ShaderVertexInputInfo>(stages.data(), stage_count),
			                           pixel_copy ? &*pixel_copy : nullptr, programs,
			                           key.static_params, cache.m_driver_cache,
			                           keyed_hook ? &keyed_hook : nullptr);
			    result.compile_ns = CompileClockNs() - begin;
			    compile_ns.fetch_add(result.compile_ns, std::memory_order_relaxed);
			    g_compile_totals.gfx_pipelines.fetch_add(1, std::memory_order_relaxed);
			    g_compile_totals.gfx_pipeline_ns.fetch_add(result.compile_ns, std::memory_order_relaxed);
			    HangTrace::RecordCompile({.kind = HangTrace::CompileKind::GraphicsPipeline,
			        .stage = "gfx", .guest_hash = vertex.stage.program->shader_hash,
			        .id = programs.vertex[0].id, .id2 = pixel_copy ? programs.pixel.id : 0,
			        .pipeline_ns = result.compile_ns, .total_ns = result.compile_ns,
			        .detail = "prefetch-worker"});
			    cache.NotePipelineCreated(result.compile_ns);
			    return result;
		    });
		auto future = task->get_future();
		const auto ticket = ++next_ticket;
		auto started = std::make_shared<std::atomic<bool>>(false);
		queued_builds.Add();
		if (queue.Submit([task, started] {
			    started->store(true, std::memory_order_release);
			    (*task)();
		    }, ticket)) {
			pending.emplace(key, Pending {std::move(future), ticket, std::move(started)});
			pending_count.fetch_add(1, std::memory_order_relaxed);
			submitted.fetch_add(1, std::memory_order_relaxed);
			return true;
		}
		queued_builds.Sub();
		return false;
	}
	Result Take(const GraphicsPipelineKey& key) {
		std::future<Result> future;
		uint64_t ticket = 0;
		{
			std::lock_guard lock(mutex);
			auto it = pending.find(key);
			if (it == pending.end()) return {};
			if (!it->second.future.valid()) return {}; // Another required lookup owns it.
			future = std::move(it->second.future);
			ticket = it->second.ticket;
			// Keep the key reserved while CP waits without the pipeline-map lock. A second
			// speculative request cannot create a duplicate before CP publishes the result.
		}
		queue.Promote(ticket);
		const auto begin = CompileClockNs();
		// KYTY_PIPELINE_SKIP_ORDER (GT7 fork round 145): once the CP has waited 100 ms, every worker
		// helps with the queue while it waits (the game is stopped; the next pipelines a draw that
		// may not be skipped waits for are then done or under way).
		while (ordered && future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
			queue.NoteStall();
		}
		auto result = future.get(); // Always wait: writes, depth and color draws all execute.
		const auto elapsed = CompileClockNs() - begin;
		used.fetch_add(1, std::memory_order_relaxed);
		wait_ns.fetch_add(elapsed, std::memory_order_relaxed);
		AtomicMax(max_wait_ns, elapsed);
		return result;
	}
	void Complete(const GraphicsPipelineKey& key) {
		std::lock_guard lock(mutex);
		if (pending.erase(key) != 0) pending_count.fetch_sub(1, std::memory_order_relaxed);
	}
	// Builds queued or running, not yet completed (a cheap check before Take).
	std::atomic<uint64_t> pending_count {0};
	// Whether a build of the key is queued or running (and not yet taken).
	bool Has(const GraphicsPipelineKey& key) {
		if (pending_count.load(std::memory_order_relaxed) == 0) return false;
		std::lock_guard lock(mutex);
		const auto it = pending.find(key);
		return it != pending.end() && it->second.future.valid();
	}
	// The skip builder's lookup: Absent (not queued: the queue was full or stopped), Pending (still
	// building after waiting up to wait_ns), or Ready (out holds the pipeline; the key stays
	// reserved until Complete).
	enum class Poll : uint8_t { Absent, Pending, Ready };
	Poll TryTake(const GraphicsPipelineKey& key, uint64_t wait_ns, Result& out) {
		std::future<Result> future;
		{
			std::lock_guard lock(mutex);
			auto it = pending.find(key);
			if (it == pending.end()) return Poll::Absent;
			if (!it->second.future.valid()) return Poll::Pending;
			// A draw asks for it now: the build moves to the front of the queue (if it has not
			// started), ahead of look-ahead requests no draw is waiting for (GT7 fork round 144's
			// "an awaited job is compiled next", usaratt/KytyPS5-GT7 df5055a09).
			queue.Promote(it->second.ticket);
			if (it->second.future.wait_for(std::chrono::nanoseconds(wait_ns)) != std::future_status::ready) {
				return Poll::Pending;
			}
			future = std::move(it->second.future);
		}
		out = future.get();
		used.fetch_add(1, std::memory_order_relaxed);
		return Poll::Ready;
	}

	PipelineCache& cache;
	const size_t max_pending;
	const bool keyed;
	const bool drop_on_stop;
	const char* const log_name;
	CompileWatch::QueueCounter queued_builds {log_name};
	const bool ordered; // KYTY_PIPELINE_SKIP_ORDER
	std::mutex mutex;
	bool stopped = false;
	uint64_t next_ticket = 0;
	std::unordered_map<GraphicsPipelineKey, Pending, GraphicsPipelineKeyHash> pending;
	std::atomic<uint64_t> submitted {0}, used {0}, compile_ns {0}, wait_ns {0}, max_wait_ns {0};
	PipelineCompileQueue queue;
};

// Background pipeline builds of the software ray-tracing kernels (TryGetComputePipeline, rtSession.h).
// A kernel's pipeline is built on a worker, outside m_mutex and off the command processor; the CP
// publishes it into m_compute_pipelines the first time it sees the build done.
struct PipelineCache::RtState {
	struct Built {
		std::unique_ptr<Pipeline> pipeline;
		std::string               error; // not empty: the build failed
		uint64_t                  ns          = 0;
		uint64_t                  shader_hash = 0;
		uint64_t                  program_id  = 0;
	};
	struct Job {
		std::shared_future<std::shared_ptr<Built>> future;
		uint64_t                                   ticket       = 0;
		uint64_t                                   shader_hash  = 0;
		uint64_t                                   requested_ns = 0;
		bool                                       asked        = false; // a dispatch has asked
		bool                                       consumed     = false; // published or failed
		bool                                       logged_skip  = false;
		bool                                       promoted     = false; // moved ahead of prewarm builds
		// KYTY_COMPUTE_BACKGROUND_WORDS: a huge compute program, not a software RT kernel (its
		// failure does not turn software RT off; its module is pinned while it builds).
		bool                                       huge         = false;
		uint64_t                                   skipped      = 0; // dispatches skipped while it built
		uint64_t                                   ran          = 0; // dispatches run (KYTY_RT_PIPELINE_LOG)
		uint64_t                                   first_wait_ns = 0; // the CP's waits for it
		uint64_t                                   asked_ns     = 0; // its first dispatch
		bool                                       flip_wait    = false; // the next flip waits for it
	};
	struct BuildContext {
		PipelineCache*                cache  = nullptr;
		const ShaderComputeInputInfo* info   = nullptr;
		vk::ShaderModule              module = nullptr;
		Built*                        built  = nullptr;
		vk::Result                    result = vk::Result::eSuccess;
	};

	// KYTY_RT_PIPELINE_THREADS (default 4, at most 8): the background builders. A software RT
	// pipeline takes about 3 s to build (Miles Morales: some 200 call-split programs for a city
	// area, so two builders kept its reflections waiting for minutes with a fresh cache).
	static size_t BuildThreads() {
		return static_cast<size_t>(std::clamp<uint64_t>(EnvU64("KYTY_RT_PIPELINE_THREADS", 4), 1, 8));
	}

	// KYTY_RT_PIPELINE_WAIT_MS (default 250): the first-dispatch wait of a huge compute program
	// (KYTY_COMPUTE_BACKGROUND_WORDS); KYTY_RT_PIPELINE_ASYNC=0 waits for every build.
	static uint32_t WaitBudgetMs() {
		if (EnvU64("KYTY_RT_PIPELINE_ASYNC", 1) == 0) return UINT32_MAX;
		return static_cast<uint32_t>(std::min<uint64_t>(EnvU64("KYTY_RT_PIPELINE_WAIT_MS", 250), 600000));
	}

	// KYTY_RT_KERNEL_WAIT_MS (default 120000, 0 = skip at once): the first-dispatch wait of a direct
	// software RT kernel (rtSession.h FirstWaitBudgetMs). Only a cold build ever takes long; a
	// dispatch past the budget is skipped until the build is ready, as before.
	static uint32_t KernelWaitBudgetMs() {
		if (EnvU64("KYTY_RT_PIPELINE_ASYNC", 1) == 0) return UINT32_MAX;
		return static_cast<uint32_t>(std::min<uint64_t>(EnvU64("KYTY_RT_KERNEL_WAIT_MS", 120000), 600000));
	}

	static std::string BudgetText(uint32_t ms) {
		return ms == UINT32_MAX ? std::string("for ever") : std::to_string(ms) + " ms";
	}

	// KYTY_RT_KERNEL_WAIT_AT=dispatch (default) | flip: where the command processor waits for a
	// direct RT kernel that is not built at its first dispatch. dispatch: the first dispatch waits
	// (as KYTY_RT_PIPELINE_ASYNC=0), so no dispatch is lost; kernels first used in one frame then
	// build one after another (cold Sky Garden: nine waits of 2-3 s). flip: that frame's dispatches
	// of it are skipped and the frame's flip waits for every such kernel at once (one wait of the
	// longest build); if no flip comes within KYTY_RT_KERNEL_FLIP_WAIT_MS (default 1000) of its first
	// dispatch, a later dispatch waits itself. Not the default: a kernel dispatched only once is lost
	// (Playroom has four such kernels).
	static bool KernelWaitAtFlip() {
		const auto* value = std::getenv("KYTY_RT_KERNEL_WAIT_AT");
		return value != nullptr && std::string_view(value) == "flip";
	}

	explicit RtState(PipelineCache& owner)
	    : cache(owner), wait_ms(WaitBudgetMs()), kernel_wait_ms(KernelWaitBudgetMs()),
	      wait_at_flip(KernelWaitAtFlip() && kernel_wait_ms != UINT32_MAX && kernel_wait_ms != 0),
	      flip_fallback_ns(std::min<uint64_t>(EnvU64("KYTY_RT_KERNEL_FLIP_WAIT_MS", 1000), 600000) * 1'000'000ull),
	      queue(BuildThreads(), 64, Common::PlaceCurrentThreadCold) {
		PipelineCacheLog("Software RT pipelines: built on {} background thread(s); an RT kernel not built at its "
		                 "first dispatch is waited for at most {} (KYTY_RT_KERNEL_WAIT_MS) {}, a huge compute "
		                 "program at most {} (KYTY_RT_PIPELINE_WAIT_MS); call-split programs never wait; past the "
		                 "budget dispatches are skipped until it is ready (KYTY_RT_PIPELINE_ASYNC=0 waits for each)",
		                 BuildThreads(), BudgetText(kernel_wait_ms),
		                 wait_at_flip ? "at that frame's flip (KYTY_RT_KERNEL_WAIT_AT)" : "at the dispatch",
		                 BudgetText(wait_ms));
	}

	// The command processor at a guest flip: waits for the kernels whose first dispatch this frame
	// found unbuilt (all at once; each within its budget from its first dispatch).
	void WaitAtFlip() {
		struct Pending {
			uint64_t                                   id;
			uint64_t                                   shader_hash;
			std::shared_future<std::shared_ptr<Built>> future;
		};
		std::vector<Pending> pending;
		uint64_t             oldest_ns = UINT64_MAX;
		{
			std::scoped_lock lock(mutex);
			for (const auto id: flip_pending) {
				if (auto found = jobs.find(id); found != jobs.end() && found->second.flip_wait) {
					pending.push_back({id, found->second.shader_hash, found->second.future});
					oldest_ns = std::min(oldest_ns, found->second.asked_ns);
				}
			}
			flip_pending.clear();
			any_flip_pending.store(false, std::memory_order_relaxed);
		}
		if (pending.empty()) return;
		const auto begin    = CompileClockNs();
		const auto deadline = oldest_ns + static_cast<uint64_t>(kernel_wait_ms) * 1'000'000ull;
		size_t     waiting  = 0;
		for (const auto& each: pending) {
			if (each.future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) waiting++;
		}
		if (waiting != 0) {
			PipelineCacheLog("Software RT: the command processor waits at the flip for {} RT kernel pipeline(s) "
			                 "first dispatched this frame (at most {}; a cold pipeline cache builds them now)",
			                 waiting, BudgetText(kernel_wait_ms));
			CompileWatch::Wait watch("flip", CompileWatch::Kind::RtPipeline, pending.front().shader_hash);
			for (const auto& each: pending) {
				const auto now = CompileClockNs();
				if (now >= deadline) break;
				(void)each.future.wait_for(std::chrono::nanoseconds(deadline - now));
			}
		}
		const auto waited = CompileClockNs() - begin;
		if (waiting != 0) {
			totals.waited.fetch_add(1, std::memory_order_relaxed);
			totals.wait_ns.fetch_add(waited, std::memory_order_relaxed);
			AddCompileStall(waited);
			FlushCompileStall();
		}
		std::scoped_lock lock(mutex);
		for (const auto& each: pending) {
			if (auto found = jobs.find(each.id); found != jobs.end()) {
				found->second.flip_wait = false;
				found->second.first_wait_ns += waited;
			}
		}
		if (waiting != 0) {
			PipelineCacheLog("Software RT: the flip waited {:.0f} ms for {} RT kernel pipeline(s)",
			                 static_cast<double>(waited) / 1.0e6, waiting);
		}
	}

	// The driver call, kept free of objects with destructors that matter: a structured exception
	// inside the driver unwinds past this frame (Common::CallCatchingStructuredException).
	static void BuildSafely(void* raw) {
		auto& context = *static_cast<BuildContext*>(raw);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
		try {
			context.built->pipeline = std::make_unique<Pipeline>();
			CreatePipelineInternal(context.cache->m_graphics, *context.built->pipeline, *context.info,
			                       context.module, context.cache->m_driver_cache, nullptr,
			                       &context.result);
		} catch (const std::exception& error) {
			context.built->error = error.what();
			context.result       = vk::Result::eErrorUnknown;
		} catch (...) {
			context.built->error = "unknown exception";
			context.result       = vk::Result::eErrorUnknown;
		}
#else
		// Builds without exceptions (the Linux build): failures come back in context.result.
		context.built->pipeline = std::make_unique<Pipeline>();
		CreatePipelineInternal(context.cache->m_graphics, *context.built->pipeline, *context.info, context.module,
		                       context.cache->m_driver_cache, nullptr, &context.result);
#endif
	}

	static const ShaderRecompiler::IR::ResourceSnapshot& EmptyResources() {
		static const ShaderRecompiler::IR::ResourceSnapshot empty {};
		return empty;
	}

	// KYTY_RT_PIPELINE_LOG=1 (diagnostics): one line per background pipeline job queued and finished.
	static bool RtPipelineLogEnabled() {
		static const bool enabled = EnvU64("KYTY_RT_PIPELINE_LOG", 0) != 0;
		return enabled;
	}

	// KYTY_RT_PIPELINE_LOG: a dispatch of a published kernel ran. Every 10 s one line with the
	// dispatches each kernel ran and skipped so far.
	void NoteRan(uint64_t id) {
		std::scoped_lock lock(mutex);
		if (const auto found = jobs.find(id); found != jobs.end()) NoteRanLocked(found->second);
	}
	void NoteRanLocked(Job& job) {
		job.ran++;
		const auto now = CompileClockNs();
		if (now - last_dispatch_log_ns < 10'000'000'000ull) return;
		last_dispatch_log_ns = now;
		std::string line;
		for (const auto& [id, each]: jobs) {
			(void)id;
			if (each.huge || (each.ran == 0 && each.skipped == 0)) continue;
			line += fmt::format(" 0x{:016x} ran {} skipped {};", each.shader_hash, each.ran, each.skipped);
		}
		PipelineCacheLog("Software RT dispatches so far:{}", line);
	}

	// Under `mutex`. The job of kernel `id`, started when new; null when the queue refused it.
	// huge: a huge compute program (KYTY_COMPUTE_BACKGROUND_WORDS), not an RT kernel. urgent: a
	// dispatch needs it now (not a prewarm of the shader precompile): it is built next, and the
	// queue's limit (full of prewarm builds) does not refuse it.
	Job* Request(uint64_t id, const ShaderComputeInputInfo& input, const ShaderProgram& program,
	             bool huge = false, bool urgent = false) {
		if (const auto found = jobs.find(id); found != jobs.end()) return &found->second;
		// A huge program's module could be destroyed with its program while it builds: pinned.
		if (huge && !g_shader_modules.PinModule(program.module)) return nullptr;
		auto& job        = jobs[id];
		job.huge         = huge;
		job.shader_hash  = input.stage.program->shader_hash;
		job.ticket       = ++next_ticket;
		job.requested_ns = CompileClockNs();
		auto promise     = std::make_shared<std::promise<std::shared_ptr<Built>>>();
		job.future       = promise->get_future().share();
		// The program (and so its module) belongs to the program cache until it is destroyed, after
		// Shutdown has joined the workers; only what the pipeline needs is copied.
		auto info             = std::make_shared<ShaderComputeInputInfo>(input);
		info->stage.resources = &EmptyResources();
		const auto module      = program.module;
		const auto shader_hash = job.shader_hash;
		const auto watch_kind  = huge ? CompileWatch::Kind::HugeCompute
		                         : input.stage.program->bindings.call_state_dword != ShaderRecompiler::IR::PushData::NoStart
		                             ? CompileWatch::Kind::SplitPipeline
		                             : CompileWatch::Kind::RtPipeline;
		queued_builds.Add();
		const bool queued      = queue.Submit(
            [this, promise, info, module, id, shader_hash, huge, watch_kind] {
                Profiler::SetThreadName("RtPipeline");
                queued_builds.Sub();
                CompileWatch::Build watch(watch_kind, shader_hash);
                auto built         = std::make_shared<Built>();
                built->shader_hash = shader_hash;
                built->program_id  = id;
                HangWatchdog::Scope compile("rt-compute-pipeline", shader_hash, id);
                const auto          begin = CompileClockNs();
                BuildContext        context {&cache, info.get(), module, built.get()};
                const auto fault = Common::CallCatchingStructuredException(&BuildSafely, &context);
                built->ns        = CompileClockNs() - begin;
                if (fault != 0) {
                    built->error = fmt::format("a structured exception (0x{:08x}) inside the driver", fault);
                } else if (context.result != vk::Result::eSuccess) {
                    if (built->error.empty()) {
                        built->error = "vkCreateComputePipelines returned " + vk::to_string(context.result);
                    }
                } else if (built->pipeline == nullptr || built->pipeline->pipeline == nullptr) {
                    built->error = "no pipeline was created";
                }
                if (RtPipelineLogEnabled()) {
                    PipelineCacheLog("Software RT: pipeline job {} of compute shader 0x{:016x} finished in {:.0f} ms{}{}",
                                     id, shader_hash, static_cast<double>(built->ns) / 1.0e6,
                                     built->error.empty() ? "" : ": ", built->error);
                }
                if (huge) g_shader_modules.Unpin(cache.m_graphics.device, module);
                promise->set_value(std::move(built));
            },
            job.ticket, urgent);
		if (!queued) {
			queued_builds.Sub();
			jobs.erase(id);
			if (huge) g_shader_modules.Unpin(cache.m_graphics.device, module);
			return nullptr;
		}
		totals.requested.fetch_add(1, std::memory_order_relaxed);
		if (RtPipelineLogEnabled()) {
			PipelineCacheLog("Software RT: pipeline job {} of compute shader 0x{:016x} queued ({} requested so far)",
			                 id, job.shader_hash, totals.requested.load(std::memory_order_relaxed));
		}
		return &jobs[id];
	}

	// The CP is done with the kernel's build (under `mutex`): publish the pipeline, or turn software
	// RT off for the session. Null on failure.
	Pipeline* Finish(uint64_t id, Job& job, Built& built) {
		job.consumed         = true;
		const auto waited_ms = (CompileClockNs() - job.requested_ns) / 1'000'000;
		if (!built.error.empty()) {
			totals.failed.fetch_add(1, std::memory_order_relaxed);
			if (job.huge) {
				// Its dispatches are skipped from now on (as KYTY_SKIP_INVALID's).
				PipelineCacheLog("Compute pipeline of shader 0x{:016x} (KYTY_COMPUTE_BACKGROUND_WORDS) could "
				                 "not be built in the background ({}); its dispatches are skipped",
				                 job.shader_hash, built.error);
				return nullptr;
			}
			if (RtSession::MarkFailed()) {
				// Black lighting until the next launch: make it stand out in the console and log.
				PipelineCacheLog("\n{0}\nWARNING: Software RT is turned off for the rest of this session: the "
				                 "pipeline of compute shader 0x{1:016x} could not be built ({2}). Dispatches of "
				                 "ray tracing kernels are skipped from now on, so the tiled lighting and GI "
				                 "passes leave their outputs untouched (black lighting, as with "
				                 "KYTY_RT_SOFTWARE=0). Restart the game to try again; set KYTY_RT_SOFTWARE=0 "
				                 "to skip them from the start.\n{0}",
				                 std::string_view("****************************************************************"),
				                 job.shader_hash, built.error);
			}
			return nullptr;
		}
		Common::LockGuard lock(cache.m_mutex);
		auto [slot, inserted] = cache.m_compute_pipelines.emplace(id, std::move(built.pipeline));
		if (!inserted) return slot->second.get();
		GpuOpProfiler::RegisterComputePipeline(slot->second->pipeline, id);
		totals.published.fetch_add(1, std::memory_order_relaxed);
		totals.build_ns.fetch_add(built.ns, std::memory_order_relaxed);
		AtomicMax(totals.max_build_ns, built.ns);
		g_compile_totals.cs_pipelines.fetch_add(1, std::memory_order_relaxed);
		g_compile_totals.cs_pipeline_ns.fetch_add(built.ns, std::memory_order_relaxed);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ComputePipelinesCreated);
		if (HangTrace::Enabled()) {
			HangTrace::RecordCompile({.kind        = HangTrace::CompileKind::ComputePipeline,
			                          .stage       = "cs",
			                          .guest_hash  = job.shader_hash,
			                          .id          = id,
			                          .pipeline_ns = built.ns,
			                          .total_ns    = built.ns,
			                          .detail      = "rt-background"});
		}
		cache.NotePipelineCreated(built.ns);
		// requested_ns is the request: a prewarm (shader precompile) or the first dispatch.
		PipelineCacheLog("{} pipeline of compute shader 0x{:016x} built in {:.0f} ms, ready {} ms "
		                 "after it was requested; its first dispatch waited {:.0f} ms, {} dispatch(es) skipped",
		                 job.huge ? "Huge compute (KYTY_COMPUTE_BACKGROUND_WORDS)" : "Software RT",
		                 job.shader_hash, static_cast<double>(built.ns) / 1.0e6, waited_ms,
		                 static_cast<double>(job.first_wait_ns) / 1.0e6, job.skipped);
		return slot->second.get();
	}

	// Returns once every requested build has finished (KYTY_PIPELINE_LIST_PREPARE).
	void WaitAll() {
		std::vector<std::shared_future<std::shared_ptr<Built>>> futures;
		{
			std::scoped_lock lock(mutex);
			for (const auto& [id, job]: jobs) {
				(void)id;
				if (job.future.valid()) futures.push_back(job.future);
			}
		}
		for (const auto& future: futures) future.wait();
	}

	// Exit: no new builds start, running ones finish, and pipelines nobody published are destroyed.
	void Shutdown() {
		if (stopped.exchange(true)) return;
		queue.DropPending();
		queue.Stop();
		std::scoped_lock lock(mutex);
		for (auto& [id, job]: jobs) {
			(void)id;
			if (job.consumed || !job.future.valid() ||
			    job.future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
				continue;
			}
			auto built = job.future.get();
			if (built != nullptr && built->pipeline != nullptr) {
				if (built->pipeline->pipeline != nullptr) {
					cache.m_graphics.device.destroyPipeline(built->pipeline->pipeline, nullptr);
				}
				ReleasePipelineLayout(cache.m_graphics, built->pipeline->pipeline_layout,
				                      built->pipeline->descriptor_set_layout);
			}
		}
		const auto totals_now = Totals();
		if (totals_now.requested != 0) {
			PipelineCacheLog("Software RT pipelines: {} requested, {} published, {} failed; the command processor "
			                 "waited {} time(s) for {:.1f} ms in all and skipped {} dispatch(es); builds took "
			                 "{:.1f} ms (longest {:.1f} ms)",
			                 totals_now.requested, totals_now.published, totals_now.failed, totals_now.waited,
			                 static_cast<double>(totals_now.wait_ns) / 1.0e6, totals_now.skipped_dispatches,
			                 static_cast<double>(totals_now.build_ns) / 1.0e6,
			                 static_cast<double>(totals_now.max_build_ns) / 1.0e6);
		}
	}

	// Per guest shader: the RT jobs requested, ready (published, or built and waiting for the CP to
	// publish it), still building, and failed (the RT status line).
	std::vector<RtShaderPipelines> Status() {
		std::vector<RtShaderPipelines> out;
		std::scoped_lock               lock(mutex);
		for (const auto& [id, job]: jobs) {
			(void)id;
			if (job.huge) continue;
			auto found = std::ranges::find(out, job.shader_hash, &RtShaderPipelines::shader_hash);
			if (found == out.end()) {
				out.push_back({.shader_hash = job.shader_hash});
				found = out.end() - 1;
			}
			found->requested++;
			const bool done =
			    job.future.valid() && job.future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
			if (!done) {
				found->building++;
				continue;
			}
			const auto& built = job.future.get();
			if (built == nullptr || !built->error.empty()) {
				found->failed++;
			} else {
				found->ready++;
			}
		}
		return out;
	}

	struct AtomicTotals {
		std::atomic<uint64_t> requested {0}, published {0}, failed {0}, skipped_dispatches {0}, waited {0};
		std::atomic<uint64_t> wait_ns {0}, build_ns {0}, max_build_ns {0};
	};
	RtPipelineTotals Totals() const {
		const auto get = [](const std::atomic<uint64_t>& value) { return value.load(std::memory_order_relaxed); };
		return {get(totals.requested), get(totals.published), get(totals.failed),
		        get(totals.skipped_dispatches), get(totals.waited), get(totals.wait_ns),
		        get(totals.build_ns), get(totals.max_build_ns)};
	}

	PipelineCache&                    cache;
	const uint32_t                    wait_ms;
	const uint32_t                    kernel_wait_ms;
	const bool                        wait_at_flip;
	const uint64_t                    flip_fallback_ns;
	std::vector<uint64_t>             flip_pending;               // under `mutex`
	std::atomic<bool>                 any_flip_pending {false};
	uint64_t                          last_dispatch_log_ns = 0; // KYTY_RT_PIPELINE_LOG dispatch counts
	CompileWatch::QueueCounter        queued_builds {"background compute"};
	std::mutex                        mutex;
	std::unordered_map<uint64_t, Job> jobs;
	uint64_t                          next_ticket = 0;
	std::atomic<bool>                 stopped {false};
	AtomicTotals                      totals;
	PipelineCompileQueue              queue;
};

namespace {

const ShaderRecompiler::IR::ResourceSnapshot& NoPrewarmResources() {
	static const ShaderRecompiler::IR::ResourceSnapshot empty {};
	return empty;
}

// The level the game was booted into, the pipeline journal's level marker: KYTY_PIPELINE_JOURNAL_LEVEL,
// else the value after -lvl in the game's /app0/args.txt (developer boot arguments), else empty.
std::string BootLevelMarker() {
	if (const auto* level = std::getenv("KYTY_PIPELINE_JOURNAL_LEVEL"); level != nullptr) return level;
	if (PipelineCacheTitleId().empty()) return {}; // no game (tests): no file system mounts
	const auto path = LibKernel::FileSystem::GetRealFilename("/app0/args.txt");
	if (path.empty()) return {};
	std::ifstream in(path);
	std::string   token;
	while (in >> token) {
		if (token == "-lvl" && in >> token) return token.substr(0, 128);
	}
	return {};
}

// The pipeline journal's record payload: EncodePipelineJournalPayload (pipelineList.h).
using JournalPayload = PipelineJournalPayload;

bool DecodeJournalPayload(std::span<const uint8_t> bytes, JournalPayload& out) {
	return DecodePipelineJournalPayload(bytes, out);
}

} // namespace

// KYTY_PIPELINE_PREWARM_COMPUTE=1 (default off, on with KYTY_PIPELINE_LIST; needs the shader
// precompile replay): the pipeline of every compute program the replay publishes is built on a
// below-normal-priority thread into the driver cache and published, so the first dispatch finds it
// instead of building it on the command processor. The software ray-tracing kernels keep their own
// path (RtState). A dispatch that comes first builds its pipeline as usual and the prewarmed one is
// dropped: wasted work, never a wait. KYTY_PIPELINE_PREWARM_THREADS (default 1, 4 when preparing, 1..8).
struct PipelineCache::ComputePrewarm {
	ComputePrewarm(PipelineCache& owner, size_t threads)
	    : cache(owner), queue(threads, SIZE_MAX, Common::PlaceCurrentThreadCold) {}
	~ComputePrewarm() { Stop(); }
	ComputePrewarm(const ComputePrewarm&)            = delete;
	ComputePrewarm& operator=(const ComputePrewarm&) = delete;

	void Stop() {
		if (stopped.exchange(true)) return;
		queue.DropPending();
		ReleaseWaiters();
		queue.Stop();
		LogTotals();
	}

	// KYTY_PIPELINE_COMPUTE_AHEAD: a dropped build is never waited for.
	void ReleaseWaiters() {
		{
			std::scoped_lock lock(inflight_mutex);
			inflight.clear();
		}
		inflight_cv.notify_all();
	}

	// KYTY_PIPELINE_COMPUTE_AHEAD=1: a compute program prepared ahead of its dispatch (draw preparation,
	// KYTY_PIPELINE_PREFETCH_PROGRAMS) gets its pipeline built here at once, each id once.
	void RequestAhead(uint64_t id, const ShaderComputeInputInfo& input, vk::ShaderModule module) {
		if (stopped.load(std::memory_order_acquire)) return;
		{
			std::scoped_lock lock(inflight_mutex);
			if (!asked.insert(id).second) return;
			inflight.insert(id);
		}
		auto info             = std::make_shared<ShaderComputeInputInfo>(input);
		info->stage.resources = &NoPrewarmResources();
		if (queue.Submit([this, id, info, module] { Build(id, *info, module); }, id)) {
			requested.fetch_add(1, std::memory_order_relaxed);
			ahead.fetch_add(1, std::memory_order_relaxed);
		} else {
			Finished(id);
		}
	}

	// The command processor needs the pipeline of `id`: when a build of it is queued or running, it
	// moves to the front and this waits for it (true). False when none is.
	bool WaitFor(uint64_t id) {
		std::unique_lock lock(inflight_mutex);
		if (!inflight.contains(id)) return false;
		queue.Promote(id);
		inflight_cv.wait(lock, [&] { return !inflight.contains(id); });
		return true;
	}

	void Finished(uint64_t id) {
		{
			std::scoped_lock lock(inflight_mutex);
			if (inflight.erase(id) == 0) return;
		}
		inflight_cv.notify_all();
	}

	// Builds everything requested so far, then stops (KYTY_PIPELINE_LIST_PREPARE).
	void Drain() {
		if (stopped.exchange(true)) return;
		queue.Stop();
		LogTotals();
	}

	// Builds finished (built, dropped or failed) and builds requested, for the prepare run's progress
	// events.
	[[nodiscard]] std::pair<uint64_t, uint64_t> Progress() const {
		const auto get = [](const std::atomic<uint64_t>& value) { return value.load(std::memory_order_relaxed); };
		return {get(g_compile_totals.prewarm_built) + get(dropped) + get(failed), get(requested)};
	}

	void LogTotals() const {
		if (requested.load() != 0) {
			PipelineCacheLog("Compute prewarm: {} requested, {} built and published ({:.1f} ms), {} dropped (a "
			                 "dispatch built it first), {} failed; {} requested ahead of their dispatch, {} "
			                 "dispatches waited for one ({:.1f} ms)",
			                 requested.load(), g_compile_totals.prewarm_built.load(),
			                 static_cast<double>(g_compile_totals.prewarm_ns.load()) / 1e6, dropped.load(),
			                 failed.load(), ahead.load(), waits.load(), static_cast<double>(wait_ns.load()) / 1e6);
		}
	}

	void Request(uint64_t id, const ShaderComputeInputInfo& input, vk::ShaderModule module) {
		if (stopped.load(std::memory_order_acquire)) return;
		auto info             = std::make_shared<ShaderComputeInputInfo>(input);
		info->stage.resources = &NoPrewarmResources();
		if (queue.Submit([this, id, info, module] { Build(id, *info, module); })) {
			requested.fetch_add(1, std::memory_order_relaxed);
		}
	}

	void Build(uint64_t id, const ShaderComputeInputInfo& info, vk::ShaderModule module) {
		BuildOnce(id, info, module);
		Finished(id);
	}

	void BuildOnce(uint64_t id, const ShaderComputeInputInfo& info, vk::ShaderModule module) {
		static thread_local bool named = false;
		if (!named) {
			named = true;
			LowerCurrentThreadPriority();
			Profiler::SetThreadName("ComputePrewarm");
		}
		{
			Common::LockGuard lock(cache.m_mutex);
			if (cache.m_compute_pipelines.contains(id)) {
				dropped.fetch_add(1, std::memory_order_relaxed);
				return;
			}
		}
		auto                      pipeline = std::make_unique<Pipeline>();
		uint64_t                  key      = 0;
		ComputePipelineCreateHook hook     = [&](const vk::ComputePipelineCreateInfo& create, vk::Pipeline* out) {
            key = cache.m_known != nullptr ? ComputePipelineKey(create) : 0;
            return cache.m_graphics.device.createComputePipelines(cache.m_driver_cache, 1, &create, nullptr, out);
		};
		vk::Result          result = vk::Result::eSuccess;
		const auto          begin  = CompileClockNs();
		HangWatchdog::Scope compile("compute-pipeline-prewarm", info.stage.program->shader_hash, id);
		CreatePipelineInternal(cache.m_graphics, *pipeline, info, module, cache.m_driver_cache, &hook, &result);
		const auto ns = CompileClockNs() - begin;
		if (result != vk::Result::eSuccess || pipeline->pipeline == nullptr) {
			failed.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		cache.NoteKnownPipeline(key);
		cache.NotePipelineCreated(ns);
		g_compile_totals.prewarm_ns.fetch_add(ns, std::memory_order_relaxed);
		Common::LockGuard lock(cache.m_mutex);
		if (cache.m_compute_pipelines.contains(id)) {
			cache.m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			ReleasePipelineLayout(cache.m_graphics, pipeline->pipeline_layout, pipeline->descriptor_set_layout);
			dropped.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		pipeline->prewarmed = true;
		GpuOpProfiler::RegisterComputePipeline(pipeline->pipeline, id);
		cache.m_compute_pipelines.emplace(id, std::move(pipeline));
		g_compile_totals.prewarm_built.fetch_add(1, std::memory_order_relaxed);
	}

	PipelineCache&               cache;
	std::atomic<bool>            stopped {false};
	std::atomic<uint64_t>        requested {0}, dropped {0}, failed {0}, ahead {0}, waits {0}, wait_ns {0};
	std::mutex                   inflight_mutex;
	std::condition_variable      inflight_cv;
	std::unordered_set<uint64_t> inflight; // ids requested ahead, not built yet
	std::unordered_set<uint64_t> asked;    // every id requested ahead (each is built once)
	PipelineCompileQueue         queue;    // last: joined before the rest goes
};

// KYTY_PIPELINE_LIST (default off): a KYPLST1 pipeline list (pipelineList.h) made from recorded runs,
// for players who never ran the game: =1 reads PipelineLists/<title>_<version>.kyplst, else
// PipelineLists/<title>.kyplst (next to the emulator, the working directory), any other value is the file's path. The list must name this game's title and
// version (else it is ignored); each shader's code is read from the player's own executable at the
// recorded offset (or found elsewhere in it by its hashes) and used only when its hashes match. The
// matching shaders' permutations join the shader precompile replay, and the pipelines whose stages all
// matched join the pipeline replay (JournalReplay; boot level first, then the pipelines recorded in the
// most levels). Compute pipelines follow from the compute prewarm, on by default with the list.
// KYTY_PIPELINE_LIST_PREPARE=1 ("Prepare shaders"): builds everything at start-up with more threads
// (KYTY_SHADER_PRECOMPILE_THREADS and KYTY_PIPELINE_JOURNAL_THREADS default 8), prints progress (also
// as "KYTY_EVENT: prepare-progress" lines about once a second, common/kytyEvents.h), saves the caches
// and exits (exit code 0; 2 when there is no usable list) before the game starts.
struct PipelineCache::ListState {
	std::vector<std::string>             levels;
	std::vector<PipelineList::Pipeline>  pipelines;  // moved into the pipeline replay
	std::vector<ShaderJournal::Source>   sources;    // index-aligned with the list's shaders; code empty: unusable
	std::vector<ShaderJournal::Entry>    entries;    // the usable permutations (source: index into `sources`)
	std::vector<uint64_t>                identities; // per list permutation: this run's identity (0: unusable)
	std::vector<uint32_t>                wave_sizes; // per list permutation: the stage's guest wave size
	bool                                prepare = false;
};

// KYTY_PIPELINE_JOURNAL=1 (default off): every graphics pipeline the game creates is journaled by its
// content key (GraphicsContent: the create info with module content hashes, and the layout
// signature) in _PipelineCache/<title>.pipelines.journal (KYTY_PIPELINE_JOURNAL_PATH), with the boot
// level as marker. At start-up, once the shader precompile replay has published the programs, the
// journaled pipelines (and the usable pipelines of a KYTY_PIPELINE_LIST) are rebuilt on
// below-normal-priority threads (KYTY_PIPELINE_JOURNAL_THREADS, default 4, 8 with a list, 1..16:
// NVIDIA ran 4 about 3.8x in parallel on a cold cache) into the driver cache: the boot level's
// first, then this PC's own journal in the order the pipelines were first needed, then the list's,
// those recorded in the most levels first. Each rebuilt key becomes known, so the draw that needs it takes it from the driver
// cache with a probe (KYTY_PIPELINE_FAST_FIRST) instead of compiling. Keys the driver cache is already
// known to hold (KYTY_PIPELINE_KNOWN) are skipped (KYTY_PIPELINE_JOURNAL_FORCE=1 rebuilds them too). A
// record whose modules are not published (another translator with other bindings, a program the
// replay skipped) is skipped: a miss is wasted work at most, never a different pipeline. List records
// name their modules only by permutation identity (they hold no module hashes). The replay stops
// adding to the driver cache once its data exceeds KYTY_PIPELINE_LIST_CACHE_LIMIT_MB (default 448):
// beyond the 512 MB file cap the next start would discard the whole cache.
// KYTY_PIPELINE_JOURNAL_REPLAY=0 only records; KYTY_PIPELINE_JOURNAL_VERIFY=1 logs each pipeline a
// draw needed that the journal did not hold at start-up (the first 20, and the count at exit).
struct PipelineCache::JournalReplay {
	JournalReplay(PipelineCache& owner, uint64_t verify_mode)
	    : cache(owner), verify(verify_mode != 0), strict(verify_mode >= 2) {}
	~JournalReplay() { Stop(); }
	JournalReplay(const JournalReplay&)            = delete;
	JournalReplay& operator=(const JournalReplay&) = delete;

	// One pipeline to rebuild: a pipeline journal record (this PC's own runs) or a pipeline list's,
	// converted to the journal's payload with the permutation identities of this run.
	struct Item {
		PipelineJournal::Record record;
		uint32_t                rank = 0;    // after the boot level's records: higher first
		std::vector<uint16_t>   list_levels; // list records: indices into list_level_names
		bool                    from_list = false;
		std::vector<uint32_t>   stage_waves; // list records: per stage, the guest wave size (0: none)
	};

	// The boot level, read once the game's file system is mounted (after the cache is created).
	const std::string& Level() {
		std::call_once(level_once, [this] { level = BootLevelMarker(); });
		return level;
	}

	void Start(uint32_t thread_count, bool force_known) {
		force    = force_known;
		begin_ns = CompileClockNs();
		queued.store(items.size(), std::memory_order_release);
		running  = std::max<uint32_t>(1, thread_count);
		for (uint32_t i = 0; i < running; i++) threads.emplace_back([this] { Worker(); });
	}

	// The boot level's records first, then the rest by rank (stable: first-needed order within one).
	void BuildOrder() {
		const auto&           boot = Level();
		std::vector<uint32_t> rest;
		order.reserve(items.size());
		for (uint32_t i = 0; i < items.size(); i++) {
			const auto&    item = items[i];
			JournalPayload payload;
			if (!DecodeJournalPayload(item.record.payload, payload)) continue;
			bool current = !boot.empty() && payload.level == boot;
			for (const auto level: item.list_levels) {
				current = current || (!boot.empty() && level < list_level_names.size() && list_level_names[level] == boot);
			}
			(current ? order : rest).push_back(i);
		}
		first_level = static_cast<uint32_t>(order.size());
		std::stable_sort(rest.begin(), rest.end(), [&](uint32_t a, uint32_t b) { return items[a].rank > items[b].rank; });
		order.insert(order.end(), rest.begin(), rest.end());
		total.store(order.size(), std::memory_order_release);
	}

	// Once, before the shader precompile replay goes (its Wait is what the workers block in).
	void Stop() {
		if (stopped_once.exchange(true)) return;
		stop.store(true, std::memory_order_release);
		if (cache.m_shader_precompiler != nullptr) cache.m_shader_precompiler->Stop();
		for (auto& thread: threads) {
			if (thread.joinable()) thread.join();
		}
		threads.clear();
		if (logged_exit.exchange(true)) return;
		PipelineCacheLog("Pipeline journal: {} recorded this run; replay built {} ({} from the pipeline list; {:.1f} ms), "
		                 "skipped {} known to the driver cache, {} with modules not published, {} failed, {} beyond "
		                 "the driver cache limit, {} not started; {} of {} pipelines created were not in the journal "
		                 "at start-up",
		                 cache.m_journal != nullptr ? cache.m_journal->GetStats().recorded : 0, built.load(),
		                 built_list.load(), static_cast<double>(g_compile_totals.journal_ns.load()) / 1e6,
		                 skipped_known.load(), missing.load(), failed.load(), skipped_full.load(),
		                 order.size() - std::min<uint64_t>(order.size(), next.load()), unpredicted.load(),
		                 created.load());
		if (cache.m_adopt) {
			PipelineCacheLog("Pipeline journal: {} records the game had already created were not built; {}",
			                 already_created.load(), cache.AdoptReport());
		}
		if (track) {
			// Coverage: the pipelines the game created that the replay rebuilt or found known (or that
			// the journal held), by the content key both compute from the same modules and state.
			std::scoped_lock lock(keys_mutex);
			uint64_t         covered = 0;
			for (const auto key: created_keys) covered += replay_keys.contains(key) || loaded_keys.contains(key) ? 1 : 0;
			PipelineCacheLog("Pipeline replay coverage: {} of {} graphics pipelines the game created were predicted "
			                 "({} replay keys)",
			                 covered, created_keys.size(), replay_keys.size());
		}
		// KYTY_PIPELINE_JOURNAL_VERIFY=2 (tests): a replay that could not rebuild a record exactly, or a
		// pipeline the journal did not predict although it was loaded, fails the run.
		if (strict && (failed.load() != 0 || mismatched.load() != 0 || missing.load() != 0 ||
		               (!loaded_keys.empty() && unpredicted.load() != 0) || (!order.empty() && built.load() == 0))) {
			PipelineCacheLog("PipelineJournalTests: failed: {} failed, {} rebuilt with another key, {} with "
			                 "missing modules, {} unpredicted, {} of {} built",
			                 failed.load(), mismatched.load(), missing.load(), unpredicted.load(), built.load(),
			                 order.size());
		}
	}

	// Blocks until the replay workers are done (KYTY_PIPELINE_JOURNAL_WAIT=1: tests, measurements;
	// KYTY_PIPELINE_LIST_PREPARE).
	void Wait() {
		for (auto& thread: threads) {
			if (thread.joinable()) thread.join();
		}
		threads.clear();
	}

	void Worker() {
		if (low_priority) LowerCurrentThreadPriority();
		Profiler::SetThreadName("PipelineJournal");
		Common::PlaceCurrentThreadCold();
		// The pipelines need the programs' modules: wait for the shader precompile replay. Early
		// (KYTY_PIPELINE_JOURNAL_EARLY=1): start at once; a record whose modules the replay has not
		// published yet is set aside and tried again once the replay is done.
		auto* precompiler = cache.m_shader_precompiler.get();
		if (precompiler != nullptr && !early) precompiler->Wait();
		std::call_once(order_once, [this] {
			BuildOrder();
			last_report_ns = CompileClockNs();
		});
		while (!stop.load(std::memory_order_acquire)) {
			const auto index = next.fetch_add(1, std::memory_order_relaxed);
			if (index >= order.size()) break;
			const bool can_defer = early && precompiler != nullptr && !precompiler->GetProgress().finished;
			if (!Replay(items[order[index]], can_defer)) {
				if (early_rounds) {
					// KYTY_PIPELINE_JOURNAL_EARLY=2: set aside at once (a large journal waited 2 s per
					// record), retried in rounds below while the shader replay publishes.
					std::scoped_lock lock(deferred_mutex);
					deferred.push_back(order[index]);
					continue;
				}
				// The replay publishes in about the order the pipelines were first needed: wait up to
				// 2 s for this record's modules, then set it aside.
				bool       replayed = false;
				const auto waited   = CompileClockNs();
				while (!stop.load(std::memory_order_acquire) && CompileClockNs() - waited < 2'000'000'000ull) {
					std::this_thread::sleep_for(std::chrono::milliseconds(20));
					const bool finished = precompiler->GetProgress().finished;
					if (Replay(items[order[index]], !finished)) {
						replayed = true;
						break;
					}
				}
				if (!replayed) {
					std::scoped_lock lock(deferred_mutex);
					deferred.push_back(order[index]);
					continue;
				}
			}
			done.fetch_add(1, std::memory_order_relaxed);
			MaybeReport();
		}
		if (early_rounds && precompiler != nullptr) RetryDeferredRounds(*precompiler);
		if (early && precompiler != nullptr && !stop.load(std::memory_order_acquire)) {
			precompiler->Wait();
			while (!stop.load(std::memory_order_acquire)) {
				uint32_t item = 0;
				{
					std::scoped_lock lock(deferred_mutex);
					if (deferred.empty()) break;
					item = deferred.back();
					deferred.pop_back();
				}
				(void)Replay(items[item], false);
				done.fetch_add(1, std::memory_order_relaxed);
				MaybeReport();
			}
		}
		if (running.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			size_t cache_bytes = 0;
			if (cache.m_driver_cache != nullptr) {
				(void)cache.m_graphics.device.getPipelineCacheData(cache.m_driver_cache, &cache_bytes, nullptr);
			}
			PipelineCacheLog("Pipeline journal: replay {} after {:.1f} s: {} built ({:.1f} ms; {} with another "
			                 "translator's modules, {} from the pipeline list), {} skipped as known, {} with modules "
			                 "not published, {} failed, {} beyond the driver cache limit ({} of the boot level '{}' "
			                 "first); driver cache {} bytes",
			                 stop.load() ? "stopped" : "done",
			                 static_cast<double>(CompileClockNs() - begin_ns) / 1e9, built.load(),
			                 static_cast<double>(g_compile_totals.journal_ns.load()) / 1e6, retargets.load(),
			                 built_list.load(), skipped_known.load(), missing.load(), failed.load(),
			                 skipped_full.load(), first_level, Level(), cache_bytes);
			items = {}; // the last worker: nothing reads the records any more
		}
	}

	// KYTY_PIPELINE_JOURNAL_EARLY=2: while the shader replay runs, the set-aside records are tried
	// again about every second once it has published more; those still missing stay set aside for
	// the final pass after the replay. Each worker takes the whole list in turn.
	void RetryDeferredRounds(ShaderPrecompiler& precompiler) {
		uint64_t last_done = UINT64_MAX;
		while (!stop.load(std::memory_order_acquire)) {
			const auto progress = precompiler.GetProgress();
			if (progress.finished) return;
			if (progress.done == last_done) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
				continue;
			}
			last_done = progress.done;
			std::vector<uint32_t> round;
			{
				std::scoped_lock lock(deferred_mutex);
				round.swap(deferred);
			}
			std::vector<uint32_t> still;
			for (const auto item: round) {
				if (stop.load(std::memory_order_acquire) || !Replay(items[item], true)) {
					still.push_back(item);
					continue;
				}
				done.fetch_add(1, std::memory_order_relaxed);
				MaybeReport();
			}
			{
				std::scoped_lock lock(deferred_mutex);
				deferred.insert(deferred.end(), still.begin(), still.end());
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		}
	}

	// Records finished and records in all for the prepare run's progress events: the queued count until
	// the workers have ordered the records (they wait for the shader replay first).
	[[nodiscard]] std::pair<uint64_t, uint64_t> Progress() const {
		const auto ordered = total.load(std::memory_order_acquire);
		return {done.load(std::memory_order_relaxed), ordered != 0 ? ordered : queued.load(std::memory_order_acquire)};
	}

	// A progress line every report_ns (any worker; the one that wins the exchange prints it).
	void MaybeReport() {
		const auto now  = CompileClockNs();
		auto       last = last_report_ns.load(std::memory_order_relaxed);
		if (now - last < report_ns || !last_report_ns.compare_exchange_strong(last, now)) return;
		const auto all = total.load(std::memory_order_acquire);
		const auto d   = done.load(std::memory_order_relaxed);
		PipelineCacheLog("Pipeline replay: {}/{} ({}%): {} built, {} known, {} not published, {} failed, {:.1f} s",
		                 d, all, all != 0 ? d * 100 / all : 100, built.load(), skipped_known.load(), missing.load(),
		                 failed.load(), static_cast<double>(now - begin_ns) / 1e9);
	}

	// Whether the driver cache has reached the replay's limit (checked every 32 builds).
	bool CacheFull() {
		if (cache_limit_bytes == 0 || cache.m_driver_cache == nullptr) return false;
		if (cache_full.load(std::memory_order_relaxed)) return true;
		if (full_checks.fetch_add(1, std::memory_order_relaxed) % 32 != 0) return false;
		size_t size = 0;
		if (cache.m_graphics.device.getPipelineCacheData(cache.m_driver_cache, &size, nullptr) == vk::Result::eSuccess &&
		    size > cache_limit_bytes && !cache_full.exchange(true)) {
			PipelineCacheLog("Pipeline replay: the driver cache holds {} bytes, over the replay limit of {} bytes "
			                 "(KYTY_PIPELINE_LIST_CACHE_LIMIT_MB); the remaining records are not built",
			                 size, cache_limit_bytes);
		}
		return cache_full.load(std::memory_order_relaxed);
	}

	// False (nothing counted) when can_defer and a module is not published yet: the caller tries
	// the record again after the shader precompile replay.
	bool Replay(const Item& item, bool can_defer) {
		const auto&    record = item.record;
		JournalPayload payload;
		if (!DecodeJournalPayload(record.payload, payload)) {
			failed.fetch_add(1, std::memory_order_relaxed);
			return true;
		}
		// Known to the driver cache: nothing to build, but only while the record's own modules are the
		// ones this run uses (after a translator or preset change the record's key is still known
		// although the game now needs the retargeted pipeline: checked below with its new key).
		const auto known_record = !force && !cache.m_adopt && !item.from_list && cache.IsKnownPipeline(record.key);
		if (known_record && (!retarget || RecordModulesPublished(payload))) {
			skipped_known.fetch_add(1, std::memory_order_relaxed);
			g_compile_totals.journal_skipped.fetch_add(1, std::memory_order_relaxed);
			NoteReplayKey(record.key);
			return true;
		}
		std::span<const uint32_t> words(payload.words);
		uint32_t                  words_snapshot = payload.snapshot_words;
		// A list pipeline carries the recording device's feedback-loop dynamic state and required subgroup
		// sizes (the list stores the canonical form, PipelineList::ApplySnapshotDeviceState): this device's
		// values go in before the pipeline is described and built, so its key is the one the game makes.
		std::vector<uint32_t> device_words;
		if (item.from_list) {
			const auto& graphics = cache.m_graphics;
			device_words         = payload.words;
			const auto subgroup  = [&](uint32_t stage_flags, uint32_t index) -> uint32_t {
				const auto wave = index < item.stage_waves.size() ? item.stage_waves[index] : 0u;
				if (wave == 0) return 0;
				if (stage_flags == static_cast<uint32_t>(vk::ShaderStageFlagBits::eMeshEXT)) {
					return graphics.GraphicsSubgroupSize(vk::ShaderStageFlagBits::eMeshEXT, wave);
				}
				if (stage_flags == static_cast<uint32_t>(vk::ShaderStageFlagBits::eFragment)) {
					return graphics.GraphicsSubgroupSize(vk::ShaderStageFlagBits::eFragment, wave);
				}
				return 0;
			};
			if (PipelineList::ApplySnapshotDeviceState(device_words, words_snapshot, graphics.attachment_feedback_loop_enabled,
			                                           subgroup)) {
				words = device_words;
			} else {
				words_snapshot = payload.snapshot_words;
			}
		}
		const auto snapshot_words = words.first(words_snapshot);
		const auto signature      = words.subspan(words_snapshot + 1);
		auto&      device         = cache.m_graphics.device;
		// Pin the modules (a program cache teardown waits for the build).
		// A module of the recorded code, or else (KYTY_PIPELINE_JOURNAL_RETARGET, default on; always for
		// list records, which carry no module hashes) the module another translator revision made from
		// the same permutation inputs with the same bindings: the pipeline is then rebuilt with the new
		// code under its new key.
		std::vector<std::pair<uint64_t, vk::ShaderModule>> modules;
		bool                                               found      = true;
		bool                                               retargeted = false;
		const auto hashes = GraphicsPipelineSnapshot::SerializedModules(snapshot_words);
		for (size_t i = 0; i < hashes.size(); i++) {
			auto module = hashes[i] != 0 ? g_shader_modules.Pin(hashes[i]) : vk::ShaderModule {};
			if (module == nullptr && (retarget || hashes[i] == 0) && i < payload.identities.size() &&
			    payload.identities[i].first != 0) {
				module     = g_shader_modules.PinIdentity(payload.identities[i].first, payload.identities[i].second);
				retargeted = retargeted || module != nullptr;
			}
			if (module == nullptr) {
				found = false;
				break;
			}
			modules.emplace_back(hashes[i], module);
		}
		const auto unpin = [&] {
			for (const auto& [hash, module]: modules) g_shader_modules.Unpin(device, module);
		};
		// The layout from its signature (pipeline/shaders.cpp: interned flag, push descriptors, push
		// stages, push size, then binding, type, count and stages per binding).
		if (!found && can_defer) {
			unpin();
			return false;
		}
		// A pipeline that samples bindless images ends its signature with one marker word and has
		// the bindless table as set 1 (pipeline/shaders.cpp).
		const bool bindless = signature.size() >= 5 && (signature.size() - 5) % 4 == 0 && signature.back() == 0xb1d1e55u;
		const auto binding_words = bindless ? signature.size() - 1 : signature.size();
		if (!found || modules.empty() || signature.size() < 4 || (!bindless && (signature.size() - 4) % 4 != 0)) {
			unpin();
			missing.fetch_add(1, std::memory_order_relaxed);
			return true;
		}
		vk::DescriptorSetLayout bindless_set = nullptr;
		if (bindless) {
			bindless_set = BindlessSetLayout(cache.m_graphics);
			if (bindless_set == nullptr) {
				unpin();
				missing.fetch_add(1, std::memory_order_relaxed);
				return true;
			}
		}
		std::vector<vk::DescriptorSetLayoutBinding> bindings;
		for (size_t i = 4; i < binding_words; i += 4) {
			bindings.push_back({signature[i], static_cast<vk::DescriptorType>(signature[i + 1]), signature[i + 2],
			                    vk::ShaderStageFlags(signature[i + 3]), nullptr});
		}
		const auto layouts = AcquirePipelineLayout(cache.m_graphics, bindings, vk::ShaderStageFlags(signature[2]),
		                                           signature[3], bindless_set);
		// Modules by stage index: list records have every hash 0.
		uint32_t   lookup   = 0;
		const auto snapshot = GraphicsPipelineSnapshot::Deserialize(
		    snapshot_words,
		    [&](uint64_t hash) {
			    const auto index = lookup++;
			    if (index < modules.size() && modules[index].first == hash) return modules[index].second;
			    for (const auto& [h, module]: modules) {
				    if (h == hash && hash != 0) return module;
			    }
			    return vk::ShaderModule {};
		    },
		    layouts.pipeline_layout);
		vk::Pipeline pipeline = nullptr;
		bool         ok       = false;
		if (snapshot != nullptr) {
			// The rebuilt create info must describe the journaled pipeline exactly.
			// The layout signature's push-descriptor word is the device's choice (maxPushDescriptors), the
			// one the layout above was made with.
			std::vector<uint32_t> device_signature(signature.begin(), signature.end());
			if (item.from_list && device_signature.size() >= 2) device_signature[1] = layouts.uses_push_descriptors ? 1u : 0u;
			const auto check = DescribeGraphicsPipeline(snapshot->Info(), device_signature);
			if (check.key != 0) NoteReplayKey(check.key);
			if (retargeted && check.key != 0 && !force && !cache.m_adopt && cache.IsKnownPipeline(check.key)) {
				skipped_known.fetch_add(1, std::memory_order_relaxed);
				g_compile_totals.journal_skipped.fetch_add(1, std::memory_order_relaxed);
				ok = true;
			} else if (cache.m_adopt && check.key != 0 && CreatedByGame(check.key)) {
				// The game made it already: an adopted copy would never be taken.
				already_created.fetch_add(1, std::memory_order_relaxed);
				ok = true;
			} else if (CacheFull()) {
				skipped_full.fetch_add(1, std::memory_order_relaxed);
				ok = true;
			} else if (check.key == record.key || (retargeted && check.key != 0)) {
				if (retargeted && !item.from_list) retargets.fetch_add(1, std::memory_order_relaxed);
				const auto          begin = CompileClockNs();
				HangWatchdog::Scope compile("graphics-pipeline-journal", record.key, 0, 0);
				// A driver fault on a rebuilt create info (another translator's modules) costs the
				// record, not the session.
				BuildContext context {&cache, &snapshot->Info(), &pipeline};
				const auto   fault = Common::CallCatchingStructuredException(&BuildSafely, &context);
				const auto   ns    = CompileClockNs() - begin;
				ok                 = fault == 0 && context.result == vk::Result::eSuccess && pipeline != nullptr;
				if (ok) {
					g_compile_totals.journal_ns.fetch_add(ns, std::memory_order_relaxed);
					g_compile_totals.journal_built.fetch_add(1, std::memory_order_relaxed);
					built.fetch_add(1, std::memory_order_relaxed);
					if (item.from_list) built_list.fetch_add(1, std::memory_order_relaxed);
					cache.NoteKnownPipeline(check.key);
					cache.NotePipelineCreated(ns);
					if (cache.m_adopt && cache.OfferAdoptedPipeline(check.key, pipeline)) pipeline = nullptr;
				}
			} else {
				mismatched.fetch_add(1, std::memory_order_relaxed);
			}
		}
		if (pipeline != nullptr) device.destroyPipeline(pipeline, nullptr);
		ReleasePipelineLayout(cache.m_graphics, layouts.pipeline_layout, layouts.set_layout);
		unpin();
		if (!ok) failed.fetch_add(1, std::memory_order_relaxed);
		return true;
	}

	// Whether every module the record names by content hash is published in this run.
	static bool RecordModulesPublished(const JournalPayload& payload) {
		const std::span<const uint32_t> words(payload.words);
		for (const auto hash: GraphicsPipelineSnapshot::SerializedModules(words.first(payload.snapshot_words))) {
			if (hash == 0 || !g_shader_modules.Contains(hash)) return false;
		}
		return true;
	}

	struct BuildContext {
		PipelineCache*                        cache    = nullptr;
		const vk::GraphicsPipelineCreateInfo* info     = nullptr;
		vk::Pipeline*                         pipeline = nullptr;
		vk::Result                            result   = vk::Result::eErrorUnknown;
	};
	static void BuildSafely(void* raw) {
		auto& context  = *static_cast<BuildContext*>(raw);
		context.result = PipelineFeedback::CreateGraphics(context.cache->m_graphics.device,
		                                                  context.cache->m_driver_cache, *context.info,
		                                                  context.pipeline, PipelineFeedback::Path::Replay);
	}

	void NoteReplayKey(uint64_t key) {
		if (!track) return;
		std::scoped_lock lock(keys_mutex);
		replay_keys.insert(key);
	}

	// The command processor created a graphics pipeline with this content key.
	[[nodiscard]] bool CreatedByGame(uint64_t key) {
		std::scoped_lock lock(game_keys_mutex);
		return game_keys.contains(key);
	}

	void NoteCreated(uint64_t key, const std::array<uint64_t, 4>& guest) {
		created.fetch_add(1, std::memory_order_relaxed);
		if (cache.m_adopt) {
			std::scoped_lock lock(game_keys_mutex);
			game_keys.insert(key);
		}
		if (track) {
			std::scoped_lock lock(keys_mutex);
			created_keys.push_back(key);
		}
		if (loaded_keys.contains(key)) return;
		unpredicted.fetch_add(1, std::memory_order_relaxed);
		g_compile_totals.journal_unpredicted.fetch_add(1, std::memory_order_relaxed);
		if (verify && logged_misses.fetch_add(1, std::memory_order_relaxed) < 20) {
			PipelineCacheLog("Pipeline journal verify: pipeline {:016x} (VS 0x{:016x}, PS 0x{:016x}) was not in the "
			                 "journal at start-up (level '{}')",
			                 key, guest[0], guest[3], Level());
		}
	}

	PipelineCache&               cache;
	std::string                  level;
	std::once_flag               level_once, order_once;
	const bool                   verify;
	const bool                   strict;
	bool                         force        = false;
	bool                         low_priority = true;
	// Coverage bookkeeping (KYTY_PIPELINE_LIST or _VERIFY): every key the replay computed and every key
	// the game created.
	bool                         track        = false;
	uint32_t                     first_level  = 0;
	uint64_t                     report_ns    = 10'000'000'000ull;
	size_t                       cache_limit_bytes = 0; // 0: no limit
	std::unordered_set<uint64_t> loaded_keys; // filled before Start; read-only afterwards
	std::vector<Item>            items;       // filled before Start
	std::vector<std::string>     list_level_names;
	std::vector<uint32_t>        order;
	std::atomic<uint64_t>        next {0}, done {0}, total {0}, queued {0}, last_report_ns {0}, full_checks {0};
	std::atomic<uint32_t>        running {0};
	std::atomic<bool>            stop {false}, logged_exit {false}, stopped_once {false}, cache_full {false};
	std::atomic<uint64_t> built {0}, built_list {0}, skipped_known {0}, missing {0}, failed {0}, mismatched {0},
	    retargets {0}, skipped_full {0};
	const bool            retarget = EnvU64("KYTY_PIPELINE_JOURNAL_RETARGET", 1) != 0;
	const bool            early    = EnvU64("KYTY_PIPELINE_JOURNAL_EARLY", 0) != 0;
	const bool            early_rounds = EnvU64("KYTY_PIPELINE_JOURNAL_EARLY", 0) >= 2;
	std::mutex            deferred_mutex;
	std::vector<uint32_t> deferred; // indices into items: modules not published at the first try
	std::atomic<uint64_t> created {0}, unpredicted {0}, logged_misses {0}, already_created {0};
	std::mutex                   game_keys_mutex; // KYTY_PIPELINE_ADOPT: content keys the game created
	std::unordered_set<uint64_t> game_keys;
	std::mutex                   keys_mutex;
	std::unordered_set<uint64_t> replay_keys;
	std::vector<uint64_t>        created_keys;
	uint64_t                     begin_ns = 0;
	std::vector<std::thread>     threads;
};

bool PipelineCache::IsKnownPipeline(uint64_t key) const {
	return key != 0 && m_known != nullptr && m_known->Contains(key);
}

void PipelineCache::NoteKnownPipeline(uint64_t key) {
	if (key != 0 && m_known != nullptr) (void)m_known->Add(key);
}

void PipelineCache::NoteJournalPipeline(uint64_t key, std::span<const uint32_t> words, uint32_t snapshot_words,
                                        std::span<const std::pair<uint64_t, uint64_t>> identities,
                                        const std::array<uint64_t, 4>& guest) {
	if (m_journal_replay != nullptr) m_journal_replay->NoteCreated(key, guest);
	if (m_journal == nullptr) return;
	if (!m_journal->Contains(key)) {
		(void)m_journal->Add(key, EncodePipelineJournalPayload(m_journal_replay != nullptr ? m_journal_replay->Level()
		                                                                                    : std::string(),
		                                                       words, snapshot_words, identities));
	}
}

// KYTY_PIPELINE_KNOWN=1 (default off; useful with KYTY_PIPELINE_FAST_FIRST on a device with pipeline
// creation cache control): the content keys of the pipelines the driver cache holds, in
// _PipelineCache/<title>.pipelines.known (KYTY_PIPELINE_KNOWN_PATH). A key is added when a pipeline
// is built optimized into the driver cache (or found there by a probe), and the list is written
// only with the driver cache file, up to the keys created before that file was serialized; it is
// tied to the driver cache signature and the cold salt, and emptied when the driver cache starts
// empty. Fast-first probes known keys instead of rebuilding them unoptimized (FastFirstState).
// KYTY_PIPELINE_JOURNAL or KYTY_PIPELINE_LIST alone keeps such a list in memory for the keys its
// replay built.
void PipelineCache::InitializeKnownPipelines() {
	const bool known   = EnvU64("KYTY_PIPELINE_KNOWN", 0) != 0;
	const bool journal = EnvU64("KYTY_PIPELINE_JOURNAL", 0) != 0;
	const bool list    = m_list != nullptr;
	if (!known && !journal && !list) return;
	// Before the shader precompile replay publishes modules: the journal and the list find them by
	// their identities.
	m_program_cache->identify_modules = journal || list;
	PipelineJournal::Settings settings;
	settings.background_writer = false;
	settings.sealed_writes     = true;
	settings.log               = [](const std::string& message) { PipelineCacheLog("{}", message); };
	if (known && m_driver_cache != nullptr && !m_driver_cache_path.empty()) {
		if (const auto* custom = std::getenv("KYTY_PIPELINE_KNOWN_PATH"); custom != nullptr && *custom != '\0') {
			settings.path = custom;
		} else {
			settings.path = m_driver_cache_path;
			settings.path.replace_extension(".pipelines.known");
		}
		const auto identity = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties()) +
		                      fmt::format("known pipelines 1; cold salt {}", EnvU64("KYTY_PIPELINE_COLD_SALT", 0));
		settings.identity.assign(identity.begin(), identity.end());
	}
	m_known = std::make_unique<PipelineJournal>(std::move(settings));
	if (!m_known->Path().empty()) {
		const auto stats = m_known->GetStats();
		if (!m_driver_cache_loaded && m_known->KnownCount() != 0) {
			m_known->Reset();
			PipelineCacheLog("Known pipelines: the driver cache started empty; {} known keys dropped", stats.loaded);
		} else {
			PipelineCacheLog("Known pipelines: {} keys loaded from {} ({:.0f} ms{})", stats.loaded,
			                 Common::PathToString(m_known->Path()), static_cast<double>(stats.load_ns) / 1e6,
			                 stats.header_rejected ? "; another driver or salt, replaced" : "");
		}
	}
	m_known->ReleaseLoaded();
}

vk::Pipeline PipelineCache::TakeAdoptedPipeline(uint64_t key) {
	if (!m_adopt || key == 0) return nullptr;
	vk::Pipeline pipeline = nullptr;
	{
		std::scoped_lock lock(m_adopt_mutex);
		if (const auto found = m_adopted.find(key); found != m_adopted.end()) {
			pipeline = found->second;
			m_adopted.erase(found);
		}
	}
	if (pipeline != nullptr) m_adopt_taken.fetch_add(1, std::memory_order_relaxed);
	return pipeline;
}

bool PipelineCache::OfferAdoptedPipeline(uint64_t key, vk::Pipeline pipeline) {
	if (!m_adopt || key == 0 || pipeline == nullptr) return false;
	std::scoped_lock lock(m_adopt_mutex);
	if (m_adopted.size() >= m_adopt_max || m_adopted.contains(key)) {
		m_adopt_refused.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	m_adopted.emplace(key, pipeline);
	m_adopt_offered.fetch_add(1, std::memory_order_relaxed);
	return true;
}

std::string PipelineCache::AdoptReport() {
	size_t waiting = 0;
	{
		std::scoped_lock lock(m_adopt_mutex);
		waiting = m_adopted.size();
	}
	return fmt::format("adopted pipelines: {} kept from the replay, {} taken by draws, {} waiting, {} not kept "
	                   "(KYTY_PIPELINE_ADOPT_MAX {} or a duplicate)",
	                   m_adopt_offered.load(), m_adopt_taken.load(), waiting, m_adopt_refused.load(), m_adopt_max);
}

void PipelineCache::InitializePipelineJournal() {
	const bool journal = EnvU64("KYTY_PIPELINE_JOURNAL", 0) != 0;
	const bool list    = m_list != nullptr && !m_list->pipelines.empty();
	if ((!journal && !list) || m_known == nullptr) return;
	m_adopt_max = std::min<uint64_t>(EnvU64("KYTY_PIPELINE_ADOPT_MAX", 12000), 1u << 20u);
	m_adopt     = EnvU64("KYTY_PIPELINE_ADOPT", 0) != 0 && m_adopt_max != 0;
	if (m_adopt) {
		PipelineCacheLog("Pipeline journal: the replay keeps up to {} built pipelines for the draws that need "
		                 "them, known ones included (KYTY_PIPELINE_ADOPT)",
		                 m_adopt_max);
	}
	const auto verify = EnvU64("KYTY_PIPELINE_JOURNAL_VERIFY", 0);
	m_journal_replay  = std::make_unique<JournalReplay>(*this, verify);
	auto& replay      = *m_journal_replay;
	replay.track      = list || verify != 0;
	if (journal) {
		std::filesystem::path path;
		if (const auto* custom = std::getenv("KYTY_PIPELINE_JOURNAL_PATH"); custom != nullptr && *custom != '\0') {
			path = custom;
		} else if (const auto title_id = PipelineCacheFileStem(); !title_id.empty()) {
			path = std::filesystem::path("_PipelineCache") / (title_id + ".pipelines.journal");
			SeedRenderDocCacheFile(path, ".pipelines.journal");
		}
		if (!path.empty()) {
			// The records hold module content hashes and Vulkan state: tied to the device, not the driver
			// (another driver rebuilds them all) nor the translator (other code hashes simply miss).
			const auto& properties = m_graphics.GetPhysicalDeviceProperties();
			const auto  identity   = fmt::format("kyty pipeline journal 1; device {:08x}:{:08x}", properties.vendorID,
			                                     properties.deviceID);
			PipelineJournal::Settings settings;
			settings.path     = path;
			settings.identity = std::vector<uint8_t>(identity.begin(), identity.end());
			settings.log      = [](const std::string& message) { PipelineCacheLog("{}", message); };
			m_journal         = std::make_unique<PipelineJournal>(std::move(settings));
			for (const auto& record: m_journal->Loaded()) {
				replay.loaded_keys.insert(record.key);
				replay.items.push_back({record, UINT32_MAX, {}, false});
			}
			m_journal->ReleaseLoaded();
			const auto stats = m_journal->GetStats();
			PipelineCacheLog("Pipeline journal: {}: {} pipelines loaded ({:.0f} ms{})", Common::PathToString(path),
			                 stats.loaded, static_cast<double>(stats.load_ns) / 1e6,
			                 stats.header_rejected ? "; another device or format, replaced" : "");
		}
	}
	if (list) {
		// The list's pipelines whose stages all resolved, with this run's permutation identities.
		uint64_t usable = 0;
		for (auto& pipeline: m_list->pipelines) {
			std::vector<std::pair<uint64_t, uint64_t>> identities;
			std::vector<uint32_t>                      waves;
			for (const auto& stage: pipeline.stages) {
				const auto identity = m_list->identities[stage.permutation];
				if (identity == 0) break;
				identities.emplace_back(stage.plain ? identity ^ PlainModuleIdentitySalt : identity, stage.bindings);
				waves.push_back(m_list->wave_sizes[stage.permutation]);
			}
			if (identities.size() != pipeline.stages.size()) continue;
			JournalReplay::Item item;
			item.record.key     = pipeline.key;
			item.record.payload = EncodePipelineJournalPayload({}, pipeline.words, pipeline.snapshot_words, identities);
			item.rank           = static_cast<uint32_t>(pipeline.levels.size());
			item.list_levels    = std::move(pipeline.levels);
			item.from_list      = true;
			item.stage_waves    = std::move(waves);
			replay.items.push_back(std::move(item));
			usable++;
		}
		replay.list_level_names = std::move(m_list->levels);
		m_list->pipelines       = {};
		const auto limit_mb     = EnvU64("KYTY_PIPELINE_LIST_CACHE_LIMIT_MB", 448);
		replay.cache_limit_bytes = static_cast<size_t>(limit_mb) * 1024 * 1024;
		PipelineCacheLog("Pipeline list: {} pipelines queued for the replay (driver cache limit {} MB)", usable,
		                 limit_mb);
	}
	// Without a driver cache a rebuilt pipeline helps nobody; KYTY_PIPELINE_JOURNAL_FORCE=1 rebuilds
	// anyway (tests).
	const bool force   = EnvU64("KYTY_PIPELINE_JOURNAL_FORCE", 0) != 0;
	const bool prepare = m_list != nullptr && m_list->prepare;
	if (EnvU64("KYTY_PIPELINE_JOURNAL_REPLAY", 1) == 0 || replay.items.empty() ||
	    (m_driver_cache == nullptr && !force)) {
		replay.items = {};
		return;
	}
	replay.low_priority = !prepare;
	replay.report_ns    = prepare ? 5'000'000'000ull : 10'000'000'000ull;
	// With a pipeline list (a first start builds hundreds at once) 8: a cold spring_punch_sky_aztec
	// stalled 5.5 s on 8 threads, 10.3 s on 4 (the level needs ~290 pipelines in its first 20 s).
	const auto threads  = std::clamp<uint64_t>(EnvU64("KYTY_PIPELINE_JOURNAL_THREADS", prepare || list ? 8 : 4), 1, 16);
	replay.Start(static_cast<uint32_t>(threads), force);
	if (EnvU64("KYTY_PIPELINE_JOURNAL_WAIT", 0) != 0) replay.Wait();
}

void PipelineCache::StopBackgroundPipelines() {
	if (m_journal_replay != nullptr) m_journal_replay->Stop();
	if (m_compute_prewarm != nullptr) m_compute_prewarm->Stop();
}

namespace {

// A pipeline list source's device-dependent stage input fields for this device (the host subgroup
// size of compute and mesh stages, mesh group splitting and passes; as GetComputeProgram and FinishMeshStage
// set them), and its static key derived again from the input info, as BuildKey does. False when the
// source cannot be used on this device.
bool NormalizeListSource(const GraphicContext& graphics, ShaderJournal::Source& source, uint32_t* stage_wave = nullptr) {
	if (stage_wave != nullptr) *stage_wave = source.wave_size;
	const auto load = [&](auto& info) {
		constexpr auto stored = ShaderStoredInputInfoSize<std::remove_cvref_t<decltype(info)>>();
		if (source.input_info.size() != stored) return false;
		std::memcpy(static_cast<void*>(&info), source.input_info.data(), stored);
		info.stage = {};
		ShaderSanitizeStoredInputInfo(info);
		return true;
	};
	const auto store = [&](const auto& info) {
		const auto* bytes = reinterpret_cast<const uint8_t*>(&info);
		source.input_info.assign(
		    bytes, bytes + ShaderStoredInputInfoSize<std::remove_cvref_t<decltype(info)>>());
		source.static_state.clear();
		BuildStageStaticKey(info, source.static_state);
	};
	switch (source.kind) {
		case ShaderJournal::Kind::Vertex: {
			auto info = std::make_unique<ShaderVertexInputInfo>();
			if (!load(*info)) return false;
			if (info->logical_stage == ShaderType::Mesh) {
				if (!graphics.mesh_shader_enabled) return false;
				auto&      mesh     = info->mesh;
				if (stage_wave != nullptr) *stage_wave = mesh.wave_size;
				const auto required = graphics.GraphicsSubgroupSize(vk::ShaderStageFlagBits::eMeshEXT, mesh.wave_size);
				mesh.host_subgroup_size = required != 0 ? required : graphics.subgroup_size;
				const auto& limits      = graphics.mesh_shader_properties;
				mesh.split_groups = limits.maxMeshWorkGroupCount[0] < limits.maxMeshWorkGroupTotalCount ? 1u : 0u;
				mesh.passes = mesh.PassesFor(std::min(limits.maxMeshWorkGroupInvocations, limits.maxMeshWorkGroupSize[0]));
				if (mesh.passes == 0) return false;
			}
			store(*info);
			return true;
		}
		case ShaderJournal::Kind::Pixel: {
			auto info = std::make_unique<ShaderPixelInputInfo>();
			if (!load(*info)) return false;
			store(*info);
			return true;
		}
		case ShaderJournal::Kind::Compute: {
			auto info = std::make_unique<ShaderComputeInputInfo>();
			if (!load(*info)) return false;
			info->host_subgroup_size = graphics.SupportsComputeWave64() ? 64u : 32u;
			store(*info);
			return true;
		}
	}
	return false;
}

} // namespace

void PipelineCache::InitializePipelineList() {
	// Unset (default): PipelineLists/<title>.kyplst next to the emulator when it exists (a release
	// ships them; nothing is logged when there is none). "1": that file, logged when missing. "0": off.
	// Any other value: that file.
	const auto* setting = std::getenv("KYTY_PIPELINE_LIST");
	const bool  prepare = EnvU64("KYTY_PIPELINE_LIST_PREPARE", 0) != 0;
	const bool  unset   = setting == nullptr || *setting == '\0';
	const bool  off     = !unset && std::strcmp(setting, "0") == 0;
	if (off && !prepare) return;
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) return;
	const auto begin = CompileClockNs();
	std::string game_version;
	(void)Loader::SystemContentParamSfoGetString("APP_VER", &game_version);
	// The default file: PipelineLists/<title>_<version>.kyplst (one list per game version; adapted from
	// Erlou93/KytyPS5-Yotei 9daeb9e), else PipelineLists/<title>.kyplst.
	auto default_path = std::filesystem::path("PipelineLists") / (title_id + "_" + game_version + ".kyplst");
	{
		std::error_code versioned_error;
		if (game_version.empty() || !std::filesystem::exists(default_path, versioned_error)) {
			default_path = std::filesystem::path("PipelineLists") / (title_id + ".kyplst");
		}
	}
	const auto path = unset || off || std::strcmp(setting, "1") == 0 ? default_path : std::filesystem::path(setting);
	std::vector<uint8_t> bytes;
	{
		std::error_code exists_error;
		if (unset && !prepare && !std::filesystem::exists(path, exists_error)) return;
		std::ifstream   in(path, std::ios::binary);
		std::error_code error;
		const auto      size = std::filesystem::file_size(path, error);
		if (!in || error || size > 256ull * 1024 * 1024) {
			PipelineCacheLog("Pipeline list: {} not found", Common::PathToString(path));
			return;
		}
		bytes.resize(static_cast<size_t>(size));
		if (!in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
			PipelineCacheLog("Pipeline list: {} could not be read", Common::PathToString(path));
			return;
		}
	}
	PipelineList::File file;
	std::string        error;
	if (!PipelineList::Decode(bytes, file, &error)) {
		PipelineCacheLog("Pipeline list: {} is not usable ({})", Common::PathToString(path), error);
		return;
	}
	bytes = {};
	std::string version;
	(void)Loader::SystemContentParamSfoGetString("APP_VER", &version);
	// KYTY_PIPELINE_LIST_CROSS=1: a list of another version or SKU is used best-effort; its shaders still
	// have to match this game's executable by their hashes.
	const bool other_game = file.info.title_id != title_id || file.info.app_version != version;
	const bool cross      = other_game && EnvU64("KYTY_PIPELINE_LIST_CROSS", 0) != 0;
	if (other_game && !cross) {
		PipelineCacheLog("Pipeline list: {} is for {} version {}, the game is {} version {}: ignored",
		                 Common::PathToString(path), file.info.title_id, file.info.app_version, title_id, version);
		return;
	}
	// A game folder's file, or a member of a game archive (a .zar game: "<game>.zar!/eboot.bin").
	const auto executable = PipelineListExecutable();
	PipelineListGameFile code_file(executable);
	if (executable.empty() || !code_file.Ok()) {
		PipelineCacheLog("Pipeline list: the game executable ({}) cannot be read: ignored",
		                 Common::PathToString(executable));
		return;
	}
	// Data files (list format with a files section): opened once each, next to the executable (/app0).
	std::map<uint32_t, std::unique_ptr<PipelineListGameFile>> data_files;
	const PipelineList::ReadCodeFn read = [&](PipelineList::CodeFile which, uint64_t offset, std::span<uint8_t> out) {
		PipelineListGameFile* in    = &code_file;
		const auto            index = static_cast<uint32_t>(which);
		if (index != 0) {
			auto& opened = data_files[index];
			if (opened == nullptr && index <= file.files.size()) {
				opened = std::make_unique<PipelineListGameFile>(executable.parent_path() /
				                                                Common::PathFromUtf8(file.files[index - 1]));
			}
			if (opened == nullptr || !opened->Ok()) return false;
			in = opened.get();
		}
		return in->Read(offset, out);
	};
	const PipelineList::ReadImageFn read_image = [&](PipelineList::CodeFile) {
		std::vector<uint8_t> image;
		uint64_t             size = 0;
		if (!code_file.Size(size) || size > 4ull * 1024 * 1024 * 1024) return image;
		image.resize(static_cast<size_t>(size));
		if (!code_file.Read(0, image)) image.clear();
		return image;
	};
	const std::array<uint32_t, 3> sizes {ShaderStoredInputInfoSize<ShaderVertexInputInfo>(),
	                                     ShaderStoredInputInfoSize<ShaderPixelInputInfo>(),
	                                     ShaderStoredInputInfoSize<ShaderComputeInputInfo>()};
	const auto                 int21 = Int21VertexLayout();
	PipelineList::UpgradeStats upgrade_stats;
	if (file.info.input_info_sizes[0] == int21.from_size) {
		PipelineList::UpgradeVertexInputInfos(file, int21, sizes[0], upgrade_stats);
		// Its image specializations end after `fmask`; they are encoded again in this build's format.
		uint64_t specializations = 0;
		for (auto& permutation: file.permutations) {
			ShaderRecompiler::IR::ResourceSpecialization specialization;
			if (ShaderRecompiler::IR::DecodeSpecialization(permutation.specialization, specialization) ||
			    !ShaderRecompiler::IR::DecodeSpecializationInt21(permutation.specialization, specialization)) {
				continue;
			}
			permutation.specialization.clear();
			ShaderRecompiler::IR::EncodeSpecialization(specialization, permutation.specialization);
			specializations++;
		}
		PipelineCacheLog("Pipeline list: int21 layout: {} vertex input infos upgraded ({} tessellation ones left "
		                 "out), {} specializations upgraded",
		                 upgrade_stats.upgraded, upgrade_stats.skipped, specializations);
	}
	PipelineList::ResolveStats stats;
	auto resolved = PipelineList::Resolve(file, sizes, read, read_image, stats);
	if (stats.resolved * 2 < stats.shaders) {
		PipelineCacheLog("Pipeline list: only {} of {} shaders match this game's files ({} differ, {} with "
		                 "another input layout, made by {}): ignored",
		                 stats.resolved, stats.shaders, stats.mismatched, stats.other_input_layout, file.info.producer);
		return;
	}
	if (file.info.state_format != PipelineList::StateFormat && !file.pipelines.empty()) {
		PipelineCacheLog("Pipeline list: {} was made by {} with pipeline state format {} (this build: {}); its {} "
		                 "pipelines are not replayed, only its shaders are precompiled",
		                 Common::PathToString(path), file.info.producer, file.info.state_format,
		                 PipelineList::StateFormat, file.pipelines.size());
		file.pipelines.clear();
	}
	// This device's inputs and static keys; the permutations' identities follow from them.
	uint64_t              not_here = 0;
	std::vector<uint32_t> source_waves(resolved.sources.size(), 0);
	for (size_t i = 0; i < resolved.sources.size(); i++) {
		if (resolved.usable[i] && !NormalizeListSource(m_graphics, resolved.sources[i], &source_waves[i])) {
			resolved.usable[i] = false;
			resolved.sources[i].code.clear();
			not_here++;
		}
	}
	auto state = std::make_unique<ListState>();
	state->identities.assign(file.permutations.size(), 0);
	state->wave_sizes.assign(file.permutations.size(), 0);
	for (size_t k = 0; k < resolved.entries.size(); k++) {
		const auto& entry = resolved.entries[k];
		if (!resolved.usable[entry.source]) continue;
		state->wave_sizes[resolved.entry_permutation[k]] = source_waves[entry.source];
		state->identities[resolved.entry_permutation[k]] = PermutationIdentity(resolved.sources[entry.source], entry);
		state->entries.push_back(entry);
	}
	uint64_t pipelines = 0;
	for (const auto& pipeline: file.pipelines) {
		pipelines += std::all_of(pipeline.stages.begin(), pipeline.stages.end(),
		                         [&](const PipelineList::Stage& stage) { return state->identities[stage.permutation] != 0; })
		                 ? 1
		                 : 0;
	}
	state->sources   = std::move(resolved.sources);
	state->levels    = std::move(file.levels);
	state->pipelines = std::move(file.pipelines);
	state->prepare   = prepare;
	if (cross) {
		PipelineCacheLog("Pipeline list: retargeted list: {} of {} shaders matched (list {} {}, game {} {})",
		                 stats.resolved - not_here, stats.shaders, file.info.title_id, file.info.app_version, title_id, version);
	}
	PipelineCacheLog("Pipeline list: {} ({} {}, by {}): {} of {} shaders match the game ({} found at another offset, "
	                 "{} differ, {} unreadable, {} with another input layout, {} not usable on this device); {} "
	                 "permutations, {} of {} pipelines usable, {} levels ({:.0f} ms)",
	                 Common::PathToString(path), file.info.title_id, file.info.app_version, file.info.producer,
	                 stats.resolved - not_here, stats.shaders, stats.relocated, stats.mismatched, stats.unreadable,
	                 stats.other_input_layout, not_here, state->entries.size(), pipelines, state->pipelines.size(),
	                 state->levels.size(), static_cast<double>(CompileClockNs() - begin) / 1e6);
	m_list = std::move(state);
}

// KYTY_PIPELINE_LIST_PREPARE=1: waits for the shader replay, the pipeline replay, the compute prewarm
// and the ray-tracing kernels, saves the caches and ends the process.
void PipelineCache::RunListPrepare() {
	const auto begin = CompileClockNs();
	if (m_list == nullptr) {
		PipelineCacheLog("Prepare shaders: no usable pipeline list (KYTY_PIPELINE_LIST); nothing to do");
		std::fflush(nullptr);
		std::_Exit(2);
	}
	PipelineCacheLog("Prepare shaders: building the pipeline list's shaders and pipelines");
	// About once a second, how far the replays are, as an event line (common/kytyEvents.h) for the
	// program that started the run (the launcher's Precompile window shows it as a progress bar).
	std::atomic<const char*> phase {"shaders"};
	const auto progress = [&] {
		ShaderPrecompiler::Progress   shaders;
		std::pair<uint64_t, uint64_t> pipelines, compute;
		if (m_shader_precompiler != nullptr) shaders = m_shader_precompiler->GetProgress();
		if (m_journal_replay != nullptr) pipelines = m_journal_replay->Progress();
		if (m_compute_prewarm != nullptr) compute = m_compute_prewarm->Progress();
		const auto counts = fmt::format("phase={} shaders={}/{} pipelines={}/{} compute={}/{}", phase.load(),
		                                shaders.done, shaders.total, pipelines.first, pipelines.second, compute.first,
		                                compute.second);
		PipelineCacheLog("{}", KytyEvents::FormatEvent("prepare-progress", counts));
	};
	std::mutex                  progress_mutex;
	std::condition_variable_any progress_cv;
	std::jthread reporter([&](const std::stop_token& stop) {
		std::unique_lock lock(progress_mutex);
		while (!progress_cv.wait_for(lock, stop, std::chrono::seconds(1), [] { return false; }) &&
		       !stop.stop_requested()) {
			progress();
		}
	});
	if (m_shader_precompiler != nullptr) m_shader_precompiler->Wait();
	const auto shaders_ns = CompileClockNs() - begin;
	phase.store("pipelines");
	if (m_journal_replay != nullptr) m_journal_replay->Wait();
	phase.store("compute");
	if (m_compute_prewarm != nullptr) m_compute_prewarm->Drain();
	if (m_rt != nullptr) m_rt->WaitAll();
	// Saving ends the replays and frees what the counts are read from: the last line goes out first.
	reporter.request_stop();
	reporter.join();
	phase.store("saving");
	progress();
	size_t     size     = 0;
	if (m_driver_cache != nullptr) (void)m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
	Save();
	const auto& t = g_compile_totals;
	PipelineCacheLog("Prepare shaders: done in {:.1f} s (shaders {:.1f} s, pipelines and saving {:.1f} s): {} "
	                 "permutations replayed, {} graphics pipelines built, {} compute pipelines prewarmed; driver "
	                 "cache {} bytes{}",
	                 static_cast<double>(CompileClockNs() - begin) / 1e9, static_cast<double>(shaders_ns) / 1e9,
	                 static_cast<double>(CompileClockNs() - begin - shaders_ns) / 1e9, t.replayed.load(),
	                 t.journal_built.load(), t.prewarm_built.load(), size,
	                 size > DriverCacheFileCap() ? " (over the file cap, KYTY_DRIVER_CACHE_MAX_MB: the next start discards it)" : "");
	// The same counts as an event line (known: pipelines the driver cache already held).
	const auto counts =
	    fmt::format("seconds={:.1f} shaders={} pipelines={} known={} compute={}",
	                static_cast<double>(CompileClockNs() - begin) / 1e9, t.replayed.load(), t.journal_built.load(),
	                t.journal_skipped.load(), t.prewarm_built.load());
	PipelineCacheLog("{}", KytyEvents::FormatEvent("prepare-done", counts));
	std::fflush(nullptr);
	std::_Exit(0);
}

// KYTY_PIPELINE_STATS=1: the "Pipelines 10s" line (pipelineStats.h), from the process-wide compile
// totals and this cache's software-RT totals, on a thread of its own; quiet intervals print nothing.
struct PipelineCache::StatsReporter {
	explicit StatsReporter(PipelineCache& owner)
	    : cache(owner), thread([this](const std::stop_token& stop) { Run(stop); }) {}
	~StatsReporter() {
		thread.request_stop();
		if (thread.joinable()) thread.join();
	}
	StatsReporter(const StatsReporter&)            = delete;
	StatsReporter& operator=(const StatsReporter&) = delete;

	PipelineStatsValues Values() const {
		const auto& t   = g_compile_totals;
		const auto  get = [](const std::atomic<uint64_t>& v) { return v.load(std::memory_order_relaxed); };
		PipelineStatsValues v;
		v.gfx_hit = get(t.gfx_hit), v.gfx_fast = get(t.gfx_fast), v.gfx_full = get(t.gfx_full);
		v.gfx_other = get(t.gfx_other), v.gfx_cp_ns = get(t.gfx_cp_ns);
		v.cs_hit = get(t.cs_hit), v.cs_fast = get(t.cs_fast), v.cs_full = get(t.cs_full);
		v.cs_cp_ns = get(t.cs_cp_ns), v.cs_prewarmed = get(t.cs_prewarmed);
		v.rt           = cache.GetRtPipelineTotals().published;
		v.known_misses = get(t.known_misses);
		v.programs = get(t.programs), v.translations = get(t.translations), v.replayed = get(t.replayed);
		v.stall_ns = get(t.stall_ns);
		if (cache.m_fast_first != nullptr) {
			v.optimized   = get(cache.m_fast_first->counters.swaps);
			v.optimize_ns = get(cache.m_fast_first->counters.optimize_ns);
		}
		v.prewarm_built = get(t.prewarm_built), v.prewarm_ns = get(t.prewarm_ns);
		v.journal_built = get(t.journal_built), v.journal_skipped = get(t.journal_skipped);
		v.journal_ns = get(t.journal_ns), v.journal_unpredicted = get(t.journal_unpredicted);
		return v;
	}

	void Run(const std::stop_token& stop) {
		Profiler::SetThreadName("PipelineStats");
		Common::PlaceCurrentThreadCold();
		auto before    = Values();
		auto before_ns = CompileClockNs();
		std::unique_lock lock(mutex);
		while (!cv.wait_for(lock, stop, std::chrono::seconds(10), [] { return false; }) &&
		       !stop.stop_requested()) {
			const auto now    = Values();
			const auto now_ns = CompileClockNs();
			const auto worst  = g_compile_totals.window_stall_max_ns.exchange(0, std::memory_order_relaxed);
			if (PipelineStatsChanged(now, before)) {
				PipelineCacheLog("{}", FormatPipelineStats(now, before, worst,
				                                           static_cast<double>(now_ns - before_ns) / 1e9));
			}
			before    = now;
			before_ns = now_ns;
		}
	}

	PipelineCache&              cache;
	std::mutex                  mutex;
	std::condition_variable_any cv;
	std::jthread                thread; // last: started once the rest is constructed
};

// KYTY_SHADER_PRECOMPILE_STATS=1: every 20 s, where the precompile replay's time goes and how much of
// what it published the game bound (g_permutation_use), with the CompileSPV phase times of the replay
// threads and of every thread, and the program cache's lookup-side decompression.
struct PipelineCache::PrecompileStats {
	explicit PrecompileStats(PipelineCache& owner)
	    : cache(owner), thread([this](const std::stop_token& stop) { Run(stop); }) {}
	~PrecompileStats() {
		thread.request_stop();
		if (thread.joinable()) thread.join();
		Log();
	}
	PrecompileStats(const PrecompileStats&)            = delete;
	PrecompileStats& operator=(const PrecompileStats&) = delete;

	void Log() const {
		const auto  get = [](const std::atomic<uint64_t>& v) { return v.load(std::memory_order_relaxed); };
		const auto  ms  = [&](const std::atomic<uint64_t>& v) { return static_cast<double>(get(v)) / 1.0e6; };
		const auto& use = g_permutation_use;
		ShaderPrecompiler::Progress progress;
		if (cache.m_shader_precompiler != nullptr) progress = cache.m_shader_precompiler->GetProgress();
		const auto all = CompileSpvTotals();
		std::string disk;
		if (cache.m_program_disk != nullptr) {
			const auto s = cache.m_program_disk->GetStats();
			disk = fmt::format("; program cache: {} permutation hits, {} misses, expanded {} in {:.0f} ms", s.permutation_hits,
			                   s.permutation_misses, s.expanded, static_cast<double>(s.expand_ns) / 1.0e6);
		}
		if (g_used_permutations.Enabled()) {
			disk += fmt::format("; {} bound permutations new to the used file", g_used_permutations.Added());
		}
		// What the game itself had to publish (reloaded or emitted on its own threads) and how long
		// draws waited for programs and pipelines.
		const auto& t = g_compile_totals;
		disk += fmt::format("; game-published permutations {} ({} disk permutation hits in all, {} translations); "
		                    "compile stalls {} totalling {:.1f} ms, worst {:.1f} ms",
		                    get(t.programs) - std::min(get(t.programs), get(t.replayed)),
		                    get(t.disk_permutation_hits), get(t.translations), get(t.stalls), ms(t.stall_ns),
		                    ms(t.stall_max_ns));
		if (cache.m_skip != nullptr) {
			disk += fmt::format("; pipeline skip: {} draws skipped", cache.GetSkipTotals().skipped_draws);
		}
		PipelineCacheLog(
		    "Shader precompile stats: replay {}/{} done ({} compiled, {} reloaded, {} present, {} skipped), cpu {:.1f} s; "
		    "game bound {} distinct permutations, {} of them replay-published ({} of {} replayed bound); replay "
		    "CompileSPV: {} modules, shrink {:.0f} ms, spirv-val {} in {:.0f} ms, {} known-valid, create {:.0f} ms; "
		    "all threads CompileSPV: {} modules, shrink {:.0f} ms, spirv-val {} in {:.0f} ms, {} known-valid, create "
		    "{:.0f} ms{}",
		    progress.done, progress.total, progress.compiled, progress.reloaded, progress.present, progress.skipped,
		    static_cast<double>(progress.cpu_ns) / 1.0e9, get(use.bound), get(use.replay_bound), get(use.replay_bound),
		    get(use.replayed), get(use.spv_calls), ms(use.spv_shrink_ns), get(use.spv_validated),
		    ms(use.spv_validate_ns), get(use.spv_validate_skipped), ms(use.spv_create_ns), all.calls,
		    static_cast<double>(all.shrink_ns) / 1.0e6, all.validated, static_cast<double>(all.validate_ns) / 1.0e6,
		    all.validate_skipped, static_cast<double>(all.create_ns) / 1.0e6, disk);
	}

	void Run(const std::stop_token& stop) {
		Profiler::SetThreadName("PrecompileStats");
		Common::PlaceCurrentThreadCold();
		std::unique_lock lock(mutex);
		while (!cv.wait_for(lock, stop, std::chrono::seconds(20), [] { return false; }) &&
		       !stop.stop_requested()) {
			Log();
		}
	}

	PipelineCache&              cache;
	std::mutex                  mutex;
	std::condition_variable_any cv;
	std::jthread                thread; // last: started once the rest is constructed
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device, graphics.maintenance8_enabled)),
      m_diagnostics(std::make_unique<PipelineDiagnostics>()) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	// Each cache counts its generations in its own range, so a per-thread lookup memo left by a
	// destroyed cache can never match a later cache allocated at the same address (tests create
	// several render contexts in one process).
	m_pipeline_generation.store(
	    (g_pipeline_cache_instances.fetch_add(1, std::memory_order_relaxed) + 1) << 32u,
	    std::memory_order_relaxed);
	m_program_cache->pixel_rate_variants =
	    graphics.mixed_attachment_samples == GraphicContext::MixedAttachmentSamples::Nv &&
	    !EqaaPerSampleShading();
	InitializeDriverCache();
	InitializePipelineList();
	InitializeKnownPipelines();
	if (m_driver_cache != nullptr && DriverCacheSaveSettings::Get().enabled) {
		m_saver = std::make_unique<DriverCacheSaver>(*this);
	}
	InitializeMadExactFilter();
	InitializeProgramDiskCache();
	InitializeShaderPrecompile();
	InitializePipelineJournal();
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
	if (PipelineFastFirstRequested(m_graphics.GetPhysicalDeviceProperties().vendorID)) {
		if (m_library != nullptr) {
			PipelineCacheLog("Pipeline fast-first: KYTY_PIPELINE_LIBRARY already links pipelines "
			                 "without optimization; fast-first is off");
		} else {
			m_fast_first = std::make_unique<FastFirstState>(*this);
			PipelineCacheLog("Pipeline fast-first: new pipelines are built unoptimized and "
			                 "optimized in the background (driver-cache probe: {}); pipeline "
			                 "prefetch (KYTY_PIPELINE_PREFETCH) is off with it",
			                 !m_fast_first->probe_supported ? "unsupported"
			                 : m_fast_first->probe_all      ? "every pipeline"
			                 : m_known != nullptr           ? "pipelines known to be cached"
			                                                : "off");
		}
	} else if (EnvU64("KYTY_PIPELINE_FAST_FIRST_COMPUTE", 0) != 0) {
		// KYTY_PIPELINE_FAST_FIRST_COMPUTE=1 (default off): fast-first for compute pipelines only;
		// graphics pipelines keep the prefetch pool. On NVIDIA Miles' unoptimized graphics builds were
		// the full compile (91ff6ece4), compute ones may not be.
		m_fast_first               = std::make_unique<FastFirstState>(*this);
		m_fast_first->compute_only = true;
		PipelineCacheLog("Pipeline fast-first: compute pipelines only (KYTY_PIPELINE_FAST_FIRST_COMPUTE)");
	}
	const bool graphics_fast_first = m_fast_first != nullptr && !m_fast_first->compute_only;
	if ((g_pipeline_prefetch.On() || EnvU64("KYTY_PIPELINE_PREFETCH_POOL", 0) != 0) && m_library == nullptr &&
	    !graphics_fast_first) {
		// Up to 16: a cold Miles start compiles hundreds of ~130 ms pipelines, and the old cap of 4
		// silently turned 8 or 12 into 4.
		const auto threads = std::clamp<uint64_t>(EnvU64("KYTY_PIPELINE_PREFETCH_THREADS", 2), 1, 16);
		// KYTY_PIPELINE_PREFETCH_JOURNAL=0: prefetched pipelines are not keyed (neither journaled nor known).
		const bool keyed = EnvU64("KYTY_PIPELINE_PREFETCH_JOURNAL", 1) != 0;
		m_prefetch = std::make_unique<PrefetchState>(*this, threads, 128, keyed);
		PipelineCacheLog("Pipeline prefetch: {} compile workers; draws always wait for their exact pipeline", threads);
	}
	if (EnvU64("KYTY_PIPELINE_SKIP", 0) != 0 && m_library == nullptr) {
		const auto threads = std::clamp<uint64_t>(EnvU64("KYTY_PIPELINE_SKIP_THREADS", 3), 1, 16);
		m_skip_wait_ns     = std::min<uint64_t>(EnvU64("KYTY_PIPELINE_SKIP_WAIT_US", 2000), 1000000) * 1000ull;
		m_skip_frame_budget_ns =
		    std::min<uint64_t>(EnvU64("KYTY_PIPELINE_SKIP_FRAME_BUDGET_US", 0), 1000000) * 1000ull;
		// KYTY_PIPELINE_SKIP_ORDER=1 (default off; adapted from usa-ratt/KytyPS5-GT7 rounds 141 and 145,
		// 3adca0037 / bf5ffc229): the builders take the pipeline a draw asked for most recently first
		// (worker 0 the oldest, so none starves: a car seen for a moment left its builds behind each
		// new frame's), builds no draw asked for in the last 2 s run on the first
		// KYTY_PIPELINE_SKIP_BACKGROUND_THREADS builders only (default 2), and all at below normal
		// priority. Once a draw that may not be skipped has waited 100 ms for a build, every builder
		// takes any build for the next 500 ms (a GT7 race start waited for 51 pipelines in a row with
		// 13 of 16 workers idle). New VS/PS translations (KYTY_ASYNC_TRANSLATION) go newest first too.
		PipelineCompileQueue::Ordering ordering;
		ordering.recent_first = SkipOrderEnabled();
		ordering.background_workers =
		    static_cast<size_t>(std::clamp<uint64_t>(EnvU64("KYTY_PIPELINE_SKIP_BACKGROUND_THREADS", 2), 1, 16));
		m_skip = std::make_unique<PrefetchState>(*this, threads, 1024, true, "Pipeline skip", true, ordering);
		// KYTY_PIPELINE_SKIP_PROBE=1 (default off): driver-cache lookups first (TryGetGraphicsPipeline).
		m_skip_probe =
		    EnvU64("KYTY_PIPELINE_SKIP_PROBE", 0) != 0 && m_graphics.pipeline_creation_cache_control_enabled;
		if (m_skip_probe) {
			PipelineCacheLog("Pipeline skip: new pipelines of skippable draws are looked up in the driver cache "
			                 "first (KYTY_PIPELINE_SKIP_PROBE)");
		}
		PipelineCacheLog("Pipeline skip: {} background builders; draws that only change the picture wait "
		                 "{} us for a new pipeline, then are skipped until it is built "
		                 "(KYTY_PIPELINE_SKIP_THREADS, KYTY_PIPELINE_SKIP_WAIT_US){}",
		                 threads, m_skip_wait_ns / 1000,
		                 ordering.recent_first
		                     ? fmt::format("; most recently asked first, {} builders take older builds "
		                                   "(KYTY_PIPELINE_SKIP_ORDER)",
		                                   ordering.background_workers)
		                     : std::string());
	}
	if (EnvU64("KYTY_PIPELINE_STATS", 0) != 0) m_stats_reporter = std::make_unique<StatsReporter>(*this);
	if (EnvU64("KYTY_PIPELINE_LIST_PREPARE", 0) != 0 && !PipelineCacheTitleId().empty()) RunListPrepare();
}

PipelineCache::~PipelineCache() {
	m_stats_reporter.reset();
	StopBackgroundPipelines();
	AaMaskCoverage::Shutdown(m_graphics.device);
	StopShaderPrecompile();
	if (m_rt != nullptr) m_rt->Shutdown();
	m_program_cache->StopBackgroundChecks();
	if (m_prefetch != nullptr) m_prefetch->Stop();
	if (m_skip != nullptr) {
		m_skip->Stop();
		const auto skip = GetSkipTotals();
		PipelineCacheLog("Pipeline skip: {} draws skipped; {} pipelines built in the background ({} ms), {} "
		                 "published, {} of them within the wait; {} lookups found the queue full",
		                 skip.skipped_draws, skip.requested, skip.compile_ns / 1000000, skip.published,
		                 skip.ready_in_wait, skip.not_queued);
	}
	if (const auto feedback = PipelineFeedback::Report(); !feedback.empty()) PipelineCacheLog("{} (exit)", feedback);
	if (m_adopt) PipelineCacheLog("Pipeline journal: {}", AdoptReport());
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
	m_prefetch.reset(); // Unused speculative pipelines before layouts and program modules.
	m_skip.reset();
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
	{
		std::scoped_lock lock(m_adopt_mutex);
		for (const auto& [key, pipeline]: m_adopted) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline, nullptr);
		}
		m_adopted.clear();
	}
	if (m_fast_first != nullptr) {
		// Replaced unoptimized pipelines still waiting for their retirement age; the optimized
		// builds that replaced them were destroyed with their entries above.
		for (const auto handle: m_fast_first->retired_handles.TakeAll()) {
			m_graphics.device.destroyPipeline(handle, nullptr);
		}
	}
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
	const auto title_id = PipelineCacheFileStem();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	// KYTY_PIPELINE_CACHE_PATH names another file (measurements of a first start: an empty cache
	// without touching the real one); the known-pipeline list goes next to it.
	if (const auto* custom = std::getenv("KYTY_PIPELINE_CACHE_PATH"); custom != nullptr && *custom != '\0') {
		m_driver_cache_path = std::filesystem::path(custom);
	} else {
		m_driver_cache_path = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
		SeedRenderDocCacheFile(m_driver_cache_path, ".bin");
		SeedRenderDocCacheFile(std::filesystem::path(m_driver_cache_path).replace_extension(".pipelines.known"),
		                       ".pipelines.known");
	}
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
	Common::RamStats::Range("driver cache load temporary", initial_data.data(), initial_data.capacity());
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
	m_driver_cache_loaded = !initial_data.empty();
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

// KYTY_PROGRAM_CACHE=0|1: the persistent translated-program cache (programDiskCache.h), next to
// the driver cache as _PipelineCache/<title>.programs.bin (KYTY_PROGRAM_CACHE_PATH names another
// file; tests use it). Default: ProgramDiskCacheDefault. KYTY_PROGRAM_CACHE_SAVE=0: no periodic
// saves, only the one at exit. KYTY_PROGRAM_CACHE_VERIFY: see ProgramCache::DiskVerifyMode.
constexpr bool ProgramDiskCacheDefault = false;

// KYTY_MAD_EXACT_SHADERS=<token>[,<token>...] (diagnostic, default unset): the selected programs
// are emitted with MadMode::Exact (as KYTY_MAD_MODE=exact) while the others keep KYTY_MAD_MODE, to
// find which programs need unfused MADs. Tokens: a stage (vs ps cs ls hs es gs; none = every stage),
// a hex guest hash, or m<mask>:<value> (hex) to select by hash bits (bisection). The setting is not
// part of the codegen fingerprint; BuildDiskKey salts the disk-cache key of each selected program.
void PipelineCache::InitializeMadExactFilter() {
	const char* value = std::getenv("KYTY_MAD_EXACT_SHADERS");
	if (value == nullptr || *value == '\0') {
		return;
	}
	ShaderRecompiler::MadExactFilter filter;
	filter.active = true;
	std::string_view text(value);
	while (!text.empty()) {
		const auto comma = text.find(',');
		std::string token(text.substr(0, comma));
		text = comma == std::string_view::npos ? std::string_view {} : text.substr(comma + 1);
		const auto stage_bit = [](ShaderType type) { return 1u << static_cast<uint32_t>(type); };
		if (token == "vs") {
			filter.stage_mask |= stage_bit(ShaderType::Vertex);
		} else if (token == "ps") {
			filter.stage_mask |= stage_bit(ShaderType::Pixel);
		} else if (token == "cs") {
			filter.stage_mask |= stage_bit(ShaderType::Compute);
		} else if (token == "ls") {
			filter.stage_mask |= stage_bit(ShaderType::Local);
		} else if (token == "hs") {
			filter.stage_mask |= stage_bit(ShaderType::TessellationControl);
		} else if (token == "es") {
			filter.stage_mask |= stage_bit(ShaderType::TessellationEvaluation);
		} else if (token == "gs") {
			filter.stage_mask |= stage_bit(ShaderType::Mesh);
		} else if (!token.empty() && token[0] == 'm' && token.find(':') != std::string::npos) {
			const auto colon  = token.find(':');
			filter.hash_mask  = std::strtoull(token.substr(1, colon - 1).c_str(), nullptr, 16);
			filter.hash_value = std::strtoull(token.substr(colon + 1).c_str(), nullptr, 16) & filter.hash_mask;
		} else if (!token.empty()) {
			filter.hashes.push_back(std::strtoull(token.c_str(), nullptr, 16));
		}
	}
	if (filter.stage_mask == 0) {
		filter.stage_mask = ~0u;
	}
	PipelineCacheLog("MAD exact filter: stages {:#x}, {} hashes, mask {:016x} value {:016x}", filter.stage_mask,
	                 filter.hashes.size(), filter.hash_mask, filter.hash_value);
	ShaderRecompiler::SetMadExactFilter(std::move(filter));
}

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
		const auto title_id = PipelineCacheFileStem();
		if (title_id.empty()) {
			return;
		}
		if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
			PipelineCacheLog("Program cache: disabled (non-Release build)");
			return;
		}
		path = std::filesystem::path("_PipelineCache") / (title_id + ".programs.bin");
		SeedRenderDocCacheFile(path, ".programs.bin");
	}
	// The file's identity: the codegen version and runtime codegen state, and the device (its
	// state that the emitter reads is part of the fingerprint already; the device and driver are
	// named as well, so that no device-dependent input can be missed).
	ProgramDiskCache::Settings settings;
	settings.path           = path;
	settings.identity       = ShaderRecompiler::CodegenFingerprint();
	settings.periodic_saves = EnvU64("KYTY_PROGRAM_CACHE_SAVE", 1) != 0;
	// KYTY_PROGRAM_CACHE_MAX_MB (default 512, 64..3584): GT7 fills 512 MiB of translated programs
	// in its first sessions, after which new programs were no longer kept.
	settings.max_file_bytes = std::clamp<uint64_t>(EnvU64("KYTY_PROGRAM_CACHE_MAX_MB", 512), 64u, 3584u) << 20u;
	// KYTY_PROGRAM_CACHE_EVICT=1 (default 0): at the limit, records this run has not used make room
	// for new ones (ProgramDiskCache::Settings::evict_unused); GT7 outgrows 3584 MiB in two runs.
	settings.evict_unused = EnvU64("KYTY_PROGRAM_CACHE_EVICT", 0) != 0;
	// KYTY_PROGRAM_CACHE_COMPRESS=<zstd level 1..19> (default 0): new permutations are stored packed
	// (ProgramDiskCache::Settings::compress_level); SPIR-V packs ~4x, so MAX_MB holds ~4x the programs.
	settings.compress_level = static_cast<int>(std::min<uint64_t>(EnvU64("KYTY_PROGRAM_CACHE_COMPRESS", 0), 19u));
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
	// Reloaded permutations skip spirv-val of their shrunk modules once a run has validated the same
	// words (vulkanCommon.h); next to the program file, so a program file per codegen set keeps its own.
	const auto verdict_path = Common::PathToString(path) + ".shrinkval";
	const auto verdicts     = UseShrinkVerdictFile(verdict_path);
	PipelineCacheLog("Program cache: {} known-valid shrunk modules ({})", verdicts, verdict_path);
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

	void NoteCreated(uint64_t create_ns) {
		last_created_ns.store(CompileClockNs(), std::memory_order_relaxed);
		created.fetch_add(1, std::memory_order_release);
		// KYTY_PCACHE_SAVE_SLOW_MS=N (default 0: off; TheCruZ/KytyPS5-GTA 988e46cd): a pipeline whose
		// creation took N ms or more has the cache saved at the saver's next wake-up (within a
		// second), not at the next period: closing the emulator soon after a minutes-long ray
		// tracing pipeline (Wolverine) threw the work away.
		static const uint64_t slow_ns = [] {
			const auto* value = std::getenv("KYTY_PCACHE_SAVE_SLOW_MS");
			return value != nullptr ? std::strtoull(value, nullptr, 10) * 1'000'000ull : 0ull;
		}();
		if (slow_ns != 0 && create_ns >= slow_ns) {
			urgent.store(true, std::memory_order_release);
		}
	}

	[[nodiscard]] bool Due(uint64_t now) const {
		const auto& settings = DriverCacheSaveSettings::Get();
		const auto  pending  = created.load(std::memory_order_acquire) - saved_created;
		if (pending == 0 || oversized || now < retry_after_ns) {
			return false;
		}
		if (urgent.load(std::memory_order_acquire)) {
			return true; // KYTY_PCACHE_SAVE_SLOW_MS
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
		Common::PlaceCurrentThreadCold();
		std::mutex                  mutex;
		std::condition_variable_any wake;
		std::unique_lock            lock(mutex);
		while (!stop.stop_requested()) {
			wake.wait_for(lock, stop, std::chrono::seconds(1), [] { return false; });
			if (stop.stop_requested() || !Due(CompileClockNs())) {
				continue;
			}
			// Count only what the serialized data can contain: pipelines created before it.
			urgent.store(false, std::memory_order_release);
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

	// WriteDriverCache result of a periodic save skipped for exceeding DriverCacheFileCap().
	static constexpr uint64_t WriteOversized = UINT64_MAX;

	PipelineCache&        cache;
	std::atomic<uint64_t> created {0};
	std::atomic<uint64_t> last_created_ns {0};
	std::atomic<bool>     urgent {false}; // KYTY_PCACHE_SAVE_SLOW_MS
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
	m_saver->NoteCreated(create_ns);
	if (m_saver->serializing.load(std::memory_order_relaxed)) {
		HangTrace::RecordPipelineCacheSaveOverlap(create_ns);
	}
}

// KYTY_PCACHE_SAVE_MAPPED (WriteDriverCache): read at each save.
static Live::Switch g_pcache_save_mapped("KYTY_PCACHE_SAVE_MAPPED", Live::ParseDefaultOff);

uint64_t PipelineCache::WriteDriverCache(bool periodic) {
	HangWatchdog::Scope snapshot(
	    "pipeline-cache-snapshot",
	    reinterpret_cast<uint64_t>(static_cast<VkPipelineCache>(m_driver_cache)), periodic);
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
	// KYTY_PCACHE_SAVE_MAPPED=1 (live, default off; based on Senaxx/KytyPS5 2708521dc): the driver
	// writes its data straight into a mapping of the temporary file, whose pages belong to the file
	// and are not charged to commit as the same bytes on the heap are (the payload, ~130 MB for Astro
	// Bot's cache and up to the KYTY_DRIVER_CACHE_MAX_MB cap, at every periodic save and at exit).
	// The file's layout is the same.
	const bool use_mapped = g_pcache_save_mapped.On();
	auto       temp_path  = m_driver_cache_path;
	temp_path += ".tmp";
	const auto signature   = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto prefix_size = signature.size() + sizeof(uint64_t);
	if (use_mapped && !Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return 0;
	}
	std::vector<uint8_t> payload;
	PipelineCacheMappedFile mapped;
	uint8_t*             data   = nullptr;
	size_t               size   = 0;
	vk::Result           result = vk::Result::eIncomplete;
	if (m_saver != nullptr) m_saver->serializing.store(true, std::memory_order_relaxed);
	// The known-pipeline list written after this save names only pipelines created before it.
	if (m_known != nullptr) m_known->Seal();
	for (uint32_t attempt = 0; attempt < 3 && result == vk::Result::eIncomplete; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		// Pipelines created between the two calls grow the data; leave room for them. Whatever
		// is returned is valid initial data, but only complete data replaces the file.
		const size_t capacity = size + size / 8u + 1024u * 1024u;
		if (use_mapped) {
			if (!mapped.Create(temp_path, prefix_size + capacity)) {
				result = vk::Result::eErrorOutOfHostMemory;
				break;
			}
			data = mapped.Data() + prefix_size;
		} else {
			payload.resize(capacity);
			data = payload.data();
		}
		size   = capacity;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, data);
	}
	if (m_saver != nullptr) m_saver->serializing.store(false, std::memory_order_relaxed);
	const auto serialized = CompileClockNs();
	const char* kind      = periodic ? "periodic" : "exit";
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		(void)mapped.Close(0);
		PipelineCacheLog("Vulkan pipeline cache: {} save failed ({}, {} bytes)", kind,
		                 vk::to_string(result), size);
		return 0;
	}
	if (!use_mapped) {
		payload.resize(size);
	}
	auto       prefix       = signature;
	const auto payload_hash = XXH3_64bits(data, size);
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (periodic && prefix.size() + size > DriverCacheFileCap()) {
		// The loader rejects such a file and starts over; keep the last file that it accepts.
		// The exit save still writes it, which is what resets an overgrown cache.
		(void)mapped.Close(0);
		PipelineCacheLog("Vulkan pipeline cache: {} bytes exceed the {} byte cap; periodic saves "
		                 "stop, the exit save still writes it",
		                 size, DriverCacheFileCap());
		return DriverCacheSaver::WriteOversized;
	}
	if (!use_mapped && !Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return 0;
	}
	uint32_t prefix_written  = 0;
	uint32_t payload_written = 0;
	bool     flushed         = false;
	if (use_mapped) {
		// The prefix goes in front of the payload the driver wrote; the file is cut to what was used.
		std::memcpy(mapped.Data(), prefix.data(), prefix.size());
		flushed         = mapped.Close(prefix.size() + size);
		prefix_written  = flushed ? static_cast<uint32_t>(prefix.size()) : 0u;
		payload_written = flushed ? static_cast<uint32_t>(size) : 0u;
	} else {
		Common::File file;
		if (file.Create(temp_path)) {
			file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
			file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
		}
		flushed = !file.IsInvalid() && file.Flush();
		file.Close();
	}
	// std::filesystem::rename replaces the destination in one step (MoveFileExW with
	// MOVEFILE_REPLACE_EXISTING on Windows, rename(2) elsewhere): a kill at any point leaves
	// either the previous or the new complete file. Common::File::RenameFile deleted the old
	// file first, so a kill between the delete and the move lost the cache.
	std::error_code error;
	if (prefix_written == prefix.size() && payload_written == size && flushed) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (prefix_written != prefix.size() || payload_written != size || !flushed || error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}{}",
		                 Common::PathToString(m_driver_cache_path),
		                 error ? " (" + error.message() + ")" : std::string());
		return 0;
	}
	if (m_known != nullptr) m_known->Flush();
	const auto written = CompileClockNs();
	HangTrace::RecordPipelineCacheSave(size, serialized - begin, written - serialized);
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {} ({}, serialize {:.1f} ms, write "
	                 "{:.1f} ms{})",
	                 size, Common::PathToString(m_driver_cache_path), kind,
	                 static_cast<double>(serialized - begin) / 1.0e6,
	                 static_cast<double>(written - serialized) / 1.0e6,
	                 use_mapped ? ", mapped" : "");
	return size;
}

void PipelineCache::Save() {
	StopBackgroundPipelines();
	StopShaderPrecompile();
	if (m_prefetch != nullptr) m_prefetch->Stop();
	if (m_skip != nullptr) m_skip->Stop();
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
	// Queued optimized compiles finish within the drain budget so that the cache saved below holds
	// them; an unoptimized pipeline still in use stays so.
	if (m_fast_first != nullptr) {
		m_fast_first->Stop();
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

void PipelineCache::WaitShaderPrecompile() {
	if (m_shader_precompiler != nullptr) m_shader_precompiler->Wait();
	if (m_shader_journal != nullptr) m_shader_journal->Flush();
}

// KYTY_SHADER_PRECOMPILE=1 (default off): journals the inputs of every program permutation this
// run compiles (shaderPrecompile.h) and replays the ones an earlier run journaled on background
// threads from start-up, so the game finds them published. KYTY_SHADER_PRECOMPILE_PATH names
// another journal file (tests; the default is _PipelineCache/<title>.shaders.journal),
// KYTY_SHADER_PRECOMPILE_REPLAY=0 only records, KYTY_SHADER_PRECOMPILE_THREADS (default 2, 1..8)
// sets the replay threads, KYTY_SHADER_PRECOMPILE_MAX (default 30000, 0 = all) the entries
// replayed, KYTY_SHADER_PRECOMPILE_WAIT=1 makes start-up wait for the replay to finish. KYTY_RESOURCE_REUSE=1 keeps it off (that mode serializes preparation on a mutex the
// replay does not take).
void PipelineCache::InitializeShaderPrecompile() {
	const bool precompile = EnvU64("KYTY_SHADER_PRECOMPILE", 0) != 0;
	const bool list       = m_list != nullptr && !m_list->entries.empty();
	if (!precompile && !list) {
		return;
	}
	if (ResourceReuseEnabled()) {
		PipelineCacheLog("Shader precompile: off (KYTY_RESOURCE_REUSE=1)");
		return;
	}
	if (precompile) {
		std::filesystem::path path;
		if (const auto* custom = std::getenv("KYTY_SHADER_PRECOMPILE_PATH");
		    custom != nullptr && *custom != '\0') {
			path = std::filesystem::path(custom);
		} else {
			const auto title_id = PipelineCacheFileStem();
			if (title_id.empty()) {
				return;
			}
			path = std::filesystem::path("_PipelineCache") / (title_id + ".shaders.journal");
			SeedRenderDocCacheFile(path, ".shaders.journal");
		}
		// The identity holds what the stored inputs depend on besides the game: the device (subgroup
		// and mesh limits are part of the stage input infos) and the sizes of the input structs. The
		// translator is not part of it: the journal holds inputs, not outputs.
		const auto& properties = m_graphics.GetPhysicalDeviceProperties();
		auto        identity   = fmt::format(
	        "kyty shader journal; device {:08x}:{:08x}; input infos {} {} {}", properties.vendorID,
	        properties.deviceID, ShaderStoredInputInfoSize<ShaderVertexInputInfo>(),
	        ShaderStoredInputInfoSize<ShaderPixelInputInfo>(), ShaderStoredInputInfoSize<ShaderComputeInputInfo>());
		// KYTY_BDA_FORCE_SHADERS changes those programs' bound buffers, which the stored
		// specializations list: a journal recorded with another list is not replayed.
		std::string forced;
		for (const auto hash: ShaderRecompiler::GetCodegenOptions().bda_force_shaders) {
			forced += fmt::format(" {:016x}", hash);
		}
		if (!forced.empty()) {
			identity += "; bda force" + forced;
		}
		ShaderJournal::Settings settings;
		settings.path     = path;
		settings.identity = std::vector<uint8_t>(identity.begin(), identity.end());
		settings.log      = [](const std::string& message) { PipelineCacheLog("{}", message); };
		m_shader_journal  = std::make_unique<ShaderJournal>(std::move(settings));
		m_program_cache->journal = m_shader_journal.get();
		const auto stats = m_shader_journal->GetStats();
		PipelineCacheLog("Shader precompile: journal {}: {} sources, {} permutations loaded ({:.0f} ms{}{})",
		                 Common::PathToString(path), m_shader_journal->Sources().size(),
		                 m_shader_journal->Entries().size(), static_cast<double>(stats.load_ns) / 1.0e6,
		                 stats.damaged_bytes != 0 ? "; damaged tail ignored" : "",
		                 stats.header_rejected ? "; another device or format, replaced" : "");
		// KYTY_SHADER_PRECOMPILE_USED=1 (default off): permutations draws bind are recorded in
		// <journal>.used, and the replay takes the journal entries an earlier run bound first.
		// KYTY_SHADER_PRECOMPILE_UNUSED=0 (with it) replays only those (once the file names any).
		if (EnvU64("KYTY_SHADER_PRECOMPILE_USED", 0) != 0) {
			auto used_path = path;
			used_path += ".used";
			m_used_loaded = g_used_permutations.Open(used_path);
			PipelineCacheLog("Shader precompile: {} permutations bound by earlier runs ({})", m_used_loaded,
			                 Common::PathToString(used_path));
		}
	} else {
		// The pipeline list alone: a journal object only to hold the replay's input (no file, and the
		// program cache records nothing into it).
		ShaderJournal::Settings settings;
		settings.background_writer = false;
		m_shader_journal           = std::make_unique<ShaderJournal>(std::move(settings));
	}
	if (list) {
		const auto added = m_shader_journal->AppendReplayOnly(std::move(m_list->sources), m_list->entries);
		m_list->sources  = {};
		m_list->entries  = {};
		PipelineCacheLog("Shader precompile: the pipeline list adds {} sources and {} permutations to the replay",
		                 added.sources, added.entries);
	}
	// KYTY_NULL_BUFFER_SHARE: the journaled permutations' live buffer slots seed the share, so the
	// first null V# of a slot in this session takes the live specialization an earlier session saw.
	if (ShaderRecompiler::GetCodegenOptions().null_buffer_share && m_shader_journal != nullptr) {
		const auto& sources = m_shader_journal->Sources();
		size_t      seeded  = 0;
		for (const auto& entry: m_shader_journal->Entries()) {
			if (entry.source >= sources.size()) continue;
			const auto& source = sources[entry.source];
			ShaderRecompiler::IR::ResourceSpecialization specialization;
			if (!ShaderRecompiler::IR::DecodeSpecialization(entry.specialization, specialization)) continue;
			ShaderRecompiler::IR::SeedNullBufferShare(source.hash, source.stage, source.user_data_base,
			                                          source.user_data_count, specialization);
			seeded++;
		}
		PipelineCacheLog("Null buffer share: seeded from {} journaled permutations", seeded);
	}
	const bool prepare = m_list != nullptr && m_list->prepare;
	// KYTY_PIPELINE_COMPUTE_AHEAD=1 (default off; needs KYTY_PIPELINE_PREWARM_COMPUTE and
	// KYTY_PIPELINE_PREFETCH_PROGRAMS): a compute program prepared ahead of its dispatch gets its
	// pipeline built on the prewarm threads at once, and the dispatch waits for that build instead of
	// compiling it on the command processor (a cold Miles start spent 20 s there). Created before the
	// replay check: a first start has an empty journal.
	m_compute_ahead = EnvU64("KYTY_PIPELINE_COMPUTE_AHEAD", 0) != 0;
	if (EnvU64("KYTY_PIPELINE_PREWARM_COMPUTE", m_list != nullptr ? 1 : 0) != 0) {
		const auto threads = static_cast<size_t>(
		    std::clamp<uint64_t>(EnvU64("KYTY_PIPELINE_PREWARM_THREADS", prepare ? 4 : 1), 1, 16));
		m_compute_prewarm  = std::make_unique<ComputePrewarm>(*this, threads);
		PipelineCacheLog("Compute prewarm: the pipelines of replayed compute programs{} are built on {} background "
		                 "thread(s)", m_compute_ahead ? " and of programs prepared ahead" : "", threads);
	}
	if (EnvU64("KYTY_SHADER_PRECOMPILE_REPLAY", 1) == 0 || m_shader_journal->Entries().empty()) {
		return;
	}
	ShaderPrecompiler::Settings replay;
	replay.threads     = static_cast<uint32_t>(
	    std::clamp<uint64_t>(EnvU64("KYTY_SHADER_PRECOMPILE_THREADS", prepare ? 8 : 2), 1, 16));
	replay.low_priority = !prepare;
	replay.max_entries = EnvU64("KYTY_SHADER_PRECOMPILE_MAX", 30000);
	replay.thread_init = [](uint32_t) {
		Profiler::SetThreadName("ShaderPrecompile");
		Common::PlaceCurrentThreadCold();
	};
	// The software ray-tracing kernels first (and their pipelines, started as each is published):
	// a cold pipeline build takes seconds, and the game needs them at its first lighting pass.
	replay.priority_source = [](const ShaderJournal::Source& source) {
		return source.kind == ShaderJournal::Kind::Compute && MayContainBvhIntersect(source.code);
	};
	if (m_compute_prewarm != nullptr) m_program_cache->prewarm_all_compute = true;
	m_program_cache->rt_prewarm = [this](const ShaderComputeInputInfo& info, const ShaderProgram& program) {
		PrewarmComputePipeline(info, program);
	};
	replay.log = [](const std::string& message) {
		if (!PipelineFeedback::Enabled()) {
			PipelineCacheLog("{}", message);
			return;
		}
		// Where the replay's time goes (all program work of the process so far, game included).
		const auto& t = g_compile_totals;
		PipelineCacheLog("{} | programs {}: disk hits {} sources {} permutations (decode {:.0f} ms), translated {} "
		                 "({:.0f} ms), emit {:.0f} ms, module {:.0f} ms",
		                 message, t.programs.load(), t.disk_source_hits.load(), t.disk_permutation_hits.load(),
		                 LoadMs(t.disk_load_ns), t.translations.load(), LoadMs(t.translate_ns), LoadMs(t.emit_ns),
		                 LoadMs(t.module_ns));
	};
	// Progress for the compile-wait lines (common/compileWatch.h).
	const uint64_t replay_total = replay.max_entries != 0
	                                  ? std::min<uint64_t>(m_shader_journal->Entries().size(), replay.max_entries)
	                                  : m_shader_journal->Entries().size();
	auto replay_done = std::make_shared<std::atomic<uint64_t>>(0);
	CompileWatch::NotePrecompile(0, replay_total);
	replay.on_finished = [this, replay_total] {
		CompileWatch::NotePrecompile(replay_total, replay_total);
		m_shader_journal->ReleaseLoaded();
	};
	if (g_used_permutations.Enabled()) {
		replay.used_entry = [](const ShaderJournal::Source& source, const ShaderJournal::Entry& entry) {
			return g_used_permutations.WasUsed(PermutationIdentity(source, entry));
		};
		replay.replay_unused = m_used_loaded == 0 || EnvU64("KYTY_SHADER_PRECOMPILE_UNUSED", 1) != 0;
	}
	const bool stats   = CompileSpvStatsEnabled();
	replay.measure_cpu = stats;
	m_shader_precompiler = std::make_unique<ShaderPrecompiler>(
	    *m_shader_journal, std::move(replay),
	    [this, stats, replay_total, replay_done](const ShaderJournal::Source& source, const ShaderJournal::Entry& entry) {
		    if (!stats) {
			    const auto outcome = m_program_cache->ReplayEntry(source, entry);
			    CompileWatch::NotePrecompile(replay_done->fetch_add(1, std::memory_order_relaxed) + 1, replay_total);
			    return outcome;
		    }
		    const auto before  = CompileSpvThreadTimes();
		    const auto outcome = m_program_cache->ReplayEntry(source, entry);
		    const auto after   = CompileSpvThreadTimes();
		    auto&      use     = g_permutation_use;
		    use.spv_calls.fetch_add(after.calls - before.calls, std::memory_order_relaxed);
		    use.spv_shrink_ns.fetch_add(after.shrink_ns - before.shrink_ns, std::memory_order_relaxed);
		    use.spv_validated.fetch_add(after.validated - before.validated, std::memory_order_relaxed);
		    use.spv_validate_ns.fetch_add(after.validate_ns - before.validate_ns, std::memory_order_relaxed);
		    use.spv_validate_skipped.fetch_add(after.validate_skipped - before.validate_skipped,
		                                       std::memory_order_relaxed);
		    use.spv_create_ns.fetch_add(after.create_ns - before.create_ns, std::memory_order_relaxed);
		    CompileWatch::NotePrecompile(replay_done->fetch_add(1, std::memory_order_relaxed) + 1, replay_total);
		    return outcome;
	    });
	if (stats) m_precompile_stats = std::make_unique<PrecompileStats>(*this);
	if (EnvU64("KYTY_SHADER_PRECOMPILE_WAIT", 0) != 0) {
		// Start-up waits for the replay (the best case of precompiling: tests, measurements).
		m_shader_precompiler->Wait();
	}
}

// Stops the replay (entries in progress finish) and drops the loaded journal content; the journal
// keeps recording until the cache goes.
void PipelineCache::StopShaderPrecompile() {
	m_precompile_stats.reset(); // its last line first; it reads the replay's progress
	if (m_shader_precompiler != nullptr) {
		m_shader_precompiler->Stop();
		m_shader_precompiler.reset(); // joins the workers
		if (m_shader_journal != nullptr) m_shader_journal->ReleaseLoaded();
	}
	if (m_shader_journal != nullptr) m_shader_journal->Flush();
}

PipelineCache::ProgramTotals PipelineCache::Totals() {
	const auto& t = g_compile_totals;
	return {.programs          = t.programs.load(std::memory_order_relaxed),
	        .translations      = t.translations.load(std::memory_order_relaxed),
	        .source_hits       = t.disk_source_hits.load(std::memory_order_relaxed),
	        .permutation_hits  = t.disk_permutation_hits.load(std::memory_order_relaxed),
	        .verify_checks     = t.disk_verify_checks.load(std::memory_order_relaxed),
	        .verify_mismatches = t.disk_verify_mismatches.load(std::memory_order_relaxed),
	        .replayed          = t.replayed.load(std::memory_order_relaxed)};
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

// KYTY_PIPELINE_SKIP_TESS=1 (default off; the GT7 preset): the skip builder and the prefetch pool
// build tessellated pipelines too, so their draws are skipped instead of waiting on the GPU thread.
bool PipelineCache::SkipTessellatedEnabled() {
	static const bool on = EnvU64("KYTY_PIPELINE_SKIP_TESS", 0) != 0;
	return on;
}

bool PipelineCache::TessellationActive(const HW::UserConfig& user_config) {
	return user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
}

void ExitWithoutMeshShaders(const GraphicContext& graphics) {
	EXIT("This game draws with mesh (NGG) shaders, which need the Vulkan extension "
	     "VK_EXT_mesh_shader with its meshShader feature. The GPU \"%s\" does not support it, "
	     "and the emulator has no fallback for these draws (NVIDIA GTX 16/RTX 20 series and "
	     "newer, AMD RX 6000 series and newer and Intel Arc support it).\n",
	     graphics.GetPhysicalDeviceProperties().deviceName.data());
	std::abort();
}

namespace {

// Static stage information shared by GetGraphicsPrograms and the draw-prep speculative
// preparation: pure functions of the registers and constant device limits.
// False when the mesh program exceeds the host's mesh limits: with KYTY_SKIP_INVALID (default on)
// its draws are skipped, as upstream 560756f34 does, instead of stopping the emulator.
[[nodiscard]] bool FinishMeshStage(const GraphicContext& graphics, ShaderVertexInputInfo& vertex_info) {
	if (!graphics.mesh_shader_enabled) {
		ExitWithoutMeshShaders(graphics);
	}
	auto& mesh = vertex_info.mesh;
	// The pipeline requires the wave size where it can (CreatePipelineInternal, shaders.cpp).
	const auto required =
	    graphics.GraphicsSubgroupSize(vk::ShaderStageFlagBits::eMeshEXT, mesh.wave_size);
	mesh.host_subgroup_size = required != 0 ? required : graphics.subgroup_size;
	const auto& limits      = graphics.mesh_shader_properties;
	// A device whose X limit is below its total limit (RADV: 65,535) gets programs that take a
	// group offset, so ExecutePreparedDraw can split a draw with more groups along X.
	mesh.split_groups =
	    limits.maxMeshWorkGroupCount[0] < limits.maxMeshWorkGroupTotalCount ? 1u : 0u;
	// A subgroup with more threads than a mesh workgroup may have (NVIDIA: 128) runs its waves in
	// passes (Senaxx a1c2a88a9; EmitMeshEntryPoint).
	mesh.passes = mesh.PassesFor(
	    std::min(limits.maxMeshWorkGroupInvocations, limits.maxMeshWorkGroupSize[0]));
	if (mesh.passes == 0 || mesh.max_vertices > limits.maxMeshOutputVertices ||
	    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
	    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
		if (!Common::SkipInvalidEnabled()) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     mesh.HostThreads(), mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
		static std::atomic_bool logged = false;
		if (Common::LogOnce(logged)) {
			std::printf("Warning: draws of mesh shaders that exceed the host's mesh limits are skipped "
			            "(KYTY_SKIP_INVALID, logged once): threads=%u vertices=%u primitives=%u LDS=%u\n",
			            mesh.HostThreads(), mesh.max_vertices, mesh.max_primitives,
			            mesh.lds_size_dwords);
		}
		return false;
	}
	return true;
}

// How the pixel stage's parameters relate to the vertex exports (upstream b38b7454a): rect lists
// expand flat aliases through separate outputs; otherwise the provoking vertex selects the flat
// value. Part of the pixel stage's static key.
static void ApplyPixelParameterMode(const HW::Context& context, const HW::UserConfig& user_config,
                                    bool mesh_active, bool tess_active,
                                    ShaderPixelInputInfo& pixel_info) {
	if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
		pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
	} else if (context.GetModeControl().provoking_vtx_last) {
		pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
	} else {
		pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
	}
}

void ApplyDualSourceBlending(const HW::Context& context, ShaderPixelInputInfo& pixel_info) {
	const auto& blend    = context.GetBlendControl(0);
	const bool  blending = blend.enable && !context.GetRenderTarget(0).info.blend_bypass;
	pixel_info.alpha_blend_source = ShaderAlphaBlendSource::None;
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
	           pixel_info.target_output_mode[0] != 7 && [&] {
		           // KYTY_UI_TEXT_BLEND: exports to render targets that are not bound are dead and
		           // do not block taking over MRT1 (Franseven).
		           const bool ui_text = UiTextBlendEnabled();
		           for (uint32_t slot = 1; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			           if (pixel_info.target_output_mode[slot] == 0) {
				           continue;
			           }
			           if (!ui_text || (context.GetRenderTarget(slot).base.addr != 0 &&
			                            render_target_mask_slot(context.GetRenderTargetMask(),
			                                                    slot) != 0)) {
				           return false;
			           }
		           }
		           return true;
	           }()) {
		// The export mapping moves alpha out of the fourth channel: MRT1 carries logical alpha,
		// or the per-channel source factors, as a second blend source (KYTY_BLEND_ALPHA_REMAP;
		// upstream 73615c31f, 0bef3fc0d, 94e7d239d).
		switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
			case BlendMappingSupport::SourceAlpha:
				pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
				break;
			case BlendMappingSupport::SourceAlphaOne:
				pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
				break;
			case BlendMappingSupport::SourceAlphaZero:
				pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
				break;
			case BlendMappingSupport::OneAlphaZero:
				pixel_info.alpha_blend_source = ShaderAlphaBlendSource::OneAlphaZero;
				break;
			default: break;
		}
		if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
			// The dead exports past MRT1 (KYTY_UI_TEXT_BLEND) are dropped.
			for (uint32_t slot = 2; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
				pixel_info.target_output_mode[slot] = 0;
			}
			pixel_info.dual_source_blending     = true;
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = {};
		}
	}
}

// KYTY_FLOAT_STORE_CLAMP (usa-ratt/KytyPS5-GT7 cfa8a479e): the MRTs that go to a 16-bit float
// render target, whose 32-bit exports then clamp to the half range as the guest GPU stores them.
// Only with the default export routing (no KYTY_CB_SHADER_MASK_EXPORTS remap).
static void ApplyHalfFloatTargets(const HW::Context& context, ShaderPixelInputInfo& pixel_info) {
	pixel_info.target_half_float_mask = 0;
	if (!ShaderRecompiler::GetCodegenOptions().float_store_clamp ||
	    pixel_info.target_shader_mask != UINT32_MAX) {
		return;
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX && slot < 8u; slot++) {
		if (pixel_info.target_output_mode[slot] == 0) {
			continue;
		}
		const auto& rt = context.GetRenderTarget(slot);
		if (rt.base.addr != 0 && rt.info.channel_type == Prospero::ChannelType::kFloat &&
		    (rt.info.format == Prospero::ChannelLayout::k16 ||
		     rt.info.format == Prospero::ChannelLayout::k16_16 ||
		     rt.info.format == Prospero::ChannelLayout::k16_16_16_16)) {
			pixel_info.target_half_float_mask |= static_cast<uint8_t>(1u << slot);
		}
	}
	if (pixel_info.dual_source_blending) {
		// MRT1 is the second blend source of target 0.
		pixel_info.target_half_float_mask =
		    static_cast<uint8_t>((pixel_info.target_half_float_mask & ~2u) |
		                         ((pixel_info.target_half_float_mask & 1u) << 1u));
	}
}

// KYTY_INT_MRT_OUTPUTS=1 (default off; upstream 26ca60923, CodegenOptions::int_mrt_outputs): the exports that go
// to UINT and SINT render targets are typed as integer vectors (the emitter's PixelOutputType), as Vulkan requires
// for integer attachments; by default they stay float vectors whose bits the attachment takes.
// Off by default: it changes the pixel programs (and static keys) of draws to integer targets.
static bool IntTargetOutputsEnabled() {
	return ShaderRecompiler::GetCodegenOptions().int_mrt_outputs;
}

static void ApplyIntegerTargets(const HW::Context& context, ShaderPixelInputInfo& pixel_info) {
	pixel_info.target_uint_mask = 0;
	pixel_info.target_sint_mask = 0;
	if (!IntTargetOutputsEnabled()) {
		return;
	}
	for (uint32_t export_index = 0; export_index < 8u; export_index++) {
		const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
		if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
		const auto& rt = context.GetRenderTarget(slot);
		if (rt.base.addr == 0 || render_target_mask_slot(context.GetRenderTargetMask(), slot) == 0u) {
			continue;
		}
		if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
			pixel_info.target_uint_mask |= static_cast<uint8_t>(1u << export_index);
		} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
			pixel_info.target_sint_mask |= static_cast<uint8_t>(1u << export_index);
		}
	}
	// MRT1 as the second blend source (or the remapped alpha) of target 0 is not an integer output.
	if (pixel_info.dual_source_blending ||
	    pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
		pixel_info.target_uint_mask &= static_cast<uint8_t>(~2u);
		pixel_info.target_sint_mask &= static_cast<uint8_t>(~2u);
	}
}

void ApplyClipSpace(const GraphicContext& graphics, const HW::Context& context,
                    ShaderVertexInputInfo& last_vertex_stage, bool tess_active = false,
                    bool rect_list = false) {
	const auto& clip_control = context.GetClipControl();
	// KYTY_VTE_CNTL: the position format belongs to the last pre-rasterization stage (the TES
	// in a tessellated pipeline).
	const bool     vte_cntl    = ShaderRecompiler::GetCodegenOptions().vte_cntl;
	const uint32_t vte_control = vte_cntl ? context.GetScreenViewport().transform_control
	                                      : ViewportTransform::DefaultControl;
	last_vertex_stage.clip_space.vte_format = ViewportTransform::PackPositionFormat(vte_control);
	if (!clip_control.clip_disable) {
		if (clip_control.user_clip_planes != 0) {
			// Upstream feed075a9. CLIP_DISABLE also disables user planes. How they combine with
			// shader clip and cull distances is unverified; draw push constants reach VS/mesh and
			// PS only, and the generated RECTLIST stages do not forward clip distances.
			const bool unmodelled = (last_vertex_stage.pa_cl_vs_out_cntl & 0xffffu) != 0 ||
			                        tess_active || rect_list;
			if (unmodelled && Common::SkipInvalidEnabled()) {
				// GT7 (usa-ratt/KytyPS5-GT7 round 104): tessellated draws (Tokyo, the LM55 VGT
				// page) set user planes; they are drawn without the planes instead of ending.
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 4u) {
					LOGF("User clip planes 0x%02x ignored (shader clip distances 0x%04x, "
					     "tessellation %d, rect list %d)\n",
					     static_cast<unsigned>(clip_control.user_clip_planes),
					     static_cast<unsigned>(last_vertex_stage.pa_cl_vs_out_cntl & 0xffffu),
					     tess_active ? 1 : 0, rect_list ? 1 : 0);
				}
				return;
			}
			EXIT_NOT_IMPLEMENTED(unmodelled);
			last_vertex_stage.user_clip_planes = clip_control.user_clip_planes;
		}
		return;
	}
	const auto& viewport = context.GetScreenViewport().viewports[0];
	const auto& limits   = graphics.GetPhysicalDeviceProperties().limits;
	auto&       clip     = last_vertex_stage.clip_space;
	clip.scale[0]        = ViewportTransform::Scale<0>(vte_control, viewport.xscale);
	clip.scale[1]        = ViewportTransform::Scale<1>(vte_control, viewport.yscale);
	clip.offset[0]       = ViewportTransform::Offset<0>(vte_control, viewport.xoffset);
	clip.offset[1]       = ViewportTransform::Offset<1>(vte_control, viewport.yoffset);
	clip.half_extent[0] = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
	clip.half_extent[1] = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
	clip.enabled        = true;
}

} // namespace

// KYTY_SKIP_INVALID (chenxiao07/KytyPS5 b310b97c9, b200a57b9): resource tables that never become
// readable (pointers into memory the guest has not mapped while an area loads) skip the draw or
// dispatch instead of ending the session, as an unmaterializable stage already does. Logged once
// per reason.
static PipelineCache::GraphicsPrograms SkipUnmaterializable(const char* why) {
	static std::mutex                      mutex;
	static std::unordered_set<std::string> logged;
	{
		std::scoped_lock lock(mutex);
		if (!logged.insert(why).second) {
			return {};
		}
	}
	std::printf("Warning: %s; it is skipped (KYTY_SKIP_INVALID, logged once)\n", why);
	return {};
}

// KYTY_UNNORMALIZED_FIX (usa-ratt/KytyPS5-GT7 45ac971cc): whether every sampler the stage samples
// through has FORCE_UNNORMALIZED set (S# word 0 bit 15) in this draw's snapshot.
static bool StageSamplersAllUnnormalized(const ShaderStageRuntime& stage) {
	if (!stage || stage.program->info.sampled_pairs.empty()) {
		return false;
	}
	const auto& info = stage.program->info;
	for (const auto& pair: info.sampled_pairs) {
		if (pair.sampler >= info.samplers.size()) {
			return false;
		}
		const auto index = info.samplers[pair.sampler].snapshot_index;
		if (index >= stage.resources->samplers.size()) {
			return false;
		}
		const auto& value = stage.resources->samplers[index];
		if (value.dword_count == 0 || ((value.dwords[0] >> 15u) & 1u) == 0) {
			return false;
		}
	}
	return true;
}

// A hull shader sampling through unnormalized samplers only runs its copy with normalized
// coordinates (pixelRateVariant.h), under an id of its own (its pipelines differ); its samplers are
// made normalized when the draw binds them (ShaderProgram::normalized_samplers).
static void ApplyNormalizedLod(PipelineCache::GraphicsPrograms&            programs,
                               std::array<const ShaderVertexInputInfo*, 3> vertex_info) {
	for (uint32_t i = 0; i < 3; i++) {
		auto& program = programs.vertex[i];
		if (!program || vertex_info[i]->logical_stage != ShaderType::TessellationControl) {
			continue;
		}
		const auto variant = NormalizedLod::Find(program.id);
		if (variant.module == nullptr || !StageSamplersAllUnnormalized(vertex_info[i]->stage)) {
			continue;
		}
		program.module              = variant.module;
		program.id                  = variant.variant_id;
		program.normalized_samplers = true;
	}
}

bool PipelineCache::AsyncTranslationEnabled() noexcept {
	return g_async_translation.On();
}

void PipelineCache::AllowDeferredTranslation(bool allow) noexcept {
	t_defer_requested = allow;
}

void PipelineCache::NoteAsyncTranslationRefused(uint64_t shader_hash) {
	GraphicsTranslations::Instance().Refuse(shader_hash);
}

// KYTY_OBJPRIM_ID=1 (ShaderObjprimIdEnabled; based on Erlou93/KytyPS5-Yotei): the hardware ObjPrimID
// write. A draw with VGT_DRAW_PAYLOAD_CNTL.OBJPRIM_ID_EN writes VGT_OBJECT_ID (plus the primitive
// index when PA_CL_OBJPRIM_ID_CNTL.ADD_PIPED_PRIM_ID is set) into an enabled 32-bit colour slot its
// pixel shader does not export. Ghost of Yotei's depth pre-pass writes its visibility image
// (R32_UINT, slot 7) so; the Vcull tile scan (CS 17444e6a) reads it to build the deferred pass's
// indirect draws, which all draw nothing without it (depth but no G-buffer).
// The slot of the ObjPrimID write: slot 7 when it qualifies (Yotei binds its ID image to slot 7),
// else the first that does. A slot qualifies when CB_TARGET_MASK enables it, a colour target is set
// (base != 0, COLOR_32) and the pixel shader does not export to it. 0xff: none.
static uint8_t ObjprimIdSlot(const HW::Context& context, const ShaderPixelInputInfo& pixel_info) {
	if (!ShaderObjprimIdEnabled() || (context.GetDrawPayloadControl() & 0x1u) == 0) {
		return 0xffu;
	}
	uint32_t exported = 0;
	for (uint32_t index = 0; index < std::size(pixel_info.target_output_mode); index++) {
		if (pixel_info.target_output_mode[index] == 0) {
			continue;
		}
		const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, index);
		if (slot < 8u) {
			exported |= 1u << slot;
		}
	}
	const auto usable = [&](uint32_t slot) {
		const auto& rt = context.GetRenderTarget(slot);
		return render_target_mask_slot(context.GetRenderTargetMask(), slot) != 0 &&
		       (exported & (1u << slot)) == 0 && rt.base.addr != 0 &&
		       rt.info.format == Prospero::ChannelLayout::k32;
	};
	uint8_t chosen = 0xffu;
	if (usable(7)) {
		chosen = 7;
	} else {
		for (uint32_t slot = 0; slot < 8u && chosen == 0xffu; slot++) {
			if (usable(slot)) {
				chosen = static_cast<uint8_t>(slot);
			}
		}
	}
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 8u) {
		std::string layout;
		for (uint32_t slot = 0; slot < 8u; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if (rt.base.addr == 0) {
				continue;
			}
			layout += fmt::format(" [{} base=0x{:x} fmt={} mask={:x}{}]", slot, rt.base.addr,
			                      static_cast<uint32_t>(rt.info.format),
			                      render_target_mask_slot(context.GetRenderTargetMask(), slot),
			                      (exported & (1u << slot)) != 0 ? " export" : "");
		}
		std::printf("ObjPrimID: payload=0x%x objprim_cntl=0x%x slot=%d%s\n", context.GetDrawPayloadControl(),
		            context.GetObjprimIdControl(), chosen == 0xffu ? -1 : static_cast<int>(chosen),
		            layout.c_str());
		std::fflush(stdout);
	}
	return chosen;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    GraphicsStagePreps& stage_preps) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	const bool tess_active = TessellationActive(user_config);
	// KYTY_ASYNC_TRANSLATION: the caller's request holds for this lookup only.
	bool defer_translation = std::exchange(t_defer_requested, false) && g_async_translation.On();
	std::array<ShaderParams, 3> vertex_params;
	// Static stage information (shader map, headers, vertex tables); ended before
	// materialization.
	static constexpr tracy::SourceLocationData prepare_static_location {
	    "Programs::PrepareStatic", TracyFunction, TracyFile, static_cast<uint32_t>(__LINE__), 0};
	Profiler::ScopedBlock static_zone(&prepare_static_location, Profiler::DetailedEnabled());
	if (tess_active) {
		bool unsupported = false;
		vertex_params    = PrepareTessellationPrograms(vertex_regs, context, vertex_info,
		                                               Common::SkipInvalidEnabled() ? &unsupported
		                                                                               : nullptr);
		if (unsupported) {
			return SkipUnmaterializable("a tessellated draw's LS/HS/TES set is not supported");
		}
	} else {
		(void)ShaderTakePrepareFailed();
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
		if (ShaderTakePrepareFailed()) {
			// KYTY_SKIP_INVALID: already logged per shader by PrepareProgram.
			return SkipUnmaterializable("a draw's vertex stage cannot be described (null or "
			                            "uncommitted vertex tables, or a GS that emits no vertex)");
		}
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active && !FinishMeshStage(m_graphics, vertex_info[0])) {
		return SkipUnmaterializable("a mesh shader exceeds the host's mesh limits");
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		ApplyPixelParameterMode(context, user_config, mesh_active, tess_active, pixel_info);
		ApplyDualSourceBlending(context, pixel_info);
		ApplyHalfFloatTargets(context, pixel_info);
		ApplyIntegerTargets(context, pixel_info);
	}
	ApplyClipSpace(m_graphics, context, vertex_info[tess_active ? 2u : 0u], tess_active,
	               Prospero::IsRectList(user_config.GetPrimType()));
	static_zone.End();
	// The program cache locks internally, so no pipeline lock is held across materialization.
	auto& scratch    = ProgramCache::ThreadScratch();
	auto& evaluation = ShaderRecompiler::IR::ThreadEvaluationScratch();
	uint32_t push_data_start = ShaderRecompiler::IR::PushData::GraphicsPushStart(
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwords(
	                      vertex_info[0].mesh.split_groups != 0)
	                : 0,
	    vertex_info[tess_active ? 2u : 0u].user_clip_planes);
	// KYTY_OBJPRIM_ID: the per-draw VGT_OBJECT_ID takes the first push dword after the draw's own
	// dwords (mesh parameters, clip planes); the stages' data follows it.
	pixel_info.objprim_slot        = 0xffu;
	pixel_info.objprim_push_dword  = 0xffu;
	pixel_info.objprim_add_prim_id = false;
	if (pixel_active) {
		pixel_info.objprim_slot = ObjprimIdSlot(context, pixel_info);
		if (pixel_info.HasObjprimOutput()) {
			if (push_data_start < ShaderRecompiler::IR::PushData::DwordCount) {
				pixel_info.objprim_add_prim_id = (context.GetObjprimIdControl() & 0x2u) != 0;
				pixel_info.objprim_push_dword  = static_cast<uint8_t>(push_data_start);
				push_data_start++;
			} else {
				pixel_info.objprim_slot = 0xffu;
			}
		}
	}
	// KYTY_ASYNC_TRANSLATION: no mesh draws, and tessellated ones only with KYTY_PIPELINE_SKIP_TESS
	// (the pipeline-skip stage refusal).
	if (mesh_active || (tess_active && !SkipTessellatedEnabled())) defer_translation = false;
	struct DeferScope {
		explicit DeferScope(bool on) {
			t_defer_translation   = on;
			t_translation_pending = false;
			t_translation_taken   = false;
		}
		~DeferScope() { t_defer_translation = false; }
		DeferScope(const DeferScope&)            = delete;
		DeferScope& operator=(const DeferScope&) = delete;
	} defer_scope(defer_translation);
	const auto serial = [&](std::array<ShaderVertexInputInfo, 3>& vertex_inputs,
	                        ShaderPixelInputInfo& pixel_input,
	                        GraphicsStagePreps& preps) -> GraphicsPrograms {
		for (uint32_t attempt = 0; attempt < 64; ++attempt) {
			ShaderReadAttempt read_attempt;
			uint32_t          push_data_cursor = push_data_start;
			GraphicsPrograms  result;
			t_translation_pending = false;
			if (pixel_active) {
				result.pixel = m_program_cache->Get(pixel_params, pixel_input, preps.pixel,
				                                   push_data_cursor, read_attempt, scratch,
				                                   evaluation);
			}
			const bool link = pixel_active && !tess_active && !mesh_active &&
			                  !Prospero::IsRectList(user_config.GetPrimType());
			LinkPixelParameters(link, vertex_inputs[0], pixel_input);
			// A pending pixel translation leaves the vertex outputs unlinked: when the pixel inputs
			// may be relocated, the vertex key is not the one the draw will have, so the vertex
			// stage waits for the next draw.
			const bool vertex_waits = t_translation_pending && link && MayRelocatePixelParameters(pixel_input);
			for (uint32_t i = 0; i < (tess_active ? 3u : 1u) &&
			                     !read_attempt.materialization_failed && !vertex_waits; ++i) {
				result.vertex[i] =
				    m_program_cache->Get(vertex_params[i], vertex_inputs[i], preps.vertex[i],
				                         push_data_cursor, read_attempt, scratch, evaluation);
			}
			result.translation_pending = t_translation_pending;
			result.translation_taken   = t_translation_taken;
			if (!read_attempt.materialization_failed || result.translation_pending) {
				if (tess_active && UnnormalizedSamplingFix()) {
					ApplyNormalizedLod(result, {&vertex_inputs[0], &vertex_inputs[1], &vertex_inputs[2]});
				}
				return result;
			}
			// No pipeline or texture-cache lock is held while the scheduler publishes bytes.
			// Restart all stages before final bindings/uploads, including their SRT refresh.
			if (!read_attempt.Synchronize()) {
				NoteReadinessFailure("draw", pixel_params.hash, read_attempt);
				if (!Common::SkipInvalidEnabled()) {
					EXIT("graphics resource readiness: a missing range cannot be synchronized\n");
				}
				return SkipUnmaterializable("a draw's resource tables read memory that cannot be synchronized");
			}
		}
		if (!Common::SkipInvalidEnabled()) {
			EXIT("graphics resource readiness did not converge after 64 attempts\n");
		}
		return SkipUnmaterializable("a draw's resource readiness did not converge after 64 attempts");
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
    StagePrep& pixel_prep, GraphicsPrograms& programs, uint64_t* compile_ns) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	EXIT_IF(!DrawPrep::Speculative());
	if (TessellationActive(user_config) || ResourceReuseEnabled()) {
		return SpeculativeResult::Ineligible;
	}
	const auto read_failed = [] { return DrawPrep::ActiveRecorder()->reads->Failed(); };
	// The same static stage preparation as GetGraphicsPrograms (non-tessellated).
	(void)ShaderTakePrepareFailed();
	const auto vertex_params = PrepareProgram(vertex_regs, context, user_config, vertex_info);
	if (read_failed()) {
		return SpeculativeResult::ReadFailed;
	}
	if (ShaderTakePrepareFailed()) {
		return SpeculativeResult::Ineligible; // the serial path skips the draw
	}
	if (vertex_info.logical_stage == ShaderType::Mesh && !FinishMeshStage(m_graphics, vertex_info)) {
		return SpeculativeResult::Ineligible; // the serial path skips the draw
	}
	Common::PrepPhases::Mark(Common::PrepPhases::Phase::VsProgram);
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		ApplyPixelParameterMode(context, user_config, vertex_info.logical_stage == ShaderType::Mesh,
		                        false, pixel_info);
		ApplyDualSourceBlending(context, pixel_info);
		ApplyHalfFloatTargets(context, pixel_info);
		ApplyIntegerTargets(context, pixel_info);
	}
	ApplyClipSpace(m_graphics, context, vertex_info, false,
	               Prospero::IsRectList(user_config.GetPrimType()));
	if (vertex_info.user_clip_planes != 0) {
		return SpeculativeResult::Ineligible; // the serial path places the plane dwords
	}
	// KYTY_OBJPRIM_ID: the serial path places the ObjPrimID push dword.
	pixel_info.objprim_slot        = 0xffu;
	pixel_info.objprim_push_dword  = 0xffu;
	pixel_info.objprim_add_prim_id = false;
	if (pixel_active && ObjprimIdSlot(context, pixel_info) != 0xffu) {
		return SpeculativeResult::Ineligible;
	}
	if (read_failed()) {
		return SpeculativeResult::ReadFailed;
	}
	Common::PrepPhases::Mark(Common::PrepPhases::Phase::PsProgram);
	const uint32_t push_data_start = vertex_info.logical_stage == ShaderType::Mesh
	                                     ? ShaderRecompiler::IR::PushData::MeshDrawDwords(
	                                           vertex_info.mesh.split_groups != 0)
	                                     : 0;
	const bool compile_ahead = g_program_prefetch.On();
	const auto result = m_program_cache->TryPrepareSpeculative(
	    pixel_active, pixel_params, pixel_info, pixel_prep, vertex_params, vertex_info, vertex_prep,
	    push_data_start, programs,
	    pixel_active && vertex_info.logical_stage == ShaderType::Vertex &&
	        !Prospero::IsRectList(user_config.GetPrimType()),
	    compile_ahead ? compile_ns : nullptr);
	if (result == SpeculativeResult::Ok && read_failed()) {
		return SpeculativeResult::ReadFailed;
	}
	return result;
}

bool PipelineCache::DrawPrepTessEnabled() {
	static const bool on = EnvU64("KYTY_DRAW_PREP_TESS", 0) != 0;
	return on;
}

PipelineCache::SpeculativeResult PipelineCache::PrepareTessellatedProgramsSpeculative(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo*, 3> vertex_info, ShaderPixelInputInfo& pixel_info,
    std::array<StagePrep*, 3> vertex_preps, StagePrep& pixel_prep, GraphicsPrograms& programs,
    uint64_t* compile_ns) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	EXIT_IF(!DrawPrep::Speculative());
	if (!TessellationActive(user_config) || ResourceReuseEnabled()) {
		return SpeculativeResult::Ineligible;
	}
	const auto read_failed = [] { return DrawPrep::ActiveRecorder()->reads->Failed(); };
	// GetGraphicsPrograms' static stage preparation for a tessellated draw.
	std::array<ShaderVertexInputInfo, 3> infos;
	bool                                 unsupported = false;
	const auto vertex_params = PrepareTessellationPrograms(vertex_regs, context, infos, &unsupported);
	if (read_failed()) {
		return SpeculativeResult::ReadFailed;
	}
	if (unsupported) {
		return SpeculativeResult::Ineligible; // the serial path skips the draw or ends
	}
	Common::PrepPhases::Mark(Common::PrepPhases::Phase::VsProgram);
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		ApplyPixelParameterMode(context, user_config, false, true, pixel_info);
		ApplyDualSourceBlending(context, pixel_info);
		ApplyHalfFloatTargets(context, pixel_info);
		ApplyIntegerTargets(context, pixel_info);
	}
	ApplyClipSpace(m_graphics, context, infos[2], true,
	               Prospero::IsRectList(user_config.GetPrimType()));
	if (infos[2].user_clip_planes != 0) {
		return SpeculativeResult::Ineligible; // the serial path places the plane dwords
	}
	if (read_failed()) {
		return SpeculativeResult::ReadFailed;
	}
	Common::PrepPhases::Mark(Common::PrepPhases::Phase::PsProgram);
	for (uint32_t i = 0; i < 3; i++) {
		*vertex_info[i] = infos[i];
	}
	const bool compile_ahead = g_program_prefetch.On();
	const auto result = m_program_cache->TryPrepareSpeculativeTess(
	    pixel_active, pixel_params, pixel_info, pixel_prep, vertex_params, vertex_info,
	    vertex_preps, 0, programs, compile_ahead ? compile_ns : nullptr);
	if (result == SpeculativeResult::Ok && read_failed()) {
		return SpeculativeResult::ReadFailed;
	}
	if (result == SpeculativeResult::Ok && UnnormalizedSamplingFix()) {
		ApplyNormalizedLod(programs, {vertex_info[0], vertex_info[1], vertex_info[2]});
	}
	return result;
}

void PipelineCache::SelectCallSplitCode(const void* code) {
	t_split_code = static_cast<const ShaderRecompiler::CallSplit::Code*>(code);
}

std::shared_ptr<const void> PipelineCache::TakePartitionRun() {
	return std::exchange(t_partition_run, nullptr);
}

void PipelineCache::ReportPartitionKeys(uint64_t program, std::span<const uint32_t> keys) {
	CallExpander().ReportPartitionKeys(program, keys);
}

std::shared_ptr<const void> PipelineCache::TakeCallSplitRun() {
	return std::exchange(t_split_run, nullptr);
}

// KYTY_SKIP_SHADER_HASHES="hash,hash,..." (hex, default empty; usa-ratt/KytyPS5-GT7's skip list,
// compute only here): the dispatches of these guest compute shaders are skipped, as a compute
// program that fails to build is. A measurement aid (what one pass costs), not for presets.
static bool SkipComputeShaderRequested(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> result;
		if (const char* value = std::getenv("KYTY_SKIP_SHADER_HASHES"); value != nullptr) {
			std::string_view list(value);
			size_t           start = 0;
			while (start < list.size()) {
				const auto end = std::min(list.find(',', start), list.size());
				if (end > start) {
					result.push_back(std::strtoull(std::string(list.substr(start, end - start)).c_str(),
					                               nullptr, 16));
				}
				start = end + 1;
			}
		}
		for (const auto hash: result) {
			PipelineCacheLog("Skip list: the dispatches of compute shader 0x{:016x} are skipped "
			                 "(KYTY_SKIP_SHADER_HASHES)",
			                 hash);
		}
		return result;
	}();
	return !hashes.empty() && std::find(hashes.begin(), hashes.end(), shader_hash) != hashes.end();
}

void PipelineCache::DropCallRuns() {
	t_split_run.reset();
	t_partition_run.reset();
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info,
                                               StagePrep&                   stage_prep) {
	auto program = LookupComputeProgram(regs, sh, input_info, stage_prep);
	if (!program) {
		// A split kernel whose R_0 is still translating (CallTranslations) or not ready: the
		// dispatch is skipped, and its run must not reach the next dispatch of this thread. A
		// draw-prep dispatch (KYTY_DRAW_PREP_DISPATCH) skips this lookup and would take it as its
		// own: the next kernel then ran the split chain with its registers (its hash and its AGC
		// scratch size) and the other kernel's split code, and stopped in resource tracking.
		t_split_run.reset();
	}
	return program;
}

ShaderProgram PipelineCache::LookupComputeProgram(const HW::ComputeShaderInfo& regs,
                                                  const HW::ShaderRegisters&   sh,
                                                  ShaderComputeInputInfo&      input_info,
                                                  StagePrep&                   stage_prep) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	std::optional<CpGaps::Scope> prepare_gap;
	prepare_gap.emplace(CpGaps::Cat::ProgPrepare);
	auto              params      = PrepareProgram(regs, sh, input_info);
	t_empty_hash = 0;
	t_empty_why.clear();
	if (g_drop_cs_hashes.Get() != 0 && DropCsHashListed(params.hash)) {
		NoteEmptyProgram(params.hash, "listed in KYTY_DROP_CS_HASHES");
		return {};
	}
	if (SkipComputeShaderRequested(params.hash)) {
		return {};
	}
	if (!ExpandComputeCalls(params, input_info)) {
		NoteEmptyProgram(params.hash, "call split plan being built");
		return {};
	}
	prepare_gap.reset();
	// Use one effective size and placement for the cache key, LDS declaration, and access bounds.
	// Above the device limit (AMD reports 32 KiB; some Astro Bot compute shaders ask for 48) the
	// LDS lives in a device buffer (upstream b0bbaef1e; RenderExecutor binds it per dispatch,
	// BindSharedMemory; KYTY_LDS_DEVICE_BUFFER, default auto), or is clamped (=0).
	const auto requested = input_info.lds_size_dwords;
	const auto lds = ShaderRecompiler::PlanComputeLds(
	    requested, m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize);
	input_info.lds_size_dwords = lds.dwords;
	input_info.lds_storage     = lds.storage;
	if (requested * 4u > lds.limit_bytes) {
		// Once per shader, named.
		static std::mutex                   warned_mutex;
		static std::unordered_set<uint64_t> warned;
		bool                                first = false;
		{
			const std::lock_guard lock(warned_mutex);
			first = warned.insert(params.hash).second;
		}
		if (first && lds.clamped) {
			PipelineCacheLog("GPU warning: game compute shader 0x{:016x} requests {} bytes of LDS, "
			                 "but the Vulkan device limit is {} bytes. Clamping LDS; rendering may "
			                 "be incorrect (KYTY_LDS_DEVICE_BUFFER=0).",
			                 params.hash, requested * 4u, lds.limit_bytes);
		} else if (first) {
			PipelineCacheLog("GPU note: game compute shader 0x{:016x} requests {} bytes of LDS, "
			                 "above the Vulkan device limit of {} bytes; its LDS lives in a device "
			                 "buffer (KYTY_LDS_DEVICE_BUFFER).",
			                 params.hash, requested * 4u, lds.limit_bytes);
		}
	}
	auto& scratch    = ProgramCache::ThreadScratch();
	auto& evaluation = ShaderRecompiler::IR::ThreadEvaluationScratch();
	if (params.call_expansion != 0 && CallTranslations::Enabled()) {
		// A new expanded call program: its dispatches are skipped while it translates.
		auto& key = scratch.key;
		ProgramCache::BuildKey(params, input_info, key);
		if (m_program_cache->FindSource(key) == nullptr) {
			auto&      translations = CallTranslations::Instance();
			const auto id = CallTranslations::Id(key.hash, key.user_data_count, key.static_state);
			switch (translations.Query(id)) {
				case CallTranslations::State::Missing:
					translations.Queue(id, params, input_info);
					NoteEmptyProgram(params.hash, "expanded call program translating");
					return {};
				case CallTranslations::State::Pending:
					NoteEmptyProgram(params.hash, "expanded call program translating");
					return {};
				case CallTranslations::State::Ready: break;
			}
		}
	}
	for (uint32_t attempt = 0; attempt < 64; ++attempt) {
		ShaderReadAttempt read_attempt;
		uint32_t push_data_cursor = 0;
		const auto result = m_program_cache->Get(params, input_info, stage_prep, push_data_cursor,
		                                         read_attempt, scratch, evaluation);
		if (!read_attempt.materialization_failed) return result;
		if (!read_attempt.Synchronize()) {
			NoteReadinessFailure("dispatch", params.hash, read_attempt);
			if (!Common::SkipInvalidEnabled()) {
				EXIT("compute resource readiness: a missing range cannot be synchronized\n");
			}
			(void)SkipUnmaterializable("a dispatch's resource tables read memory that cannot be synchronized");
			NoteEmptyProgram(params.hash, "resource tables read memory that cannot be synchronized");
			return {};
		}
	}
	if (!Common::SkipInvalidEnabled()) {
		EXIT("compute resource readiness did not converge after 64 attempts\n");
	}
	(void)SkipUnmaterializable("a dispatch's resource readiness did not converge after 64 attempts");
	NoteEmptyProgram(params.hash, "resource readiness did not converge");
	return {};
}

PipelineCache::SpeculativeResult PipelineCache::PrepareComputeProgramSpeculative(
    const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
    ShaderComputeInputInfo& input_info, StagePrep& stage_prep, ShaderProgram& program,
    uint64_t* compile_ns) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	EXIT_IF(!DrawPrep::Speculative());
	if (ResourceReuseEnabled()) {
		return SpeculativeResult::Ineligible;
	}
	const auto read_failed = [] { return DrawPrep::ActiveRecorder()->reads->Failed(); };
	// The same static preparation as GetComputeProgram.
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto params             = PrepareProgram(regs, sh, input_info);
	if (read_failed()) {
		return SpeculativeResult::ReadFailed;
	}
	// A program with calls is expanded on the serial path only (ExpandComputeCalls).
	if (CallProgramsHandled() && params.back_code.empty() && ComputeProgramMayCall(params)) {
		return SpeculativeResult::Ineligible;
	}
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	const bool compile_ahead = g_program_prefetch.On();
	const auto result        = m_program_cache->TryPrepareSpeculativeCompute(
        params, input_info, stage_prep, program, compile_ahead ? compile_ns : nullptr);
	if (result == SpeculativeResult::Ok && read_failed()) {
		return SpeculativeResult::ReadFailed;
	}
	if (result == SpeculativeResult::Ok && m_compute_ahead && m_compute_prewarm != nullptr && program &&
	    input_info.stage.program != nullptr && !input_info.stage.program->info.uses_bvh) {
		// Each id once; a build finds a pipeline the command processor made first and drops itself.
		m_compute_prewarm->RequestAhead(program.id, input_info, program.module);
	}
	return result;
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

namespace {
std::mutex g_spirv_variant_mutex;
// (base program id, group index) -> edited copy
std::unordered_map<uint64_t, ShaderProgram> g_spirv_variants;
constexpr uint64_t SpirvVariantKey(uint64_t id, size_t group) {
	return (id << 8u) | (group & 0xffu);
}
} // namespace

static void RegisterSpirvVariant(uint64_t id, size_t group, ShaderProgram variant) {
	std::scoped_lock lock(g_spirv_variant_mutex);
	g_spirv_variants[SpirvVariantKey(id, group)] = variant;
}

ShaderProgram PipelineCache::SpirvVariantProgram(const ShaderProgram& program, uint32_t group) {
	if (!program || group == 0) return {};
	std::scoped_lock lock(g_spirv_variant_mutex);
	const auto       it = g_spirv_variants.find(SpirvVariantKey(program.id, group - 1u));
	if (it == g_spirv_variants.end()) return {};
	auto variant                = it->second;
	variant.normalized_samplers = program.normalized_samplers;
	return variant;
}

bool PipelineCache::SpirvVariantsConfigured() {
	return !SpirvVariantSettings().empty();
}

// KYTY_SPIRV_VARIANT_LIVE (diagnostic, live, default 0): n = draws (renderDraw.cpp) and dispatches
// (TryGetComputePipeline) run group n's KYTY_SPIRV_VARIANT copies of their programs.
static Live::Switch g_spirv_variant_live("KYTY_SPIRV_VARIANT_LIVE", [](const char* value) -> int64_t {
	return value != nullptr ? static_cast<int64_t>(std::strtoul(value, nullptr, 10)) : 0;
});

uint32_t PipelineCache::SpirvVariantLiveGroup() {
	return static_cast<uint32_t>(g_spirv_variant_live.Get());
}

// KYTY_LAYER_CLAMP: the least layer count of the bound attachments (0 without any).
static uint32_t BoundLayerCount(std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
                                bool with_depth) {
	if (!ShaderRecompiler::GetCodegenOptions().layer_clamp) {
		return 0;
	}
	uint32_t layers = UINT32_MAX;
	for (const auto& color: colors) {
		layers = std::min(layers, color.desc.view_info.layer_count);
	}
	if (with_depth) {
		layers = std::min(layers, depth.desc.view_info.layer_count);
	}
	return layers == UINT32_MAX ? 0u : std::max(layers, 1u);
}

// KYTY_LAYER_CLAMP: whether the vertex-side stage exports the render target array index.
static bool ExportsLayer(const ShaderVertexInputInfo& info) {
	return info.stage && std::ranges::any_of(info.stage.program->info.outputs, [](const auto& output) {
		       return output.kind == ShaderRecompiler::IR::StageOutputKind::Layer;
	       });
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
	rendering.layer_limit       = ShaderRecompiler::GetCodegenOptions().layer_clamp &&
	                                      targets.layer_count != 0 && ExportsLayer(vs_input_info)
	                                  ? targets.layer_count
	                                  : 0u;
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
			// Upstream 73615c31f, 0bef3fc0d, 94e7d239d, 96011cc5c: logical alpha or the mapped
			// source factors through MRT1 (ApplyDualSourceBlending), blended as the second source,
			// or no blending where the export mapping cannot be blended correctly.
			auto alpha_source = ShaderAlphaBlendSource::None;
			if (slot == 0 && ps_input_info != nullptr) {
				alpha_source = ps_input_info->alpha_blend_source;
			}
			const bool alpha_remap = alpha_source != ShaderAlphaBlendSource::None;
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
			if (alpha_remap && static_params.blend_enable[slot]) {
				// The second source's factors replace the source-alpha ones; one shared equation.
				auto blend = bc;
				switch (alpha_source) {
					case ShaderAlphaBlendSource::SourceAlpha:
						blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
						blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
						break;
					case ShaderAlphaBlendSource::SourceAlphaOne:
					case ShaderAlphaBlendSource::SourceAlphaZero:
						// The second source carries the mapped source factor; its alpha stays
						// logical Sa.
						blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
						blend.color_destblend =
						    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
						break;
					case ShaderAlphaBlendSource::OneAlphaZero:
						// The second source: 1 or 0 for each physical colour channel (0 where
						// logical alpha lies), logical Sa in its A for the destination factor.
						// Physical A holds a logical colour channel, so its source factor is One.
						blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
						blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
						blend.alpha_srcblend  = static_cast<uint8_t>(Prospero::BlendFactor::kOne);
						blend.alpha_destblend = blend.color_destblend;
						blend.alpha_comb_fcn  = blend.color_comb_fcn;
						break;
					case ShaderAlphaBlendSource::None: break;
				}
				static_params.color_srcblend[slot]       = blend.color_srcblend;
				static_params.color_destblend[slot]      = blend.color_destblend;
				static_params.separate_alpha_blend[slot] = false;
				if (alpha_source == ShaderAlphaBlendSource::OneAlphaZero) {
					static_params.alpha_srcblend[slot]       = blend.alpha_srcblend;
					static_params.alpha_destblend[slot]      = blend.alpha_destblend;
					static_params.alpha_comb_fcn[slot]       = blend.alpha_comb_fcn;
					static_params.separate_alpha_blend[slot] = true;
				}
			}
			if (alpha_remap) {
				// Kept in the key (its layout is persisted); the factors above carry the remap.
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
			// KYTY_EQAA_MIXED=1 (usa-ratt/KytyPS5-GT7 18979e65b): PS5 EQAA, colour at the
			// fragment count and depth at the coverage count, through mixed attachment samples
			// (AMD: either side larger; NV: colour < depth, rasterizing at the depth count).
			using MixedSamples             = GraphicContext::MixedAttachmentSamples;
			const auto     mixed           = m_graphics.mixed_attachment_samples;
			const uint32_t color_samples   = attachment_samples;
			const uint32_t depth_samples   = targets.depth_samples;
			const bool     allowed = mixed == MixedSamples::Amd ||
			                     (mixed == MixedSamples::Nv && color_samples < depth_samples);
			if (!allowed) {
				if (fatal) {
					EXIT("mixed color/depth sample counts are unsupported: %u and %u\n",
					     attachment_samples, targets.depth_samples);
				}
				return false;
			}
			rendering.color_samples = color_samples;
			rendering.depth_samples = depth_samples;
			attachment_samples      = std::max(color_samples, depth_samples);
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				LOGF("Pipeline: mixed attachment samples (%s, KYTY_EQAA_MIXED): colour %u, depth "
				     "%u, raster %u\n",
				     mixed == MixedSamples::Amd ? "AMD" : "NV", color_samples, depth_samples,
				     attachment_samples);
			}
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
	if (static_params.sample_shading_enable && rendering.color_samples != 0 &&
	    m_graphics.mixed_attachment_samples == GraphicContext::MixedAttachmentSamples::Nv) {
		// NV mixed samples may not shade per sample (VUID-VkPipelineMultisampleStateCreateInfo-
		// rasterizationSamples-01415): per pixel, with the per-pixel copy of a shader reading its
		// sample index (pixelRateVariant.h).
		static_params.sample_shading_enable = false;
	}
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		if (fatal) {
			EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
		}
		return false;
	}
	static_params.alpha_to_coverage_enable = ps_active && ctx.GetShaderRegisters().AlphaToMask();
	// KYTY_AA_MASK_COVERAGE (aaMaskCoverage.h; the GT7 fork's fc4111066): PA_SC_AA_MASK reduced to
	// the raster samples. A draw whose masks leave samples out runs the pixel shader's copy that
	// applies them (or a coverage-only shader without one) once the copy is made; until then, and
	// for shaders without a copy, the key keeps all ones and the draw runs as before.
	if (AaMaskCoverage::Enabled()) {
		const uint32_t samples = std::clamp<uint32_t>(attachment_samples, 1u, 16u);
		const uint32_t full    = samples >= 16u ? 0xffffu : (1u << samples) - 1u;
		const uint32_t both    = full | (full << 16u);
		const uint32_t mask01  = ctx.GetAaMask(0) & both;
		const uint32_t mask23  = ctx.GetAaMask(1) & both;
		if (mask01 != both || mask23 != both) {
			// The EQAA per-pixel copy the fragment stage runs (shaders.cpp), whose AA copy it takes.
			const bool pixel_rate =
			    ps_active && rendering.color_samples != 0 &&
			    m_graphics.mixed_attachment_samples == GraphicContext::MixedAttachmentSamples::Nv &&
			    !EqaaPerSampleShading() && PixelRate::Find(pixel_program.id).module != nullptr;
			if (AaMaskCoverage::Request(m_graphics.device, ps_active ? pixel_program.id : 0u, pixel_rate,
			                            mask01, mask23) != nullptr) {
				static_params.aa_mask[0] = mask01;
				static_params.aa_mask[1] = mask23;
			}
		}
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
	const auto iter  = m_graphics_pipelines.find(key);
	const auto found = iter != m_graphics_pipelines.end() ? iter->second.get() : nullptr;
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
	FlushCompileStall();
}

PipelineCache::PrefetchTotals PipelineCache::GetPrefetchTotals() const {
	if (m_prefetch == nullptr) return {};
	return {m_prefetch->submitted.load(), m_prefetch->used.load(), m_prefetch->compile_ns.load(),
	        m_prefetch->wait_ns.load(), m_prefetch->max_wait_ns.load(),
	        m_program_cache->speculative_programs.load()};
}

FastFirstSnapshot PipelineCache::GetFastFirstTotals() const {
	if (m_fast_first == nullptr) return {};
	return SnapshotFastFirst(m_fast_first->counters, m_fast_first->scheduler.Pending());
}

bool PipelineCache::PipelinePrefetchEnabled() const noexcept {
	return m_prefetch != nullptr && g_pipeline_prefetch.On();
}

void PipelineCache::NoteProgramPrefetchWait(uint64_t ns) {
	if (ns != 0) AddCompileStall(ns);
}

void PipelineCache::PrefetchGraphicsPipeline(const PipelineTargets& targets, const HW::Context& ctx,
    const HW::UserConfig& user_config, const ShaderVertexInputInfo& vertex_info,
    const ShaderPixelInputInfo* pixel_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	if (!PipelinePrefetchEnabled() || programs.VertexStageCount() != 1 ||
	    topology == vk::PrimitiveTopology::ePatchList || graphics_debug_dump_enabled()) return;
	const Pipeline* pipeline = nullptr;
	uint64_t generation = 0;
	if (FindGraphicsPipelineForPlan(targets, ctx, user_config, vertex_info, pixel_info,
	    topology, primitive_restart_enable, programs, pipeline, generation) != PlanLookup::Absent) return;
	GraphicsPipelineKey key {};
	if (!BuildGraphicsPipelineKey(targets, ctx, user_config, vertex_info, pixel_info, topology,
	    primitive_restart_enable, programs, false, key) || !m_mutex.TryLock()) return;
	// Recheck under the map lock: CP creation cannot race a speculative request into a second job.
	if (!m_graphics_pipelines.contains(key)) m_prefetch->Request(key, {&vertex_info, 1}, pixel_info, programs);
	m_mutex.Unlock();
}

void PipelineCache::PrefetchGraphicsPipeline(std::span<const RenderColorInfo> colors,
    const RenderDepthInfo& depth, std::span<const ShaderVertexInputInfo> vertex_info,
    CommandBuffer& command, const ShaderPixelInputInfo* pixel_info,
    vk::PrimitiveTopology topology, bool primitive_restart_enable, const GraphicsPrograms& programs) {
	if (!PipelinePrefetchEnabled() || vertex_info.size() != 1) return;
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
	targets.layer_count              = BoundLayerCount(colors, depth, targets.with_depth);

	PrefetchGraphicsPipeline(targets, command.GetRegisters(), command.GetUserConfig(),
	    vertex_info.front(), pixel_info, topology, primitive_restart_enable, programs);
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	return *GetGraphicsPipelineImpl(colors, depth, vertex_info, command, ps_input_info, topology,
	                                primitive_restart_enable, programs, false);
}

PipelineCache::Pipeline* PipelineCache::TryGetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs, bool skippable) {
	return GetGraphicsPipelineImpl(colors, depth, vertex_info, command, ps_input_info, topology,
	                               primitive_restart_enable, programs, skippable && m_skip != nullptr);
}

PipelineCache::SkipTotals PipelineCache::GetSkipTotals() const {
	if (m_skip == nullptr) return {};
	return {.requested = m_skip->submitted.load(std::memory_order_relaxed),
	        .published = m_skip->used.load(std::memory_order_relaxed),
	        .skipped_draws = m_skip_draws.load(std::memory_order_relaxed),
	        .ready_in_wait = m_skip_ready_in_wait.load(std::memory_order_relaxed),
	        .not_queued = m_skip_not_queued.load(std::memory_order_relaxed),
	        .compile_ns = m_skip->compile_ns.load(std::memory_order_relaxed)};
}

PipelineCache::Pipeline* PipelineCache::GetGraphicsPipelineImpl(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs, bool skippable) {
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
	targets.layer_count              = BoundLayerCount(colors, depth, targets.with_depth);

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;
	// For the compile-wait lines (common/compileWatch.h).
	const uint64_t watch_vs_hash = vs_input_info.stage.program != nullptr ? vs_input_info.stage.program->shader_hash : 0;
	const uint64_t watch_ps_hash =
	    ps_active && ps_input_info->stage.program != nullptr ? ps_input_info->stage.program->shader_hash : 0;

	GraphicsPipelineKey key {};
	(void)BuildGraphicsPipelineKey(targets, command.GetRegisters(), command.GetUserConfig(),
	                               vs_input_info, ps_input_info, topology, primitive_restart_enable,
	                               programs, true, key);

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
	const auto remember = [&](Pipeline& pipeline) -> Pipeline* {
		if (memo_enabled) {
			last.cache      = this;
			last.pipeline   = &pipeline;
			last.generation = m_pipeline_generation.load(std::memory_order_relaxed);
			last.key        = key;
		}
		FlushCompileStall();
		return &pipeline;
	};

	PrefetchState::Result ready;
	PrefetchState*        ready_from = nullptr;
	uint64_t wait_begin = 0;
	// KYTY_PIPELINE_SKIP: a draw that only changes the picture does not wait for a pipeline the
	// driver has not built: a background thread builds it (optimized, journaled when published)
	// and the draw is skipped until it is there, after waiting KYTY_PIPELINE_SKIP_WAIT_US for it
	// (a driver-cache hit is usually done by then). A full queue falls back to building it here.
	// KYTY_PIPELINE_SKIP_TESS: tessellated draws (three vertex stages, patch lists) as well.
	const bool background_ok =
	    (vertex_info.size() == 1 && topology != vk::PrimitiveTopology::ePatchList) ||
	    (vertex_info.size() == 3 && SkipTessellatedEnabled());
	// KYTY_PIPELINE_SKIP_PROBE=1 (default off; adapted from usa-ratt/KytyPS5-GT7 round 87,
	// 3a8591ae2): a skippable draw's new pipeline that no builder has yet is first looked up in the
	// driver cache alone, here (VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT): one the
	// driver has is published at once instead of waiting behind the builders' queue (where a cache
	// hit was skipped for seconds behind second-long compiles), one it does not have goes to the
	// builders without a wait for the lookup. Needs pipelineCreationCacheControl.
	bool probe_hit = false;
	if (m_skip_probe && skippable && background_ok && !graphics_debug_dump_enabled()) {
		bool try_probe = false;
		{
			Common::LockGuard lock(m_mutex);
			if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
				return remember(*iter->second);
			}
			try_probe = !(m_prefetch != nullptr && m_prefetch->Has(key)) && !m_skip->Has(key);
		}
		if (try_probe) {
			const auto probe_begin = CompileClockNs();
			auto       probed      = std::make_unique<Pipeline>();
			GraphicsContent probed_content;
			const bool keyed_probe = m_known != nullptr || m_journal != nullptr;
			GraphicsPipelineCreateHook probe_hook = [&](const vk::GraphicsPipelineCreateInfo& info,
			                                            std::span<const uint32_t>         layout_signature,
			                                            vk::Pipeline*                     pipeline) {
				if (m_adopt) {
					probed_content = DescribeGraphicsPipeline(info, layout_signature);
					if (auto adopted = TakeAdoptedPipeline(probed_content.key); adopted != nullptr) {
						PipelineFeedback::NoteAdopted(PipelineFeedback::Path::Probe);
						*pipeline = adopted;
						return vk::Result::eSuccess;
					}
				}
				auto probe_info = info;
				probe_info.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
				const auto made = PipelineFeedback::CreateGraphics(m_graphics.device, m_driver_cache, probe_info,
				                                                   pipeline, PipelineFeedback::Path::Probe);
				if (made == vk::Result::eSuccess && *pipeline != nullptr && keyed_probe) {
					if (probed_content.key == 0) probed_content = DescribeGraphicsPipeline(info, layout_signature);
					NoteKnownPipeline(probed_content.key);
				}
				return made;
			};
			vk::Result probe_result = vk::Result::eSuccess;
			{
				Common::CrashContext::Scope note("graphics pipeline driver-cache lookup (VS, PS)",
				                                 vs_input_info.stage.program != nullptr
				                                     ? vs_input_info.stage.program->shader_hash
				                                     : 0,
				                                 ps_active && ps_input_info->stage.program != nullptr
				                                     ? ps_input_info->stage.program->shader_hash
				                                     : 0);
				CreatePipelineInternal(m_graphics, *probed, key.rendering, key.vertex_input, vertex_info,
				                       ps_input_info, programs, key.static_params, m_driver_cache, &probe_hook,
				                       &probe_result);
			}
			const auto probe_ns = CompileClockNs() - probe_begin;
			m_skip_probe_ns.fetch_add(probe_ns, std::memory_order_relaxed);
			if (probe_result == vk::Result::eSuccess) {
				m_skip_probe_hits.fetch_add(1, std::memory_order_relaxed);
				ready.pipeline   = std::move(probed);
				ready.content    = std::move(probed_content);
				ready.compile_ns = probe_ns;
				probe_hit        = true;
				wait_begin       = probe_begin;
			} else {
				m_skip_probe_misses.fetch_add(1, std::memory_order_relaxed);
			}
		}
	}
	if (skippable && background_ok && !graphics_debug_dump_enabled() && !probe_hit) {
		bool           queued = false;
		PrefetchState* source = m_skip.get();
		{
			Common::LockGuard lock(m_mutex);
			if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
				return remember(*iter->second);
			}
			// A build the prefetch pool already has for the key (draw preparation asked ahead) is
			// polled there instead of building the same pipeline a second time on the skip builder.
			if (m_prefetch != nullptr && m_prefetch->Has(key)) {
				source = m_prefetch.get();
				queued = true;
			} else {
				queued = m_skip->Request(key, vertex_info, ps_input_info, programs);
			}
		}
		uint64_t wait_ns = queued ? m_skip_wait_ns : 0;
		if (m_skip_frame_budget_ns != 0) {
			wait_ns = std::min(wait_ns, m_skip_frame_budget_ns - std::min(m_skip_frame_budget_ns, m_skip_frame_used));
		}
		const auto poll_begin = wait_ns != 0 ? CompileClockNs() : 0;
		const auto poll       = source->TryTake(key, wait_ns, ready);
		if (wait_ns != 0) m_skip_frame_used += CompileClockNs() - poll_begin;
		// Totals at most every 2 s while skips or skip builds happen (the exit report may not run).
		const auto report = [this] {
			static uint64_t last_ns = 0;
			const auto      now     = CompileClockNs();
			if (now - last_ns < 2'000'000'000ull) return;
			last_ns         = now;
			const auto skip = GetSkipTotals();
			PipelineCacheLog("Pipeline skip: {} draws skipped so far; {} pipelines queued ({} ms building), {} "
			                 "published ({} within the wait); {} lookups found the queue full",
			                 skip.skipped_draws, skip.requested, skip.compile_ns / 1000000, skip.published,
			                 skip.ready_in_wait, skip.not_queued);
			if (m_skip_probe) {
				PipelineCacheLog("Pipeline skip: driver-cache lookups found {} pipelines, sent {} to the "
				                 "builders ({} ms looking; KYTY_PIPELINE_SKIP_PROBE)",
				                 m_skip_probe_hits.load(std::memory_order_relaxed),
				                 m_skip_probe_misses.load(std::memory_order_relaxed),
				                 m_skip_probe_ns.load(std::memory_order_relaxed) / 1000000);
			}
			static uint64_t feedback_ns = 0;
			if (now - feedback_ns >= 10'000'000'000ull) {
				feedback_ns = now;
				if (const auto feedback = PipelineFeedback::Report(); !feedback.empty()) {
					PipelineCacheLog("{}", feedback);
				}
				if (m_adopt) PipelineCacheLog("Pipeline journal: {}", AdoptReport());
			}
		};
		if (poll == PrefetchState::Poll::Pending) {
			g_frame_pipeline_skips.fetch_add(1, std::memory_order_relaxed);
			PassCensus::Drop('d', vs_input_info.stage.program != nullptr ? vs_input_info.stage.program->shader_hash : 0,
			                 "skipped: pipeline building (KYTY_PIPELINE_SKIP)");
			if (m_skip_draws.fetch_add(1, std::memory_order_relaxed) == 0) {
				PipelineCacheLog("Pipeline skip: first draw skipped while its pipeline builds "
				                 "(VS 0x{:016x})", vs_input_info.stage.program != nullptr
				                                       ? vs_input_info.stage.program->shader_hash
				                                       : 0);
			}
			report();
			return nullptr;
		}
		if (poll == PrefetchState::Poll::Ready) {
			ready_from = source;
			wait_begin = CompileClockNs();
			if (queued) m_skip_ready_in_wait.fetch_add(1, std::memory_order_relaxed);
			report();
		} else {
			m_skip_not_queued.fetch_add(1, std::memory_order_relaxed);
		}
	}
	// A draw that cannot be skipped waits for a build the skip builder already has for its key
	// (moved to the front of its queue) instead of building the pipeline a second time.
	if (!skippable && ready_from == nullptr && m_skip != nullptr &&
	    m_skip->pending_count.load(std::memory_order_relaxed) != 0) {
		{
			Common::LockGuard lock(m_mutex);
			if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
				return remember(*iter->second);
			}
		}
		wait_begin = CompileClockNs();
		CompileWatch::Wait watch("draw", CompileWatch::Kind::GraphicsPipeline, watch_vs_hash, watch_ps_hash);
		ready      = m_skip->Take(key);
		if (ready.pipeline != nullptr) {
			ready_from = m_skip.get();
		} else {
			wait_begin = 0;
		}
	}
	if (ready_from == nullptr && !probe_hit && m_prefetch != nullptr && background_ok &&
	    !graphics_debug_dump_enabled()) {
		{
			Common::LockGuard lock(m_mutex);
			if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
				return remember(*iter->second);
			}
			// Reserve the key before releasing the map. Other preparation threads may submit
			// independent keys while this draw waits, allowing compiles to overlap each other.
			if (PipelinePrefetchEnabled()) m_prefetch->Request(key, vertex_info, ps_input_info, programs);
		}
		wait_begin = CompileClockNs();
		CompileWatch::Wait watch("draw", CompileWatch::Kind::GraphicsPipeline, watch_vs_hash, watch_ps_hash);
		ready = m_prefetch->Take(key);
		if (ready.pipeline != nullptr) ready_from = m_prefetch.get();
	}
	const auto        lock_begin = PaceLog::StallTraceEnabled() ? CompileClockNs() : 0;
	Common::LockGuard lock(m_mutex);
	if (lock_begin != 0) {
		PaceLog::NoteStall(lock_begin, CompileClockNs(), PaceLog::StallPipelineMutex, watch_vs_hash, 0,
		                   "graphics pipeline lock");
	}
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		if (ready.pipeline != nullptr) {
			m_graphics.device.destroyPipeline(ready.pipeline->pipeline, nullptr);
			ReleasePipelineLayout(m_graphics, ready.pipeline->pipeline_layout,
			                      ready.pipeline->descriptor_set_layout);
			if (ready_from != nullptr) ready_from->Complete(key); // null: a driver-cache lookup's
		}
		return remember(*iter->second);
	}
	const auto create_begin = wait_begin != 0 ? wait_begin : CompileClockNs();

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	const uint64_t vs_hash =
	    vertex_info[0].stage.program ? vertex_info[0].stage.program->shader_hash : 0;
	const uint64_t ps_hash =
	    ps_active && ps_input_info->stage.program ? ps_input_info->stage.program->shader_hash : 0;
	const bool prefetched = ready.pipeline != nullptr;
	auto cached = prefetched ? std::move(ready.pipeline) : std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	// Pipeline libraries (6/7): the monolithic create info goes to the library path, which may
	// fast-link it and hand back the create info for the background optimized compile.
	GraphicsPipelineLibrary::Result library_result;
	GraphicsPipelineCreateHook      library_hook;
	FastFirstState::GraphicsFast    fast_result;
	GraphicsContent                 content;
	const bool                      keyed = m_known != nullptr || m_journal != nullptr;
	if (m_fast_first != nullptr && !m_fast_first->compute_only && !prefetched && !m_fast_first->scheduler.Stopped()) {
		library_hook = [&](const vk::GraphicsPipelineCreateInfo& info,
		                   std::span<const uint32_t> layout_signature, vk::Pipeline* pipeline) {
			if (keyed) content = DescribeGraphicsPipeline(info, layout_signature);
			return m_fast_first->CreateGraphics(info, content, pipeline, fast_result);
		};
	} else if (keyed && !prefetched) {
		library_hook = [&](const vk::GraphicsPipelineCreateInfo& info,
		                   std::span<const uint32_t> layout_signature, vk::Pipeline* pipeline) {
			content = DescribeGraphicsPipeline(info, layout_signature);
			if (auto adopted = TakeAdoptedPipeline(content.key); adopted != nullptr) {
				PipelineFeedback::NoteAdopted(PipelineFeedback::Path::Sync);
				*pipeline = adopted;
				return vk::Result::eSuccess;
			}
			const auto result = PipelineFeedback::CreateGraphics(m_graphics.device, m_driver_cache, info, pipeline,
			                                                     PipelineFeedback::Path::Sync);
			if (result == vk::Result::eSuccess) NoteKnownPipeline(content.key);
			return result;
		};
	}
	if (m_library != nullptr) {
		library_hook = [&](const vk::GraphicsPipelineCreateInfo& info,
		                   std::span<const uint32_t> layout_signature, vk::Pipeline* pipeline) {
			library_result = m_library->library.Create(info, layout_signature, m_driver_cache);
			*pipeline      = library_result.pipeline;
			return library_result.result;
		};
	}
	if (!prefetched) {
		Profiler::ScopedFrameWait pipeline_create(Profiler::FrameWait::GraphicsPipelineCreate);
		CompileWatch::Build       watch_build(CompileWatch::Kind::GraphicsPipeline, watch_vs_hash, watch_ps_hash);
		CompileWatch::Wait        watch_wait("draw", CompileWatch::Kind::GraphicsPipeline, watch_vs_hash, watch_ps_hash);
		HangWatchdog::Scope       compile(
		    "graphics-pipeline",
		    vertex_info[0].stage.program ? vertex_info[0].stage.program->shader_hash : 0, vs_id,
		    ps_id);
		// Always in the log, and named by a crash report on this thread: AMD test 2 players' driver
		// crashes inside pipeline builds left no trace of the pipeline.
		Common::CrashContext::Scope note("graphics pipeline build (VS, PS)", vs_hash, ps_hash);
		LOGF("PipelineTrace: graphics build begin VS=0x%016" PRIx64 " PS=0x%016" PRIx64 "\n",
		     vs_hash, ps_hash);
		CreatePipelineInternal(m_graphics, *cached, key.rendering, key.vertex_input, vertex_info,
		                       ps_input_info, programs, key.static_params, m_driver_cache,
		                       library_hook ? &library_hook : nullptr);
		LOGF("PipelineTrace: graphics build done VS=0x%016" PRIx64 " PS=0x%016" PRIx64 "\n",
		     vs_hash, ps_hash);
	}
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GraphicsPipelinesCreated);
	GpuOpProfiler::RegisterGraphicsPipeline(cached->pipeline, key.vertex_shader_ids.data(),
	                                        static_cast<uint32_t>(key.vertex_shader_ids.size()),
	                                        ps_id);

	const auto created_pipeline = cached->pipeline;
	auto [iter, inserted]       = m_graphics_pipelines.emplace(key, std::move(cached));
	EXIT_IF(!inserted);
	if (prefetched && ready_from != nullptr) ready_from->Complete(key); // null: a driver-cache lookup's
	if (prefetched && ready.content.key != 0) content = std::move(ready.content); // journaled below
	const bool built_fast = fast_result.snapshot != nullptr; // EnqueueGraphics takes the snapshot
	if (built_fast) {
		m_fast_first->EnqueueGraphics(&iter->first, created_pipeline, fast_result, vs_hash,
		                              ps_hash);
	}
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
	auto&       totals = g_compile_totals;
	if (!prefetched) {
		totals.gfx_pipelines.fetch_add(1, std::memory_order_relaxed);
		totals.gfx_pipeline_ns.fetch_add(create_ns, std::memory_order_relaxed);
	}
	auto* made = prefetched || library_result.path == GraphicsPipelineLibrary::Path::Linked
	                 ? &totals.gfx_other
	             : fast_result.cache_hit || library_result.path == GraphicsPipelineLibrary::Path::CacheHit
	                 ? &totals.gfx_hit
	             : built_fast ? &totals.gfx_fast
	                          : &totals.gfx_full;
	made->fetch_add(1, std::memory_order_relaxed);
	totals.gfx_cp_ns.fetch_add(create_ns, std::memory_order_relaxed);
	auto* origin_total = origin == HangTrace::PipelineOrigin::New           ? &totals.gfx_new
	                     : origin == HangTrace::PipelineOrigin::Permutation ? &totals.gfx_permutation
	                                                                         : &totals.gfx_variant;
	origin_total->fetch_add(1, std::memory_order_relaxed);
	if (HangTrace::Enabled() && !prefetched) {
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
	// A fast build left the driver cache untouched; its optimized build notes the creation. A probe
	// hit added nothing to it.
	if (!prefetched && !built_fast && !fast_result.cache_hit) NotePipelineCreated(create_ns);
	if (content.key != 0 && (m_journal != nullptr || m_journal_replay != nullptr)) {
		NoteJournalPipeline(content.key, content.words, content.snapshot_words, content.identities, guest);
	} else if (m_journal != nullptr) {
		// A pipeline the journal cannot hold (no content key): counted by kind, logged at powers of two.
		static std::atomic<uint64_t> unkeyed[4] {};
		const auto kind = topology == vk::PrimitiveTopology::ePatchList ? (vertex_info.size() > 1 ? 1 : 2)
		                  : vertex_info.size() > 1                       ? 3
		                                                                 : 0;
		const auto n    = unkeyed[kind].fetch_add(1, std::memory_order_relaxed) + 1;
		if ((n & (n - 1)) == 0) {
			PipelineCacheLog("Pipeline journal: {} {} pipelines created without a journal key so far (VS 0x{:016x})", n,
			                 kind == 1 ? "tessellation" : kind == 2 ? "rect-list" : kind == 3 ? "multi-stage" : "other",
			                 vs_hash);
		}
	}
	return remember(*iter->second);
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	// KYTY_PIPELINE_COMPUTE_AHEAD: wait for a build already under way (outside m_mutex, which the
	// build takes to publish) instead of building the pipeline a second time here.
	uint64_t ahead_wait_ns = 0;
	if (m_compute_ahead && m_compute_prewarm != nullptr) {
		bool present = false;
		{
			Common::LockGuard lock(m_mutex);
			present = m_compute_pipelines.contains(compute_program.id);
		}
		const auto begin = CompileClockNs();
		if (!present && m_compute_prewarm->WaitFor(compute_program.id)) {
			ahead_wait_ns = CompileClockNs() - begin;
			m_compute_prewarm->waits.fetch_add(1, std::memory_order_relaxed);
			m_compute_prewarm->wait_ns.fetch_add(ahead_wait_ns, std::memory_order_relaxed);
		}
	}

	const auto        lock_begin = PaceLog::StallTraceEnabled() ? CompileClockNs() : 0;
	Common::LockGuard lock(m_mutex);
	if (lock_begin != 0) {
		PaceLog::NoteStall(lock_begin, CompileClockNs(), PaceLog::StallPipelineMutex, 0, 1, "compute pipeline lock");
	}

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		if (iter->second->prewarmed) {
			// Built by the start-up prewarm (KYTY_PIPELINE_PREWARM_COMPUTE) before any dispatch asked.
			iter->second->prewarmed = false;
			g_compile_totals.cs_prewarmed.fetch_add(1, std::memory_order_relaxed);
		}
		if (ahead_wait_ns != 0) {
			g_compile_totals.cs_cp_ns.fetch_add(ahead_wait_ns, std::memory_order_relaxed);
			AddCompileStall(ahead_wait_ns);
		}
		FlushCompileStall();
		return *iter->second;
	}
	const auto create_begin = CompileClockNs();

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	HangWatchdog::Scope compile(
	    "compute-pipeline", input_info.stage.program ? input_info.stage.program->shader_hash : 0,
	    compute_program.id);
	const uint64_t      watch_hash = input_info.stage.program ? input_info.stage.program->shader_hash : 0;
	CompileWatch::Build watch_build(CompileWatch::Kind::ComputePipeline, watch_hash);
	CompileWatch::Wait  watch_wait("dispatch", CompileWatch::Kind::ComputePipeline, watch_hash);
	const uint64_t cs_hash = input_info.stage.program ? input_info.stage.program->shader_hash : 0;
	Common::CrashContext::Scope note("compute pipeline build (CS)", cs_hash);
	FastFirstState::ComputeFast fast_result;
	ComputePipelineCreateHook   fast_hook;
	if (m_fast_first != nullptr && m_fast_first->scheduler.Stopped() == false) {
		fast_hook = [&](const vk::ComputePipelineCreateInfo& info, vk::Pipeline* pipeline) {
			return m_fast_first->CreateCompute(info, m_known != nullptr ? ComputePipelineKey(info) : 0,
			                                   pipeline, fast_result);
		};
	}
	// KYTY_SKIP_INVALID (usa-ratt/KytyPS5-GT7 ef6f66053, 7ecb6aec3): a compute pipeline the driver
	// cannot build is kept as failed (build_failed) and its dispatches are skipped instead of
	// ending the emulator (GT7 after several races). The log names the shader, the driver's answer
	// and the device memory in use.
	const bool tolerate_failure = Common::SkipInvalidEnabled();
	vk::Result build_result     = vk::Result::eSuccess;
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache,
	                       fast_hook ? &fast_hook : nullptr,
	                       tolerate_failure ? &build_result : nullptr);
	if (tolerate_failure && (build_result != vk::Result::eSuccess || cached->pipeline == nullptr)) {
		static std::atomic<uint32_t> reported {0};
		const auto                   count = reported.fetch_add(1, std::memory_order_relaxed) + 1u;
		if (count <= 64u || std::has_single_bit(count)) {
			const auto text = fmt::format(
			    "Compute pipeline CS=0x{:016x}: the graphics driver could not build it "
			    "(vkCreateComputePipelines returned {}; device memory in use: {} MB). Its dispatches "
			    "are skipped from now on (KYTY_SKIP_INVALID; {} so far)\n",
			    cs_hash, vk::to_string(build_result),
			    m_graphics.CanReportMemoryUsage() ? m_graphics.GetDeviceMemoryUsage() >> 20u : 0,
			    count);
			LOGF("%s", text.c_str());
			std::fprintf(stderr, "%s", text.c_str());
		}
		cached->pipeline        = nullptr;
		cached->pipeline_layout = nullptr;
		cached->build_failed    = true;
		auto [failed, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
		EXIT_IF(!inserted);
		FlushCompileStall();
		return *failed->second;
	}
	GpuOpProfiler::RegisterComputePipeline(cached->pipeline, compute_program.id);
	const auto created_pipeline = cached->pipeline;

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	const bool built_fast = fast_result.snapshot != nullptr; // EnqueueCompute takes the snapshot
	if (built_fast) {
		m_fast_first->EnqueueCompute(compute_program.id, created_pipeline, fast_result, cs_hash);
	}

	const auto create_ns = CompileClockNs() - create_begin;
	g_compile_totals.cs_pipelines.fetch_add(1, std::memory_order_relaxed);
	g_compile_totals.cs_pipeline_ns.fetch_add(create_ns, std::memory_order_relaxed);
	(fast_result.cache_hit ? g_compile_totals.cs_hit
	 : built_fast          ? g_compile_totals.cs_fast
	                       : g_compile_totals.cs_full)
	    .fetch_add(1, std::memory_order_relaxed);
	g_compile_totals.cs_cp_ns.fetch_add(create_ns, std::memory_order_relaxed);
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
	if (!built_fast && !fast_result.cache_hit) NotePipelineCreated(create_ns);
	return *iter->second;
}

PipelineCache::RtState& PipelineCache::Rt() {
	std::call_once(m_rt_once, [this] { m_rt = std::make_unique<RtState>(*this); });
	return *m_rt;
}

void PipelineCache::WaitRtKernelsAtFlip() {
	if (m_rt == nullptr || !m_rt->any_flip_pending.load(std::memory_order_acquire)) return;
	m_rt->WaitAtFlip();
}

PipelineCache::RtPipelineTotals PipelineCache::GetRtPipelineTotals() const {
	return m_rt != nullptr ? m_rt->Totals() : RtPipelineTotals {};
}

std::vector<PipelineCache::RtShaderPipelines> PipelineCache::GetRtPipelineStatus() const {
	return m_rt != nullptr ? m_rt->Status() : std::vector<RtShaderPipelines> {};
}

PipelineCache::CallTranslationTotals PipelineCache::GetCallTranslationTotals() {
	const auto get = [](const std::atomic<uint64_t>& value) { return value.load(std::memory_order_relaxed); };
	return {get(g_call_translation_totals.queued), get(g_call_translation_totals.finished),
	        get(g_call_translation_totals.words), get(g_call_translation_totals.ms)};
}

void PipelineCache::PrewarmComputePipeline(const ShaderComputeInputInfo& input_info,
                                           const ShaderProgram&          compute_program) {
	if (!compute_program || input_info.stage.program == nullptr) return;
	if (!input_info.stage.program->info.uses_bvh) {
		if (m_compute_prewarm != nullptr) m_compute_prewarm->Request(compute_program.id, input_info, compute_program.module);
		return;
	}
	if (RtSession::Failed()) return;
	{
		Common::LockGuard lock(m_mutex);
		if (m_compute_pipelines.contains(compute_program.id)) return;
	}
	auto&            rt = Rt();
	std::scoped_lock lock(rt.mutex);
	if (!rt.stopped.load(std::memory_order_acquire)) {
		(void)rt.Request(compute_program.id, input_info, compute_program);
	}
}

PipelineCache::Pipeline* PipelineCache::TryGetComputePipeline(const ShaderComputeInputInfo& input_info,
                                                              const ShaderProgram& program_in) {
	EXIT_IF(!program_in || !input_info.stage);
	// KYTY_SPIRV_VARIANT_LIVE: the group's edited copy (same bindings and layout) when it has one.
	ShaderProgram compute_program = program_in;
	if (const auto group = SpirvVariantLiveGroup(); group != 0 && SpirvVariantsConfigured()) [[unlikely]] {
		if (const auto variant = SpirvVariantProgram(program_in, group)) {
			compute_program = variant;
		}
	}
	// A call-split program (CallSplit.h) builds in the background too, and never waits.
	const bool call_split =
	    input_info.stage.program->bindings.call_state_dword != ShaderRecompiler::IR::PushData::NoStart;
	const bool rt_kernel = input_info.stage.program->info.uses_bvh;
	// KYTY_COMPUTE_BACKGROUND_WORDS=<n> (default 0 = off; usa-ratt/KytyPS5-GT7 767c1e25e,
	// 518b92220, 5a41b9f8a, 1000000 there): a compute program of n SPIR-V words or more whose
	// pipeline is not built yet goes the RT kernels' way: it builds on the RtState threads
	// (KYTY_RT_PIPELINE_THREADS), its first dispatch waits up to KYTY_RT_PIPELINE_WAIT_MS for it
	// (a driver-cache hit is done by then), and later dispatches are skipped until it is ready, so
	// a cold compile of a huge program does not freeze the command processor for seconds.
	static const uint64_t huge_words = EnvU64("KYTY_COMPUTE_BACKGROUND_WORDS", 0);
	bool huge = false;
	if (!rt_kernel && !call_split && huge_words != 0 &&
	    g_shader_modules.WordsOf(compute_program.module) >= huge_words) {
		Common::LockGuard lock(m_mutex);
		if (const auto found = m_compute_pipelines.find(compute_program.id);
		    found != m_compute_pipelines.end()) {
			return found->second->build_failed ? nullptr : found->second.get();
		}
		huge = true;
	}
	if (!rt_kernel && !call_split && !huge) {
		auto& built = GetComputePipeline(input_info, compute_program);
		return built.build_failed ? nullptr : &built; // KYTY_SKIP_INVALID: the dispatch is skipped
	}
	const auto id = compute_program.id;
	auto&      rt = Rt();
	if (!huge && RtSession::Failed()) {
		rt.totals.skipped_dispatches.fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	{
		Pipeline* published = nullptr;
		{
			Common::LockGuard lock(m_mutex);
			if (const auto found = m_compute_pipelines.find(id); found != m_compute_pipelines.end()) {
				published = found->second.get();
			}
		}
		if (published != nullptr) {
			// rt.mutex is taken before m_mutex elsewhere (Finish): never under it.
			if (RtState::RtPipelineLogEnabled()) rt.NoteRan(id);
			return published;
		}
	}
	std::shared_future<std::shared_ptr<RtState::Built>> future;
	uint64_t                                            ticket = 0;
	bool                                                first  = false;
	uint64_t                                            shader_hash = input_info.stage.program->shader_hash;
	const auto kernel_class = huge ? RtSession::KernelClass::Huge
	                          : call_split ? RtSession::KernelClass::CallSplit
	                                       : RtSession::KernelClass::Direct;
	const auto first_budget_ms = RtSession::FirstWaitBudgetMs(kernel_class, rt.kernel_wait_ms, rt.wait_ms);
	{
		std::scoped_lock lock(rt.mutex);
		auto*            job = rt.Request(id, input_info, compute_program, huge, !huge);
		if (job == nullptr) {
			if (huge) {
				// Queue full (or the module is going): built here, as without the switch.
				auto& built = GetComputePipeline(input_info, compute_program);
				return built.build_failed ? nullptr : &built;
			}
			rt.totals.skipped_dispatches.fetch_add(1, std::memory_order_relaxed);
			return nullptr;
		}
		if (job->consumed) {
			// Built and published or failed earlier: the map lookup above missed, so it failed.
			rt.totals.skipped_dispatches.fetch_add(1, std::memory_order_relaxed);
			return nullptr;
		}
		first       = !job->asked;
		job->asked  = true;
		if (first) job->asked_ns = CompileClockNs();
		// A prewarmed build a dispatch now waits for: ahead of the prewarm builds queued before it
		// (the shader precompile queues hundreds of RT programs of other areas).
		if (!huge && !job->promoted) {
			job->promoted = true;
			rt.queue.Promote(job->ticket);
		}
		future      = job->future;
		ticket      = job->ticket;
	}
	bool ready = future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
	if (huge && first) {
		PipelineCacheLog("Compute pipeline of shader 0x{:016x} ({} SPIR-V words) builds in the background "
		                 "(KYTY_COMPUTE_BACKGROUND_WORDS); its dispatches are skipped until it is ready",
		                 shader_hash, g_shader_modules.WordsOf(compute_program.module));
	}
	// KYTY_RT_KERNEL_WAIT_AT=flip: a direct kernel's dispatches are skipped until the frame's flip,
	// which waits for it (RtState::WaitAtFlip); with no flip for flip_fallback_ns a dispatch waits.
	bool     wait_here = first;
	uint32_t budget_ms = first_budget_ms;
	if (!ready && kernel_class == RtSession::KernelClass::Direct && rt.wait_at_flip) {
		std::scoped_lock lock(rt.mutex);
		if (auto found = rt.jobs.find(id); found != rt.jobs.end()) {
			auto&      job = found->second;
			const auto now = CompileClockNs();
			if (first) {
				job.flip_wait = true;
				rt.flip_pending.push_back(id);
				rt.any_flip_pending.store(true, std::memory_order_release);
				wait_here = false;
			} else if (job.flip_wait) {
				const auto age = now - job.asked_ns;
				wait_here      = age >= rt.flip_fallback_ns;
				if (wait_here) {
					const auto budget_ns = static_cast<uint64_t>(rt.kernel_wait_ms) * 1'000'000ull;
					budget_ms = age >= budget_ns ? 0 : static_cast<uint32_t>((budget_ns - age) / 1'000'000ull);
					job.flip_wait = false; // waited for here, not at the flip
				}
			}
		}
	}
	uint64_t waited_ns = 0;
	switch (RtSession::DecideDispatch(false, ready, wait_here, budget_ms)) {
		case RtSession::DispatchAction::Run: break;
		case RtSession::DispatchAction::WaitBounded: {
			rt.queue.Promote(ticket);
			if (budget_ms > 1000) {
				PipelineCacheLog("{}: the command processor waits for the pipeline of compute shader 0x{:016x} "
				                 "(at most {}; a cold pipeline cache builds it now)",
				                 huge ? "Huge compute" : "Software RT", shader_hash, RtState::BudgetText(budget_ms));
			}
			const auto         begin = CompileClockNs();
			CompileWatch::Wait watch("dispatch", huge ? CompileWatch::Kind::HugeCompute : CompileWatch::Kind::RtPipeline,
			                         shader_hash);
			if (budget_ms == UINT32_MAX) {
				future.wait();
			} else {
				(void)future.wait_for(std::chrono::milliseconds(budget_ms));
			}
			waited_ns = CompileClockNs() - begin;
			rt.totals.waited.fetch_add(1, std::memory_order_relaxed);
			rt.totals.wait_ns.fetch_add(waited_ns, std::memory_order_relaxed);
			AddCompileStall(waited_ns);
			FlushCompileStall();
			ready = future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
			break;
		}
		case RtSession::DispatchAction::Skip: break;
	}
	if (!ready) {
		rt.totals.skipped_dispatches.fetch_add(1, std::memory_order_relaxed);
		std::scoped_lock lock(rt.mutex);
		if (const auto job = rt.jobs.find(id); job != rt.jobs.end()) {
			job->second.skipped++;
			job->second.first_wait_ns += waited_ns;
			if (!job->second.logged_skip) {
				job->second.logged_skip = true;
				PipelineCacheLog("{}: the pipeline of compute shader 0x{:016x} is still being built; "
				                 "its dispatches are skipped until it is ready{}",
				                 huge ? "Huge compute" : "Software RT", shader_hash,
				                 job->second.flip_wait ? " (this frame's flip waits for it)"
				                 : huge                ? ""
				                                       : " (the lighting it feeds lags)");
			}
		}
		return nullptr;
	}
	std::scoped_lock lock(rt.mutex);
	auto             job = rt.jobs.find(id);
	if (job == rt.jobs.end()) return nullptr;
	job->second.first_wait_ns += waited_ns;
	if (RtState::RtPipelineLogEnabled()) rt.NoteRanLocked(job->second);
	if (job->second.consumed) {
		Common::LockGuard map_lock(m_mutex);
		const auto        found = m_compute_pipelines.find(id);
		return found != m_compute_pipelines.end() ? found->second.get() : nullptr;
	}
	auto built = future.get();
	return rt.Finish(id, job->second, *built);
}

std::string PipelineCache::TakeEmptyProgramReason(uint64_t* hash) {
	return TakeEmptyProgramReasonImpl(hash);
}

} // namespace Libs::Graphics
