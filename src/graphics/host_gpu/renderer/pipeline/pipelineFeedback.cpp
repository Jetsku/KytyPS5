#include "graphics/host_gpu/renderer/pipeline/pipelineFeedback.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::PipelineFeedback {

namespace {

struct Counters {
	std::atomic<uint64_t> hits {0};
	std::atomic<uint64_t> compiles {0};
	std::atomic<uint64_t> unreported {0}; // the driver left the feedback invalid
	std::atomic<uint64_t> failed {0};     // not created (a probe miss, an error)
	std::atomic<uint64_t> adopted {0};
	std::atomic<uint64_t> hit_ns {0};
	std::atomic<uint64_t> compile_ns {0};
	std::atomic<uint64_t> failed_ns {0};
	std::atomic<uint64_t> hit_max_ns {0};
	std::atomic<uint64_t> compile_max_ns {0};
};

std::array<Counters, static_cast<size_t>(Path::Count)> g_counters;

constexpr std::array<const char*, static_cast<size_t>(Path::Count)> PathNames = {"sync", "builder", "probe",
                                                                                 "replay"};

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void Max(std::atomic<uint64_t>& target, uint64_t value) {
	auto current = target.load(std::memory_order_relaxed);
	while (value > current && !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
	}
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PIPELINE_FEEDBACK");
		return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

vk::Result CreateGraphics(vk::Device device, vk::PipelineCache cache, const vk::GraphicsPipelineCreateInfo& original,
                          vk::Pipeline* pipeline, Path path) {
	// KYTY_EMPTY_FS: the driver's copy only (the caller keyed the original).
	EmptyFragment::Storage storage;
	const auto&            plain_info = EmptyFragment::Apply(original, storage);
	// KYTY_PIPELINE_STATS: the driver's copy asks for statistics.
	vk::GraphicsPipelineCreateInfo stats_info;
	const bool                     stats = PipelineStats::Enabled() && PipelineStats::Selected(plain_info);
	if (stats) {
		stats_info = plain_info;
		stats_info.flags |= vk::PipelineCreateFlagBits::eCaptureStatisticsKHR;
	}
	const auto& info = stats ? stats_info : plain_info;
	if (!Enabled()) {
		const auto made = device.createGraphicsPipelines(cache, 1, &info, nullptr, pipeline);
		if (stats && made == vk::Result::eSuccess && *pipeline != nullptr) {
			PipelineStats::Print(device, *pipeline, PipelineStats::Describe(info.pStages, info.stageCount), info.flags);
		}
		return made;
	}
	vk::PipelineCreationFeedback           whole {};
	vk::PipelineCreationFeedbackCreateInfo feedback {};
	feedback.pPipelineCreationFeedback          = &whole;
	feedback.pipelineStageCreationFeedbackCount = 0;
	feedback.pNext                              = info.pNext;
	auto copy                                   = info;
	copy.pNext                                  = &feedback;
	const auto begin                            = NowNs();
	const auto result                           = device.createGraphicsPipelines(cache, 1, &copy, nullptr, pipeline);
	const auto ns                               = NowNs() - begin;
	auto&      c                                = g_counters[static_cast<size_t>(path)];
	if (result != vk::Result::eSuccess || *pipeline == nullptr) {
		c.failed.fetch_add(1, std::memory_order_relaxed);
		c.failed_ns.fetch_add(ns, std::memory_order_relaxed);
	} else if (!(whole.flags & vk::PipelineCreationFeedbackFlagBits::eValid)) {
		c.unreported.fetch_add(1, std::memory_order_relaxed);
		c.compile_ns.fetch_add(ns, std::memory_order_relaxed);
	} else if (whole.flags & vk::PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit) {
		c.hits.fetch_add(1, std::memory_order_relaxed);
		c.hit_ns.fetch_add(ns, std::memory_order_relaxed);
		Max(c.hit_max_ns, ns);
	} else {
		c.compiles.fetch_add(1, std::memory_order_relaxed);
		c.compile_ns.fetch_add(ns, std::memory_order_relaxed);
		Max(c.compile_max_ns, ns);
	}
	if (stats && result == vk::Result::eSuccess && *pipeline != nullptr) {
		PipelineStats::Print(device, *pipeline, PipelineStats::Describe(info.pStages, info.stageCount), info.flags);
	}
	return result;
}

void NoteAdopted(Path path) {
	g_counters[static_cast<size_t>(path)].adopted.fetch_add(1, std::memory_order_relaxed);
}

std::string Report() {
	std::string out;
	for (size_t i = 0; i < g_counters.size(); i++) {
		const auto& c        = g_counters[i];
		const auto  hits     = c.hits.load(std::memory_order_relaxed);
		const auto  compiles = c.compiles.load(std::memory_order_relaxed);
		const auto  other    = c.unreported.load(std::memory_order_relaxed);
		const auto  failed   = c.failed.load(std::memory_order_relaxed);
		const auto  adopted  = c.adopted.load(std::memory_order_relaxed);
		if (hits + compiles + other + failed + adopted == 0) continue;
		const auto avg = [](uint64_t ns, uint64_t n) { return n != 0 ? static_cast<double>(ns) / 1e6 / n : 0.0; };
		out += fmt::format("{}{} {} driver-cache hits ({:.2f} ms avg, {:.1f} max), {} compiled ({:.1f} ms avg, "
		                   "{:.0f} max), {} unreported, {} failed/missed ({:.2f} ms avg), {} adopted",
		                   out.empty() ? "" : " | ", PathNames[i], hits, avg(c.hit_ns.load(), hits),
		                   static_cast<double>(c.hit_max_ns.load()) / 1e6, compiles,
		                   avg(c.compile_ns.load(), compiles + other),
		                   static_cast<double>(c.compile_max_ns.load()) / 1e6, other, failed,
		                   avg(c.failed_ns.load(), failed), adopted);
	}
	return out.empty() ? out : "Pipeline feedback: " + out;
}

} // namespace Libs::Graphics::PipelineFeedback

namespace Libs::Graphics::PipelineStats {

namespace {

struct Selection {
	bool                  all = false;
	std::vector<uint64_t> hashes;
};

const Selection& Settings() {
	static const Selection selection = [] {
		Selection parsed;
		const char* value = std::getenv("KYTY_PIPELINE_STATS");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) return parsed;
		const std::string text(value);
		if (text == "all" || text == "1") {
			parsed.all = true;
			return parsed;
		}
		size_t at = 0;
		while (at < text.size()) {
			const auto next = text.find(',', at);
			const auto item = text.substr(at, next == std::string::npos ? std::string::npos : next - at);
			if (!item.empty()) parsed.hashes.push_back(std::strtoull(item.c_str(), nullptr, 16));
			if (next == std::string::npos) break;
			at = next + 1;
		}
		return parsed;
	}();
	return selection;
}

struct ModuleName {
	const char* stage = "";
	uint64_t    hash  = 0;
};

std::mutex                                        g_mutex;
std::unordered_map<VkShaderModule, ModuleName>    g_modules;
std::unordered_set<VkPipeline>                    g_printed;
std::atomic<bool>                                 g_enabled {false};

bool HashSelected(uint64_t hash) {
	const auto& s = Settings();
	return s.all || std::ranges::find(s.hashes, hash) != s.hashes.end() ||
	       std::ranges::find(s.hashes, hash & 0xffffffffull) != s.hashes.end() ||
	       std::ranges::find(s.hashes, hash >> 32u) != s.hashes.end();
}

} // namespace

bool Requested() {
	const auto& s = Settings();
	return s.all || !s.hashes.empty();
}

void SetEnabled(bool on) {
	g_enabled.store(on && Requested(), std::memory_order_relaxed);
	if (on && Requested()) {
		std::printf("Kyty pipeline stats: on (KYTY_PIPELINE_STATS)\n");
	}
}

bool Enabled() {
	return g_enabled.load(std::memory_order_relaxed);
}

void RegisterModule(vk::ShaderModule module, const char* stage, uint64_t hash) {
	if (!Enabled() || module == nullptr) return;
	std::scoped_lock lock(g_mutex);
	g_modules[static_cast<VkShaderModule>(module)] = {stage, hash};
}

bool Selected(vk::ShaderModule module) {
	if (!Enabled() || module == nullptr) return false;
	std::scoped_lock lock(g_mutex);
	const auto it = g_modules.find(static_cast<VkShaderModule>(module));
	return it != g_modules.end() && HashSelected(it->second.hash);
}

bool Selected(const vk::GraphicsPipelineCreateInfo& info) {
	if (!Enabled() || info.pStages == nullptr) return false;
	for (uint32_t i = 0; i < info.stageCount; i++) {
		if (Selected(info.pStages[i].module)) return true;
	}
	return false;
}

std::string Describe(const vk::PipelineShaderStageCreateInfo* stages, uint32_t count) {
	std::string out;
	std::scoped_lock lock(g_mutex);
	for (uint32_t i = 0; i < count && stages != nullptr; i++) {
		const auto it = g_modules.find(static_cast<VkShaderModule>(stages[i].module));
		if (it == g_modules.end()) continue;
		out += fmt::format("{}{}:0x{:016x}", out.empty() ? "" : "|", it->second.stage, it->second.hash);
	}
	return out;
}

void Print(vk::Device device, vk::Pipeline pipeline, const std::string& what, vk::PipelineCreateFlags flags) {
	{
		std::scoped_lock lock(g_mutex);
		if (!g_printed.insert(static_cast<VkPipeline>(pipeline)).second || g_printed.size() > 4096) return;
	}
	vk::PipelineInfoKHR pipeline_info {};
	pipeline_info.pipeline = pipeline;
	uint32_t count         = 0;
	if (device.getPipelineExecutablePropertiesKHR(&pipeline_info, &count, nullptr) != vk::Result::eSuccess) return;
	std::vector<vk::PipelineExecutablePropertiesKHR> executables(count);
	if (device.getPipelineExecutablePropertiesKHR(&pipeline_info, &count, executables.data()) != vk::Result::eSuccess) {
		return;
	}
	std::string line = fmt::format("Pipeline stats: {}{}:", what,
	                               (flags & vk::PipelineCreateFlagBits::eDisableOptimization) ? " (unoptimized)" : "");
	for (uint32_t e = 0; e < count; e++) {
		vk::PipelineExecutableInfoKHR exec {};
		exec.pipeline        = pipeline;
		exec.executableIndex = e;
		uint32_t n           = 0;
		if (device.getPipelineExecutableStatisticsKHR(&exec, &n, nullptr) != vk::Result::eSuccess) continue;
		std::vector<vk::PipelineExecutableStatisticKHR> stats(n);
		if (device.getPipelineExecutableStatisticsKHR(&exec, &n, stats.data()) != vk::Result::eSuccess) continue;
		line += fmt::format(" [{} sg{}", executables[e].name.data(), executables[e].subgroupSize);
		for (uint32_t s = 0; s < n; s++) {
			const auto& st = stats[s];
			switch (st.format) {
				case vk::PipelineExecutableStatisticFormatKHR::eBool32:
					line += fmt::format(" {}={}", st.name.data(), st.value.b32 ? 1 : 0);
					break;
				case vk::PipelineExecutableStatisticFormatKHR::eInt64:
					line += fmt::format(" {}={}", st.name.data(), st.value.i64);
					break;
				case vk::PipelineExecutableStatisticFormatKHR::eUint64:
					line += fmt::format(" {}={}", st.name.data(), st.value.u64);
					break;
				case vk::PipelineExecutableStatisticFormatKHR::eFloat64:
					line += fmt::format(" {}={:.3f}", st.name.data(), st.value.f64);
					break;
			}
		}
		line += "]";
	}
	std::printf("%s\n", line.c_str());
}

} // namespace Libs::Graphics::PipelineStats

namespace Libs::Graphics::EmptyFragment {

namespace {

// main() { } with OriginUpperLeft and EarlyFragmentTests, no inputs or outputs (SPIR-V 1.0;
// chenxiao07/KytyPS5 5a7fa9a1b).
constexpr uint32_t kEmptyFragmentSpirv[] = {
    0x07230203, 0x00010000, 0x00070000, 0x00000005, 0x00000000, 0x00020011, 0x00000001, 0x0003000e,
    0x00000000, 0x00000001, 0x0005000f, 0x00000004, 0x00000001, 0x6e69616d, 0x00000000, 0x00030010,
    0x00000001, 0x00000007, 0x00030010, 0x00000001, 0x00000009, 0x00020013, 0x00000002, 0x00030021,
    0x00000003, 0x00000002, 0x00050036, 0x00000002, 0x00000001, 0x00000000, 0x00000003, 0x000200f8,
    0x00000004, 0x000100fd, 0x00010038};

constexpr uint32_t kVendorNvidia = 0x10deu;

vk::ShaderModule                  g_module = nullptr;
vk::PipelineShaderStageCreateInfo g_stage {};
std::atomic<uint64_t>             g_applied {0};

} // namespace

void Configure(vk::Device device, uint32_t vendor_id) {
	const char* value = std::getenv("KYTY_EMPTY_FS");
	const int   mode  = value != nullptr ? std::atoi(value) : 0;
	if (g_module != nullptr || mode <= 0 || (mode == 1 && vendor_id != kVendorNvidia)) {
		if (mode == 1 && vendor_id != kVendorNvidia) {
			std::printf("Kyty empty fragment shader: off (KYTY_EMPTY_FS=1 applies to NVIDIA only; vendor 0x%x)\n",
			            vendor_id);
		}
		return;
	}
	vk::ShaderModuleCreateInfo create {};
	create.codeSize = sizeof(kEmptyFragmentSpirv);
	create.pCode    = kEmptyFragmentSpirv;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&create, nullptr, &module) != vk::Result::eSuccess || module == nullptr) {
		std::printf("Kyty empty fragment shader: module creation failed, off (KYTY_EMPTY_FS)\n");
		return;
	}
	g_module       = module;
	g_stage        = vk::PipelineShaderStageCreateInfo {};
	g_stage.stage  = vk::ShaderStageFlagBits::eFragment;
	g_stage.module = module;
	g_stage.pName  = "main";
	std::printf("Kyty empty fragment shader: on for pipelines without a pixel shader (KYTY_EMPTY_FS=%d, vendor 0x%x)\n",
	            mode, vendor_id);
}

vk::ShaderModule Module() noexcept {
	return g_module;
}

bool Eligible(bool library, bool linked, uint32_t stage_count, bool has_fragment, bool rasterizer_discard) noexcept {
	return !library && !linked && stage_count != 0 && stage_count < Storage {}.stages.size() && !has_fragment &&
	       !rasterizer_discard;
}

const vk::GraphicsPipelineCreateInfo& Apply(const vk::GraphicsPipelineCreateInfo& info, Storage& storage) noexcept {
	if (g_module == nullptr) {
		return info;
	}
	bool linked = false;
	for (const auto* next = static_cast<const vk::BaseInStructure*>(info.pNext); next != nullptr; next = next->pNext) {
		if (next->sType == vk::StructureType::ePipelineLibraryCreateInfoKHR ||
		    next->sType == vk::StructureType::eGraphicsPipelineLibraryCreateInfoEXT) {
			linked = true;
		}
	}
	bool has_fragment = false;
	for (uint32_t i = 0; i < info.stageCount && info.pStages != nullptr; i++) {
		has_fragment |= info.pStages[i].stage == vk::ShaderStageFlagBits::eFragment;
	}
	const bool discard = info.pRasterizationState != nullptr && info.pRasterizationState->rasterizerDiscardEnable;
	if (!Eligible(static_cast<bool>(info.flags & vk::PipelineCreateFlagBits::eLibraryKHR), linked, info.stageCount,
	              has_fragment, discard) ||
	    info.pStages == nullptr) {
		return info;
	}
	for (uint32_t i = 0; i < info.stageCount; i++) {
		storage.stages[i] = info.pStages[i];
	}
	storage.stages[info.stageCount] = g_stage;
	storage.info                    = info;
	storage.info.stageCount         = info.stageCount + 1;
	storage.info.pStages            = storage.stages.data();
	g_applied.fetch_add(1, std::memory_order_relaxed);
	return storage.info;
}

const vk::PipelineShaderStageCreateInfo* LibraryStage() noexcept {
	return g_module != nullptr ? &g_stage : nullptr;
}

uint64_t AppliedCount() noexcept {
	return g_applied.load(std::memory_order_relaxed);
}

void SetModuleForTest(vk::ShaderModule module) noexcept {
	g_module       = module;
	g_stage        = vk::PipelineShaderStageCreateInfo {};
	g_stage.stage  = vk::ShaderStageFlagBits::eFragment;
	g_stage.module = module;
	g_stage.pName  = "main";
}

} // namespace Libs::Graphics::EmptyFragment
