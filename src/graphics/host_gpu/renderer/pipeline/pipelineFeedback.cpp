#include "graphics/host_gpu/renderer/pipeline/pipelineFeedback.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>

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
	const auto&            info = EmptyFragment::Apply(original, storage);
	if (!Enabled()) {
		return device.createGraphicsPipelines(cache, 1, &info, nullptr, pipeline);
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
