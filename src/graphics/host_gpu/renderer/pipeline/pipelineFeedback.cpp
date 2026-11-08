#include "graphics/host_gpu/renderer/pipeline/pipelineFeedback.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
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

vk::Result CreateGraphics(vk::Device device, vk::PipelineCache cache, const vk::GraphicsPipelineCreateInfo& info,
                          vk::Pipeline* pipeline, Path path) {
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
