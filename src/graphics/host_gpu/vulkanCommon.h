#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANCOMMON_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANCOMMON_H_

#define VK_NO_PROTOTYPES
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC    1
#define VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL 1
#define VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_EXCEPTIONS

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/gpu_defs.h"

#include <cstdint>
#include <fmt/format.h>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_hash.hpp>

namespace Libs::Graphics {

vk::Format  VulkanFormat(Prospero::BufferFormat guest_format);
void        RequireVulkanSuccess(vk::Result result, const char* operation);
vk::ShaderModule CompileSPV(std::span<const uint32_t> code, vk::Device device);
// specialization: optional specialization constants of the compute stage.
vk::Pipeline     CreateComputePipeline(vk::Device device, std::span<const uint32_t> code,
                                       vk::PipelineLayout layout, const char* operation,
                                       const vk::SpecializationInfo* specialization = nullptr);

// CompileSPV phase times (KYTY_SHADER_PRECOMPILE_STATS=1; all zero otherwise): the function-array
// shrink, the spirv-val run on a shrunk module (or its skip, a known-valid module) and the driver's
// vkCreateShaderModule.
struct CompileSpvTimes {
	uint64_t calls            = 0;
	uint64_t shrunk           = 0; // the rewrite changed the module
	uint64_t validated        = 0; // spirv-val runs
	uint64_t validate_skipped = 0; // known-valid shrunk modules
	uint64_t shrink_ns        = 0;
	uint64_t validate_ns      = 0;
	uint64_t create_ns        = 0;
};
[[nodiscard]] bool            CompileSpvStatsEnabled();
[[nodiscard]] CompileSpvTimes CompileSpvThreadTimes(); // the calling thread's totals
[[nodiscard]] CompileSpvTimes CompileSpvTotals();      // every thread's

// Shrunk modules that passed spirv-val are remembered across runs in this file (their XXH3-128
// content hashes; the header names the validator build), so a module reloaded from the program
// cache is not validated again. Loads the file, rewrites it with its intact records and appends
// new verdicts as they come. KYTY_SHRINK_VERDICTS=0 keeps it off. Returns the number loaded.
uint64_t UseShrinkVerdictFile(const std::string& path);

[[nodiscard]] constexpr bool IsHdrSwapchainFormat(vk::Format format) {
	switch (format) {
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32: return true;
		default: return false;
	}
}

// KYTY_ADDRESS_BINDING_REPORT (graphicContext.h): object names kept for the device-fault report.
// Only the first name of a handle is kept (per-draw range names would rename game buffers).
[[nodiscard]] bool AddressBindingReportActive();
[[nodiscard]] bool AddressBindingReportWantsName(uint64_t handle);
void               AddressBindingReportNoteName(uint64_t handle, std::string name);

template <typename Handle, typename... Args>
void SetVulkanObjectNameF(vk::Device device, Handle handle, fmt::format_string<Args...> format,
                          Args&&... args) {
	if constexpr (Handle::objectType == vk::ObjectType::eBuffer || Handle::objectType == vk::ObjectType::eImage) {
		if (AddressBindingReportActive() && handle != nullptr) {
			const auto key = static_cast<uint64_t>(
			    reinterpret_cast<uintptr_t>(static_cast<typename Handle::CType>(handle)));
			if (AddressBindingReportWantsName(key)) {
				AddressBindingReportNoteName(key, fmt::format(format, std::forward<Args>(args)...));
			}
		}
	}
	// Names are set for the debug dump and under --rd (RenderDoc shows them: images by guest address).
	if ((!Config::GraphicsDebugDumpEnabled() && !Config::RenderDocEnabled()) || device == nullptr || handle == nullptr ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkSetDebugUtilsObjectNameEXT == nullptr) {
		return;
	}

	const auto                      name = fmt::format(format, std::forward<Args>(args)...);
	vk::DebugUtilsObjectNameInfoEXT info {};
	info.objectType   = Handle::objectType;
	info.objectHandle = static_cast<uint64_t>(
	    reinterpret_cast<uintptr_t>(static_cast<typename Handle::CType>(handle)));
	info.pObjectName = name.c_str();
	(void)device.setDebugUtilsObjectNameEXT(&info);
}

template <typename T, typename Enumerator>
[[nodiscard]] std::vector<T> EnumerateVulkan(const char* operation, Enumerator&& enumerate) {
	for (;;) {
		uint32_t count = 0;
		RequireVulkanSuccess(enumerate(&count, nullptr), operation);
		if (count == 0) {
			return {};
		}

		std::vector<T> values(count);
		const auto     result = enumerate(&count, values.data());
		if (result == vk::Result::eSuccess) {
			values.resize(count);
			return values;
		}
		if (result != vk::Result::eIncomplete) {
			RequireVulkanSuccess(result, operation);
		}
	}
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANCOMMON_H_
