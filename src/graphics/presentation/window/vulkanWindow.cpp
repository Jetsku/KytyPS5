#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hangWatchdog.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/uploadDma.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineFastFirst.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/spirvLocalArrays.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/presentation/window/windowInternal.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "kernel/memory.h"
#include "libs/controller.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <fmt/format.h>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vk_platform.h>

// IWYU pragma: no_include <intrin.h>

#define KYTY_ENABLE_DEBUG_PRINTF

namespace Libs::Graphics {

struct VulkanExtensions {
	bool enable_validation_layers = false;

	std::vector<const char*>             required_extensions;
	std::vector<vk::ExtensionProperties> available_extensions;
	std::vector<const char*>             required_layers;
	std::vector<vk::LayerProperties>     available_layers;
};

vk::PhysicalDeviceVulkan12Features WindowContext::RequiredVulkan12Features() noexcept {
	vk::PhysicalDeviceVulkan12Features features {};
	features.samplerMirrorClampToEdge  = VK_TRUE;
	features.timelineSemaphore         = VK_TRUE;
	features.shaderOutputLayer         = VK_TRUE;
	features.shaderOutputViewportIndex = VK_TRUE;
	features.bufferDeviceAddress       = VK_TRUE;
	features.shaderBufferInt64Atomics  = VK_TRUE;
	return features;
}

vk::PhysicalDeviceVulkan13Features WindowContext::RequiredVulkan13Features() noexcept {
	vk::PhysicalDeviceVulkan13Features features {};
	features.dynamicRendering = VK_TRUE;
	features.synchronization2 = VK_TRUE;
	return features;
}

static bool HasExtension(const std::vector<vk::ExtensionProperties>& extensions, const char* name) {
	return std::any_of(extensions.begin(), extensions.end(),
	                   [name](const auto& ext) { return strcmp(ext.extensionName, name) == 0; });
}

static bool HasExtension(const std::vector<const char*>& extensions, const char* name) {
	return std::any_of(extensions.begin(), extensions.end(),
	                   [name](const char* ext) { return strcmp(ext, name) == 0; });
}

static bool HasLayer(const std::vector<vk::LayerProperties>& layers, const char* name) {
	return std::any_of(layers.begin(), layers.end(),
	                   [name](const auto& layer) { return strcmp(layer.layerName, name) == 0; });
}

static void GetSurfaceCapabilities(vk::PhysicalDevice physical_device, vk::SurfaceKHR surface,
                                   SurfaceCapabilities& r) {
	RequireVulkanSuccess(physical_device.getSurfaceCapabilitiesKHR(surface, &r.capabilities),
	                     "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

	r.formats = EnumerateVulkan<vk::SurfaceFormatKHR>( // @suppress("Ambiguous problem")
	    "vkGetPhysicalDeviceSurfaceFormatsKHR", [&](uint32_t* count, vk::SurfaceFormatKHR* values) {
		    return physical_device.getSurfaceFormatsKHR(surface, count, values);
	    });
	EXIT_NOT_IMPLEMENTED(r.formats.empty());

	r.present_modes = EnumerateVulkan<vk::PresentModeKHR>( // @suppress("Ambiguous problem")
	    "vkGetPhysicalDeviceSurfacePresentModesKHR",
	    [&](uint32_t* count, vk::PresentModeKHR* values) {
		    return physical_device.getSurfacePresentModesKHR(surface, count, values);
	    });
	EXIT_NOT_IMPLEMENTED(r.present_modes.empty());
}

static bool CheckFormat(vk::PhysicalDevice device, vk::Format format, bool tile,
                        vk::FormatFeatureFlags features) {
	vk::FormatProperties format_props {};
	device.getFormatProperties(format, &format_props);

	const auto supported_features =
	    (tile ? format_props.optimalTilingFeatures : format_props.linearTilingFeatures);
	return (supported_features & features) == features;
}

static uint32_t VulkanFindQueueFamily(vk::PhysicalDevice device, vk::SurfaceKHR surface) {
	EXIT_IF(device == nullptr);
	EXIT_IF(surface == nullptr);

	uint32_t queue_family_count = 0;
	device.getQueueFamilyProperties(&queue_family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> queue_families(queue_family_count);
	device.getQueueFamilyProperties(&queue_family_count, queue_families.data());

	const auto required = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
	for (uint32_t family = 0; family < queue_family_count; family++) {
		const auto& properties             = queue_families[family];
		vk::Bool32  presentation_supported = VK_FALSE;
		RequireVulkanSuccess(device.getSurfaceSupportKHR(family, surface, &presentation_supported),
		                     "vkGetPhysicalDeviceSurfaceSupportKHR");

		LOGF("\tqueue family: %s [count = %u], [present = %s]\n",
		     vk::to_string(properties.queueFlags).c_str(), properties.queueCount,
		     (presentation_supported == VK_TRUE ? "true" : "false"));
		if (properties.queueCount != 0 && (properties.queueFlags & required) == required &&
		    presentation_supported == VK_TRUE) {
			LOGF("\tselected universal queue family %u\n", family);
			return family;
		}
	}
	return static_cast<uint32_t>(-1);
}

// On failure out_device is null and out_rejections says why each device was skipped.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static void VulkanFindPhysicalDevice(vk::Instance instance, vk::SurfaceKHR surface,
                                     const std::vector<const char*>& device_extensions,
                                     SurfaceCapabilities&            out_capabilities,
                                     vk::PhysicalDevice& out_device, uint32_t& out_queue_family,
                                     std::string& out_rejections) {
	EXIT_IF(instance == nullptr);
	EXIT_IF(surface == nullptr);

	auto devices = EnumerateVulkan<vk::PhysicalDevice>(
	    "vkEnumeratePhysicalDevices", [&](uint32_t* count, vk::PhysicalDevice* values) {
		    return instance.enumeratePhysicalDevices(count, values);
	    });
	EXIT_NOT_IMPLEMENTED(devices.empty());

	if (Config::GetGpuIndex() >= 0) {
		if (static_cast<size_t>(Config::GetGpuIndex()) < devices.size()) {
			devices = {devices[Config::GetGpuIndex()]};
		} else {
			LOGF("Vulkan GPU index %d is unavailable; selecting automatically\n",
			     Config::GetGpuIndex());
		}
	}

	vk::PhysicalDevice  best_device       = nullptr;
	uint32_t            best_queue_family = static_cast<uint32_t>(-1);
	SurfaceCapabilities best_capabilities;
	std::tuple<int, int, uint64_t> best_rank {};
	std::string                    best_name;

	for (const auto& device: devices) {
		bool skip_device = false;

		vk::PhysicalDeviceProperties device_properties {};
		device.getProperties(&device_properties);

		const auto reject = [&](const std::string& reason) {
			LOGF("%s\n", reason.c_str());
			out_rejections +=
			    skip_device ? "; " : fmt::format("\n  {}: ", device_properties.deviceName.data());
			out_rejections += reason;
			skip_device = true;
		};

		LOGF("Vulkan device: %s\n", device_properties.deviceName.data());
		if (device_properties.apiVersion < VULKAN_TARGET_API_VERSION) {
			reject(fmt::format("Vulkan {}.{} is required, but device supports only {}.{}.{}",
			                   VK_VERSION_MAJOR(VULKAN_TARGET_API_VERSION),
			                   VK_VERSION_MINOR(VULKAN_TARGET_API_VERSION),
			                   VK_VERSION_MAJOR(device_properties.apiVersion),
			                   VK_VERSION_MINOR(device_properties.apiVersion),
			                   VK_VERSION_PATCH(device_properties.apiVersion)));
			continue;
		}

		vk::PhysicalDeviceFeatures2 device_features2 {};

		vk::PhysicalDeviceVulkan13Features features13 {};

		vk::PhysicalDeviceColorWriteEnableFeaturesEXT color_write_ext {};
		vk::PhysicalDeviceImageViewMinLodFeaturesEXT  image_view_min_lod {};
		color_write_ext.pNext = &image_view_min_lod;

		vk::PhysicalDeviceDepthClipEnableFeaturesEXT depth_clip_enable {};
		depth_clip_enable.pNext = &color_write_ext;

		vk::PhysicalDeviceDepthClipControlFeaturesEXT depth_clip_control {};
		depth_clip_control.pNext = &depth_clip_enable;

		vk::PhysicalDeviceVulkan12Features features12 {};
#if defined(__APPLE__)
		features12.pNext = &depth_clip_control;
#else
		vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR fragment_barycentric {};
		fragment_barycentric.pNext = &depth_clip_control;
		features12.pNext           = &fragment_barycentric;
#endif
		features13.pNext       = &features12;
		device_features2.pNext = &features13;

		device.getFeatures2(&device_features2);
		const auto required_features12 = WindowContext::RequiredVulkan12Features();
		const auto required_features13 = WindowContext::RequiredVulkan13Features();

		const auto queue_family = VulkanFindQueueFamily(device, surface);
		if (queue_family == static_cast<uint32_t>(-1)) {
			reject("No universal graphics, compute, and presentation queue");
		}

		const auto check_feature = [&](vk::Bool32 supported, const char* name,
		                               vk::Bool32 required = VK_TRUE) {
			if (required == VK_TRUE && supported != VK_TRUE) {
				reject(fmt::format("{} is not supported", name));
			}
		};

		// Optional (VulkanCreateDevice): without them the renderer keeps static colour-write masks
		// and lets depth clamping stand in for the depth-clip switch, as on MoltenVK.
		if (color_write_ext.colorWriteEnable != VK_TRUE) {
			LOGF("colorWriteEnable is not supported\n");
		}
		check_feature(image_view_min_lod.minLod, "image view minLod");

		check_feature(depth_clip_control.depthClipControl, "depthClipControl");
		if (depth_clip_enable.depthClipEnable != VK_TRUE) {
			LOGF("depthClipEnable is not supported\n");
		}
#if !defined(__APPLE__)
		check_feature(device_features2.features.depthClamp, "depthClamp");
		check_feature(fragment_barycentric.fragmentShaderBarycentric, "fragmentShaderBarycentric");
#endif

		check_feature(features12.samplerMirrorClampToEdge, "samplerMirrorClampToEdge",
		              required_features12.samplerMirrorClampToEdge);
		check_feature(features12.timelineSemaphore, "timelineSemaphore",
		              required_features12.timelineSemaphore);
		check_feature(features12.shaderOutputLayer, "shaderOutputLayer",
		              required_features12.shaderOutputLayer);
		check_feature(features12.shaderOutputViewportIndex, "shaderOutputViewportIndex",
		              required_features12.shaderOutputViewportIndex);
		check_feature(features12.bufferDeviceAddress, "bufferDeviceAddress",
		              required_features12.bufferDeviceAddress);
		check_feature(features12.shaderBufferInt64Atomics, "shaderBufferInt64Atomics",
		              required_features12.shaderBufferInt64Atomics);
		check_feature(features13.robustImageAccess, "robustImageAccess");
		check_feature(features13.dynamicRendering, "dynamicRendering",
		              required_features13.dynamicRendering);
		check_feature(features13.synchronization2, "synchronization2",
		              required_features13.synchronization2);
		check_feature(device_features2.features.sampleRateShading, "sampleRateShading");
		check_feature(device_features2.features.depthBiasClamp, "depthBiasClamp");
		check_feature(device_features2.features.shaderClipDistance, "shaderClipDistance");
		check_feature(device_features2.features.shaderCullDistance, "shaderCullDistance");
		check_feature(device_features2.features.largePoints, "largePoints");
		check_feature(device_features2.features.multiViewport, "multiViewport");
		check_feature(device_features2.features.fillModeNonSolid, "fillModeNonSolid");
		check_feature(device_features2.features.fragmentStoresAndAtomics,
		              "fragmentStoresAndAtomics");

		check_feature(device_features2.features.samplerAnisotropy, "samplerAnisotropy");
		check_feature(device_features2.features.robustBufferAccess, "robustBufferAccess");
#if defined(__APPLE__)
		if (device_features2.features.depthBounds != VK_TRUE) {
			LOGF("depthBounds is not supported\n");
		}
#else
		check_feature(device_features2.features.depthBounds, "depthBounds");
#endif
		check_feature(device_features2.features.shaderStorageImageWriteWithoutFormat,
		              "shaderStorageImageWriteWithoutFormat");

		check_feature(device_features2.features.shaderImageGatherExtended,
		              "shaderImageGatherExtended");

		check_feature(device_features2.features.independentBlend, "independentBlend");
		check_feature(device_features2.features.tessellationShader, "tessellationShader");

		if (!skip_device) {
			auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
			    "vkEnumerateDeviceExtensionProperties",
			    [&](uint32_t* count, vk::ExtensionProperties* values) {
				    return device.enumerateDeviceExtensionProperties(nullptr, count, values);
			    });
			EXIT_NOT_IMPLEMENTED(available_extensions.empty());

			for (const char* ext: device_extensions) {
				if (!HasExtension(available_extensions, ext)) {
					reject(fmt::format("{} is not supported", ext));
				}
			}

			if (skip_device) {
				for (const auto& ext: available_extensions) {
					LOGF("Vulkan available extension: %s, version = %u\n", ext.extensionName.data(),
					     ext.specVersion);
				}
			}
		}

		SurfaceCapabilities candidate_capabilities;
		if (!skip_device) {
			GetSurfaceCapabilities(device, surface, candidate_capabilities);

			if (!(candidate_capabilities.capabilities.supportedUsageFlags &
			      vk::ImageUsageFlagBits::eTransferDst)) {
				reject("Surface cannot be destination of blit");
			}
		}

		const auto check_format = [&](vk::Format format, vk::FormatFeatureFlags features,
		                              const char* usage) {
			if (!skip_device && !CheckFormat(device, format, true, features)) {
				reject(fmt::format("Format vk::Format::e{} cannot be used as {}",
				                   vk::to_string(format), usage));
			}
		};

		const auto depth_features = vk::FormatFeatureFlagBits::eDepthStencilAttachment;
		const auto texture_features =
		    vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferDst;
		const auto storage_features =
		    vk::FormatFeatureFlagBits::eStorageImage | vk::FormatFeatureFlagBits::eTransferDst;

		check_format(vk::Format::eD32Sfloat, depth_features, "depth buffer");
		check_format(vk::Format::eD32SfloatS8Uint, depth_features, "depth buffer");
		check_format(vk::Format::eD16Unorm, depth_features, "depth buffer");
		check_format(vk::Format::eBc3SrgbBlock, texture_features, "texture");
		check_format(vk::Format::eR8G8B8A8Srgb, texture_features, "texture");
		check_format(vk::Format::eR8Unorm, texture_features, "texture");
		check_format(vk::Format::eR8G8Unorm, texture_features, "texture");

		if (!skip_device &&
		    !CheckFormat(device, vk::Format::eR8G8B8A8Srgb, true, storage_features)) {
			LOGF("Format vk::Format::eR8G8B8A8Srgb cannot be used as storage image\n");
			check_format(vk::Format::eR8G8B8A8Unorm, storage_features, "storage image");
		}

		if (!skip_device &&
		    !CheckFormat(device, vk::Format::eB8G8R8A8Srgb, true, storage_features)) {
			LOGF("Format vk::Format::eB8G8R8A8Srgb cannot be used as storage image\n");
			check_format(vk::Format::eB8G8R8A8Unorm, storage_features, "storage image");
		}

		if (!skip_device && device_properties.limits.maxSamplerAnisotropy < 16.0f) {
			reject("maxSamplerAnisotropy < 16.0f");
		}

		if (skip_device) {
			continue;
		}

		// Automatic choice: a discrete GPU over an integrated one (a Ryzen iGPU next to an RTX card),
		// a native driver over a layered one (Microsoft's D3D12-based "Dozen" driver can list the same
		// card again as a discrete GPU), then the most device-local memory; the first wins a tie.
		vk::PhysicalDeviceDriverProperties driver_properties {};
		vk::PhysicalDeviceProperties2      properties2 {};
		properties2.pNext = &driver_properties;
		device.getProperties2(&properties2);
		const auto memory_properties = device.getMemoryProperties();
		uint64_t   local_bytes       = 0;
		for (uint32_t i = 0; i < memory_properties.memoryHeapCount; i++) {
			if (memory_properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
				local_bytes = std::max<uint64_t>(local_bytes, memory_properties.memoryHeaps[i].size);
			}
		}
		const bool layered = driver_properties.driverID == vk::DriverId::eMesaDozen;
		const int  type_rank =
		    device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu     ? 3
		    : device_properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu ? 2
		    : device_properties.deviceType == vk::PhysicalDeviceType::eVirtualGpu    ? 1
		                                                                             : 0;
		const auto rank = std::make_tuple(layered ? 0 : 1, type_rank, local_bytes);
		LOGF("Vulkan device candidate: %s (type %d, driver %d, %.1f GiB device-local)\n",
		     device_properties.deviceName.data(), type_rank, static_cast<int>(driver_properties.driverID),
		     static_cast<double>(local_bytes) / (1024.0 * 1024.0 * 1024.0));
		if (best_device == nullptr || rank > best_rank) {
			best_device       = device;
			best_queue_family = queue_family;
			best_capabilities = std::move(candidate_capabilities);
			best_rank         = rank;
			best_name         = device_properties.deviceName.data();
		}
	}

	if (best_device != nullptr) {
		std::printf("Kyty GPU: %s (%s)\n", best_name.c_str(),
		            Config::GetGpuIndex() >= 0 ? "selected in the launcher" : "automatic choice");
	}
	out_device       = best_device;
	out_queue_family = best_queue_family;
	if (best_device != nullptr) {
		out_capabilities = std::move(best_capabilities);
	}
}

// Guest shaders flush f32 denormals and keep f16/f64 ones (FLOAT_MODE 0xC0). Tell the SPIR-V
// emitter which float-control execution modes this device accepts. With independence NONE all
// widths must share one mode, so nothing is declared there. KYTY_SHADER_FLOAT_CONTROLS=0
// keeps the host defaults.
static void ConfigureShaderFloatControls(const vk::PhysicalDeviceVulkan12Properties& properties) {
	namespace Spirv = ShaderRecompiler::Spirv;
	Spirv::HostFloatControls controls {};
	const auto*              env     = std::getenv("KYTY_SHADER_FLOAT_CONTROLS");
	const bool               enabled = env == nullptr || std::strcmp(env, "0") != 0;
	const auto independence          = properties.denormBehaviorIndependence;
	if (enabled && independence != vk::ShaderFloatControlsIndependence::eNone) {
		controls.denorm_flush_f32 = properties.shaderDenormFlushToZeroFloat32 == VK_TRUE;
		const bool preserve16     = properties.shaderDenormPreserveFloat16 == VK_TRUE;
		const bool preserve64     = properties.shaderDenormPreserveFloat64 == VK_TRUE;
		if (independence == vk::ShaderFloatControlsIndependence::eAll) {
			controls.denorm_preserve_f16 = preserve16;
			controls.denorm_preserve_f64 = preserve64;
		} else {
			// 32_BIT_ONLY: 16- and 64-bit types must share a mode.
			controls.denorm_preserve_f16 = preserve16 && preserve64;
			controls.denorm_preserve_f64 = preserve16 && preserve64;
		}
	}
	Spirv::SetHostFloatControls(controls);
	LOGF("Vulkan float controls: independence=%s ftz32=%s preserve16=%s preserve64=%s%s\n",
	     vk::to_string(independence).c_str(), controls.denorm_flush_f32 ? "true" : "false",
	     controls.denorm_preserve_f16 ? "true" : "false",
	     controls.denorm_preserve_f64 ? "true" : "false", enabled ? "" : " (disabled by env)");
}

static std::string FormatBytes(uint64_t bytes) {
	if (bytes >= (1ull << 30u)) {
		return fmt::format("{:.1f} GiB", static_cast<double>(bytes) / static_cast<double>(1ull << 30u));
	}
	return fmt::format("{} MiB", bytes >> 20u);
}

// "Kyty GPU capabilities:" lines on the console, so a player's log answers device questions
// (driver, architecture, memory, mesh limits, float controls, subgroups, features) without tests.
// Also resolves KYTY_VS_OUTPUT_GUARD / KYTY_MESH_OUTPUT_GUARD (auto: on for NVIDIA Turing only).
static void PrintGpuCapabilities(GraphicContext& graphics, const vk::PhysicalDeviceProperties& props,
                                 const vk::PhysicalDeviceVulkan11Properties&            p11,
                                 const vk::PhysicalDeviceVulkan12Properties&            p12,
                                 const vk::PhysicalDeviceSubgroupSizeControlProperties& subgroup,
                                 const vk::PhysicalDeviceFeatures&                      f10,
                                 const vk::PhysicalDeviceVulkan12Features&              f12,
                                 const vk::PhysicalDeviceVulkan13Features&              f13,
                                 const vk::PhysicalDeviceRobustness2FeaturesEXT*        robustness2,
                                 bool workgroup_explicit_layout) {
	namespace Compat        = DeviceCompat;
	const auto architecture = Compat::GuessNvidiaArchitecture(props.vendorID, props.deviceID);
	const auto driver       = props.driverVersion;
	const std::string driver_text =
	    props.vendorID == Compat::NvidiaVendorId
	        ? fmt::format("NVIDIA {}.{:02}", (driver >> 22u) & 0x3ffu, (driver >> 14u) & 0xffu)
	        : fmt::format("{}.{}.{} (0x{:08x})", VK_VERSION_MAJOR(driver), VK_VERSION_MINOR(driver),
	                      VK_VERSION_PATCH(driver), driver);
	std::printf("Kyty GPU capabilities: %s, vendor 0x%04x device 0x%04x = %s; driver %s (%s %s); "
	            "Vulkan %u.%u.%u\n",
	            props.deviceName.data(), props.vendorID, props.deviceID,
	            Compat::NvidiaArchitectureName(architecture), driver_text.c_str(),
	            p12.driverName.data(), p12.driverInfo.data(), VK_VERSION_MAJOR(props.apiVersion),
	            VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion));

	// Memory: VRAM, the host-visible part of it (Resizable BAR), and the driver budget.
	const auto& memory = graphics.physical_device_memory_properties;
	uint64_t    vram   = 0;
	uint64_t    bar    = 0;
	uint64_t    host   = 0;
	uint32_t    vram_heap = 0;
	for (uint32_t heap = 0; heap < memory.memoryHeapCount; heap++) {
		const auto size = memory.memoryHeaps[heap].size;
		if (memory.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
			if (size > vram) {
				vram      = size;
				vram_heap = heap;
			}
		} else {
			host = std::max(host, size);
		}
	}
	for (uint32_t type = 0; type < memory.memoryTypeCount; type++) {
		const auto flags = memory.memoryTypes[type].propertyFlags;
		if ((flags & vk::MemoryPropertyFlagBits::eDeviceLocal) &&
		    (flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
			bar = std::max(bar, memory.memoryHeaps[memory.memoryTypes[type].heapIndex].size);
		}
	}
	std::string budget = "no VK_EXT_memory_budget";
	if (graphics.memory_budget_ext_enabled) {
		vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget_props {};
		vk::PhysicalDeviceMemoryProperties2         memory2 {};
		memory2.pNext = &budget_props;
		graphics.physical_device.getMemoryProperties2(&memory2);
		budget = fmt::format("budget {} (used by this process {})",
		                     FormatBytes(budget_props.heapBudget[vram_heap]),
		                     FormatBytes(budget_props.heapUsage[vram_heap]));
	}
	std::printf("Kyty GPU capabilities: VRAM %s, %s; device-local host-visible heap %s (Resizable "
	            "BAR %s); system heap %s\n",
	            FormatBytes(vram).c_str(), budget.c_str(), FormatBytes(bar).c_str(),
	            bar >= (2ull << 30u) ? "on" : "off", FormatBytes(host).c_str());

	if (graphics.mesh_shader_enabled) {
		const auto& m = graphics.mesh_shader_properties;
		std::printf("Kyty GPU capabilities: mesh shaders: invocations %u (preferred %u), group size "
		            "%u, output vertices %u primitives %u components %u, output memory %u, "
		            "payload+output %u, shared %u; granularity vertex %u primitive %u; prefers "
		            "local-invocation output vertex %u primitive %u, compact vertex %u primitive %u\n",
		            m.maxMeshWorkGroupInvocations, m.maxPreferredMeshWorkGroupInvocations,
		            m.maxMeshWorkGroupSize[0], m.maxMeshOutputVertices, m.maxMeshOutputPrimitives,
		            m.maxMeshOutputComponents, m.maxMeshOutputMemorySize,
		            m.maxMeshPayloadAndOutputMemorySize, m.maxMeshSharedMemorySize,
		            m.meshOutputPerVertexGranularity, m.meshOutputPerPrimitiveGranularity,
		            static_cast<uint32_t>(m.prefersLocalInvocationVertexOutput),
		            static_cast<uint32_t>(m.prefersLocalInvocationPrimitiveOutput),
		            static_cast<uint32_t>(m.prefersCompactVertexOutput),
		            static_cast<uint32_t>(m.prefersCompactPrimitiveOutput));
	} else {
		std::printf("Kyty GPU capabilities: mesh shaders: unavailable\n");
	}

	const auto independence = [](vk::ShaderFloatControlsIndependence value) {
		return value == vk::ShaderFloatControlsIndependence::eAll         ? "all"
		       : value == vk::ShaderFloatControlsIndependence::e32BitOnly ? "32-bit only"
		                                                                   : "none";
	};
	const auto bits3 = [](vk::Bool32 a, vk::Bool32 b, vk::Bool32 c) {
		return fmt::format("{}{}{}", a ? 1 : 0, b ? 1 : 0, c ? 1 : 0);
	};
	const auto declared = ShaderRecompiler::Spirv::GetHostFloatControls();
	std::printf("Kyty GPU capabilities: float controls (f16 f32 f64 bits): denorm independence %s, "
	            "rounding independence %s; denorm preserve %s, flush %s; RTE %s, RTZ %s; signed "
	            "zero/inf/nan %s; Kyty declares ftz32 %u preserve16 %u preserve64 %u\n",
	            independence(p12.denormBehaviorIndependence),
	            independence(p12.roundingModeIndependence),
	            bits3(p12.shaderDenormPreserveFloat16, p12.shaderDenormPreserveFloat32,
	                  p12.shaderDenormPreserveFloat64)
	                .c_str(),
	            bits3(p12.shaderDenormFlushToZeroFloat16, p12.shaderDenormFlushToZeroFloat32,
	                  p12.shaderDenormFlushToZeroFloat64)
	                .c_str(),
	            bits3(p12.shaderRoundingModeRTEFloat16, p12.shaderRoundingModeRTEFloat32,
	                  p12.shaderRoundingModeRTEFloat64)
	                .c_str(),
	            bits3(p12.shaderRoundingModeRTZFloat16, p12.shaderRoundingModeRTZFloat32,
	                  p12.shaderRoundingModeRTZFloat64)
	                .c_str(),
	            bits3(p12.shaderSignedZeroInfNanPreserveFloat16,
	                  p12.shaderSignedZeroInfNanPreserveFloat32,
	                  p12.shaderSignedZeroInfNanPreserveFloat64)
	                .c_str(),
	            declared.denorm_flush_f32 ? 1u : 0u, declared.denorm_preserve_f16 ? 1u : 0u,
	            declared.denorm_preserve_f64 ? 1u : 0u);

	std::printf("Kyty GPU capabilities: subgroups: size %u (min %u max %u), required-size stages "
	            "0x%x, size control %s; supported stages 0x%x operations 0x%x, quad ops in all "
	            "stages %u\n",
	            p11.subgroupSize, subgroup.minSubgroupSize, subgroup.maxSubgroupSize,
	            static_cast<vk::ShaderStageFlags::MaskType>(subgroup.requiredSubgroupSizeStages),
	            f13.subgroupSizeControl ? "yes" : "no",
	            static_cast<vk::ShaderStageFlags::MaskType>(p11.subgroupSupportedStages),
	            static_cast<vk::SubgroupFeatureFlags::MaskType>(p11.subgroupSupportedOperations),
	            static_cast<uint32_t>(p11.subgroupQuadOperationsInAllStages));

	std::printf("Kyty GPU capabilities: features: int16 %u int64 %u float64 %u float16 %u int8 %u "
	            "storage8 %u BDA %u memory model %u demote %u maintenance4 %u robustBufferAccess %u "
	            "robustBufferAccess2 %u robustImageAccess2 %u nullDescriptor %u workgroup layout %u "
	            "buffer int64 atomics %u\n",
	            static_cast<uint32_t>(f10.shaderInt16), static_cast<uint32_t>(f10.shaderInt64),
	            static_cast<uint32_t>(f10.shaderFloat64), static_cast<uint32_t>(f12.shaderFloat16),
	            static_cast<uint32_t>(f12.shaderInt8),
	            static_cast<uint32_t>(f12.storageBuffer8BitAccess),
	            static_cast<uint32_t>(f12.bufferDeviceAddress),
	            static_cast<uint32_t>(f12.vulkanMemoryModel),
	            static_cast<uint32_t>(f13.shaderDemoteToHelperInvocation),
	            static_cast<uint32_t>(f13.maintenance4), static_cast<uint32_t>(f10.robustBufferAccess),
	            robustness2 != nullptr ? static_cast<uint32_t>(robustness2->robustBufferAccess2) : 0u,
	            robustness2 != nullptr ? static_cast<uint32_t>(robustness2->robustImageAccess2) : 0u,
	            robustness2 != nullptr ? static_cast<uint32_t>(robustness2->nullDescriptor) : 0u,
	            workgroup_explicit_layout ? 1u : 0u,
	            static_cast<uint32_t>(f12.shaderBufferInt64Atomics));

	// Undefined geometry outputs (SpirvEmitter.h, HostGeometryGuard).
	const auto vertex_mode = Compat::ParseQuirkMode(std::getenv("KYTY_VS_OUTPUT_GUARD"));
	const auto mesh_mode   = Compat::ParseQuirkMode(std::getenv("KYTY_MESH_OUTPUT_GUARD"));
	const bool automatic   = Compat::GeometryOutputGuardAutomatic(architecture);
	const ShaderRecompiler::Spirv::HostGeometryGuard guard {
	    .vertex_outputs = Compat::ResolveQuirk(vertex_mode, automatic),
	    .mesh_outputs   = Compat::ResolveQuirk(mesh_mode, automatic),
	};
	ShaderRecompiler::Spirv::SetHostGeometryGuard(guard);
	std::printf("Kyty GPU capabilities: geometry output guard: vertex %s (KYTY_VS_OUTPUT_GUARD=%s), "
	            "mesh %s (KYTY_MESH_OUTPUT_GUARD=%s); auto is %s on this GPU\n",
	            guard.vertex_outputs ? "on" : "off", Compat::QuirkModeName(vertex_mode),
	            guard.mesh_outputs ? "on" : "off", Compat::QuirkModeName(mesh_mode),
	            automatic ? "on (NVIDIA Turing)" : "off");
}

static vk::Device VulkanCreateDevice(GraphicContext& graphics,
	                                 const std::vector<const char*>& device_extensions) {
	const auto physical_device = graphics.physical_device;
	const auto queue_family    = graphics.queue_family;
	EXIT_IF(physical_device == nullptr);
	EXIT_IF(queue_family == static_cast<uint32_t>(-1));

	// Queue 0 carries every scheduler submission and presentation. A second queue of the same
	// family (KYTY_SIDE_QUEUE, default on when the family has one) runs side-copy readbacks, so a
	// copy waits only for its producer instead of for everything queued ahead of it on queue 0.
	// Buffers stay EXCLUSIVE: ownership is per family, and both queues share it.
	uint32_t family_queue_count = 0;
	{
		uint32_t count = 0;
		physical_device.getQueueFamilyProperties(&count, nullptr);
		std::vector<vk::QueueFamilyProperties> families(count);
		physical_device.getQueueFamilyProperties(&count, families.data());
		if (queue_family < count) {
			family_queue_count = families[queue_family].queueCount;
		}
	}
	const bool side_queue_requested = [] {
		const auto* value = std::getenv("KYTY_SIDE_QUEUE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	graphics.side_queue_index = side_queue_requested && family_queue_count >= 2 ? 1u : 0u;
	std::printf("Kyty side-copy queue: %s (family %u has %u queues, KYTY_SIDE_QUEUE)\n",
	       graphics.side_queue_index != 0 ? "queue 1" : "shared with queue 0", queue_family,
	       family_queue_count);

	const std::array<float, 2> queue_priorities {1.0f, 1.0f};
	std::array<vk::DeviceQueueCreateInfo, 2> queue_create_infos {};
	auto&                      queue_create_info = queue_create_infos[0];
	queue_create_info.queueFamilyIndex = queue_family;
	queue_create_info.queueCount       = graphics.side_queue_index != 0 ? 2u : 1u;
	queue_create_info.pQueuePriorities = queue_priorities.data();
	// KYTY_UPLOAD_DMA: one queue of a transfer-only family (the copy engines), for UploadDma.
	// Not under RenderDoc, whose captures of the extra queue are not needed for analysis.
	graphics.transfer_queue_family = static_cast<uint32_t>(-1);
	if (UploadDmaRequested() && !Config::RenderDocEnabled()) {
		uint32_t count = 0;
		physical_device.getQueueFamilyProperties(&count, nullptr);
		std::vector<vk::QueueFamilyProperties> families(count);
		physical_device.getQueueFamilyProperties(&count, families.data());
		for (uint32_t family = 0; family < count; family++) {
			const auto flags = families[family].queueFlags;
			if (family != queue_family && families[family].queueCount != 0 &&
			    (flags & vk::QueueFlagBits::eTransfer) &&
			    !(flags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) &&
			    !(flags & (vk::QueueFlagBits::eVideoDecodeKHR | vk::QueueFlagBits::eVideoEncodeKHR |
			               vk::QueueFlagBits::eOpticalFlowNV))) {
				graphics.transfer_queue_family = family;
				break;
			}
		}
	}
	uint32_t queue_create_count = 1;
	if (graphics.transfer_queue_family != static_cast<uint32_t>(-1)) {
		auto& transfer            = queue_create_infos[queue_create_count++];
		transfer.queueFamilyIndex = graphics.transfer_queue_family;
		transfer.queueCount       = 1;
		transfer.pQueuePriorities = queue_priorities.data();
	}
	std::printf("Kyty upload DMA queue: %s (KYTY_UPLOAD_DMA)\n",
	            graphics.transfer_queue_family != static_cast<uint32_t>(-1)
	                ? fmt::format("family {}", graphics.transfer_queue_family).c_str()
	                : "none");

	// Chained after image_view_min_lod below, each only where enabled (MoltenVK and older drivers
	// lack them; the renderer then keeps static colour-write masks and depth-clamp-based clipping).
	vk::PhysicalDeviceColorWriteEnableFeaturesEXT color_write_ext {};
	color_write_ext.colorWriteEnable = VK_TRUE;

	vk::PhysicalDeviceDepthClipEnableFeaturesEXT depth_clip_enable {};
	depth_clip_enable.depthClipEnable = VK_TRUE;

	vk::PhysicalDeviceDepthClipControlFeaturesEXT depth_clip_control {};
	vk::PhysicalDeviceImageViewMinLodFeaturesEXT  image_view_min_lod {};
	image_view_min_lod.minLod = VK_TRUE;
	depth_clip_control.pNext  = &image_view_min_lod;
	depth_clip_control.depthClipControl = VK_TRUE;

	// Indirect draws' start instance reaches the shaders as gl_BaseInstance.
	vk::PhysicalDeviceShaderDrawParametersFeatures draw_parameters {};
	draw_parameters.pNext                = &depth_clip_control;
	draw_parameters.shaderDrawParameters = VK_TRUE;

	auto features12  = WindowContext::RequiredVulkan12Features();
	features12.pNext = &draw_parameters;
	// drawIndirectCount is set below, once the supported features are known.

	vk::PhysicalDeviceVulkan13Features supported_features13 {};

	const auto robustness2_ext_enabled =
	    HasExtension(device_extensions, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);

	vk::PhysicalDeviceRobustness2FeaturesEXT supported_robustness2 {};
	if (robustness2_ext_enabled) {
		supported_robustness2.pNext = supported_features13.pNext;
		supported_features13.pNext = &supported_robustness2;
	}

	const bool mesh_extension = HasExtension(device_extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME);
	vk::PhysicalDeviceMeshShaderFeaturesEXT supported_mesh {};
	supported_mesh.pNext = &supported_features13;
	vk::PhysicalDeviceFeatures2 supported_features2 {};
	supported_features2.pNext = mesh_extension ? static_cast<void*>(&supported_mesh)
	                                           : static_cast<void*>(&supported_features13);
	const bool feedback_extensions =
	    HasExtension(device_extensions, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) &&
	    HasExtension(device_extensions, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME);
	vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT feedback_layout {};
	vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT feedback_dynamic {};
	if (feedback_extensions) {
		feedback_dynamic.pNext = supported_features2.pNext;
		feedback_layout.pNext  = &feedback_dynamic;
		supported_features2.pNext = &feedback_layout;
	}
	const bool provoking_extension =
	    HasExtension(device_extensions, VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME);
	vk::PhysicalDeviceProvokingVertexFeaturesEXT provoking_vertex {};
	if (provoking_extension) {
		provoking_vertex.pNext = supported_features2.pNext;
		supported_features2.pNext = &provoking_vertex;
	}
	const bool index_type_uint8_extension =
	    HasExtension(device_extensions, VK_KHR_INDEX_TYPE_UINT8_EXTENSION_NAME) ||
	    HasExtension(device_extensions, VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME);
	vk::PhysicalDeviceIndexTypeUint8FeaturesKHR supported_index_type_uint8 {};
	if (index_type_uint8_extension) {
		supported_index_type_uint8.pNext = supported_features2.pNext;
		supported_features2.pNext        = &supported_index_type_uint8;
	}
	const bool maintenance8_extension =
	    HasExtension(device_extensions, VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
	vk::PhysicalDeviceMaintenance8FeaturesKHR supported_maintenance8 {};
	if (maintenance8_extension) {
		supported_maintenance8.pNext = supported_features2.pNext;
		supported_features2.pNext    = &supported_maintenance8;
	}
	const auto advertised_extensions = physical_device.enumerateDeviceExtensionProperties();
	RequireVulkanSuccess(advertised_extensions.result, "vkEnumerateDeviceExtensionProperties");
	const bool pipeline_library_extension =
	    HasExtension(advertised_extensions.value, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
	vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT supported_pipeline_library {};
	if (pipeline_library_extension) {
		supported_pipeline_library.pNext = supported_features2.pNext;
		supported_features2.pNext        = &supported_pipeline_library;
	}
	vk::PhysicalDeviceShaderObjectFeaturesEXT supported_shader_object {};
	if (HasExtension(advertised_extensions.value, VK_EXT_SHADER_OBJECT_EXTENSION_NAME)) {
		supported_shader_object.pNext = supported_features2.pNext;
		supported_features2.pNext = &supported_shader_object;
	}
	// Native 64-bit LDS atomics use typed views of shared memory (upstream 6799ecbc5).
	const bool workgroup_layout_extension =
	    HasExtension(device_extensions, VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
	vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR supported_workgroup_layout {};
	if (workgroup_layout_extension) {
		supported_workgroup_layout.pNext = supported_features2.pNext;
		supported_features2.pNext        = &supported_workgroup_layout;
	}
	vk::PhysicalDeviceVulkan12Features supported_features12 {};
	supported_features12.pNext = supported_features2.pNext;
	supported_features2.pNext  = &supported_features12;
	vk::PhysicalDeviceShaderDrawParametersFeatures supported_draw_parameters {};
	supported_draw_parameters.pNext = supported_features2.pNext;
	supported_features2.pNext       = &supported_draw_parameters;
	const bool image_atomic_int64_extension =
	    HasExtension(device_extensions, VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME);
	vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT image_atomic_int64 {};
	if (image_atomic_int64_extension) {
		image_atomic_int64.pNext = supported_features2.pNext;
		supported_features2.pNext = &image_atomic_int64;
	}
	const bool color_write_extension =
	    HasExtension(device_extensions, VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME);
	vk::PhysicalDeviceColorWriteEnableFeaturesEXT supported_color_write {};
	if (color_write_extension) {
		supported_color_write.pNext = supported_features2.pNext;
		supported_features2.pNext   = &supported_color_write;
	}
	const bool depth_clip_extension =
	    HasExtension(device_extensions, VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME);
	vk::PhysicalDeviceDepthClipEnableFeaturesEXT supported_depth_clip {};
	if (depth_clip_extension) {
		supported_depth_clip.pNext = supported_features2.pNext;
		supported_features2.pNext  = &supported_depth_clip;
	}
	physical_device.getFeatures2(&supported_features2);
	graphics.color_write_enable_enabled =
	    color_write_extension && supported_color_write.colorWriteEnable == VK_TRUE;
	graphics.depth_clip_enable_enabled =
	    depth_clip_extension && supported_depth_clip.depthClipEnable == VK_TRUE;
	{
		void* optional = nullptr;
		if (graphics.color_write_enable_enabled) {
			color_write_ext.pNext = optional;
			optional              = &color_write_ext;
		}
		if (graphics.depth_clip_enable_enabled) {
			depth_clip_enable.pNext = optional;
			optional                = &depth_clip_enable;
		}
		image_view_min_lod.pNext = optional;
	}
	LOGF("Vulkan colorWriteEnable: %s, depthClipEnable: %s\n",
	     graphics.color_write_enable_enabled ? "true" : "false (static colour-write masks)",
	     graphics.depth_clip_enable_enabled ? "true" : "false (depth clamp only without Z clipping)");
	graphics.shader_image_int64_atomics_enabled = image_atomic_int64.shaderImageInt64Atomics;
	features12.shaderSharedInt64Atomics = supported_features12.shaderSharedInt64Atomics;
	vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup_layout {};
	workgroup_layout.workgroupMemoryExplicitLayout =
	    supported_workgroup_layout.workgroupMemoryExplicitLayout;
	if (workgroup_layout_extension) {
		workgroup_layout.pNext = features12.pNext;
		features12.pNext       = &workgroup_layout;
	}
	if (!features12.shaderSharedInt64Atomics || !workgroup_layout.workgroupMemoryExplicitLayout) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "WARNING: Native 64-bit LDS atomics are unavailable: shaderSharedInt64Atomics={}, "
		    "workgroupMemoryExplicitLayout={}. Continuing with the selected Vulkan device.\n",
		    features12.shaderSharedInt64Atomics != VK_FALSE,
		    workgroup_layout.workgroupMemoryExplicitLayout != VK_FALSE));
	}
	graphics.maintenance8_enabled =
	    maintenance8_extension && supported_maintenance8.maintenance8 == VK_TRUE;
	LOGF("Vulkan maintenance8 (depth/color image copies): %s\n",
	     graphics.maintenance8_enabled ? "true" : "false");
	// Graphics pipeline libraries need fast linking: without it a link is a full compile.
	bool library_fast_linking = false;
	if (pipeline_library_extension &&
	    supported_pipeline_library.graphicsPipelineLibrary == VK_TRUE) {
		vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT library_properties {};
		vk::PhysicalDeviceProperties2                          properties {};
		properties.pNext = &library_properties;
		physical_device.getProperties2(&properties);
		library_fast_linking = library_properties.graphicsPipelineLibraryFastLinking == VK_TRUE;
	}
	graphics.pipeline_library_enabled = library_fast_linking &&
	    HasExtension(device_extensions, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
	// Only used by the driver-cache probes of the library path and KYTY_PIPELINE_FAST_FIRST; the
	// default device stays unchanged.
	graphics.pipeline_creation_cache_control_enabled =
	    (graphics.pipeline_library_enabled || Libs::Graphics::PipelineFastFirstRequested()) &&
	    supported_features13.pipelineCreationCacheControl == VK_TRUE;
	LOGF("Vulkan pipeline support: GPL extension=%s, feature=%s, fast linking=%s, cache control=%s\n",
	     pipeline_library_extension ? "true" : "false",
	     supported_pipeline_library.graphicsPipelineLibrary ? "true" : "false",
	     library_fast_linking ? "true" : "false",
	     supported_features13.pipelineCreationCacheControl ? "true" : "false");
	LOGF("Vulkan graphics pipeline library enabled: %s, pipeline creation cache control enabled: %s\n",
	     graphics.pipeline_library_enabled ? "true" : "false",
	     graphics.pipeline_creation_cache_control_enabled ? "true" : "false");
	LOGF("Vulkan shader object support: %s (renderer uses pipelines)\n",
	     supported_shader_object.shaderObject ? "true" : "false");
	graphics.mesh_shader_enabled = mesh_extension && supported_mesh.meshShader;
	if (!graphics.mesh_shader_enabled) {
		// Not a reason to reject the device: games without mesh (NGG) shaders still run.
		Log::WriteToConsoleAndLog(fmt::format(
		    "WARNING: the GPU \"{}\" has no mesh shaders ({}). Games that draw with mesh (NGG) "
		    "shaders stop at their first such draw; other games are not affected.\n",
		    graphics.GetPhysicalDeviceProperties().deviceName.data(),
		    mesh_extension ? "VK_EXT_mesh_shader without the meshShader feature"
		                   : "no VK_EXT_mesh_shader"));
	}
	// Optional: native indirect draws fall back to CPU-read arguments without these.
	graphics.draw_indirect_first_instance_enabled =
	    supported_features2.features.drawIndirectFirstInstance == VK_TRUE;
	graphics.multi_draw_indirect_enabled =
	    supported_features2.features.multiDrawIndirect == VK_TRUE;
	graphics.draw_indirect_count_enabled = supported_features12.drawIndirectCount == VK_TRUE;
	graphics.sampler_filter_minmax_enabled = supported_features12.samplerFilterMinmax == VK_TRUE;
	LOGF("Vulkan sampler min/max reduction: %s\n",
	     graphics.sampler_filter_minmax_enabled ? "true" : "false");
	graphics.index_type_uint8_enabled =
	    index_type_uint8_extension && supported_index_type_uint8.indexTypeUint8 == VK_TRUE;
	LOGF("Vulkan indirect draws: firstInstance=%s multiDraw=%s count=%s indexUint8=%s\n",
	     graphics.draw_indirect_first_instance_enabled ? "true" : "false",
	     graphics.multi_draw_indirect_enabled ? "true" : "false",
	     graphics.draw_indirect_count_enabled ? "true" : "false",
	     graphics.index_type_uint8_enabled ? "true" : "false");

	vk::PhysicalDeviceSubgroupSizeControlProperties subgroup_size_control {};

	vk::PhysicalDeviceVulkan11Properties properties11 {};
	properties11.pNext = &subgroup_size_control;

	vk::PhysicalDeviceVulkan12Properties properties12 {};
	properties12.pNext = &properties11;

	vk::PhysicalDeviceRobustness2PropertiesEXT robustness2_properties {};
	vk::PhysicalDeviceProperties2 properties2 {};
	properties2.pNext = &properties12;

	if (graphics.mesh_shader_enabled) {
		subgroup_size_control.pNext = &graphics.mesh_shader_properties;
	}
	if (robustness2_ext_enabled) {
		robustness2_properties.pNext = properties2.pNext;
		properties2.pNext            = &robustness2_properties;
	}
	physical_device.getProperties2(&properties2);
	ConfigureShaderFloatControls(properties12);
	// robustBufferAccess2 (enabled below whenever supported) makes a storage-buffer load return 0
	// when any byte lies past the descriptor range rounded up to this alignment; at 1 byte that is
	// exactly the shaders' own dword bounds check, which they can then leave to the device.
	{
		ShaderRecompiler::Spirv::HostBufferRobustness robustness {};
		robustness.storage_dword_loads_return_zero =
		    robustness2_ext_enabled && supported_robustness2.robustBufferAccess2 == VK_TRUE &&
		    robustness2_properties.robustStorageBufferAccessSizeAlignment == 1u;
		ShaderRecompiler::Spirv::SetHostBufferRobustness(robustness);
		LOGF("Vulkan robustness: robustBufferAccess2=%s storage alignment=%" PRIu64
		     " shader dword bounds checks=%s\n",
		     robustness2_ext_enabled && supported_robustness2.robustBufferAccess2 == VK_TRUE
		         ? "true"
		         : "false",
		     static_cast<uint64_t>(robustness2_properties.robustStorageBufferAccessSizeAlignment),
		     robustness.storage_dword_loads_return_zero ? "device" : "shader");
	}
	// Optional: IMAGE_SAMPLE*_CL clamps become the MinLod image operand.
	const bool shader_resource_min_lod =
	    supported_features2.features.shaderResourceMinLod == VK_TRUE;
	// Derivatives in compute shaders (IMAGE_GET_LOD): VK_KHR_compute_shader_derivatives, else the NV
	// extension (one feature structure for both); the emitter declares the matching SPIR-V extension.
	// Without either, such a shader keeps the KHR declaration and is named when it is compiled.
	vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR compute_derivatives {};
	const bool derivatives_khr =
	    HasExtension(device_extensions, VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
	const bool derivatives_nv =
	    HasExtension(device_extensions, VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
	if (derivatives_khr || derivatives_nv) {
		vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR supported_derivatives {};
		vk::PhysicalDeviceFeatures2                           derivatives_query {};
		derivatives_query.pNext = &supported_derivatives;
		physical_device.getFeatures2(&derivatives_query);
		compute_derivatives.computeDerivativeGroupQuads =
		    supported_derivatives.computeDerivativeGroupQuads;
	}
	{
		namespace Spirv = ShaderRecompiler::Spirv;
		const auto derivatives = compute_derivatives.computeDerivativeGroupQuads != VK_TRUE
		                             ? Spirv::HostComputeDerivatives::None
		                         : derivatives_khr ? Spirv::HostComputeDerivatives::Khr
		                                           : Spirv::HostComputeDerivatives::Nv;
		Spirv::SetHostImageFeatures(
		    {.min_lod = shader_resource_min_lod, .compute_derivatives = derivatives});
		LOGF("Vulkan compute shader derivatives (IMAGE_GET_LOD): %s\n",
		     derivatives == Spirv::HostComputeDerivatives::Khr
		         ? VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME
		     : derivatives == Spirv::HostComputeDerivatives::Nv
		         ? VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME
		         : "unavailable (compute shaders that need them are named when compiled)");
	}
	LOGF("Vulkan shaderResourceMinLod (IMAGE_SAMPLE*_CL): %s\n",
	     shader_resource_min_lod ? "true" : "false");
	// S_MEMREALTIME (KYTY_REALTIME_CLOCK): the device clock, else the subgroup clock, else the
	// placeholder UINT64_MAX. Without a clock a guest spin-wait timed by S_MEMREALTIME never ends.
	vk::PhysicalDeviceShaderClockFeaturesKHR shader_clock {};
	{
		namespace Spirv = ShaderRecompiler::Spirv;
		Spirv::HostShaderClock clock {};
		if (HasExtension(device_extensions, VK_KHR_SHADER_CLOCK_EXTENSION_NAME)) {
			vk::PhysicalDeviceShaderClockFeaturesKHR supported_clock {};
			vk::PhysicalDeviceFeatures2              clock_query {};
			clock_query.pNext = &supported_clock;
			physical_device.getFeatures2(&clock_query);
			shader_clock.shaderSubgroupClock = supported_clock.shaderSubgroupClock;
			shader_clock.shaderDeviceClock   = supported_clock.shaderDeviceClock;
			clock.scope = supported_clock.shaderDeviceClock     ? Spirv::HostClockScope::Device
			              : supported_clock.shaderSubgroupClock ? Spirv::HostClockScope::Subgroup
			                                                    : Spirv::HostClockScope::None;
		}
		const double period = properties2.properties.limits.timestampPeriod;
		clock.shift         = Spirv::RealtimeClockShift(period);
		Spirv::SetHostShaderClock(clock);
		const bool requested = ShaderRecompiler::GetCodegenOptions().realtime_clock;
		std::printf("Kyty shader clock (S_MEMREALTIME, KYTY_REALTIME_CLOCK): %s; timestamp period "
		            "%.3f ns, shift %d\n",
		            !requested                                      ? "off (placeholder)"
		            : clock.scope == Spirv::HostClockScope::Device   ? "device clock"
		            : clock.scope == Spirv::HostClockScope::Subgroup ? "subgroup clock"
		                                                             : "none (placeholder)",
		            period, clock.shift);
		if (requested && clock.scope == Spirv::HostClockScope::Subgroup) {
			Log::WriteToConsoleAndLog("Warning: the Vulkan device has no shaderDeviceClock; "
			                          "S_MEMREALTIME reads the subgroup clock\n");
		} else if (requested && clock.scope == Spirv::HostClockScope::None) {
			Log::WriteToConsoleAndLog("Warning: the Vulkan device has no shader clock "
			                          "(VK_KHR_shader_clock); S_MEMREALTIME returns the placeholder "
			                          "UINT64_MAX, so guest timed waits may not end\n");
		}
	}

	graphics.subgroup_size                 = properties11.subgroupSize;
	graphics.min_subgroup_size             = subgroup_size_control.minSubgroupSize;
	graphics.max_subgroup_size             = subgroup_size_control.maxSubgroupSize;
	// KYTY_FUNCTION_ARRAY_SHRINK bounds SubgroupLocalInvocationId by the largest subgroup.
	SpirvLocalArrays::SetMaxSubgroupSize(
	    std::max(properties11.subgroupSize, subgroup_size_control.maxSubgroupSize));
	graphics.required_subgroup_size_stages = subgroup_size_control.requiredSubgroupSizeStages;
	graphics.compute_subgroup_size_control_enabled =
	    supported_features13.subgroupSizeControl == VK_TRUE &&
	    (graphics.required_subgroup_size_stages & vk::ShaderStageFlagBits::eCompute) &&
	    subgroup_size_control.minSubgroupSize <= 64 &&
	    subgroup_size_control.maxSubgroupSize >= 64;
	graphics.subgroup_size_control_enabled =
	    graphics.compute_subgroup_size_control_enabled ||
	    (supported_features13.subgroupSizeControl == VK_TRUE &&
	     subgroup_size_control.minSubgroupSize < subgroup_size_control.maxSubgroupSize);

	LOGF("Vulkan subgroup: default=%u min=%u max=%u stages=0x%08x size_control=%s wave64=%s\n",
	     graphics.subgroup_size, graphics.min_subgroup_size, graphics.max_subgroup_size,
	     static_cast<vk::ShaderStageFlags::MaskType>(graphics.required_subgroup_size_stages),
	     graphics.compute_subgroup_size_control_enabled ? "true" : "false",
	     graphics.SupportsComputeWave64() ? "true" : "false");
	graphics.provoking_vertex_last_enabled = provoking_extension && provoking_vertex.provokingVertexLast;
	graphics.attachment_feedback_loop_enabled =
	    feedback_extensions && feedback_layout.attachmentFeedbackLoopLayout &&
	    feedback_dynamic.attachmentFeedbackLoopDynamicState;
	LOGF("Vulkan depth feedback support: %s\n",
	     graphics.attachment_feedback_loop_enabled ? "true" : "false");
	if (graphics.mesh_shader_enabled) {
		LOGF("Vulkan MeshEXT: invocations=%u vertices=%u primitives=%u shared=%u\n",
		     graphics.mesh_shader_properties.maxMeshWorkGroupInvocations,
		     graphics.mesh_shader_properties.maxMeshOutputVertices,
		     graphics.mesh_shader_properties.maxMeshOutputPrimitives,
		     graphics.mesh_shader_properties.maxMeshSharedMemorySize);
	}
	PrintGpuCapabilities(graphics, properties2.properties, properties11, properties12,
	                     subgroup_size_control, supported_features2.features, supported_features12,
	                     supported_features13, robustness2_ext_enabled ? &supported_robustness2 : nullptr,
	                     workgroup_layout_extension &&
	                         supported_workgroup_layout.workgroupMemoryExplicitLayout == VK_TRUE);
	// VulkanFindPhysicalDevice already checked the required creation features. These
	// requirements are specific to this creation path and are not part of device selection.
	EXIT_NOT_IMPLEMENTED(supported_features2.features.shaderInt64 != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported_features2.features.vertexPipelineStoresAndAtomics != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported_features2.features.dualSrcBlend != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported_draw_parameters.shaderDrawParameters != VK_TRUE);
	vk::PhysicalDeviceFeatures device_features {};
	device_features.fragmentStoresAndAtomics = VK_TRUE;
	device_features.samplerAnisotropy        = VK_TRUE;
	device_features.robustBufferAccess       = VK_TRUE;
#if !defined(__APPLE__)
	device_features.depthBounds = VK_TRUE; // unsupported by MoltenVK
	device_features.depthClamp  = VK_TRUE;
#endif
	device_features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
	// Optional: TileManager::TileFromImage reads native images through format-less storage views.
	device_features.shaderStorageImageReadWithoutFormat =
	    supported_features2.features.shaderStorageImageReadWithoutFormat;
	graphics.storage_image_read_without_format_enabled =
	    device_features.shaderStorageImageReadWithoutFormat == VK_TRUE;
	device_features.shaderImageGatherExtended            = VK_TRUE;
	device_features.shaderResourceMinLod = shader_resource_min_lod ? VK_TRUE : VK_FALSE;
	device_features.independentBlend                     = VK_TRUE;
	device_features.dualSrcBlend                         = VK_TRUE;
	device_features.tessellationShader                   = VK_TRUE;
	device_features.sampleRateShading                    = VK_TRUE;
	device_features.depthBiasClamp                       = VK_TRUE;
	device_features.shaderClipDistance                   = VK_TRUE;
	device_features.shaderCullDistance                   = VK_TRUE;
	device_features.largePoints                          = VK_TRUE;
	device_features.multiViewport                        = VK_TRUE;
	device_features.fillModeNonSolid                      = VK_TRUE;
	device_features.vertexPipelineStoresAndAtomics       = VK_TRUE;
	graphics.sample_rate_shading_enabled                 = true;
	device_features.shaderInt64 = VK_TRUE;
	device_features.occlusionQueryPrecise = supported_features2.features.occlusionQueryPrecise;
	device_features.drawIndirectFirstInstance =
	    graphics.draw_indirect_first_instance_enabled ? VK_TRUE : VK_FALSE;
	device_features.multiDrawIndirect = graphics.multi_draw_indirect_enabled ? VK_TRUE : VK_FALSE;
	graphics.precise_occlusion_enabled = device_features.occlusionQueryPrecise == VK_TRUE;
	// KYTY_BDA_PAGETABLE_SPARSE=1 (default off): the BDA page table (BufferCache, 512 MiB for the
	// 40-bit guest address space, almost all of it zero) becomes a sparse residency buffer with
	// memory only behind the entries of registered buffers. Needs unbound ranges to read as zero.
	if (const auto* sparse = std::getenv("KYTY_BDA_PAGETABLE_SPARSE");
	    sparse != nullptr && std::strcmp(sparse, "1") == 0) {
		uint32_t count = 0;
		physical_device.getQueueFamilyProperties(&count, nullptr);
		std::vector<vk::QueueFamilyProperties> families(count);
		physical_device.getQueueFamilyProperties(&count, families.data());
		const bool queue_binds = queue_family < count &&
		                         static_cast<bool>(families[queue_family].queueFlags &
		                                           vk::QueueFlagBits::eSparseBinding);
		const auto strict = physical_device.getProperties().sparseProperties.residencyNonResidentStrict;
		graphics.sparse_residency_buffer_enabled =
		    supported_features2.features.sparseBinding == VK_TRUE &&
		    supported_features2.features.sparseResidencyBuffer == VK_TRUE && strict == VK_TRUE &&
		    queue_binds;
		LOGF("BDA page table sparse residency (KYTY_BDA_PAGETABLE_SPARSE): %s (sparseBinding=%u "
		     "sparseResidencyBuffer=%u residencyNonResidentStrict=%u queue=%u)\n",
		     graphics.sparse_residency_buffer_enabled ? "on" : "unavailable",
		     static_cast<uint32_t>(supported_features2.features.sparseBinding),
		     static_cast<uint32_t>(supported_features2.features.sparseResidencyBuffer),
		     static_cast<uint32_t>(strict), queue_binds ? 1u : 0u);
		if (graphics.sparse_residency_buffer_enabled) {
			device_features.sparseBinding         = VK_TRUE;
			device_features.sparseResidencyBuffer = VK_TRUE;
		}
	}
	// KYTY_TEXTURE_SPARSE_RESIDENCY=1 (default off): a partially resident texture (only the levels its
	// views can sample are resident, KYTY_TEXTURE_RESIDENT_MIPS) is a sparse residency image with
	// memory only behind those levels instead of a full native mip chain (GraphicContext).
	if (const auto* sparse = std::getenv("KYTY_TEXTURE_SPARSE_RESIDENCY");
	    sparse != nullptr && std::strcmp(sparse, "1") == 0) {
		uint32_t count = 0;
		physical_device.getQueueFamilyProperties(&count, nullptr);
		std::vector<vk::QueueFamilyProperties> families(count);
		physical_device.getQueueFamilyProperties(&count, families.data());
		const bool queue_binds = queue_family < count &&
		                         static_cast<bool>(families[queue_family].queueFlags &
		                                           vk::QueueFlagBits::eSparseBinding);
		const auto sparse_properties = physical_device.getProperties().sparseProperties;
		graphics.sparse_residency_image_enabled =
		    supported_features2.features.sparseBinding == VK_TRUE &&
		    supported_features2.features.sparseResidencyImage2D == VK_TRUE &&
		    sparse_properties.residencyNonResidentStrict == VK_TRUE && queue_binds;
		LOGF("Texture sparse residency (KYTY_TEXTURE_SPARSE_RESIDENCY): %s (sparseBinding=%u "
		     "sparseResidencyImage2D=%u residencyNonResidentStrict=%u standard2DBlockShape=%u queue=%u)\n",
		     graphics.sparse_residency_image_enabled ? "on" : "unavailable",
		     static_cast<uint32_t>(supported_features2.features.sparseBinding),
		     static_cast<uint32_t>(supported_features2.features.sparseResidencyImage2D),
		     static_cast<uint32_t>(sparse_properties.residencyNonResidentStrict),
		     static_cast<uint32_t>(sparse_properties.residencyStandard2DBlockShape), queue_binds ? 1u : 0u);
		if (graphics.sparse_residency_image_enabled) {
			device_features.sparseBinding          = VK_TRUE;
			device_features.sparseResidencyImage2D = VK_TRUE;
		}
	}
	// Native FP64 arithmetic (upstream 16b83a034) declares SignedZeroInfNanPreserve for 64-bit
	// floats; 32-bit rounding stays native, like ordinary FP32 arithmetic (upstream 7992aecb7).
	device_features.shaderFloat64 =
	    supported_features2.features.shaderFloat64 &&
	    properties12.shaderSignedZeroInfNanPreserveFloat64;

	vk::PhysicalDeviceRobustness2FeaturesEXT robustness2 {};
#if defined(__APPLE__)
	robustness2.pNext = &features12;
#else
	vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR fragment_barycentric {};
	fragment_barycentric.pNext                     = &features12;
	fragment_barycentric.fragmentShaderBarycentric = VK_TRUE;
	robustness2.pNext                              = &fragment_barycentric;
#endif
	if (robustness2_ext_enabled) {
		robustness2.robustBufferAccess2 = supported_robustness2.robustBufferAccess2;
		robustness2.robustImageAccess2  = supported_robustness2.robustImageAccess2;
		robustness2.nullDescriptor      = supported_robustness2.nullDescriptor;
	}

	auto features13 = WindowContext::RequiredVulkan13Features();
#if defined(__APPLE__)
	features13.pNext = robustness2_ext_enabled ? static_cast<void*>(&robustness2)
	                                           : static_cast<void*>(&features12);
#else
	features13.pNext = robustness2_ext_enabled ? static_cast<void*>(&robustness2)
	                                           : static_cast<void*>(&fragment_barycentric);
#endif
	features13.robustImageAccess   = supported_features13.robustImageAccess;
	features13.subgroupSizeControl = graphics.subgroup_size_control_enabled ? VK_TRUE : VK_FALSE;
	features13.pipelineCreationCacheControl =
	    graphics.pipeline_creation_cache_control_enabled ? VK_TRUE : VK_FALSE;

	LOGF("Vulkan robustness: robustImageAccess=%s robustImageAccess2=%s\n",
	     features13.robustImageAccess == VK_TRUE ? "true" : "false",
	     robustness2_ext_enabled && robustness2.robustImageAccess2 == VK_TRUE ? "true" : "false");

	vk::DeviceCreateInfo create_info {};
	vk::PhysicalDeviceMeshShaderFeaturesEXT mesh_features {};
	mesh_features.pNext                 = &features13;
	mesh_features.meshShader            = graphics.mesh_shader_enabled;
	feedback_dynamic.pNext =
	    mesh_extension ? static_cast<void*>(&mesh_features) : static_cast<void*>(&features13);
	create_info.pNext = graphics.attachment_feedback_loop_enabled
	                        ? static_cast<void*>(&feedback_layout)
	                        : feedback_dynamic.pNext;
	if (graphics.provoking_vertex_last_enabled) {
		provoking_vertex.pNext = const_cast<void*>(create_info.pNext);
		provoking_vertex.transformFeedbackPreservesProvokingVertex = VK_FALSE;
		create_info.pNext = &provoking_vertex;
	}
	features12.drawIndirectCount = graphics.draw_indirect_count_enabled ? VK_TRUE : VK_FALSE;
	features12.samplerFilterMinmax = graphics.sampler_filter_minmax_enabled ? VK_TRUE : VK_FALSE;
	vk::PhysicalDeviceIndexTypeUint8FeaturesKHR index_type_uint8 {};
	if (graphics.index_type_uint8_enabled) {
		index_type_uint8.indexTypeUint8 = VK_TRUE;
		index_type_uint8.pNext          = const_cast<void*>(create_info.pNext);
		create_info.pNext               = &index_type_uint8;
	}
	vk::DeviceDiagnosticsConfigCreateInfoNV diagnostics_config {};
	vk::PhysicalDeviceDiagnosticsConfigFeaturesNV diagnostics_features {};
	if (HasExtension(device_extensions, VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME)) {
		vk::PhysicalDeviceDiagnosticsConfigFeaturesNV supported_diagnostics {};
		vk::PhysicalDeviceFeatures2                   diagnostics_query {};
		diagnostics_query.pNext = &supported_diagnostics;
		physical_device.getFeatures2(&diagnostics_query);
		if (supported_diagnostics.diagnosticsConfig) {
			diagnostics_config.flags =
			    vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderDebugInfo |
			    vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableResourceTracking |
			    vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderErrorReporting;
			diagnostics_config.pNext = const_cast<void*>(create_info.pNext);
			diagnostics_features.diagnosticsConfig = VK_TRUE;
			diagnostics_features.pNext = &diagnostics_config;
			create_info.pNext = &diagnostics_features;
		}
	}
	vk::PhysicalDeviceFaultFeaturesEXT device_fault {};
	if (HasExtension(device_extensions, VK_EXT_DEVICE_FAULT_EXTENSION_NAME)) {
		vk::PhysicalDeviceFaultFeaturesEXT supported_fault {};
		vk::PhysicalDeviceFeatures2        fault_query {};
		fault_query.pNext = &supported_fault;
		physical_device.getFeatures2(&fault_query);
		if (supported_fault.deviceFault) {
			device_fault.deviceFault      = VK_TRUE;
			device_fault.deviceFaultVendorBinary = supported_fault.deviceFaultVendorBinary;
			device_fault.pNext            = const_cast<void*>(create_info.pNext);
			create_info.pNext             = &device_fault;
			graphics.device_fault_enabled = true;
		}
	}
	vk::PhysicalDeviceMaintenance8FeaturesKHR maintenance8 {};
	if (graphics.maintenance8_enabled) {
		maintenance8.maintenance8 = VK_TRUE;
		maintenance8.pNext        = const_cast<void*>(create_info.pNext);
		create_info.pNext         = &maintenance8;
	}
	vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT pipeline_library {};
	if (graphics.pipeline_library_enabled) {
		pipeline_library.graphicsPipelineLibrary = VK_TRUE;
		pipeline_library.pNext                   = const_cast<void*>(create_info.pNext);
		create_info.pNext                        = &pipeline_library;
	}
	if (shader_clock.shaderDeviceClock == VK_TRUE || shader_clock.shaderSubgroupClock == VK_TRUE) {
		shader_clock.pNext = const_cast<void*>(create_info.pNext);
		create_info.pNext  = &shader_clock;
	}
	if (compute_derivatives.computeDerivativeGroupQuads == VK_TRUE) {
		compute_derivatives.pNext = const_cast<void*>(create_info.pNext);
		create_info.pNext         = &compute_derivatives;
	}
	if (graphics.shader_image_int64_atomics_enabled) {
		image_atomic_int64.pNext = const_cast<void*>(create_info.pNext);
		image_atomic_int64.sparseImageInt64Atomics = VK_FALSE;
		create_info.pNext = &image_atomic_int64;
	}
	create_info.pQueueCreateInfos       = queue_create_infos.data();
	create_info.queueCreateInfoCount    = queue_create_count;
	create_info.enabledExtensionCount   = static_cast<uint32_t>(device_extensions.size());
	create_info.ppEnabledExtensionNames = device_extensions.data();
	create_info.pEnabledFeatures        = &device_features;

	vk::Device device = nullptr;

	auto result = physical_device.createDevice(&create_info, nullptr, &device);
	if (result != vk::Result::eSuccess) {
		LOGF("vkCreateDevice failed: %s\n", vk::to_string(result).c_str());
		return nullptr;
	}

	if (DeviceFaultDiagnosticsEnabled()) {
		std::printf("Device-fault diagnostics: EXT fault=%u vendor binary=%u NV checkpoints=%u NV tracking/debug=%u\n",
		            static_cast<unsigned>(device_fault.deviceFault),
		            static_cast<unsigned>(device_fault.deviceFaultVendorBinary),
		            graphics.diagnostic_checkpoints_enabled ? 1u : 0u,
		            static_cast<unsigned>(diagnostics_features.diagnosticsConfig));
		std::fflush(stdout);
	}
	return device;
}

static void VulkanGetExtensions(VulkanExtensions& r) {
	uint32_t required_extensions_count = 0;

	const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&required_extensions_count);
	EXIT_NOT_IMPLEMENTED(extensions == nullptr);
	EXIT_NOT_IMPLEMENTED(required_extensions_count == 0);
	r.required_extensions.assign(extensions, extensions + required_extensions_count);

	r.available_extensions =
	    EnumerateVulkan<vk::ExtensionProperties>( // @suppress("Ambiguous problem")
	        "vkEnumerateInstanceExtensionProperties",
	        [](uint32_t* count, vk::ExtensionProperties* values) {
		        return vk::enumerateInstanceExtensionProperties(nullptr, count, values);
	        });

	r.enable_validation_layers = Config::VulkanValidationEnabled();

	if (HasExtension(r.available_extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
		r.required_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	} else {
		r.enable_validation_layers = false;
	}

	for (const char* ext: r.required_extensions) {
		LOGF("Vulkan required extension: %s\n", ext);
	}

	for (const auto& ext: r.available_extensions) {
		LOGF("Vulkan available extension: %s, version = %u\n", ext.extensionName.data(),
		     ext.specVersion);
	}

	r.available_layers = EnumerateVulkan<vk::LayerProperties>( // @suppress("Ambiguous problem")
	    "vkEnumerateInstanceLayerProperties", [](uint32_t* count, vk::LayerProperties* values) {
		    return vk::enumerateInstanceLayerProperties(count, values);
	    });

	for (const auto& l: r.available_layers) {
		LOGF("Vulkan available layer: %s, specVersion = %u, implVersion = %u, %s\n",
		     l.layerName.data(), l.specVersion, l.implementationVersion, l.description.data());
	}

	r.required_layers = {"VK_LAYER_KHRONOS_validation"};

	if (r.enable_validation_layers) {
		for (const char* l: r.required_layers) {
			if (!HasLayer(r.available_layers, l)) {
				LOGF("no validation layer: %s\n", l);
				r.enable_validation_layers = false;
				break;
			}
		}
	}

	if (r.enable_validation_layers) {
		auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
		    "vkEnumerateInstanceExtensionProperties",
		    [](uint32_t* count, vk::ExtensionProperties* values) {
			    return vk::enumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation",
			                                                    count, values);
		    });

		for (const auto& ext: available_extensions) {
			LOGF("VK_LAYER_KHRONOS_validation available extension: %s, version = %u\n",
			     ext.extensionName.data(), ext.specVersion);
		}

		if (HasExtension(available_extensions, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME)) {
			r.required_extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
		} else {
			r.enable_validation_layers = false;
		}
	}
}

// Validation errors and warnings are recorded in a file (KYTY_VULKAN_VALIDATION_LOG, default
// _kyty_vulkan_validation.log) and the game keeps running: users turn the launcher's validation
// option on while troubleshooting, and a known message would otherwise stop every game at boot.
// KYTY_VULKAN_VALIDATION_MODE=exit restores exiting on the first validation error. Each message
// id is written in full for its first five occurrences, then as a count at every power of two.
static bool VulkanValidationLogOnly() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_VULKAN_VALIDATION_MODE");
		return value == nullptr || std::strcmp(value, "exit") != 0;
	}();
	return enabled;
}

static void RecordVulkanValidationMessage(const char*                                   severity,
                                          const vk::DebugUtilsMessengerCallbackDataEXT* data) {
	static std::mutex                                mutex;
	static std::unordered_map<std::string, uint64_t> counts;
	static std::FILE*                                file = [] {
		const char* path = std::getenv("KYTY_VULKAN_VALIDATION_LOG");
		return std::fopen(path != nullptr && path[0] != 0 ? path : "_kyty_vulkan_validation.log",
		                  "a");
	}();
	// Key on the message id plus its first line with handles and numbers blanked, so the same id
	// raised by different commands (e.g. two unrelated hazards) is reported separately.
	std::string first_line(data->pMessage, std::strcspn(data->pMessage, "\r\n"));
	std::string shape;
	shape.reserve(first_line.size());
	for (size_t i = 0; i < first_line.size(); i++) {
		const char c = first_line[i];
		if (c >= '0' && c <= '9') {
			if (shape.empty() || shape.back() != '#') {
				shape.push_back('#');
			}
			if (c == '0' && i + 1 < first_line.size() && (first_line[i + 1] == 'x')) {
				i++;
			}
			while (i + 1 < first_line.size() &&
			       std::isxdigit(static_cast<unsigned char>(first_line[i + 1])) != 0) {
				i++;
			}
			continue;
		}
		shape.push_back(c);
	}
	const std::string id = std::string(severity) + " " +
	                       (data->pMessageIdName != nullptr ? data->pMessageIdName : "?") + " | " +
	                       shape.substr(0, 160);
	std::scoped_lock lock(mutex);
	if (counts.size() >= 4096 && counts.find(id) == counts.end()) {
		return;
	}
	const auto count = ++counts[id];
	if (file == nullptr) {
		return;
	}
	if (count <= 5) {
		std::fprintf(file, "[%s] occurrence %llu\n%s\n\n", id.c_str(),
		             static_cast<unsigned long long>(count), data->pMessage);
	} else if ((count & (count - 1)) == 0) {
		std::fprintf(file, "[%s] repeated %llu times\n", id.c_str(),
		             static_cast<unsigned long long>(count));
	}
	std::fflush(file);
}

static VKAPI_ATTR vk::Bool32 VKAPI_CALL VulkanDebugMessengerCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT      message_severity,
    vk::DebugUtilsMessageTypeFlagsEXT             message_types,
    const vk::DebugUtilsMessengerCallbackDataEXT* callback_data, void* /*user_data*/) {
	EXIT_IF(callback_data == nullptr);
	EXIT_IF(callback_data->pMessage == nullptr);

	if (VulkanValidationLogOnly() &&
	    (message_severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
	     message_severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)) {
		static std::once_flag announce;
		std::call_once(announce, [] {
			LOGF("Vulkan validation: errors and warnings are written to the validation log file and "
			     "do not stop the game (KYTY_VULKAN_VALIDATION_MODE=exit stops at the first error); "
			     "validation makes the game much slower\n");
		});
		RecordVulkanValidationMessage(
		    message_severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ? "E" : "W",
		    callback_data);
		return VK_FALSE;
	}

	const char*     severity_str   = nullptr;
	fmt::text_style severity_style = Log::Color::Default;
	bool            skip           = false;
	bool            error          = false;
	bool            debug_printf   = false;
	switch (message_severity) {
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
			severity_str   = "V";
			severity_style = Log::Color::BrightWhite;
			skip           = true;
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
			if ((message_types & vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation) &&
			    Config::SpirvDebugPrintfEnabled() && callback_data->pMessageIdName != nullptr &&
			    strcmp(callback_data->pMessageIdName, "UNASSIGNED-DEBUG-PRINTF") == 0) {
				debug_printf   = true;
				severity_style = Log::Color::BrightYellow;
				skip           = true;
			} else {
				severity_str   = "I";
				severity_style = Log::Color::Default;
				skip           = true;
			}
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
			severity_str   = "W";
			severity_style = Log::Color::Red;
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
			severity_str   = "E";
			severity_style = Log::Color::BrightRed;
			// Only validation errors are fatal; GENERAL-type errors can come
			// from unrelated loader/layer issues (e.g. a broken overlay).
			error = static_cast<bool>(message_types &
			                          vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
			break;
		default: severity_str = "?";
	}

	if (error) {
		EXIT_COLOR(severity_style, "[Vulkan][%s][%u]: %s\n", severity_str,
		           static_cast<uint32_t>(message_types), callback_data->pMessage);
	}

	if (!skip) {
		LOGF_COLOR(severity_style, "[Vulkan][%s][%u]: %s\n", severity_str,
		           static_cast<uint32_t>(message_types), callback_data->pMessage);
	}

	if (debug_printf) {
		auto strs = Common::Split(std::string(callback_data->pMessage), '|');
		if (!strs.empty()) {
			LOGF_COLOR(severity_style, "%s\n", strs[strs.size() - 1].c_str());
		}
	}

	return VK_FALSE;
}

static VKAPI_ATTR vk::Result VKAPI_CALL VulkanCreateDebugUtilsMessengerEXT(
    vk::Instance instance, const vk::DebugUtilsMessengerCreateInfoEXT* create_info,
    const vk::AllocationCallbacks* allocator, vk::DebugUtilsMessengerEXT* messenger) {
	EXIT_IF(instance == nullptr);

	if (auto func = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDebugUtilsMessengerEXT; func != nullptr) {
		return instance.createDebugUtilsMessengerEXT(create_info, allocator, messenger);
	}
	return vk::Result::eErrorExtensionNotPresent;
}

static void VulkanCheckInstanceVersion() {
	uint32_t version = VK_API_VERSION_1_0;

	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumerateInstanceVersion != nullptr) {
		auto result = vk::enumerateInstanceVersion(&version);
		if (result != vk::Result::eSuccess) {
			EXIT("Could not query Vulkan loader version: %s\n", vk::to_string(result).c_str());
		}
	}

	LOGF("Vulkan loader version: %u.%u.%u\n", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),
	     VK_VERSION_PATCH(version));
	if (version < VULKAN_TARGET_API_VERSION) {
		EXIT("Vulkan %u.%u is required, but loader supports only %u.%u.%u\n",
		     VK_VERSION_MAJOR(VULKAN_TARGET_API_VERSION),
		     VK_VERSION_MINOR(VULKAN_TARGET_API_VERSION), VK_VERSION_MAJOR(version),
		     VK_VERSION_MINOR(version), VK_VERSION_PATCH(version));
	}
}

void WindowContext::CreateVulkan() {
	EXIT_IF(window == nullptr);
	EXIT_IF(graphic_ctx.instance != nullptr);
	EXIT_IF(graphic_ctx.physical_device != nullptr);
	EXIT_IF(graphic_ctx.device != nullptr);
	EXIT_IF(surface != nullptr);

	auto get_instance_proc_addr =
	    reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
	if (get_instance_proc_addr == nullptr) {
		EXIT("Could not load Vulkan: %s\n", SDL_GetError());
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(get_instance_proc_addr);

	VulkanExtensions r;
	VulkanGetExtensions(r);
	VulkanCheckInstanceVersion();

	vk::ApplicationInfo app_info {};
	app_info.pApplicationName   = "Kyty";
	app_info.applicationVersion = 1;
	app_info.pEngineName        = "Kyty";
	app_info.engineVersion      = 1;
	app_info.apiVersion         = VULKAN_TARGET_API_VERSION; // NOLINT

	if (Config::SpirvDebugPrintfEnabled() && Config::GpuAssistedValidationEnabled()) {
		EXIT("--spirv-debug-printf and --gpu-assisted-validation are mutually exclusive\n");
	}

	vk::ValidationFeatureEnableEXT enabled_features[4]    = {};
	uint32_t                       enabled_features_count = 0;
	// KYTY_VULKAN_SYNC_VALIDATION=1: synchronization validation (missing or wrong barriers).
	if (const char* sync = std::getenv("KYTY_VULKAN_SYNC_VALIDATION"); sync != nullptr && sync[0] == '1') {
		enabled_features[enabled_features_count++] =
		    vk::ValidationFeatureEnableEXT::eSynchronizationValidation;
	}
#ifdef KYTY_ENABLE_BEST_PRACTICES
	enabled_features[enabled_features_count++] = vk::ValidationFeatureEnableEXT::eBestPractices;
#endif
#ifdef KYTY_ENABLE_DEBUG_PRINTF
	if (Config::SpirvDebugPrintfEnabled()) {
		enabled_features[enabled_features_count++] = vk::ValidationFeatureEnableEXT::eDebugPrintf;
	}
#endif
	if (Config::GpuAssistedValidationEnabled()) {
		enabled_features[enabled_features_count++] = vk::ValidationFeatureEnableEXT::eGpuAssisted;
		LOGF("Vulkan GPU-assisted validation is enabled; expect a large slowdown\n");
		std::printf("Vulkan GPU-assisted validation is enabled; expect a large slowdown\n");
		std::fflush(stdout);
	}

	vk::ValidationFeaturesEXT validation_features {};
	validation_features.enabledValidationFeatureCount  = enabled_features_count;
	validation_features.pEnabledValidationFeatures     = enabled_features;

	vk::DebugUtilsMessengerCreateInfoEXT dbg_create_info {};
	dbg_create_info.pNext           = &validation_features;
	dbg_create_info.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
	dbg_create_info.messageType     = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
	                                  vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
	                                  vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
	dbg_create_info.pfnUserCallback = VulkanDebugMessengerCallback;

	vk::InstanceCreateInfo inst_info {};
	inst_info.pNext                   = (r.enable_validation_layers ? &dbg_create_info : nullptr);
#if defined(__APPLE__)
	// MoltenVK requires VK_KHR_portability_enumeration + flag to surface
	// portability devices. Without this, enumeratePhysicalDevices hides the
	// MoltenVK adapter on some driver versions.
	if (HasExtension(r.available_extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
		if (!HasExtension(r.required_extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
			r.required_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		}
		inst_info.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
		LOGF("Vulkan instance: enabled %s\n", VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
	}
#endif
	inst_info.pApplicationInfo        = &app_info;
	inst_info.enabledExtensionCount   = static_cast<uint32_t>(r.required_extensions.size());
	inst_info.ppEnabledExtensionNames = r.required_extensions.data();
	inst_info.enabledLayerCount =
	    (r.enable_validation_layers ? static_cast<uint32_t>(r.required_layers.size()) : 0);
	inst_info.ppEnabledLayerNames =
	    (r.enable_validation_layers ? r.required_layers.data() : nullptr);

	const vk::Result result = vk::createInstance(&inst_info, nullptr, &graphic_ctx.instance);
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eErrorIncompatibleDriver:
			EXIT("Unable to find a compatible Vulkan Driver");
		default: EXIT("Could not create a Vulkan instance (for unknown reasons)");
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(graphic_ctx.instance);

	if (r.enable_validation_layers) {
		dbg_create_info.pNext = nullptr;
		if (VulkanCreateDebugUtilsMessengerEXT(graphic_ctx.instance, &dbg_create_info, nullptr,
		                                       &graphic_ctx.debug_messenger) !=
		    vk::Result::eSuccess) {
			EXIT("Could not create debug messenger");
		}
	}

	vk::SurfaceKHR::CType native_surface = VK_NULL_HANDLE;
	if (!SDL_Vulkan_CreateSurface(window, static_cast<vk::Instance::CType>(graphic_ctx.instance),
	                              nullptr, &native_surface)) {
		EXIT("Could not create a Vulkan surface");
	}
	surface = native_surface;

	std::vector<const char*> device_extensions = {
	    VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME,
	    VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
	    "VK_KHR_maintenance1"};

#if defined(__APPLE__)
	// MoltenVK lacks VK_EXT_depth_clip_enable and VK_EXT_color_write_enable; the renderer
	// falls back to default depth clipping and static color-write masks on macOS. It also
	// requires VK_KHR_portability_subset per the Vulkan portability spec.
	device_extensions.push_back("VK_KHR_portability_subset");
#else
	// VK_EXT_depth_clip_enable and VK_EXT_color_write_enable are optional (added below).
	device_extensions.push_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
#endif

#ifdef KYTY_ENABLE_DEBUG_PRINTF
	if (Config::SpirvDebugPrintfEnabled()) {
		device_extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
	}
#endif

	std::string rejected_devices;
	VulkanFindPhysicalDevice(graphic_ctx.instance, surface, device_extensions, surface_capabilities,
	                         graphic_ctx.physical_device, graphic_ctx.queue_family,
	                         rejected_devices);

	if (graphic_ctx.physical_device == nullptr) {
		EXIT("Could not find suitable device:%s", rejected_devices.c_str());
	}

	vk::PhysicalDevicePushDescriptorProperties push_descriptor_properties {};
	vk::PhysicalDeviceProperties2              physical_device_properties {};
	physical_device_properties.pNext = &push_descriptor_properties;
	graphic_ctx.physical_device.getProperties2(&physical_device_properties);
	graphic_ctx.physical_device_properties = physical_device_properties.properties;
	graphic_ctx.max_push_descriptors       = push_descriptor_properties.maxPushDescriptors;
	graphic_ctx.physical_device.getMemoryProperties(&graphic_ctx.physical_device_memory_properties);
	const auto& device_properties = graphic_ctx.GetPhysicalDeviceProperties();

	LOGF("Select device: %s\n", device_properties.deviceName.data());
	if (const auto driver = device_properties.driverVersion; device_properties.vendorID == 0x10de) {
		LOGF("Vulkan driver: NVIDIA %u.%02u\n", (driver >> 22) & 0x3ffu, (driver >> 14) & 0xffu);
	} else {
		LOGF("Vulkan driver: vendor 0x%04x version %u.%u.%u (0x%08x)\n", device_properties.vendorID,
		     VK_VERSION_MAJOR(driver), VK_VERSION_MINOR(driver), VK_VERSION_PATCH(driver), driver);
	}
	switch (HangWatchdog::ResolveAutoForDevice(device_properties.vendorID, device_properties.deviceID,
	                                           device_properties.deviceName.data())) {
		case HangWatchdog::AutoResult::On:
			LOGF("Kyty hang watchdog: on for this RTX 50 GPU (KYTY_HANG_WATCHDOG=auto); if the picture stops "
			     "for 5 s it writes _HangTrace\\watchdog-*\\watchdog.txt\n");
			break;
		case HangWatchdog::AutoResult::Off:
			LOGF("Kyty hang watchdog: off, not an RTX 50 GPU (KYTY_HANG_WATCHDOG=auto)\n");
			break;
		case HangWatchdog::AutoResult::NotAuto: break;
	}

	const vk::PhysicalDeviceImageFormatInfo2 block_texel_view_info {
	    .format = vk::Format::eBc1RgbaUnormBlock,
	    .type = vk::ImageType::e2D,
	    .tiling = vk::ImageTiling::eOptimal,
	    .usage = vk::ImageUsageFlagBits::eSampled,
	    .flags = vk::ImageCreateFlagBits::eBlockTexelViewCompatible,
	};
	const auto block_texel_view_props =
	    graphic_ctx.physical_device.getImageFormatProperties2(block_texel_view_info);
	graphic_ctx.supports_block_texel_view = block_texel_view_props.result == vk::Result::eSuccess;
	LOGF("Block Texel View support: %s\n", graphic_ctx.supports_block_texel_view ? "Yes" : "No");

	{
		auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
		    "vkEnumerateDeviceExtensionProperties",
		    [&](uint32_t* count, vk::ExtensionProperties* values) {
			    return graphic_ctx.physical_device.enumerateDeviceExtensionProperties(
			        nullptr, count, values);
		    });

		if (HasExtension(available_extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
			device_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
			graphic_ctx.memory_budget_ext_enabled = true;
		}
		// S_MEMREALTIME reads a shader clock (KYTY_REALTIME_CLOCK, default on; VulkanCreateDevice).
		if (ShaderRecompiler::GetCodegenOptions().realtime_clock &&
		    HasExtension(available_extensions, VK_KHR_SHADER_CLOCK_EXTENSION_NAME)) {
			device_extensions.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
		}
		if (DeviceFaultDiagnosticsEnabled()) {
		if (HasExtension(available_extensions, VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME)) {
			device_extensions.push_back(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
			graphic_ctx.diagnostic_checkpoints_enabled = true;
		}
		if (HasExtension(available_extensions, VK_EXT_DEVICE_FAULT_EXTENSION_NAME)) {
			device_extensions.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
		}
		if (HasExtension(available_extensions, VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME)) {
			device_extensions.push_back(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME);
		}
		}
		for (const auto* extension: {VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
		                             VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME,
		                             VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME,
		                             VK_EXT_MESH_SHADER_EXTENSION_NAME,
		                             VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME,
		                             VK_EXT_DEPTH_RANGE_UNRESTRICTED_EXTENSION_NAME}) {
			if (HasExtension(available_extensions, extension)) {
				device_extensions.push_back(extension);
			}
		}
#if !defined(__APPLE__)
		// Dynamic colour-write enables and the depth-clip switch: without them (older drivers) the
		// renderer uses its MoltenVK path (GraphicContext::color_write_enable_enabled and
		// depth_clip_enable_enabled).
		for (const auto* extension:
		     {VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME, VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME}) {
			if (HasExtension(available_extensions, extension)) {
				device_extensions.push_back(extension);
			}
		}
#endif
		// Derivatives in compute shaders (IMAGE_GET_LOD); the NV extension has the same feature.
		if (HasExtension(available_extensions, VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME)) {
			device_extensions.push_back(VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
		} else if (HasExtension(available_extensions,
		                        VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME)) {
			device_extensions.push_back(VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
		}
		// Native 8-bit index buffers for indirect draws; the KHR and EXT features are identical.
		if (HasExtension(available_extensions, VK_KHR_INDEX_TYPE_UINT8_EXTENSION_NAME)) {
			device_extensions.push_back(VK_KHR_INDEX_TYPE_UINT8_EXTENSION_NAME);
		} else if (HasExtension(available_extensions, VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME)) {
			device_extensions.push_back(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME);
		}
		// Direct depth <-> color image copies (texture-cache reinterpretation without a staging
		// buffer). KYTY_DIRECT_IMAGE_COPY=0 or KYTY_DIRECT_IMAGE_COPY_M8=0 leaves it disabled.
		if (HasExtension(available_extensions, VK_KHR_MAINTENANCE_8_EXTENSION_NAME)) {
			const auto* all = std::getenv("KYTY_DIRECT_IMAGE_COPY");
			const auto* m8  = std::getenv("KYTY_DIRECT_IMAGE_COPY_M8");
			if ((all == nullptr || std::strcmp(all, "0") != 0) &&
			    (m8 == nullptr || std::strcmp(m8, "0") != 0)) {
				device_extensions.push_back(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
			}
		}
		// Graphics pipeline libraries (KYTY_PIPELINE_LIBRARY, pipeline/pipelineLibrary.h): only
		// enabled when requested, so the default device is unchanged.
		if (PipelineLibraryRequested() &&
		    HasExtension(available_extensions, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) &&
		    HasExtension(available_extensions, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME)) {
			device_extensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
			device_extensions.push_back(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
		}
		if (HasExtension(available_extensions, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) &&
		    HasExtension(available_extensions, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME)) {
			device_extensions.push_back(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME);
			device_extensions.push_back(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME);
		}
		// KYTY_GPU_TIMING (diagnostic) and KYTY_EOP_TIMESTAMPS=gpu only: maps GPU timestamps to the
		// CPU clock and extends timestamp ordering guarantees across submissions. Not enabled for
		// normal play.
		if (const auto* calibration = GpuTiming::SelectCalibrationExtension(available_extensions);
		    calibration != nullptr) {
			device_extensions.push_back(calibration);
			GpuTiming::NoteCalibrationExtensionEnabled(calibration);
		}
	}

	graphic_ctx.device = VulkanCreateDevice(graphic_ctx, device_extensions);
	if (graphic_ctx.device == nullptr) {
		EXIT("Could not create device");
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(graphic_ctx.device);
	// Diagnostic only (KYTY_GPU_OP_PROFILE / KYTY_GPU_OP_COUNTERS): wraps dispatcher entries.
	GpuOpProfiler::InstallHooks(graphic_ctx);
	// KYTY_CP_RECORDER_VERIFY ownership hooks wrap the GpuOpProfiler's.
	CommandRecorder::InstallVerifyHooks();
	graphic_ctx.device.getQueue(graphic_ctx.queue_family, 0, &graphic_ctx.queue);
	EXIT_IF(graphic_ctx.queue == nullptr);
	if (graphic_ctx.side_queue_index != 0) {
		graphic_ctx.device.getQueue(graphic_ctx.queue_family, graphic_ctx.side_queue_index,
		                            &graphic_ctx.side_queue);
		EXIT_IF(graphic_ctx.side_queue == nullptr);
	}
	if (graphic_ctx.transfer_queue_family != static_cast<uint32_t>(-1)) {
		graphic_ctx.device.getQueue(graphic_ctx.transfer_queue_family, 0,
		                            &graphic_ctx.transfer_queue);
		EXIT_IF(graphic_ctx.transfer_queue == nullptr);
	}

	if (!graphic_ctx.CreateAllocator()) {
		EXIT("Could not create Vulkan memory allocator");
	}

	graphic_ctx.submission_queue.Initialize(graphic_ctx);
	render_context = std::make_unique<RenderContext>(graphic_ctx);
	LibKernel::Memory::InstallGpuResources(render_context.get());
	presenter = std::make_unique<Presenter>(*this);
}

void WindowContext::RefreshSurfaceCapabilities() {
	EXIT_IF(graphic_ctx.physical_device == nullptr || surface == nullptr);
	GetSurfaceCapabilities(graphic_ctx.physical_device, surface, surface_capabilities);
}

void WindowContext::RecreateSurface() {
	EXIT_IF(window == nullptr || graphic_ctx.instance == nullptr);
	if (surface != nullptr) {
		graphic_ctx.instance.destroySurfaceKHR(surface, nullptr);
		surface = nullptr;
	}
	vk::SurfaceKHR::CType native_surface = VK_NULL_HANDLE;
	if (!SDL_Vulkan_CreateSurface(window, static_cast<vk::Instance::CType>(graphic_ctx.instance),
	                              nullptr, &native_surface)) {
		EXIT("Could not recreate the Vulkan surface: %s\n", SDL_GetError());
	}
	surface = native_surface;
}

WindowContext::~WindowContext() {
	ShutdownSystemOverlayInput();
	presenter.reset();
	LibKernel::Memory::InstallGpuResources(nullptr);
	render_context.reset();
	graphic_ctx.submission_queue.Shutdown();

	if (graphic_ctx.device != nullptr) {
		RequireVulkanSuccess(graphic_ctx.device.waitIdle(), "wait for Vulkan device shutdown");
		graphic_ctx.DestroyAllocator();
		graphic_ctx.device.destroy(nullptr);
		graphic_ctx.device = nullptr;
		graphic_ctx.queue  = nullptr;
	}
	if (surface != nullptr) {
		graphic_ctx.instance.destroySurfaceKHR(surface, nullptr);
		surface = nullptr;
	}
	if (graphic_ctx.debug_messenger != nullptr) {
		graphic_ctx.instance.destroyDebugUtilsMessengerEXT(graphic_ctx.debug_messenger, nullptr);
		graphic_ctx.debug_messenger = nullptr;
	}
	if (graphic_ctx.instance != nullptr) {
		graphic_ctx.instance.destroy(nullptr);
		graphic_ctx.instance        = nullptr;
		graphic_ctx.physical_device = nullptr;
	}
	if (window != nullptr) {
		SDL_DestroyWindow(window);
		window = nullptr;
	}
	SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD);
}

} // namespace Libs::Graphics
