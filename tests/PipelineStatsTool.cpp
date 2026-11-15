// Prints the driver's statistics (registers, binary size, spill) of SPIR-V modules, offline.
//
// usage: pipeline_stats_tool <file.spv | folder> [...]
//
// A vertex or mesh shader becomes a graphics pipeline with rasterizer discard and no fragment
// stage (dynamic rendering, no attachments); a compute shader a compute pipeline; a fragment
// shader is reported as "skipped". Descriptor set layouts, push constants and the vertex input
// state come from the module's own declarations. Every pipeline is created with
// VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR (VK_KHR_pipeline_executable_properties) and a fresh
// OpModuleProcessed string, so no driver cache answers. Use: compare the register count of a
// translated shader with edited copies of it (kyty's KYTY_PIPELINE_STATS reports the same numbers
// in game).
//
// A tool, not a test: it needs a GPU and its driver.

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

PFN_vkGetInstanceProcAddr gipa = nullptr;

#define KYTY_FN(name) PFN_##name name##_ = nullptr;
KYTY_FN(vkCreateInstance)
KYTY_FN(vkEnumeratePhysicalDevices)
KYTY_FN(vkGetPhysicalDeviceProperties)
KYTY_FN(vkGetPhysicalDeviceFeatures2)
KYTY_FN(vkEnumerateDeviceExtensionProperties)
KYTY_FN(vkCreateDevice)
KYTY_FN(vkGetDeviceProcAddr)
KYTY_FN(vkDestroyInstance)
KYTY_FN(vkDestroyDevice)
KYTY_FN(vkCreateDescriptorSetLayout)
KYTY_FN(vkDestroyDescriptorSetLayout)
KYTY_FN(vkCreatePipelineLayout)
KYTY_FN(vkDestroyPipelineLayout)
KYTY_FN(vkCreateShaderModule)
KYTY_FN(vkDestroyShaderModule)
KYTY_FN(vkCreateGraphicsPipelines)
KYTY_FN(vkCreateComputePipelines)
KYTY_FN(vkDestroyPipeline)
KYTY_FN(vkGetPipelineExecutablePropertiesKHR)
KYTY_FN(vkGetPipelineExecutableStatisticsKHR)
#undef KYTY_FN

bool LoadLoader() {
#if defined(_WIN32)
	HMODULE library = LoadLibraryA("vulkan-1.dll");
	if (library == nullptr) return false;
	gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(library, "vkGetInstanceProcAddr"));
#else
	void* library = dlopen("libvulkan.so.1", RTLD_NOW);
	if (library == nullptr) return false;
	gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
#endif
	return gipa != nullptr;
}

struct Input {
	uint32_t location = 0;
	VkFormat format   = VK_FORMAT_R32G32B32A32_SFLOAT;
	uint32_t bytes    = 16;
};

struct Module {
	std::vector<uint32_t> words;
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_ALL;
	std::string           entry;
	std::map<uint32_t, std::map<uint32_t, std::pair<VkDescriptorType, uint32_t>>> bindings;
	bool                  push_constants = false;
	std::vector<Input>    inputs;
};

constexpr uint32_t RuntimeArrayCount = 16384;

VkFormat InputFormat(bool is_float, bool is_signed, uint32_t components) {
	static const VkFormat f[4] = {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT,
	                              VK_FORMAT_R32G32B32A32_SFLOAT};
	static const VkFormat s[4] = {VK_FORMAT_R32_SINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32B32_SINT,
	                              VK_FORMAT_R32G32B32A32_SINT};
	static const VkFormat u[4] = {VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32_UINT,
	                              VK_FORMAT_R32G32B32A32_UINT};
	const uint32_t i = std::clamp(components, 1u, 4u) - 1u;
	return is_float ? f[i] : is_signed ? s[i] : u[i];
}

bool Read(const std::filesystem::path& path, Module& module, std::string& error) {
	std::ifstream     file(path, std::ios::binary);
	std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (bytes.size() < 20 || bytes.size() % 4 != 0) {
		error = "not a SPIR-V module";
		return false;
	}
	module.words.resize(bytes.size() / 4);
	std::memcpy(module.words.data(), bytes.data(), bytes.size());
	const auto& w = module.words;
	if (w[0] != 0x07230203u) {
		error = "bad magic";
		return false;
	}
	struct Pointer {
		uint32_t storage = 0;
		uint32_t type    = 0;
	};
	struct Image {
		uint32_t dim     = 0;
		uint32_t sampled = 0;
	};
	struct Scalar {
		bool     is_float  = true;
		bool     is_signed = false;
		uint32_t width     = 32;
	};
	std::map<uint32_t, uint32_t> set_of, binding_of, constant, location_of;
	std::map<uint32_t, bool>     buffer_block, builtin;
	std::map<uint32_t, Pointer>  pointers;
	std::map<uint32_t, std::pair<uint32_t, uint32_t>> arrays;
	std::map<uint32_t, Image>    images;
	std::map<uint32_t, uint32_t> sampled_images;
	std::map<uint32_t, int>      kinds;
	std::map<uint32_t, Scalar>   scalars;
	std::map<uint32_t, std::pair<uint32_t, uint32_t>> vectors; // id -> (component type, count)
	std::vector<std::pair<uint32_t, uint32_t>> variables;
	for (size_t i = 5; i < w.size();) {
		const uint32_t count = w[i] >> 16u;
		const uint32_t op    = w[i] & 0xffffu;
		if (count == 0 || i + count > w.size()) {
			error = "truncated instruction";
			return false;
		}
		const uint32_t* a = &w[i + 1];
		switch (op) {
			case 15:
				if (module.entry.empty()) {
					switch (a[0]) {
						case 0: module.stage = VK_SHADER_STAGE_VERTEX_BIT; break;
						case 4: module.stage = VK_SHADER_STAGE_FRAGMENT_BIT; break;
						case 5: module.stage = VK_SHADER_STAGE_COMPUTE_BIT; break;
						case 5365: module.stage = VK_SHADER_STAGE_MESH_BIT_EXT; break;
						default: error = "unsupported execution model"; return false;
					}
					module.entry = reinterpret_cast<const char*>(&a[2]);
				}
				break;
			case 71:
				if (a[1] == 34) set_of[a[0]] = a[2];
				if (a[1] == 33) binding_of[a[0]] = a[2];
				if (a[1] == 3) buffer_block[a[0]] = true;
				if (a[1] == 30) location_of[a[0]] = a[2];
				if (a[1] == 11) builtin[a[0]] = true;
				break;
			case 21: scalars[a[0]] = {false, a[2] != 0, a[1]}; break; // OpTypeInt
			case 22: scalars[a[0]] = {true, true, a[1]}; break;      // OpTypeFloat
			case 23: vectors[a[0]] = {a[1], a[2]}; break;            // OpTypeVector
			case 32: pointers[a[0]] = {a[1], a[2]}; break;
			case 28: arrays[a[0]] = {a[1], a[2]}; break;
			case 29: arrays[a[0]] = {a[1], 0}; break;
			case 25: images[a[0]] = {a[2], a[6]}; break;
			case 26: kinds[a[0]] = 1; break;
			case 27: sampled_images[a[0]] = a[1]; break;
			case 5341: kinds[a[0]] = 2; break;
			case 43: constant[a[1]] = a[2]; break;
			case 59: variables.emplace_back(a[0], a[1]); break;
			default: break;
		}
		i += count;
	}
	if (module.entry.empty()) {
		error = "no entry point";
		return false;
	}
	for (const auto& [pointer_type, id]: variables) {
		const auto pointer = pointers.find(pointer_type);
		if (pointer == pointers.end()) continue;
		if (pointer->second.storage == 9) {
			module.push_constants = true;
			continue;
		}
		if (pointer->second.storage == 1 && module.stage == VK_SHADER_STAGE_VERTEX_BIT &&
		    location_of.contains(id) && !builtin.contains(id)) {
			uint32_t type = pointer->second.type, components = 1;
			if (vectors.contains(type)) {
				components = vectors[type].second;
				type       = vectors[type].first;
			}
			const auto scalar = scalars[type];
			module.inputs.push_back({location_of[id], InputFormat(scalar.is_float, scalar.is_signed, components),
			                         components * 4u});
			continue;
		}
		if (!set_of.contains(id) || !binding_of.contains(id)) continue;
		uint32_t type  = pointer->second.type;
		uint32_t count = 1;
		while (arrays.contains(type)) {
			const auto [element, length] = arrays[type];
			count *= length == 0 ? RuntimeArrayCount : std::max(constant[length], 1u);
			type = element;
		}
		VkDescriptorType descriptor = VK_DESCRIPTOR_TYPE_MAX_ENUM;
		switch (pointer->second.storage) {
			case 12: descriptor = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; break;
			case 2:
				descriptor = buffer_block.contains(type) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
				                                         : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				break;
			case 0:
				if (images.contains(type)) {
					const auto image  = images[type];
					const bool buffer = image.dim == 5;
					descriptor        = image.sampled == 2
					                        ? (buffer ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
					                                  : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
					                        : (buffer ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
					                                  : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
				} else if (sampled_images.contains(type)) {
					descriptor = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				} else if (kinds[type] == 1) {
					descriptor = VK_DESCRIPTOR_TYPE_SAMPLER;
				} else if (kinds[type] == 2) {
					descriptor = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
				}
				break;
			default: break;
		}
		if (descriptor == VK_DESCRIPTOR_TYPE_MAX_ENUM) {
			error = "unknown descriptor type";
			return false;
		}
		module.bindings[set_of[id]][binding_of[id]] = {descriptor, count};
	}
	return true;
}

// Inserts OpModuleProcessed "<text>" after the debug section.
std::vector<uint32_t> Salted(const std::vector<uint32_t>& words, const std::string& text) {
	size_t i = 5;
	for (;;) {
		const uint32_t op       = words[i] & 0xffffu;
		const bool     preamble = op == 17 || op == 10 || op == 11 || op == 14 || op == 15 || op == 16 ||
		                      op == 331 || op == 7 || op == 4 || op == 3 || op == 2 || op == 5 || op == 6 ||
		                      op == 330;
		if (!preamble) break;
		i += words[i] >> 16u;
	}
	std::vector<uint32_t> operand((text.size() + 4) / 4, 0u);
	std::memcpy(operand.data(), text.data(), text.size());
	std::vector<uint32_t> out(words.begin(), words.begin() + static_cast<std::ptrdiff_t>(i));
	out.push_back(static_cast<uint32_t>((operand.size() + 1) << 16u) | 330u);
	out.insert(out.end(), operand.begin(), operand.end());
	out.insert(out.end(), words.begin() + static_cast<std::ptrdiff_t>(i), words.end());
	out[1] = std::max(out[1], 0x00010100u);
	return out;
}

struct Device {
	VkInstance       instance = VK_NULL_HANDLE;
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	VkDevice         device   = VK_NULL_HANDLE;
};

bool HasExtension(const std::vector<VkExtensionProperties>& list, const char* name) {
	return std::any_of(list.begin(), list.end(), [name](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
}

template <typename T>
void LoadInstance(VkInstance instance, T& function, const char* name) {
	function = reinterpret_cast<T>(gipa(instance, name));
	if (function == nullptr) {
		std::fprintf(stderr, "missing %s\n", name);
		std::exit(1);
	}
}

bool CreateDevice(Device& d) {
	vkCreateInstance_ = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.pApplicationName = "pipeline_stats_tool";
	app.apiVersion       = VK_API_VERSION_1_3;
	VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instance_info.pApplicationInfo = &app;
	if (vkCreateInstance_(&instance_info, nullptr, &d.instance) != VK_SUCCESS) return false;
	LoadInstance(d.instance, vkEnumeratePhysicalDevices_, "vkEnumeratePhysicalDevices");
	LoadInstance(d.instance, vkGetPhysicalDeviceProperties_, "vkGetPhysicalDeviceProperties");
	LoadInstance(d.instance, vkGetPhysicalDeviceFeatures2_, "vkGetPhysicalDeviceFeatures2");
	LoadInstance(d.instance, vkEnumerateDeviceExtensionProperties_, "vkEnumerateDeviceExtensionProperties");
	LoadInstance(d.instance, vkCreateDevice_, "vkCreateDevice");
	LoadInstance(d.instance, vkGetDeviceProcAddr_, "vkGetDeviceProcAddr");
	LoadInstance(d.instance, vkDestroyInstance_, "vkDestroyInstance");
	uint32_t count = 0;
	vkEnumeratePhysicalDevices_(d.instance, &count, nullptr);
	std::vector<VkPhysicalDevice> devices(count);
	vkEnumeratePhysicalDevices_(d.instance, &count, devices.data());
	for (auto device: devices) {
		VkPhysicalDeviceProperties properties {};
		vkGetPhysicalDeviceProperties_(device, &properties);
		if (d.physical == VK_NULL_HANDLE || properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			d.physical = device;
			if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) break;
		}
	}
	if (d.physical == VK_NULL_HANDLE) return false;
	uint32_t extension_count = 0;
	vkEnumerateDeviceExtensionProperties_(d.physical, nullptr, &extension_count, nullptr);
	std::vector<VkExtensionProperties> available(extension_count);
	vkEnumerateDeviceExtensionProperties_(d.physical, nullptr, &extension_count, available.data());
	if (!HasExtension(available, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
		std::fprintf(stderr, "the device has no %s\n", VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
		return false;
	}
	std::vector<const char*> extensions;
	void*                    chain = nullptr;
	auto add = [&](const char* name, auto& features, VkStructureType type) {
		if (name != nullptr && !HasExtension(available, name)) return;
		if (name != nullptr) extensions.push_back(name);
		features       = {};
		features.sType = type;
		features.pNext = chain;
		chain          = &features;
	};
	VkPhysicalDeviceVulkan11Features v11 {};
	VkPhysicalDeviceVulkan12Features v12 {};
	VkPhysicalDeviceVulkan13Features v13 {};
	VkPhysicalDeviceMeshShaderFeaturesEXT mesh {};
	VkPhysicalDeviceRobustness2FeaturesEXT robustness2 {};
	VkPhysicalDeviceShaderClockFeaturesKHR clock {};
	VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric {};
	VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup_layout {};
	VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT image_atomic64 {};
	VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executable {};
	add(nullptr, v11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
	add(nullptr, v12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
	add(nullptr, v13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
	add(VK_EXT_MESH_SHADER_EXTENSION_NAME, mesh, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT);
	add(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, robustness2, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT);
	add(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, clock, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR);
	add(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, barycentric,
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR);
	add(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME, workgroup_layout,
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR);
	add(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, image_atomic64,
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT);
	add(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, executable,
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR);
	if (HasExtension(available, VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)) {
		extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
	}
	VkPhysicalDeviceFeatures2 features {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	features.pNext = chain;
	vkGetPhysicalDeviceFeatures2_(d.physical, &features);
	mesh.multiviewMeshShader                    = VK_FALSE;
	mesh.primitiveFragmentShadingRateMeshShader = VK_FALSE;
	float                   priority = 1.0f;
	VkDeviceQueueCreateInfo queue {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue.queueCount       = 1;
	queue.pQueuePriorities = &priority;
	VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	device_info.pNext                   = &features;
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue;
	device_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
	device_info.ppEnabledExtensionNames = extensions.data();
	if (const auto result = vkCreateDevice_(d.physical, &device_info, nullptr, &d.device); result != VK_SUCCESS) {
		std::fprintf(stderr, "vkCreateDevice: %d\n", static_cast<int>(result));
		return false;
	}
	auto load = [&](auto& function, const char* name) {
		function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(vkGetDeviceProcAddr_(d.device, name));
		if (function == nullptr) {
			std::fprintf(stderr, "missing %s\n", name);
			std::exit(1);
		}
	};
	load(vkDestroyDevice_, "vkDestroyDevice");
	load(vkCreateDescriptorSetLayout_, "vkCreateDescriptorSetLayout");
	load(vkDestroyDescriptorSetLayout_, "vkDestroyDescriptorSetLayout");
	load(vkCreatePipelineLayout_, "vkCreatePipelineLayout");
	load(vkDestroyPipelineLayout_, "vkDestroyPipelineLayout");
	load(vkCreateShaderModule_, "vkCreateShaderModule");
	load(vkDestroyShaderModule_, "vkDestroyShaderModule");
	load(vkCreateGraphicsPipelines_, "vkCreateGraphicsPipelines");
	load(vkCreateComputePipelines_, "vkCreateComputePipelines");
	load(vkDestroyPipeline_, "vkDestroyPipeline");
	load(vkGetPipelineExecutablePropertiesKHR_, "vkGetPipelineExecutablePropertiesKHR");
	load(vkGetPipelineExecutableStatisticsKHR_, "vkGetPipelineExecutableStatisticsKHR");
	return true;
}

std::string Stats(const Device& d, VkPipeline pipeline) {
	VkPipelineInfoKHR info {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
	info.pipeline  = pipeline;
	uint32_t count = 0;
	vkGetPipelineExecutablePropertiesKHR_(d.device, &info, &count, nullptr);
	std::vector<VkPipelineExecutablePropertiesKHR> executables(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
	vkGetPipelineExecutablePropertiesKHR_(d.device, &info, &count, executables.data());
	std::string out;
	for (uint32_t e = 0; e < count; e++) {
		VkPipelineExecutableInfoKHR exec {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
		exec.pipeline        = pipeline;
		exec.executableIndex = e;
		uint32_t n           = 0;
		vkGetPipelineExecutableStatisticsKHR_(d.device, &exec, &n, nullptr);
		std::vector<VkPipelineExecutableStatisticKHR> stats(n, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
		vkGetPipelineExecutableStatisticsKHR_(d.device, &exec, &n, stats.data());
		out += std::string(" [") + executables[e].name;
		for (const auto& s: stats) {
			char text[160];
			switch (s.format) {
				case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
					std::snprintf(text, sizeof(text), " %s=%u", s.name, s.value.b32);
					break;
				case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
					std::snprintf(text, sizeof(text), " %s=%lld", s.name, static_cast<long long>(s.value.i64));
					break;
				case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
					std::snprintf(text, sizeof(text), " %s=%llu", s.name, static_cast<unsigned long long>(s.value.u64));
					break;
				default: std::snprintf(text, sizeof(text), " %s=%.3f", s.name, s.value.f64); break;
			}
			out += text;
		}
		out += "]";
	}
	return out;
}

std::string Build(const Device& d, const Module& module, uint64_t salt) {
	if (module.stage == VK_SHADER_STAGE_FRAGMENT_BIT) return "skipped (fragment)";
	std::vector<VkDescriptorSetLayout> layouts;
	const uint32_t set_count = module.bindings.empty() ? 0u : module.bindings.rbegin()->first + 1u;
	for (uint32_t set = 0; set < set_count; set++) {
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		std::vector<VkDescriptorBindingFlags>     flags;
		if (const auto found = module.bindings.find(set); found != module.bindings.end()) {
			for (const auto& [binding, info]: found->second) {
				bindings.push_back({binding, info.first, info.second, VK_SHADER_STAGE_ALL, nullptr});
				flags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
			}
		}
		VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
		flags_info.bindingCount  = static_cast<uint32_t>(flags.size());
		flags_info.pBindingFlags = flags.data();
		VkDescriptorSetLayoutCreateInfo info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		info.pNext        = &flags_info;
		info.bindingCount = static_cast<uint32_t>(bindings.size());
		info.pBindings    = bindings.data();
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		vkCreateDescriptorSetLayout_(d.device, &info, nullptr, &layout);
		layouts.push_back(layout);
	}
	VkPushConstantRange        push {VK_SHADER_STAGE_ALL, 0, 256};
	VkPipelineLayoutCreateInfo layout_info {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	layout_info.setLayoutCount         = static_cast<uint32_t>(layouts.size());
	layout_info.pSetLayouts            = layouts.data();
	layout_info.pushConstantRangeCount = module.push_constants ? 1u : 0u;
	layout_info.pPushConstantRanges    = &push;
	VkPipelineLayout layout            = VK_NULL_HANDLE;
	vkCreatePipelineLayout_(d.device, &layout_info, nullptr, &layout);
	const auto               code = Salted(module.words, "kyty-stats-" + std::to_string(salt));
	VkShaderModuleCreateInfo module_info {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	module_info.codeSize        = code.size() * sizeof(uint32_t);
	module_info.pCode           = code.data();
	VkShaderModule shader       = VK_NULL_HANDLE;
	vkCreateShaderModule_(d.device, &module_info, nullptr, &shader);
	VkPipelineShaderStageCreateInfo stage {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	stage.stage  = module.stage;
	stage.module = shader;
	stage.pName  = module.entry.c_str();
	VkPipeline pipeline = VK_NULL_HANDLE;
	VkResult   result   = VK_SUCCESS;
	if (module.stage == VK_SHADER_STAGE_COMPUTE_BIT) {
		VkComputePipelineCreateInfo info {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
		info.flags  = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
		info.stage  = stage;
		info.layout = layout;
		result      = vkCreateComputePipelines_(d.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
	} else {
		std::vector<VkVertexInputBindingDescription>   vb;
		std::vector<VkVertexInputAttributeDescription> va;
		for (const auto& input: module.inputs) {
			vb.push_back({static_cast<uint32_t>(vb.size()), input.bytes, VK_VERTEX_INPUT_RATE_VERTEX});
			va.push_back({input.location, static_cast<uint32_t>(vb.size() - 1), input.format, 0});
		}
		VkPipelineVertexInputStateCreateInfo vi {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
		vi.vertexBindingDescriptionCount   = static_cast<uint32_t>(vb.size());
		vi.pVertexBindingDescriptions      = vb.data();
		vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(va.size());
		vi.pVertexAttributeDescriptions    = va.data();
		VkPipelineInputAssemblyStateCreateInfo ia {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
		ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		VkPipelineViewportStateCreateInfo vp {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
		vp.viewportCount = 1;
		vp.scissorCount  = 1;
		VkPipelineRasterizationStateCreateInfo rs {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
		rs.rasterizerDiscardEnable = VK_TRUE;
		rs.lineWidth               = 1.0f;
		VkPipelineMultisampleStateCreateInfo ms {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
		ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		const VkDynamicState             dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo ds {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
		ds.dynamicStateCount = 2;
		ds.pDynamicStates    = dynamic;
		VkPipelineRenderingCreateInfo rendering {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
		VkGraphicsPipelineCreateInfo  info {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
		const bool                    mesh = module.stage == VK_SHADER_STAGE_MESH_BIT_EXT;
		info.pNext               = &rendering;
		info.flags               = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
		info.stageCount          = 1;
		info.pStages             = &stage;
		info.pVertexInputState   = mesh ? nullptr : &vi;
		info.pInputAssemblyState = mesh ? nullptr : &ia;
		info.pViewportState      = &vp;
		info.pRasterizationState = &rs;
		info.pMultisampleState   = &ms;
		info.pDynamicState       = &ds;
		info.layout              = layout;
		result = vkCreateGraphicsPipelines_(d.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
	}
	std::string out = result == VK_SUCCESS && pipeline != VK_NULL_HANDLE ? Stats(d, pipeline)
	                                                                     : "failed " + std::to_string(result);
	if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline_(d.device, pipeline, nullptr);
	vkDestroyShaderModule_(d.device, shader, nullptr);
	vkDestroyPipelineLayout_(d.device, layout, nullptr);
	for (auto l: layouts) vkDestroyDescriptorSetLayout_(d.device, l, nullptr);
	return out;
}

} // namespace

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: pipeline_stats_tool <file.spv | folder> [...]\n");
		return 1;
	}
	std::vector<std::filesystem::path> files;
	for (int i = 1; i < argc; i++) {
		const std::filesystem::path path = argv[i];
		if (std::filesystem::is_directory(path)) {
			for (const auto& entry: std::filesystem::directory_iterator(path)) {
				if (entry.path().extension() == ".spv") files.push_back(entry.path());
			}
		} else {
			files.push_back(path);
		}
	}
	std::sort(files.begin(), files.end());
	if (!LoadLoader()) {
		std::fprintf(stderr, "no Vulkan loader\n");
		return 1;
	}
	Device device;
	if (!CreateDevice(device)) {
		std::fprintf(stderr, "no device\n");
		return 1;
	}
	const auto salt = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
	for (size_t f = 0; f < files.size(); f++) {
		Module      module;
		std::string error;
		if (!Read(files[f], module, error)) {
			std::printf("%s: %s\n", files[f].filename().string().c_str(), error.c_str());
			continue;
		}
		std::printf("%s:%s\n", files[f].filename().string().c_str(), Build(device, module, salt + f).c_str());
		std::fflush(stdout);
	}
	vkDestroyDevice_(device.device, nullptr);
	vkDestroyInstance_(device.instance, nullptr);
	return 0;
}
