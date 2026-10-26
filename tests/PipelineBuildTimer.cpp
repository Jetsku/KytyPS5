// Times how long the Vulkan driver takes to compile SPIR-V modules, offline.
//
// usage: pipeline_build_timer <out.csv> <file.spv | folder> [...] [--repeat N]
//
// Each module is compiled on its own with VK_EXT_shader_object (vkCreateShadersEXT), so a pixel
// shader needs no matching vertex shader and no render state. Descriptor set layouts and the
// push-constant range come from the module's own declarations. Before every compile the module
// gets a fresh OpModuleProcessed "kyty-timer-<n>", so neither the driver's disk cache nor its
// memory answers: every number is a cold compile, as on a first visit in a game. The result is
// the median of --repeat compiles (default 3), one line per module in the CSV.
//
// A tool, not a test: it needs a GPU and its driver, and its numbers depend on both.

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

PFN_vkGetInstanceProcAddr                 vkGetInstanceProcAddr_ = nullptr;
PFN_vkCreateInstance                      vkCreateInstance_      = nullptr;
PFN_vkEnumeratePhysicalDevices            vkEnumeratePhysicalDevices_ = nullptr;
PFN_vkGetPhysicalDeviceProperties         vkGetPhysicalDeviceProperties_ = nullptr;
PFN_vkGetPhysicalDeviceFeatures2          vkGetPhysicalDeviceFeatures2_ = nullptr;
PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties_ = nullptr;
PFN_vkEnumerateDeviceExtensionProperties  vkEnumerateDeviceExtensionProperties_ = nullptr;
PFN_vkCreateDevice                        vkCreateDevice_ = nullptr;
PFN_vkGetDeviceProcAddr                   vkGetDeviceProcAddr_ = nullptr;
PFN_vkCreateDescriptorSetLayout           vkCreateDescriptorSetLayout_ = nullptr;
PFN_vkDestroyDescriptorSetLayout          vkDestroyDescriptorSetLayout_ = nullptr;
PFN_vkCreateShadersEXT                    vkCreateShadersEXT_ = nullptr;
PFN_vkDestroyShaderEXT                    vkDestroyShaderEXT_ = nullptr;
PFN_vkDestroyDevice                       vkDestroyDevice_ = nullptr;
PFN_vkDestroyInstance                     vkDestroyInstance_ = nullptr;

bool LoadLoader() {
#if defined(_WIN32)
	HMODULE library = LoadLibraryA("vulkan-1.dll");
	if (library == nullptr) {
		return false;
	}
	vkGetInstanceProcAddr_ =
	    reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(library, "vkGetInstanceProcAddr"));
#else
	void* library = dlopen("libvulkan.so.1", RTLD_NOW);
	if (library == nullptr) {
		return false;
	}
	vkGetInstanceProcAddr_ =
	    reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
#endif
	return vkGetInstanceProcAddr_ != nullptr;
}

template <typename T>
void Load(VkInstance instance, T& function, const char* name) {
	function = reinterpret_cast<T>(vkGetInstanceProcAddr_(instance, name));
	if (function == nullptr) {
		std::fprintf(stderr, "missing %s\n", name);
		std::exit(1);
	}
}

// --- SPIR-V reading ---------------------------------------------------------------------------

struct Module {
	std::vector<uint32_t> words;
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_ALL;
	std::string           entry;
	// set -> binding -> (type, count)
	std::map<uint32_t, std::map<uint32_t, std::pair<VkDescriptorType, uint32_t>>> bindings;
	bool                  push_constants = false;
};

constexpr uint32_t RuntimeArrayCount = 16384;

bool Read(const std::filesystem::path& path, Module& module, std::string& error) {
	std::ifstream file(path, std::ios::binary);
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
	std::map<uint32_t, uint32_t> set_of, binding_of, constant;
	std::map<uint32_t, bool>     buffer_block;
	std::map<uint32_t, Pointer>  pointers;
	std::map<uint32_t, std::pair<uint32_t, uint32_t>> arrays; // id -> (element, length id or 0)
	std::map<uint32_t, Image>    images;
	std::map<uint32_t, uint32_t> sampled_images; // id -> image type
	std::map<uint32_t, int>      kinds;           // id -> 1 sampler, 2 acceleration structure
	std::vector<std::pair<uint32_t, uint32_t>> variables; // (pointer type, id)

	for (size_t i = 5; i < w.size();) {
		const uint32_t count = w[i] >> 16u;
		const uint32_t op    = w[i] & 0xffffu;
		if (count == 0 || i + count > w.size()) {
			error = "truncated instruction";
			return false;
		}
		const uint32_t* a = &w[i + 1];
		switch (op) {
			case 15: { // OpEntryPoint
				if (module.entry.empty()) {
					switch (a[0]) {
						case 0: module.stage = VK_SHADER_STAGE_VERTEX_BIT; break;
						case 1: module.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT; break;
						case 2: module.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT; break;
						case 3: module.stage = VK_SHADER_STAGE_GEOMETRY_BIT; break;
						case 4: module.stage = VK_SHADER_STAGE_FRAGMENT_BIT; break;
						case 5: module.stage = VK_SHADER_STAGE_COMPUTE_BIT; break;
						case 5364: module.stage = VK_SHADER_STAGE_TASK_BIT_EXT; break;
						case 5365: module.stage = VK_SHADER_STAGE_MESH_BIT_EXT; break;
						default: error = "unknown execution model"; return false;
					}
					module.entry = reinterpret_cast<const char*>(&a[2]);
				}
				break;
			}
			case 71: // OpDecorate
				if (a[1] == 34) set_of[a[0]] = a[2];
				if (a[1] == 33) binding_of[a[0]] = a[2];
				if (a[1] == 3) buffer_block[a[0]] = true;
				break;
			case 32: pointers[a[0]] = {a[1], a[2]}; break;              // OpTypePointer
			case 28: arrays[a[0]] = {a[1], a[2]}; break;                // OpTypeArray
			case 29: arrays[a[0]] = {a[1], 0}; break;                   // OpTypeRuntimeArray
			case 25: images[a[0]] = {a[2], a[6]}; break;                // OpTypeImage
			case 26: kinds[a[0]] = 1; break;                            // OpTypeSampler
			case 27: sampled_images[a[0]] = a[1]; break;                // OpTypeSampledImage
			case 5341: kinds[a[0]] = 2; break;                          // OpTypeAccelerationStructureKHR
			case 43: constant[a[1]] = a[2]; break;                      // OpConstant
			case 59: variables.emplace_back(a[0], a[1]); break;         // OpVariable
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
		if (pointer == pointers.end()) {
			continue;
		}
		if (pointer->second.storage == 9) { // PushConstant
			module.push_constants = true;
			continue;
		}
		if (!set_of.contains(id) || !binding_of.contains(id)) {
			continue;
		}
		uint32_t type  = pointer->second.type;
		uint32_t count = 1;
		while (arrays.contains(type)) {
			const auto [element, length] = arrays[type];
			count *= length == 0 ? RuntimeArrayCount : std::max(constant[length], 1u);
			type = element;
		}
		VkDescriptorType descriptor = VK_DESCRIPTOR_TYPE_MAX_ENUM;
		switch (pointer->second.storage) {
			case 12: descriptor = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; break; // StorageBuffer
			case 2:                                                          // Uniform
				descriptor = buffer_block.contains(type) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
				                                         : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				break;
			case 0: // UniformConstant
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
		const uint32_t op = words[i] & 0xffffu;
		const bool preamble = op == 17 || op == 10 || op == 11 || op == 14 || op == 15 ||
		                      op == 16 || op == 331 || op == 7 || op == 4 || op == 3 || op == 2 ||
		                      op == 5 || op == 6 || op == 330;
		if (!preamble) {
			break;
		}
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

// --- device -------------------------------------------------------------------------------------

struct Device {
	VkInstance       instance = VK_NULL_HANDLE;
	VkPhysicalDevice physical = VK_NULL_HANDLE;
	VkDevice         device   = VK_NULL_HANDLE;
	std::string      name;
	uint32_t         driver_version = 0;
	bool             task_shader    = false;
};

bool HasExtension(const std::vector<VkExtensionProperties>& list, const char* name) {
	return std::any_of(list.begin(), list.end(),
	                   [name](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
}

bool CreateDevice(Device& d) {
	vkCreateInstance_ =
	    reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr_(nullptr, "vkCreateInstance"));
	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.pApplicationName = "pipeline_build_timer";
	app.apiVersion       = VK_API_VERSION_1_3;
	VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instance_info.pApplicationInfo = &app;
	if (vkCreateInstance_(&instance_info, nullptr, &d.instance) != VK_SUCCESS) {
		return false;
	}
	Load(d.instance, vkEnumeratePhysicalDevices_, "vkEnumeratePhysicalDevices");
	Load(d.instance, vkGetPhysicalDeviceProperties_, "vkGetPhysicalDeviceProperties");
	Load(d.instance, vkGetPhysicalDeviceFeatures2_, "vkGetPhysicalDeviceFeatures2");
	Load(d.instance, vkGetPhysicalDeviceQueueFamilyProperties_,
	     "vkGetPhysicalDeviceQueueFamilyProperties");
	Load(d.instance, vkEnumerateDeviceExtensionProperties_, "vkEnumerateDeviceExtensionProperties");
	Load(d.instance, vkCreateDevice_, "vkCreateDevice");
	Load(d.instance, vkGetDeviceProcAddr_, "vkGetDeviceProcAddr");
	Load(d.instance, vkDestroyInstance_, "vkDestroyInstance");

	uint32_t count = 0;
	vkEnumeratePhysicalDevices_(d.instance, &count, nullptr);
	std::vector<VkPhysicalDevice> devices(count);
	vkEnumeratePhysicalDevices_(d.instance, &count, devices.data());
	// The first discrete GPU, as the emulator picks.
	for (auto device: devices) {
		VkPhysicalDeviceProperties properties {};
		vkGetPhysicalDeviceProperties_(device, &properties);
		if (d.physical == VK_NULL_HANDLE ||
		    properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			d.physical       = device;
			d.name           = properties.deviceName;
			d.driver_version = properties.driverVersion;
			if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
				break;
			}
		}
	}
	if (d.physical == VK_NULL_HANDLE) {
		return false;
	}

	uint32_t extension_count = 0;
	vkEnumerateDeviceExtensionProperties_(d.physical, nullptr, &extension_count, nullptr);
	std::vector<VkExtensionProperties> available(extension_count);
	vkEnumerateDeviceExtensionProperties_(d.physical, nullptr, &extension_count, available.data());
	if (!HasExtension(available, VK_EXT_SHADER_OBJECT_EXTENSION_NAME)) {
		std::fprintf(stderr, "the device has no %s\n", VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
		return false;
	}

	// Every feature the device has, so every module the emulator can build compiles here too.
	std::vector<const char*> extensions;
	void*                    chain = nullptr;
	auto add = [&](const char* name, auto& features, VkStructureType type) {
		if (name != nullptr && !HasExtension(available, name)) {
			return;
		}
		if (name != nullptr) {
			extensions.push_back(name);
		}
		features       = {};
		features.sType = type;
		features.pNext = chain;
		chain          = &features;
	};
	VkPhysicalDeviceVulkan11Features v11 {};
	VkPhysicalDeviceVulkan12Features v12 {};
	VkPhysicalDeviceVulkan13Features v13 {};
	VkPhysicalDeviceShaderObjectFeaturesEXT shader_object {};
	VkPhysicalDeviceMeshShaderFeaturesEXT mesh {};
	VkPhysicalDeviceRobustness2FeaturesEXT robustness2 {};
	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float {};
	VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT atomic_float2 {};
	VkPhysicalDeviceShaderClockFeaturesKHR clock {};
	VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric {};
	VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup_layout {};
	VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT image_atomic64 {};
	VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT interlock {};
	add(nullptr, v11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
	add(nullptr, v12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
	add(nullptr, v13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
	add(VK_EXT_SHADER_OBJECT_EXTENSION_NAME, shader_object,
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT);
	add(VK_EXT_MESH_SHADER_EXTENSION_NAME, mesh, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT);
	add(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, robustness2,
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT);
	add(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, atomic_float,
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT);
	add(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME, atomic_float2,
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT);
	add(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, clock, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR);
	add(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, barycentric,
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR);
	add(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME, workgroup_layout,
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR);
	add(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, image_atomic64,
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT);
	add(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME, interlock,
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT);
	for (const char* name: {VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
	                        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME}) {
		if (HasExtension(available, name)) {
			extensions.push_back(name);
		}
	}
	VkPhysicalDeviceFeatures2 features {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	features.pNext = chain;
	vkGetPhysicalDeviceFeatures2_(d.physical, &features);
	d.task_shader = mesh.taskShader == VK_TRUE;
	// Not needed to compile, and they need extra extensions or state.
	mesh.multiviewMeshShader                    = VK_FALSE;
	mesh.primitiveFragmentShadingRateMeshShader = VK_FALSE;

	uint32_t family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties_(d.physical, &family_count, nullptr);
	float                   priority = 1.0f;
	VkDeviceQueueCreateInfo queue {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue.queueFamilyIndex = 0;
	queue.queueCount       = 1;
	queue.pQueuePriorities = &priority;
	VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	device_info.pNext                   = &features;
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue;
	device_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
	device_info.ppEnabledExtensionNames = extensions.data();
	const auto result = vkCreateDevice_(d.physical, &device_info, nullptr, &d.device);
	if (result != VK_SUCCESS) {
		std::fprintf(stderr, "vkCreateDevice: %d\n", static_cast<int>(result));
		return false;
	}
	auto load_device = [&](auto& function, const char* name) {
		function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
		    vkGetDeviceProcAddr_(d.device, name));
		if (function == nullptr) {
			std::fprintf(stderr, "missing %s\n", name);
			std::exit(1);
		}
	};
	load_device(vkCreateDescriptorSetLayout_, "vkCreateDescriptorSetLayout");
	load_device(vkDestroyDescriptorSetLayout_, "vkDestroyDescriptorSetLayout");
	load_device(vkCreateShadersEXT_, "vkCreateShadersEXT");
	load_device(vkDestroyShaderEXT_, "vkDestroyShaderEXT");
	load_device(vkDestroyDevice_, "vkDestroyDevice");
	return true;
}

// --- timing -------------------------------------------------------------------------------------

struct Result {
	double   ms     = 0.0;
	VkResult result = VK_SUCCESS;
};

Result Compile(const Device& d, const Module& module, uint64_t salt) {
	std::vector<VkDescriptorSetLayout> layouts;
	const uint32_t set_count = module.bindings.empty() ? 0u : module.bindings.rbegin()->first + 1u;
	for (uint32_t set = 0; set < set_count; set++) {
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		std::vector<VkDescriptorBindingFlags>     flags;
		if (const auto found = module.bindings.find(set); found != module.bindings.end()) {
			for (const auto& [binding, info]: found->second) {
				VkDescriptorSetLayoutBinding b {};
				b.binding         = binding;
				b.descriptorType  = info.first;
				b.descriptorCount = info.second;
				b.stageFlags      = VK_SHADER_STAGE_ALL;
				bindings.push_back(b);
				flags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
			}
		}
		VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info {
		    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
		flags_info.bindingCount  = static_cast<uint32_t>(flags.size());
		flags_info.pBindingFlags = flags.data();
		VkDescriptorSetLayoutCreateInfo info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		info.pNext        = &flags_info;
		info.bindingCount = static_cast<uint32_t>(bindings.size());
		info.pBindings    = bindings.data();
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (vkCreateDescriptorSetLayout_(d.device, &info, nullptr, &layout) != VK_SUCCESS) {
			for (auto l: layouts) {
				vkDestroyDescriptorSetLayout_(d.device, l, nullptr);
			}
			return {0.0, VK_ERROR_INITIALIZATION_FAILED};
		}
		layouts.push_back(layout);
	}

	const auto code = Salted(module.words, "kyty-timer-" + std::to_string(salt));
	VkPushConstantRange push {VK_SHADER_STAGE_ALL, 0, 256};
	VkShaderCreateInfoEXT info {VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT};
	info.stage    = module.stage;
	info.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
	info.codeSize = code.size() * sizeof(uint32_t);
	info.pCode    = code.data();
	info.pName    = module.entry.c_str();
	info.setLayoutCount = static_cast<uint32_t>(layouts.size());
	info.pSetLayouts    = layouts.data();
	info.pushConstantRangeCount = module.push_constants ? 1u : 0u;
	info.pPushConstantRanges    = &push;
	if (module.stage == VK_SHADER_STAGE_MESH_BIT_EXT && d.task_shader) {
		info.flags |= VK_SHADER_CREATE_NO_TASK_SHADER_BIT_EXT;
	}
	if (module.stage == VK_SHADER_STAGE_VERTEX_BIT) {
		info.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	}

	VkShaderEXT shader = VK_NULL_HANDLE;
	const auto  begin  = std::chrono::steady_clock::now();
	const auto  result = vkCreateShadersEXT_(d.device, 1, &info, nullptr, &shader);
	const auto  end    = std::chrono::steady_clock::now();
	if (shader != VK_NULL_HANDLE) {
		vkDestroyShaderEXT_(d.device, shader, nullptr);
	}
	for (auto l: layouts) {
		vkDestroyDescriptorSetLayout_(d.device, l, nullptr);
	}
	return {std::chrono::duration<double, std::milli>(end - begin).count(), result};
}

const char* StageName(VkShaderStageFlagBits stage) {
	switch (stage) {
		case VK_SHADER_STAGE_VERTEX_BIT: return "VS";
		case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT: return "HS";
		case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: return "DS";
		case VK_SHADER_STAGE_GEOMETRY_BIT: return "GS";
		case VK_SHADER_STAGE_FRAGMENT_BIT: return "PS";
		case VK_SHADER_STAGE_COMPUTE_BIT: return "CS";
		case VK_SHADER_STAGE_TASK_BIT_EXT: return "TS";
		case VK_SHADER_STAGE_MESH_BIT_EXT: return "MS";
		default: return "?";
	}
}

} // namespace

int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: pipeline_build_timer <out.csv> <file.spv | folder> [...] "
		                     "[--repeat N]\n");
		return 1;
	}
	int repeat = 3;
	std::vector<std::filesystem::path> files;
	for (int i = 2; i < argc; i++) {
		if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
			repeat = std::max(1, std::atoi(argv[++i]));
			continue;
		}
		const std::filesystem::path path = argv[i];
		if (std::filesystem::is_directory(path)) {
			for (const auto& entry: std::filesystem::directory_iterator(path)) {
				if (entry.path().extension() == ".spv") {
					files.push_back(entry.path());
				}
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
	std::printf("%s, driver 0x%08x, %zu modules, %d compiles each\n", device.name.c_str(),
	            device.driver_version, files.size(), repeat);

	std::ofstream out(argv[1]);
	out << "file,stage,words,median_ms,min_ms,max_ms,result\n";
	const uint64_t run_salt =
	    static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
	uint64_t salt = 0;
	for (size_t f = 0; f < files.size(); f++) {
		Module      module;
		std::string error;
		const auto  name = files[f].filename().string();
		if (!Read(files[f], module, error)) {
			out << name << ",?,0,,,," << error << "\n";
			continue;
		}
		std::vector<double> times;
		VkResult            result = VK_SUCCESS;
		for (int r = 0; r < repeat; r++) {
			const auto one = Compile(device, module, run_salt + salt++);
			result         = one.result;
			if (result != VK_SUCCESS) {
				break;
			}
			times.push_back(one.ms);
		}
		std::sort(times.begin(), times.end());
		if (result != VK_SUCCESS || times.empty()) {
			out << name << "," << StageName(module.stage) << "," << module.words.size() << ",,,,"
			    << static_cast<int>(result) << "\n";
		} else {
			char line[512];
			std::snprintf(line, sizeof(line), "%s,%s,%zu,%.2f,%.2f,%.2f,ok\n", name.c_str(),
			              StageName(module.stage), module.words.size(), times[times.size() / 2],
			              times.front(), times.back());
			out << line;
		}
		out.flush();
		if ((f + 1) % 25 == 0 || f + 1 == files.size()) {
			std::printf("%zu / %zu\n", f + 1, files.size());
			std::fflush(stdout);
		}
	}
	vkDestroyDevice_(device.device, nullptr);
	vkDestroyInstance_(device.instance, nullptr);
	return 0;
}
