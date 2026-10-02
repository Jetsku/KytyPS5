#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/queueSubmission.h"
#include "graphics/host_gpu/vulkanCommon.h" // IWYU pragma: export

#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct VulkanImage;

inline constexpr uint32_t VULKAN_TARGET_API_VERSION = VK_API_VERSION_1_3;

struct GraphicContext {
	vk::Instance                       instance                              = nullptr;
	vk::DebugUtilsMessengerEXT         debug_messenger                       = nullptr;
	vk::PhysicalDevice                 physical_device                       = nullptr;
	vk::PhysicalDeviceProperties       physical_device_properties            = {};
	vk::PhysicalDeviceMemoryProperties physical_device_memory_properties     = {};
	vk::Device                         device                                = nullptr;
	VmaAllocator                       allocator                             = nullptr;
	bool                               memory_budget_ext_enabled             = false;
	bool                               device_fault_enabled                  = false;
	bool                               compute_subgroup_size_control_enabled = false;
	bool                               sample_rate_shading_enabled           = false;
	bool                               precise_occlusion_enabled             = false;
	bool                               attachment_feedback_loop_enabled      = false;
	bool                               provoking_vertex_last_enabled         = false;
	bool                               supports_block_texel_view              = false;
	// shaderStorageImageReadWithoutFormat (TileManager::TileFromImage).
	bool                               storage_image_read_without_format_enabled = false;
	// Vulkan 1.2 samplerFilterMinmax: S# FILTER_MODE min/max (SamplerCache).
	bool                               sampler_filter_minmax_enabled         = false;
	// VK_KHR_maintenance8: vkCmdCopyImage between a depth aspect and a compatible color format
	// (D32 <-> R32, D16 <-> R16). KYTY_DIRECT_IMAGE_COPY_M8=0 leaves the extension disabled.
	bool                               maintenance8_enabled                  = false;
	// Native indirect draws (vkCmdDraw*Indirect*). Each form is used only when enabled here.
	bool                               draw_indirect_first_instance_enabled  = false;
	bool                               multi_draw_indirect_enabled           = false;
	bool                               draw_indirect_count_enabled           = false;
	bool                               index_type_uint8_enabled              = false;
	// VK_EXT_graphics_pipeline_library with fast linking, enabled only when KYTY_PIPELINE_LIBRARY
	// asks for it (pipeline/pipelineLibrary.h), and core Vulkan 1.3 pipelineCreationCacheControl
	// (VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT).
	bool                               pipeline_library_enabled              = false;
	bool                               pipeline_creation_cache_control_enabled = false;
	// VK_EXT_conditional_rendering, enabled only for KYTY_PREDICATION_MODE=gpu
	// (renderer/gpuPredication.h).
	bool                               conditional_rendering_enabled         = false;
	bool                                      mesh_shader_enabled                   = false;
	vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties                = {};
	uint32_t                           subgroup_size                         = 0;
	uint32_t                           min_subgroup_size                     = 0;
	uint32_t                           max_subgroup_size                     = 0;
	uint32_t                           max_push_descriptors                  = 0;
	vk::ShaderStageFlags               required_subgroup_size_stages         = {};
	Common::Mutex                      queue_mutex;
	QueueSubmissionBroker              submission_queue;
	uint32_t                           queue_family = static_cast<uint32_t>(-1);
	vk::Queue                          queue        = nullptr;
	// Second queue of queue_family for side-copy readbacks (KYTY_SIDE_QUEUE); null when absent,
	// then side copies share `queue`. Submissions to it hold side_queue_mutex, not queue_mutex.
	uint32_t                           side_queue_index = 0;
	vk::Queue                          side_queue       = nullptr;
	Common::Mutex                      side_queue_mutex;
	// Queue of a transfer-only family (the copy engines) for asynchronous uploads (KYTY_UPLOAD_DMA,
	// renderer/cache/uploadDma.h); null when the device has none or the path is off. Only the
	// UploadDma worker submits to it. Buffers it touches are created shared by both families.
	uint32_t                           transfer_queue_family = static_cast<uint32_t>(-1);
	vk::Queue                          transfer_queue        = nullptr;

	[[nodiscard]] const vk::PhysicalDeviceProperties& GetPhysicalDeviceProperties() const {
		return physical_device_properties;
	}

	[[nodiscard]] const vk::PhysicalDeviceMemoryProperties&
	GetPhysicalDeviceMemoryProperties() const {
		return physical_device_memory_properties;
	}

	[[nodiscard]] vk::FormatProperties GetFormatProperties(vk::Format format) const {
		std::scoped_lock lock(m_format_properties_mutex);
		auto [it, inserted] = m_format_properties.try_emplace(format);
		if (inserted) {
			physical_device.getFormatProperties(format, &it->second);
		}
		return it->second;
	}

	[[nodiscard]] vk::Result GetImageFormatProperties(vk::Format format, vk::ImageType type,
	                                                  vk::ImageTiling            tiling,
	                                                  vk::ImageUsageFlags        usage,
	                                                  vk::ImageCreateFlags       flags,
	                                                  vk::ImageFormatProperties* properties) const {
		using Key = std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
		                       vk::ImageCreateFlags>;
		std::scoped_lock lock(m_image_format_properties_mutex);
		auto [it, inserted] =
		    m_image_format_properties.try_emplace(Key {format, type, tiling, usage, flags});
		if (inserted) {
			it->second.first = physical_device.getImageFormatProperties(format, type, tiling, usage,
			                                                            flags, &it->second.second);
		}
		if (properties != nullptr) {
			*properties = it->second.second;
		}
		return it->second.first;
	}

	[[nodiscard]] bool SupportsComputeWave64() const noexcept {
		return subgroup_size == 64u || compute_subgroup_size_control_enabled;
	}

	[[nodiscard]] vk::DeviceSize StorageMinAlignment() const {
		const auto alignment = physical_device_properties.limits.minStorageBufferOffsetAlignment;
		return alignment != 0 ? alignment : 1;
	}

	[[nodiscard]] bool CreateAllocator();
	void               DestroyAllocator();
	void               LogMemoryBudget() const;
	[[nodiscard]] bool CanReportMemoryUsage() const noexcept { return memory_budget_ext_enabled; }
	[[nodiscard]] uint64_t GetDeviceMemoryUsage() const;
	[[nodiscard]] uint64_t GetTotalMemoryBudget() const;
	[[nodiscard]] bool     CreateImage(const vk::ImageCreateInfo& info, VulkanImage& image);
	void                   DeleteImage(VulkanImage& image);

	uint32_t screen_width  = 0;
	uint32_t screen_height = 0;

private:
	struct RetiredNativeImage {
		vk::ImageCreateInfo create;
		vk::Image image;
		VmaAllocation allocation;
		uint64_t bytes;
	};
	void ClearRetiredImages();
	std::mutex m_retired_image_mutex;
	std::vector<RetiredNativeImage> m_retired_images;
	uint64_t m_retired_image_bytes = 0;
	mutable std::mutex                                 m_format_properties_mutex;
	mutable std::map<vk::Format, vk::FormatProperties> m_format_properties;
	mutable std::mutex                                 m_image_format_properties_mutex;
	mutable std::map<std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
	                            vk::ImageCreateFlags>,
	                 std::pair<vk::Result, vk::ImageFormatProperties>>
	    m_image_format_properties;
};

struct VulkanImageState {
	vk::PipelineStageFlags2 pl_stage    = vk::PipelineStageFlagBits2::eAllCommands;
	vk::AccessFlags2        access_mask = vk::AccessFlagBits2::eNone;
	vk::ImageLayout         layout      = vk::ImageLayout::eUndefined;
};

struct VulkanImage {
	VulkanImage() = default;
	KYTY_CLASS_NO_COPY(VulkanImage);

	vk::Format                    format      = vk::Format::eUndefined;
	vk::ImageType                 image_type  = vk::ImageType::e2D;
	vk::Extent3D                  extent      = {1, 1, 1};
	uint32_t                      layers      = 1;
	uint32_t                      mip_levels  = 1;
	uint32_t                      samples     = 1;
	vk::ImageUsageFlags           usage       = {};
	vk::ImageCreateFlags          flags       = {};
	vk::Image                     image       = nullptr;
	VulkanImageState              state;
	std::vector<VulkanImageState> subresource_states;
	VmaAllocation                allocation = nullptr;
	// Only pointer-free, ordinary optimal images are eligible for native recycling.
	// Guest contents, views and layout tracking are never retained across owners.
	bool                         pool_eligible = false;
	vk::ImageCreateInfo          pool_create_info {};
};



} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_ */
