#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/deviceCompat.h"
#include "graphics/host_gpu/queueSubmission.h"
#include "graphics/host_gpu/vulkanCommon.h" // IWYU pragma: export

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <tuple>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct VulkanImage;

inline constexpr uint32_t VULKAN_TARGET_API_VERSION = VK_API_VERSION_1_3;

struct DiagnosticCheckpoint {
	uint32_t op        = 0;
	uint64_t submit_id = 0;
	uint32_t arg0      = 0;
	uint32_t arg1      = 0;
	uint32_t arg2      = 0;
	uint32_t arg3      = 0;
	uint64_t arg4      = 0;
	uint64_t sequence  = 0;
	uint64_t tick = 0;
	uint64_t vs = 0;
	uint64_t ps = 0;
	uint64_t cs = 0;
	uint32_t queue = 0; // GpuBreadcrumbs::Queue
};

struct GraphicContext;

[[nodiscard]] const void*
RecordDiagnosticCheckpoint(const DiagnosticCheckpoint& checkpoint);
[[nodiscard]] bool DeviceFaultDiagnosticsEnabled();
// KYTY_ADDRESS_BINDING_REPORT=1 (diagnostic, default off): VK_EXT_device_address_binding_report.
// Every GPU virtual address range the driver binds or unbinds for an object is recorded (live
// ranges, plus a ring of the newest events); a device fault then prints the objects that cover the
// faulting address, end just below it or were unbound there, with their Kyty names (the names are
// kept for this report even without the debug dump) and how long before the fault they changed.
[[nodiscard]] bool AddressBindingReportRequested();
void               AddressBindingReportCreateMessenger(vk::Instance instance);
void               AddressBindingReportDump(uint64_t address);
void DumpDeviceLossDiagnostics(GraphicContext& graphics, uint64_t tick = 0, bool queue_locked = false);
// A component's own lines for the device-loss report (printed after the device fault; up to 4 notes).
void SetDeviceLossNote(void (*note)());
// A thread about to exit on a device loss: lets a device-loss report another thread is writing
// (the VK_EXT_device_fault vendor dump, _device_fault.nv-gpudmp) finish first. Waits up to
// `start_ms` for one to begin and then up to 10 s for it to end.
void WaitForDeviceLossDiagnostics(uint32_t start_ms = 500);
// Breadcrumb ring access (GpuBreadcrumbs): records and returns the sequence number; looks one up
// (false once it left the bounded history); the newest sequence; an operation's name.
[[nodiscard]] uint64_t RecordDiagnosticSequence(const DiagnosticCheckpoint& checkpoint);
[[nodiscard]] bool     LookupDiagnosticCheckpoint(uint64_t sequence, DiagnosticCheckpoint* checkpoint);
[[nodiscard]] uint64_t NewestDiagnosticSequence();
[[nodiscard]] const char* DiagnosticOpName(uint32_t op);

// KYTY_DEAD_VIEW_SCRUB=1 (default off; the GT7 preset; usa-ratt/KytyPS5-GT7 ae60f82a5): destroyed
// image views are remembered (the last 64K); one named as a rendering attachment or a storage-image
// descriptor when the command runs is reported ("DEAD image view") and replaced with a null view
// instead of being left to fault the GPU. The callers check DeadViewScrubEnabled(); the functions
// themselves always work (tests).
[[nodiscard]] bool DeadViewScrubEnabled();
void               NoteImageViewCreated(vk::ImageView view);
void               NoteImageViewDestroyed(vk::ImageView view, uint64_t guest_address, vk::Format format);
[[nodiscard]] bool DiagnosticViewIsLive(vk::ImageView view);
void               ScrubDeadRenderingViews(const vk::RenderingInfo& info, const char* where);
void ScrubDeadDescriptorViews(const vk::WriteDescriptorSet* writes, uint32_t count, const char* where);

struct GraphicContext {
	vk::Instance                       instance                              = nullptr;
	vk::DebugUtilsMessengerEXT         debug_messenger                       = nullptr;
	vk::PhysicalDevice                 physical_device                       = nullptr;
	vk::PhysicalDeviceProperties       physical_device_properties            = {};
	vk::PhysicalDeviceMemoryProperties physical_device_memory_properties     = {};
	vk::Device                         device                                = nullptr;
	VmaAllocator                       allocator                             = nullptr;
	bool                               memory_budget_ext_enabled             = false;
	// VK_EXT_external_memory_host, requested only with KYTY_INPLACE (InplaceHeap).
	bool                               external_memory_host_enabled          = false;
	// KYTY_FULLSCREEN_EXCLUSIVE=0|1 (Windows): the swapchain asks the driver to keep out of exclusive
	// fullscreen (0) or allows it (1), through VK_EXT_full_screen_exclusive; -1 (unset, or the
	// extension missing): the driver's own choice.
	int                                full_screen_exclusive                 = -1;
	bool                               diagnostic_checkpoints_enabled        = false;
	bool                               device_fault_enabled                  = false;
	// KYTY_GPU_BREADCRUMBS (gpuBreadcrumbs.h): VK_AMD_buffer_marker enabled for breadcrumbs, and
	// VK_AMD_device_coherent_memory's deviceCoherentMemory for their buffer.
	bool                               gpu_breadcrumbs_requested             = false;
	bool                               device_coherent_memory_enabled        = false;
	// KYTY_COMPUTE_WAVE64=0: wave64 compute programs run two guest lanes per invocation on 32-wide
	// subgroups (the path NVIDIA takes) even where the device could run them natively.
	bool                               compute_wave64_split                  = false;
	bool                               compute_subgroup_size_control_enabled = false;
	// subgroupSizeControl is enabled: the device has more than one subgroup size.
	bool                               subgroup_size_control_enabled         = false;
	bool                               sample_rate_shading_enabled           = false;
	// KYTY_EQAA_MIXED=1 (from usa-ratt/KytyPS5-GT7 18979e65b, df310f7a9): attachments with
	// different sample counts in one draw (PS5 EQAA: depth at the coverage count, colour at the
	// fragment count). AMD allows either side to be larger; NV only colour < depth, rasterizing at
	// the depth count. None: the switch is off or the device has neither extension.
	enum class MixedAttachmentSamples : uint8_t { None, Amd, Nv };
	MixedAttachmentSamples             mixed_attachment_samples              = MixedAttachmentSamples::None;
	// Whether one draw can attach colour and depth at these sample counts (0 colour: depth only).
	[[nodiscard]] bool AcceptsMixedSamples(uint32_t color_samples,
	                                       uint32_t depth_samples) const noexcept {
		if (color_samples == 0 || color_samples == depth_samples) {
			return true;
		}
		switch (mixed_attachment_samples) {
			case MixedAttachmentSamples::Amd: return true;
			case MixedAttachmentSamples::Nv: return color_samples < depth_samples;
			case MixedAttachmentSamples::None: return false;
		}
		return false;
	}
	// The (raster samples, depth/stencil sample mask, colour sample mask) combinations the device
	// lists for VK_NV_coverage_reduction_mode's TRUNCATE (colour sample i covered by raster sample
	// i alone, as the PS5's colour buffer sees only the samples exposed to it; MERGE lets both
	// triangles of a shared edge blend into its pixels). Empty: MERGE.
	std::vector<std::array<uint32_t, 3>> coverage_truncate_combinations;
	[[nodiscard]] bool CoverageTruncate(uint32_t raster, uint32_t depth, uint32_t color) const {
		for (const auto& combination: coverage_truncate_combinations) {
			if (combination[0] == raster && (combination[1] & depth) != 0 &&
			    (combination[2] & color) != 0) {
				return true;
			}
		}
		return false;
	}
	bool                               precise_occlusion_enabled             = false;
	bool                               shader_image_int64_atomics_enabled    = false;
	bool                               attachment_feedback_loop_enabled      = false;
	bool                               provoking_vertex_last_enabled         = false;
	// VK_EXT_color_write_enable: draws turn attachments without an image off through dynamic
	// colour-write enables. Without it (older drivers, MoltenVK) pipelines keep their static write
	// masks; an attachment without an image view discards its writes anyway.
	bool                               color_write_enable_enabled            = false;
	// VK_EXT_depth_clip_enable: depth clipping set apart from the always-on depth clamp (the DB's
	// behaviour). Without it, pipelines clamp only where the guest turns Z clipping off.
	bool                               depth_clip_enable_enabled             = false;
	// robustBufferAccess2 (ee034be4b): out-of-range storage accesses read zero / drop writes.
	bool                               robust_buffer_access_enabled          = false;
	// R8/R8G8 sRGB formats sample with linear filtering (e914ec6db): 8 and 8_8 sRGB textures.
	bool                               r8_srgb_views                         = false;
	bool                               r8g8_srgb_views                       = false;
	bool                               conditional_rendering_enabled         = false;
	bool                               supports_block_texel_view              = false;
	// shaderStorageImageReadWithoutFormat (TileManager::TileFromImage).
	bool                               storage_image_read_without_format_enabled = false;
	// Vulkan 1.2 samplerFilterMinmax: S# FILTER_MODE min/max (SamplerCache).
	bool                               sampler_filter_minmax_enabled         = false;
	// Vulkan 1.2 shaderStorageBufferArrayNonUniformIndexing, enabled when supported (upstream
	// made it a hard requirement in fd66f580b; shaders that index storage-buffer arrays
	// non-uniformly need it).
	bool                               storage_buffer_array_non_uniform_indexing_enabled = false;
	// Descriptor indexing for bindless images (BindlessTable, Senaxx): runtime arrays indexed
	// non-uniformly, partially bound, updated after bind. Enabled when supported unless
	// KYTY_BINDLESS=0. The limit is the update-after-bind sampled images one set and one stage may
	// hold; bindless_layout is created on first use (BindlessSetLayout, under its mutex).
	bool                               bindless_enabled                      = false;
	uint32_t                           bindless_max_sampled_images           = 0;
	// KYTY_BINDLESS_SAMPLERS: the update-after-bind samplers one set and one stage may hold.
	uint32_t                           bindless_max_samplers                 = 0;
	vk::DescriptorSetLayout            bindless_layout                       = nullptr;
	Common::Mutex                      bindless_layout_mutex;
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
	// KYTY_BDA_PAGETABLE_SPARSE=1: sparseBinding + sparseResidencyBuffer enabled, unbound buffer
	// ranges read as zero (residencyNonResidentStrict) and queue_family can bind sparse memory.
	bool                               sparse_residency_buffer_enabled       = false;
	// KYTY_TEXTURE_SPARSE_RESIDENCY=1: sparseBinding + sparseResidencyImage2D enabled, strict.
	bool                               sparse_residency_image_enabled        = false;
	bool                                      mesh_shader_enabled                   = false;
	// KYTY_RT_HW: VK_KHR_acceleration_structure + VK_KHR_ray_query enabled (NativeRayTracing).
	bool                                      native_ray_tracing_enabled            = false;
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

	// Sampled views of format filter linearly, as guest samplers may.
	[[nodiscard]] bool SamplesFiltered(vk::Format format) const {
		constexpr vk::FormatFeatureFlags features =
		    vk::FormatFeatureFlagBits::eSampledImage |
		    vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
		return (GetFormatProperties(format).optimalTilingFeatures & features) == features;
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
		return !compute_wave64_split && (subgroup_size == 64u || compute_subgroup_size_control_enabled);
	}

	// The subgroup size a graphics stage must require so that one host subgroup is one guest
	// wave, or 0 (DeviceCompat::GraphicsSubgroupSize). On AMD the default is 64; a wave32 pixel or
	// mesh shader would otherwise share a subgroup with another wave, and its ballots, lane reads
	// and reductions would mix the two. Where the driver picks the width per shader (Intel: 8 to
	// 32) the default width is required too. Devices with a single subgroup size (NVIDIA: 32)
	// always get 0.
	[[nodiscard]] uint32_t GraphicsSubgroupSize(vk::ShaderStageFlagBits stage,
	                                            uint32_t                wave_size) const noexcept {
		return DeviceCompat::GraphicsSubgroupSize(
		    {.min_size       = min_subgroup_size,
		     .max_size       = max_subgroup_size,
		     .enabled        = subgroup_size_control_enabled,
		     .compute        = false,
		     .compute_wave64 = compute_subgroup_size_control_enabled},
		    static_cast<bool>(required_subgroup_size_stages & stage), wave_size, subgroup_size);
	}

	// The subgroup size a compute pipeline must require so that one host subgroup holds one guest
	// wave, or 0 (DeviceCompat::ComputeSubgroupSize): the wave size on AMD as before, 32 on devices
	// whose drivers pick a width per shader (Intel: 8 to 32), nothing on NVIDIA (always 32).
	[[nodiscard]] uint32_t ComputeSubgroupSize(uint32_t wave_size,
	                                           uint32_t host_subgroup_size) const noexcept {
		return DeviceCompat::ComputeSubgroupSize(
		    {.min_size       = min_subgroup_size,
		     .max_size       = max_subgroup_size,
		     .enabled        = subgroup_size_control_enabled,
		     .compute        = static_cast<bool>(required_subgroup_size_stages &
		                                         vk::ShaderStageFlagBits::eCompute),
		     .compute_wave64 = compute_subgroup_size_control_enabled},
		    wave_size, host_subgroup_size);
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

	struct HeapUsage {
		uint64_t usage        = 0; // Whole process, as reported by VK_EXT_memory_budget.
		uint64_t budget       = 0;
		uint32_t allocations  = 0;
		bool     device_local = false;
	};
	[[nodiscard]] uint32_t GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const;
	[[nodiscard]] bool     CreateImage(const vk::ImageCreateInfo& info, VulkanImage& image);
	void                   DeleteImage(VulkanImage& image);
	// KYTY_VRAM_STATS report lines (vramStats.h): VMA heaps, allocation kinds, native image pool.
	void                   ReportVramStats();
	// KYTY_NATIVE_IMAGE_POOL_IDLE_MS (vma.cpp): destroys retained native images unused that long
	// (GPU thread, from the garbage collector; rate-limited).
	void                   TrimRetiredImages();
	// An allocation failed for lack of device memory: destroys what is only retained for reuse
	// (the native image pool) before the caller retries. Returns the bytes released.
	uint64_t               ReleaseRetainedMemory();
	// KYTY_TEXTURE_SPARSE_RESIDENCY (sparse_residency_image_enabled): creates `info` as a sparse
	// residency image with memory bound behind levels >= first_level only (a single-layer,
	// single-sample 2D image whose format, usage and flags support sparse residency; false and
	// nothing created otherwise). BindSparseImageLevels binds memory behind levels >= first_level
	// that have none yet: unbound levels read as zero (residencyNonResidentStrict) and writes to them
	// are discarded, so only levels a view can sample need memory. DeleteImage frees the memory.
	//
	// Ordering (Bug A): the binds run on the side queue and are ordered on the GPU, never waited for
	// on the host. A bind waits for `wait_semaphore` >= `wait_value` (the master timeline at the last
	// queue-0 tick that may use the image; that tick must already be submitted) and signals
	// SparseBindSemaphore(); every queue-0 batch submitted after it waits for that signal
	// (SparseBindPendingValue, CommandScheduler::Submit). Vulkan forbids other queues to access a
	// range while its binding changes, and the image's whole-chain layout transitions (needed
	// because sampled views span non-resident levels) access the levels being bound.
	[[nodiscard]] bool     CreateSparseImage(const vk::ImageCreateInfo& info, uint32_t first_level,
	                                         VulkanImage& image);
	void                   BindSparseImageLevels(VulkanImage& image, uint32_t first_level,
	                                             vk::Semaphore wait_semaphore = nullptr,
	                                             uint64_t      wait_value     = 0);
	// The timeline semaphore image binds signal, and the newest value signalled by a bind that may
	// not have completed (0: none pending). Null semaphore until the first bind.
	[[nodiscard]] vk::Semaphore SparseBindSemaphore() const noexcept { return m_sparse_timeline; }
	[[nodiscard]] uint64_t      SparseBindPendingValue();
	// Host wait for every image bind submitted so far (teardown, tests).
	void                        WaitSparseBinds();
	// The device memory behind an image: its allocation's size, or the bound bytes of a sparse one.
	[[nodiscard]] uint64_t NativeImageBytes(const VulkanImage& image) const;

	uint32_t screen_width  = 0;
	uint32_t screen_height = 0;

private:
	[[nodiscard]] bool SparseImageSupported(const vk::ImageCreateInfo& info);
	std::mutex                                 m_sparse_mutex;
	vk::Fence                                  m_sparse_fence = nullptr;
	vk::Semaphore                              m_sparse_timeline = nullptr;
	std::atomic<uint64_t>                      m_sparse_signalled {0}; // newest value a bind signals
	std::atomic<uint64_t>                      m_sparse_completed {0}; // known reached on the GPU
	std::map<std::tuple<vk::Format, vk::ImageUsageFlags, vk::ImageCreateFlags>, bool>
	                                           m_sparse_support;
	struct RetiredNativeImage {
		vk::ImageCreateInfo create;
		vk::Image image;
		VmaAllocation allocation;
		uint64_t bytes;
		std::chrono::steady_clock::time_point retired;
	};
	void ClearRetiredImages();
	std::mutex m_retired_image_mutex;
	std::vector<RetiredNativeImage> m_retired_images;
	uint64_t m_retired_image_bytes = 0;
	std::chrono::steady_clock::time_point m_retired_trim_next {};
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
	// Set by a guest draw's or dispatch's bindings (KYTY_GUEST_STORAGE_REPEAT, Image::GetBarriers).
	bool guest = false;
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
	// KYTY_TEXTURE_SPARSE_RESIDENCY: a sparse residency image (allocation is null) and its memory.
	struct SparseState {
		uint32_t                   first_bound      = UINT32_MAX; // levels >= this have memory
		uint32_t                   tail_first       = UINT32_MAX; // imageMipTailFirstLod
		uint64_t                   tail_offset      = 0;
		uint64_t                   tail_size        = 0;
		bool                       tail_bound       = false;
		vk::Extent3D               granularity      = {1, 1, 1};
		uint64_t                   block_size       = 0;
		uint32_t                   memory_type_bits = 0;
		uint64_t                   bytes            = 0;
		std::vector<VmaAllocation> allocations;
	};
	std::unique_ptr<SparseState> sparse;
};



} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_ */
