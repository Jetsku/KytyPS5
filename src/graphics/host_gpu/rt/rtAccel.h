#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RT_RTACCEL_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RT_RTACCEL_H_

// Thin helpers for host acceleration structures (VK_KHR_acceleration_structure) built from
// converted Psr BVHs. Memory is owned by the caller; these only describe and record builds.
// Nothing here runs unless the optional hardware ray-tracing path (KYTY_RT_HARDWARE) is used.

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics::Rt {

struct RtDeviceSupport {
	bool     acceleration_structure = false;
	bool     ray_query              = false;
	bool     buffer_device_address  = false;
	bool     indirect_build         = false;
	uint32_t scratch_alignment      = 0;
	uint64_t max_primitive_count    = 0;
	uint64_t max_instance_count     = 0;

	[[nodiscard]] bool Usable() const {
		return acceleration_structure && ray_query && buffer_device_address;
	}
};

RtDeviceSupport          QueryRtSupport(vk::PhysicalDevice physical_device);
std::vector<const char*> RequiredRtDeviceExtensions();

enum class AccelGeometry : uint32_t {
	Aabbs,     // procedural boxes: one per Psr leaf (exact mode)
	Triangles, // non-indexed float3 triangles (fast, approximate mode)
	Instances, // top level
};

struct AccelBuildDesc {
	AccelGeometry     geometry        = AccelGeometry::Aabbs;
	vk::DeviceAddress data            = 0; // AABBs, vertices or VkAccelerationStructureInstanceKHR
	uint32_t          stride          = 0; // AABB or vertex stride; ignored for instances
	uint32_t          primitive_count = 0; // capacity; inactive entries are NaN / null
	bool              opaque          = true;
	bool              allow_update    = false;
	bool              prefer_fast_build = false;
};

vk::AccelerationStructureBuildSizesInfoKHR QueryBuildSizes(vk::Device            device,
                                                           const AccelBuildDesc& desc);

// Records one build (or an in-place update when `source` is set and the build allowed updates).
void RecordBuild(vk::CommandBuffer cmd, const AccelBuildDesc& desc,
                 vk::AccelerationStructureKHR destination, vk::AccelerationStructureKHR source,
                 vk::DeviceAddress scratch);

// Barrier between AS builds (or conversion writes) and later AS builds / ray queries.
void RecordAccelBarrier(vk::CommandBuffer cmd, bool from_compute_writes);

} // namespace Libs::Graphics::Rt

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RT_RTACCEL_H_ */
