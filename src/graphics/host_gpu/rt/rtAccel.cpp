#include "graphics/host_gpu/rt/rtAccel.h"

#include <algorithm>
#include <cstring>

namespace Libs::Graphics::Rt {

namespace {

bool HasExtension(const std::vector<vk::ExtensionProperties>& extensions, const char* name) {
	return std::any_of(extensions.begin(), extensions.end(),
	                   [name](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
}

struct GeometryInfo {
	vk::AccelerationStructureGeometryKHR          geometry {};
	vk::AccelerationStructureBuildGeometryInfoKHR info {};
};

void Describe(const AccelBuildDesc& desc, GeometryInfo& out) {
	out.geometry = vk::AccelerationStructureGeometryKHR {};
	switch (desc.geometry) {
		case AccelGeometry::Aabbs: {
			vk::AccelerationStructureGeometryAabbsDataKHR aabbs {};
			aabbs.data.deviceAddress    = desc.data;
			aabbs.stride                = desc.stride;
			out.geometry.geometryType   = vk::GeometryTypeKHR::eAabbs;
			out.geometry.geometry.aabbs = aabbs;
			break;
		}
		case AccelGeometry::Triangles: {
			vk::AccelerationStructureGeometryTrianglesDataKHR triangles {};
			triangles.vertexFormat             = vk::Format::eR32G32B32Sfloat;
			triangles.vertexData.deviceAddress = desc.data;
			triangles.vertexStride             = desc.stride;
			triangles.maxVertex = desc.primitive_count == 0 ? 0 : desc.primitive_count * 3u - 1u;
			triangles.indexType = vk::IndexType::eNoneKHR;
			out.geometry.geometryType       = vk::GeometryTypeKHR::eTriangles;
			out.geometry.geometry.triangles = triangles;
			break;
		}
		case AccelGeometry::Instances: {
			vk::AccelerationStructureGeometryInstancesDataKHR instances {};
			instances.arrayOfPointers       = VK_FALSE;
			instances.data.deviceAddress    = desc.data;
			out.geometry.geometryType       = vk::GeometryTypeKHR::eInstances;
			out.geometry.geometry.instances = instances;
			break;
		}
	}
	if (desc.opaque) {
		out.geometry.flags = vk::GeometryFlagBitsKHR::eOpaque;
	}

	out.info       = vk::AccelerationStructureBuildGeometryInfoKHR {};
	out.info.type  = desc.geometry == AccelGeometry::Instances
	                     ? vk::AccelerationStructureTypeKHR::eTopLevel
	                     : vk::AccelerationStructureTypeKHR::eBottomLevel;
	out.info.flags = desc.prefer_fast_build
	                     ? vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild
	                     : vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
	if (desc.allow_update) {
		out.info.flags |= vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
	}
	out.info.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
	out.info.geometryCount = 1;
	out.info.pGeometries   = &out.geometry;
}

} // namespace

RtDeviceSupport QueryRtSupport(vk::PhysicalDevice physical_device) {
	RtDeviceSupport support {};
	uint32_t        count = 0;
	if (physical_device.enumerateDeviceExtensionProperties(nullptr, &count, nullptr) !=
	    vk::Result::eSuccess) {
		return support;
	}
	std::vector<vk::ExtensionProperties> extensions(count);
	if (physical_device.enumerateDeviceExtensionProperties(nullptr, &count, extensions.data()) !=
	    vk::Result::eSuccess) {
		return support;
	}
	const bool has_as = HasExtension(extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
	                    HasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
	const bool has_rq = HasExtension(extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);
	if (!has_as || !has_rq) {
		return support;
	}

	vk::PhysicalDeviceAccelerationStructureFeaturesKHR as_features {};
	vk::PhysicalDeviceRayQueryFeaturesKHR              rq_features {};
	vk::PhysicalDeviceVulkan12Features                 features12 {};
	as_features.pNext = &rq_features;
	rq_features.pNext = &features12;
	vk::PhysicalDeviceFeatures2 features {};
	features.pNext = &as_features;
	physical_device.getFeatures2(&features);

	vk::PhysicalDeviceAccelerationStructurePropertiesKHR as_properties {};
	vk::PhysicalDeviceProperties2                        properties {};
	properties.pNext = &as_properties;
	physical_device.getProperties2(&properties);

	support.acceleration_structure = as_features.accelerationStructure == VK_TRUE;
	support.indirect_build         = as_features.accelerationStructureIndirectBuild == VK_TRUE;
	support.ray_query              = rq_features.rayQuery == VK_TRUE;
	support.buffer_device_address  = features12.bufferDeviceAddress == VK_TRUE;
	support.scratch_alignment      = as_properties.minAccelerationStructureScratchOffsetAlignment;
	support.max_primitive_count    = as_properties.maxPrimitiveCount;
	support.max_instance_count     = as_properties.maxInstanceCount;
	return support;
}

std::vector<const char*> RequiredRtDeviceExtensions() {
	return {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
	        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME};
}

vk::AccelerationStructureBuildSizesInfoKHR QueryBuildSizes(vk::Device            device,
                                                           const AccelBuildDesc& desc) {
	GeometryInfo g;
	Describe(desc, g);
	vk::AccelerationStructureBuildSizesInfoKHR sizes {};
	device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice,
	                                             &g.info, &desc.primitive_count, &sizes);
	return sizes;
}

void RecordBuild(vk::CommandBuffer cmd, const AccelBuildDesc& desc,
                 vk::AccelerationStructureKHR destination, vk::AccelerationStructureKHR source,
                 vk::DeviceAddress scratch) {
	GeometryInfo g;
	Describe(desc, g);
	g.info.dstAccelerationStructure  = destination;
	g.info.scratchData.deviceAddress = scratch;
	if (source) {
		g.info.mode                     = vk::BuildAccelerationStructureModeKHR::eUpdate;
		g.info.srcAccelerationStructure = source;
	}
	vk::AccelerationStructureBuildRangeInfoKHR range {};
	range.primitiveCount                                     = desc.primitive_count;
	const vk::AccelerationStructureBuildRangeInfoKHR* ranges = &range;
	cmd.buildAccelerationStructuresKHR(1, &g.info, &ranges);
}

void RecordAccelBarrier(vk::CommandBuffer cmd, bool from_compute_writes) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR |
	                        (from_compute_writes ? vk::PipelineStageFlagBits2::eComputeShader
	                                             : vk::PipelineStageFlagBits2::eNone);
	barrier.srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR |
	                        (from_compute_writes ? vk::AccessFlagBits2::eShaderStorageWrite
	                                             : vk::AccessFlagBits2::eNone);
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR |
	                        vk::PipelineStageFlagBits2::eComputeShader;
	barrier.dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR |
	                        vk::AccessFlagBits2::eShaderRead |
	                        vk::AccessFlagBits2::eShaderStorageRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	cmd.pipelineBarrier2(&dependency);
}

} // namespace Libs::Graphics::Rt
