#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"

#include "common/assert.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderBindings.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/render.h"

#include <cstring>

namespace Libs::Graphics {

vk::ShaderStageFlagBits NativeShaderStage(ShaderType stage) {
	switch (stage) {
		case ShaderType::Local:
		case ShaderType::Vertex: return vk::ShaderStageFlagBits::eVertex;
		case ShaderType::Mesh: return vk::ShaderStageFlagBits::eMeshEXT;
		case ShaderType::TessellationControl: return vk::ShaderStageFlagBits::eTessellationControl;
		case ShaderType::TessellationEvaluation:
			return vk::ShaderStageFlagBits::eTessellationEvaluation;
		case ShaderType::Pixel: return vk::ShaderStageFlagBits::eFragment;
		case ShaderType::Compute: return vk::ShaderStageFlagBits::eCompute;
		default: EXIT("unknown native shader stage\n");
	}
}

vk::PipelineStageFlags ShaderPipelineStages(vk::ShaderStageFlags stages) {
	vk::PipelineStageFlags result = {};
	if (stages & vk::ShaderStageFlagBits::eVertex) {
		result |= vk::PipelineStageFlagBits::eVertexShader;
	}
	if (stages & vk::ShaderStageFlagBits::eMeshEXT) {
		result |= vk::PipelineStageFlagBits::eMeshShaderEXT;
	}
	if (stages & vk::ShaderStageFlagBits::eTessellationControl) {
		result |= vk::PipelineStageFlagBits::eTessellationControlShader;
	}
	if (stages & vk::ShaderStageFlagBits::eTessellationEvaluation) {
		result |= vk::PipelineStageFlagBits::eTessellationEvaluationShader;
	}
	if (stages & vk::ShaderStageFlagBits::eFragment) {
		result |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (stages & vk::ShaderStageFlagBits::eCompute) {
		result |= vk::PipelineStageFlagBits::eComputeShader;
	}
	EXIT_IF(!result);
	return result;
}

vk::MemoryBarrier MakeShaderWriteDependency() {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
	                        vk::AccessFlagBits::eVertexAttributeRead |
	                        vk::AccessFlagBits::eIndexRead | vk::AccessFlagBits::eUniformRead |
	                        vk::AccessFlagBits::eIndirectCommandRead |
	                        vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eColorAttachmentRead |
	                        vk::AccessFlagBits::eColorAttachmentWrite;
	return barrier;
}

vk::MemoryBarrier MakeShaderAccessDependency() {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	return barrier;
}

vk::MemoryBarrier MakeShaderWriteHazardDependency() {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	return barrier;
}

vk::BufferMemoryBarrier MakeGdsDependency(vk::Buffer buffer) {
	EXIT_IF(buffer == nullptr);

	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer;
	barrier.offset              = 0;
	barrier.size                = VK_WHOLE_SIZE;
	return barrier;
}

bool HasShaderBufferWrites(const ShaderStageRuntime& runtime) {
	const auto& program   = *runtime.program;
	const auto& resources = *runtime.resources;
	EXIT_IF(resources.buffers.size() != program.info.buffers.size());
	// Address writes reach memory outside the bound buffer descriptors.
	bool has_writes = program.has_address_writes;
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		if (!program.info.buffers[i].written) {
			continue;
		}
		const auto& value = resources.buffers[i];
		EXIT_IF(value.dword_count < 4);
		ShaderBufferResource descriptor;
		std::memcpy(descriptor.fields, value.dwords.data(), sizeof(descriptor.fields));
		// A zero stride means byte addressing. For either addressing mode a nonzero record
		// count is exactly the condition for a nonempty descriptor range.
		has_writes |= descriptor.Base48() != 0 && descriptor.NumRecords() != 0;
	}
	return has_writes;
}

// Sync1 masks share their bit values with the synchronization2 ones.
static vk::PipelineStageFlags2 Stages2(vk::PipelineStageFlags stages) {
	return vk::PipelineStageFlags2(static_cast<VkPipelineStageFlags2>(
	    static_cast<VkPipelineStageFlags>(stages)));
}

static vk::AccessFlags2 Access2(vk::AccessFlags access) {
	return vk::AccessFlags2(static_cast<VkAccessFlags2>(static_cast<VkAccessFlags>(access)));
}

static void RecordShaderBarrier(const CommandBuffer& buffer, vk::PipelineStageFlags source_stages,
                                vk::PipelineStageFlags destination_stages,
                                const vk::MemoryBarrier& barrier, BarrierOrigin origin) {
	if (BarrierBatchEnabled()) {
		buffer.RequestMemoryBarrier(Stages2(source_stages), Access2(barrier.srcAccessMask),
		                            Stages2(destination_stages), Access2(barrier.dstAccessMask),
		                            origin);
		return;
	}
	buffer.Handle().pipelineBarrier(source_stages, destination_stages, vk::DependencyFlags {}, 1,
	                                &barrier, 0, nullptr, 0, nullptr);
}

void ShaderAccessBarrier(const CommandBuffer& buffer, vk::PipelineStageFlags source_stages) {
	KYTY_GPU_OP_SITE("shader.access_barrier");
	RecordShaderBarrier(buffer, source_stages, vk::PipelineStageFlagBits::eAllCommands,
	                    MakeShaderAccessDependency(), BarrierOrigin::ShaderAccess);
}

void ShaderWriteHazardBarrier(const CommandBuffer&   buffer,
                              vk::PipelineStageFlags destination_stages) {
	KYTY_GPU_OP_SITE("shader.write_hazard_barrier");
	RecordShaderBarrier(buffer, vk::PipelineStageFlagBits::eAllCommands, destination_stages,
	                    MakeShaderWriteHazardDependency(), BarrierOrigin::ShaderWriteHazard);
}

void ShaderWriteBarrier(const CommandBuffer& buffer, vk::PipelineStageFlags source_stages) {
	KYTY_GPU_OP_SITE("shader.write_barrier");
	RecordShaderBarrier(buffer, source_stages,
	                    vk::PipelineStageFlagBits::eComputeShader |
	                        vk::PipelineStageFlagBits::eAllGraphics |
	                        vk::PipelineStageFlagBits::eTransfer,
	                    MakeShaderWriteDependency(), BarrierOrigin::ShaderWrite);
}

} // namespace Libs::Graphics
