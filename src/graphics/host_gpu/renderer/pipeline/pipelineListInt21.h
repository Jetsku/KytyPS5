#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELISTINT21_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELISTINT21_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// The vertex input info that int21 (744705ee1) journals and lists stored: this build added the mesh
// fast launch flag and passes, and the tessellation clamp levels and carried VGPRs. Tessellation
// stages are left out (their clamp levels come from guest registers the list does not keep).
// The emulator upgrades such a list when it loads it (pipelineCache.cpp, with its specializations);
// kyty_pipeline_list only recognizes the layout.
inline PipelineList::VertexLayoutUpgrade Int21VertexLayout() {
	static_assert(sizeof(ShaderMeshInputInfo) == 64 && sizeof(ShaderTessellationInputInfo) == 76 &&
	                  ShaderStoredInputInfoSize<ShaderVertexInputInfo>() == 10216,
	              "the vertex input info changed again: describe the int21 layout from the new one");
	const ShaderVertexInputInfo info;
	const auto offset = [&](const void* field) {
		return static_cast<uint32_t>(reinterpret_cast<const uint8_t*>(field) - reinterpret_cast<const uint8_t*>(&info));
	};
	// Defaults of the added fields as a draw of an ordinary vertex shader sets them.
	const auto mesh_at    = offset(&info.mesh.fast_launch);
	const auto mesh_added = offset(&info.tess) - mesh_at;
	const auto tess_at    = offset(&info.tess.min_tess_level);
	const auto tess_added = offset(&info.fetch_external) - tess_at;
	std::vector<uint8_t> mesh_bytes(mesh_added, 0);
	const uint32_t       passes = info.mesh.passes;
	std::memcpy(mesh_bytes.data() + (offset(&info.mesh.passes) - mesh_at), &passes, sizeof(passes));
	PipelineList::VertexLayoutUpgrade upgrade;
	upgrade.from_size = ShaderStoredInputInfoSize<ShaderVertexInputInfo>() - mesh_added - tess_added;
	upgrade.inserts.push_back({mesh_at, std::move(mesh_bytes)});
	upgrade.inserts.push_back({tess_at - mesh_added, std::vector<uint8_t>(tess_added, 0)});
	upgrade.skip_if_nonzero_at = offset(&info.tess.input_control_points) - mesh_added;
	return upgrade;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELISTINT21_H_ */
