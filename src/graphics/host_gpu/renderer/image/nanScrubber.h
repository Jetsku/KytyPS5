#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_NANSCRUBBER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_NANSCRUBBER_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

class CommandBuffer;
struct GraphicContext;

// KYTY_NAN_SCRUB (TextureCache::RunNanScrub): rewrites, on the GPU, the float fields of a buffer of
// image words that are not finite: a NaN to zero, an infinity to the largest finite value
// (nan_scrub.comp). The caller copies an image's mips tightly packed into the buffer and back.
class NanScrubber {
public:
	// How the float bit fields sit in a word of the texels.
	enum class Packing : uint32_t { B10G11R11 = 0, TwoHalves = 1 };

	explicit NanScrubber(GraphicContext& graphics);
	~NanScrubber();
	KYTY_CLASS_NO_COPY(NanScrubber);

	// False when a pipeline could not be created (logged); Scrub then records nothing.
	[[nodiscard]] bool Valid() const { return m_pipelines[0] != nullptr && m_pipelines[1] != nullptr; }

	// Records the scrub of the words of [offset, offset + size) of `buffer` (storage usage), with
	// barriers from and to transfers on either side. Binds a compute pipeline and push descriptors.
	void Scrub(CommandBuffer& command, vk::Buffer buffer, uint64_t offset, uint64_t size,
	           Packing packing) const;

private:
	GraphicContext&             m_graphics;
	vk::DescriptorSetLayout     m_set_layout      = nullptr;
	vk::PipelineLayout          m_pipeline_layout = nullptr;
	std::array<vk::Pipeline, 2> m_pipelines {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_NANSCRUBBER_H_
