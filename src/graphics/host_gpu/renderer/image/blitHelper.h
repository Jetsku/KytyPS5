#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <compare>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
class Image;
struct GraphicContext;

class BlitHelper final {
public:
	inline static constexpr auto ColorToMsDepthLayout =
	    vk::ImageLayout::eDepthStencilAttachmentOptimal;

	BlitHelper(GraphicContext& graphics, CommandScheduler& scheduler);
	~BlitHelper();
	KYTY_CLASS_NO_COPY(BlitHelper);

	void ReinterpretColorAsMsDepth(Image& source, Image& destination);
	// KYTY_MS_DEPTH_SAMPLE0 (from usa-ratt/KytyPS5-GT7 rounds 122-124): the first sample of a
	// multisample depth image into a single-sample D32 image of its size and layers
	// (CopyMsDepthSample0), and the first sample of its stencil into an R8_UINT image
	// (CopyMsStencilSample0), for texture reads that see it as single-sample.
	void CopyMsDepthSample0(Image& source, Image& destination);
	void CopyMsStencilSample0(Image& source, Image& destination);

	// Single-pass, bit-exact replacements for Image::CopyImageWithBuffer between a D32 depth
	// image (D32_SFLOAT or the depth aspect of D32_SFLOAT_S8_UINT, whose stencil is preserved)
	// and a 32-bit color image (both single-sampled 2D images with equal base extents).
	// The color side is accessed through an R32_UINT storage view (images are MUTABLE_FORMAT),
	// the depth side is sampled (depth -> color, compute) or written as gl_FragDepth
	// (color -> depth, fullscreen draw). Copies min(levels) levels and min(layers) layers.
	[[nodiscard]] bool SupportsDepthToColor32(const Image& source, const Image& destination) const;
	[[nodiscard]] bool SupportsColor32ToDepth(const Image& source, const Image& destination) const;
	void               CopyDepthToColor32(Image& source, Image& destination);
	void               CopyColor32ToDepth(Image& source, Image& destination);

private:
	enum class PipelineKind : uint32_t {
		ColorToMsDepth,
		Color32ToDepth,
		MsDepthSample0,
		MsStencilSample0
	};

	struct PipelineKey {
		uint32_t     samples = 1;
		vk::Format   format  = vk::Format::eUndefined;
		PipelineKind kind    = PipelineKind::ColorToMsDepth;

		auto operator<=>(const PipelineKey&) const = default;
	};

	struct Pipeline {
		PipelineKey  key;
		vk::Pipeline handle = nullptr;
	};

	struct DepthToColorPush {
		uint32_t width;
		uint32_t height;
		uint32_t level;
	};

	struct ColorToDepthPush {
		uint32_t layer;
	};

	[[nodiscard]] vk::Pipeline GetPipeline(PipelineKey key);
	void                       CopyMsSample0(Image& source, Image& destination, bool stencil);
	[[nodiscard]] bool         SupportsColor32Side(const Image& color) const;

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::ShaderModule        m_vertex_shader     = nullptr;
	vk::ShaderModule        m_fragment_shader   = nullptr;
	vk::ShaderModule        m_sample0_shader    = nullptr;
	vk::ShaderModule        m_stencil0_shader   = nullptr;
	std::vector<Pipeline>   m_pipelines;

	bool                    m_reinterpret_supported = false;
	vk::DescriptorSetLayout m_d2c_descriptor_layout = nullptr;
	vk::PipelineLayout      m_d2c_pipeline_layout   = nullptr;
	vk::Pipeline            m_d2c_pipeline          = nullptr;
	vk::DescriptorSetLayout m_c2d_descriptor_layout = nullptr;
	vk::PipelineLayout      m_c2d_pipeline_layout   = nullptr;
	vk::ShaderModule        m_c2d_fragment_shader   = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_
