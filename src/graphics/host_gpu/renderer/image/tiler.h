#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TILER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TILER_H_

#include "common/common.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <chrono>
#include <mutex>
#include <span>
#include <utility>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

class CommandScheduler;
class Image;
class StreamBuffer;
struct GraphicContext;
struct TileManagerTestAccess;

struct GpuTileInfo {
	TileBlockFamily family              = TileBlockFamily::Count;
	uint32_t        bytes_per_element   = 0;
	uint64_t        linear_offset       = 0;
	uint64_t        linear_size         = 0;
	uint64_t        tiled_offset        = 0;
	uint64_t        tiled_size          = 0;
	uint64_t        linear_slice_stride = 0;
	uint32_t        width               = 0;
	uint32_t        height              = 0;
	uint32_t        depth               = 1;
	uint32_t        pitch               = 0;
	uint32_t        tail_x              = 0;
	uint32_t        tail_y              = 0;
	bool            tail                = false;
	uint32_t        tiled_width         = 0;
	uint32_t        tiled_height        = 0;
	uint32_t        surface_z           = 0;
};

class TileManager final {
public:
	enum class D16Direction { Promote, Demote };

	struct Result {
		vk::Buffer buffer = nullptr;
		uint64_t   offset = 0;
		uint64_t   size   = 0;
	};
	struct D16Layout {
		uint32_t width               = 0;
		uint32_t height              = 0;
		uint32_t layers              = 0;
		uint64_t source_row_stride   = 0;
		uint64_t target_row_stride   = 0;
		uint64_t source_slice_stride = 0;
		uint64_t target_slice_stride = 0;
	};

	TileManager(GraphicContext& graphics, CommandScheduler& scheduler, StreamBuffer& stream_buffer);
	~TileManager();
	KYTY_CLASS_NO_COPY(TileManager);

	// The returned device-local buffer remains alive through the current scheduler tick.
	[[nodiscard]] Result Detile(vk::Buffer tiled, uint64_t tiled_offset, uint64_t tiled_capacity,
	                            uint64_t linear_capacity, std::span<const GpuTileInfo> infos,
	                            ColorTransform transform = ColorTransform::None);
	void Tile(vk::Buffer linear, uint64_t linear_offset, uint64_t linear_capacity, vk::Buffer tiled,
	          uint64_t tiled_offset, uint64_t tiled_capacity, std::span<const GpuTileInfo> infos);
	void TileImage(Image& image, std::span<const vk::BufferImageCopy> regions, vk::Buffer tiled,
	               uint64_t tiled_offset, uint64_t tiled_capacity, uint64_t linear_capacity,
	               std::span<const GpuTileInfo> infos,
	               ColorTransform               transform = ColorTransform::None);
	// Direct image transfers (KYTY_TILER_IMAGE_DIRECT, default on): one compute pass moves each
	// element between the tiled buffer and element (x, y) of the image through an unsigned-integer
	// storage view, replacing Detile + Image::Upload (buffer->image copy) and the image->buffer
	// copy of TileImage. The bytes that reach the image or the tiled buffer are identical. Only
	// for single-sample 2D colour images with storage usage whose texel size equals the element
	// size; infos[i] describes regions[i]. Returns false without recording anything otherwise,
	// and the caller keeps its buffer path.
	[[nodiscard]] bool DetileToImage(Image& image, vk::Buffer tiled, uint64_t tiled_offset,
	                                 uint64_t tiled_capacity, uint64_t linear_capacity,
	                                 std::span<const GpuTileInfo>         infos,
	                                 std::span<const vk::BufferImageCopy> regions);
	[[nodiscard]] bool TileFromImage(Image& image, std::span<const vk::BufferImageCopy> regions,
	                                 vk::Buffer tiled, uint64_t tiled_offset,
	                                 uint64_t tiled_capacity, uint64_t linear_capacity,
	                                 std::span<const GpuTileInfo> infos);
	[[nodiscard]] static bool ImageDirectEnabled();
	// KYTY_TILER_IMAGE_DIRECT_VERIFY=1: every direct transfer also runs the buffer path (into
	// scratch) and a completion callback compares both results byte for byte (FrameEvent
	// TilerImageVerifyChecks / TilerImageVerifyMismatches; the first mismatches are logged).
	[[nodiscard]] static bool ImageDirectVerifyEnabled();
	[[nodiscard]] Result GetScratchBuffer(uint64_t size);
	// KYTY_VRAM_STATS: idle pooled scratch bytes and their limit.
	[[nodiscard]] std::pair<uint64_t, uint64_t> ScratchPoolBytes();
	// KYTY_TILER_SCRATCH_POOL_IDLE_MS: destroys pooled scratch buffers unused that long
	// (rate-limited to every 100 ms; TextureCache's garbage collector calls it).
	void                 TrimScratchPool();
	void                 ConvertD16(Result source, Result target, D16Direction direction, bool d32,
	                                const D16Layout& layout);
	[[nodiscard]] Result TransformColor(Result input, ColorTransform transform, bool to_host);
	void TransformColor(Result input, Result output, ColorTransform transform, bool to_host);

private:
	friend struct TileManagerTestAccess;

	static constexpr uint32_t FamilyCount          = static_cast<uint32_t>(TileBlockFamily::Count);
	static constexpr uint32_t BytesPerElementCount = 5;
	static constexpr uint32_t DirectionCount       = 2;
	static constexpr uint32_t PipelineCount = FamilyCount * BytesPerElementCount * DirectionCount;

	struct Push {
		uint32_t src_base;
		uint32_t dst_base;
		uint32_t width;
		uint32_t height;
		uint32_t depth;
		uint32_t surface_z;
		uint32_t pitch_bytes;
		uint32_t slice_bytes;
		uint32_t blocks_per_row;
		uint32_t blocks_per_slice;
		uint32_t tail_x;
		uint32_t tail_y;
		uint32_t tail;
		// ColorTransform: applied by the buffer tiling passes and TransformColor (whose push
		// constants end at to_host, gpu_tiler_color_transform.comp).
		uint32_t color_transform;
		uint32_t to_host;
		// Image variants only: the element's image texel is (image_x + x, image_y + y) of layer
		// image_layer + z.
		uint32_t image_x;
		uint32_t image_y;
		uint32_t image_layer;
	};
	struct Dispatch {
		Push     push {};
		uint32_t pipeline_slot = 0;
		uint64_t params_offset = 0;
	};
	struct Scratch {
		vk::Buffer    buffer     = nullptr;
		VmaAllocation allocation = nullptr;
		uint64_t      size       = 0;
		// Allocated capacity (a power-of-two size class when pooled, else size).
		uint64_t      capacity   = 0;
		// When it entered the pool (KYTY_TILER_SCRATCH_POOL_IDLE_MS).
		std::chrono::steady_clock::time_point released {};
	};
	struct StorageBinding {
		vk::DescriptorBufferInfo info;
		uint32_t                 base = 0;
	};

	[[nodiscard]] Scratch         AllocateScratch(uint64_t size);
	[[nodiscard]] StorageBinding  BindStorage(Result buffer, uint64_t size) const;
	[[nodiscard]] static uint32_t ConversionRows(uint64_t offset, uint64_t row_stride,
	                                             uint64_t active, uint32_t remaining,
	                                             uint64_t alignment, uint64_t max_range,
	                                             uint32_t max_groups) noexcept;
	void                          DeferDestroy(Scratch scratch);
	void                          ReleaseScratch(Scratch scratch);
	// image_regions (image variants): element (x, y) of infos[i] is view texel
	// (imageOffset / image_texel + (x, y)) of image_regions[i]; with image_layer_views each region
	// has its own single-layer view (layer 0 of the view), otherwise the view holds every layer.
	void Prepare(bool tile, uint64_t tiled_capacity, uint64_t linear_capacity,
	             std::span<const GpuTileInfo> infos, uint64_t source_base, uint64_t target_base,
	             std::vector<Dispatch>& dispatches,
	             std::span<const vk::BufferImageCopy> image_regions = {}, uint32_t image_texel = 1,
	             bool image_layer_views = false, ColorTransform transform = ColorTransform::None);
	void Record(vk::Buffer source, uint64_t source_offset, uint64_t source_capacity,
	            vk::Buffer target, uint64_t target_offset, uint64_t target_capacity,
	            std::span<Dispatch> dispatches, bool clear_target);
	[[nodiscard]] vk::Pipeline GetPipeline(uint32_t slot);
	// Direct image transfers: the storage view format for an element size (eUndefined when the
	// device cannot use it in that direction), eligibility, pipelines and recording.
	[[nodiscard]] vk::Format ImageViewFormat(uint32_t bytes_per_element, bool load);
	// Uncompressed views of block-compressed images hold one level and one layer.
	[[nodiscard]] static bool ImageLayerViews(const Image& image);
	[[nodiscard]] bool       ImageTransferEligible(const Image& image, bool load,
	                                               std::span<const GpuTileInfo>         infos,
	                                               std::span<const vk::BufferImageCopy> regions);
	[[nodiscard]] vk::Pipeline GetImagePipeline(bool load, TileBlockFamily family,
	                                            uint32_t bytes_per_element);
	void RecordImage(Image& image, bool load, vk::Buffer tiled, uint64_t tiled_offset,
	                 uint64_t tiled_capacity, std::span<const GpuTileInfo> infos,
	                 std::span<const vk::BufferImageCopy> regions, std::span<Dispatch> dispatches);
	// Verify mode: copies the two results into host memory and compares them on completion.
	// ranges: {offset, size} byte ranges compared (relative to both sources).
	void VerifyOnCompletion(const char* operation, uint64_t guest_address, vk::Buffer expected,
	                        uint64_t expected_offset, vk::Buffer actual, uint64_t actual_offset,
	                        uint64_t size, std::vector<std::pair<uint64_t, uint64_t>> ranges);

	GraphicContext&                         m_graphics;
	CommandScheduler&                       m_scheduler;
	StreamBuffer&                           m_stream_buffer;
	vk::DescriptorSetLayout                 m_descriptor_layout = nullptr;
	vk::PipelineLayout                      m_pipeline_layout   = nullptr;
	std::array<vk::Pipeline, PipelineCount> m_pipelines {};
	// Direct image transfers: [load][family][log2 element bytes], created on first use.
	vk::DescriptorSetLayout                 m_image_descriptor_layout = nullptr;
	vk::PipelineLayout                      m_image_pipeline_layout   = nullptr;
	std::array<vk::Pipeline, DirectionCount * FamilyCount * BytesPerElementCount>
	                                        m_image_pipelines {};
	// Per [load][log2 element bytes]: 0 unknown, 1 usable, 2 unusable.
	std::array<uint8_t, DirectionCount * BytesPerElementCount> m_image_view_support {};
	vk::Pipeline                            m_d16_to_d24      = nullptr;
	vk::Pipeline                            m_d16_to_d32      = nullptr;
	vk::Pipeline                            m_d24_to_d16      = nullptr;
	vk::Pipeline                            m_d32_to_d16      = nullptr;
	vk::Pipeline                            m_color_transform = nullptr;
	// Completed scratch buffers kept for reuse (KYTY_TILER_SCRATCH_POOL). Returned by the
	// deferred completion callback, so a pooled buffer is never in use by the GPU.
	std::mutex                              m_scratch_mutex;
	std::vector<Scratch>                    m_scratch_pool;
	uint64_t                                m_scratch_pool_bytes = 0;
	uint64_t                                m_scratch_pool_limit = 0;
	std::chrono::milliseconds               m_scratch_idle {0};
	std::chrono::steady_clock::time_point   m_scratch_trim_next {};
	bool                                    m_clear_detile_scratch = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TILER_H_
