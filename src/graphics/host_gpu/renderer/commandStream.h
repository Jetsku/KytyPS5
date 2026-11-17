#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSTREAM_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSTREAM_H_

// Command stream of the CP recorder (KYTY_CP_RECORDER, commandRecorder.h): fully resolved native
// command packets that the command processor encodes and one recorder thread replays, in order,
// into the guest scheduler's VkCommandBuffer (design: Profiling/analysis/CP-RECORDER-P2.md).
//
// Everything here is independent of the renderer runtime, so it is unit tested without a device
// (tests/CpRecorderTests.cpp):
//  - Ring: a single-producer single-consumer byte ring of 8-byte aligned packets that never
//    straddle its end (a Wrap packet pads to the end instead).
//  - Encoder: one method per native command, with the vk::CommandBuffer argument lists. Every
//    argument is copied into the packet (values and handles only, never a pointer to the caller's
//    memory), so the caller may reuse its arrays as soon as the method returns.
//  - Replay<Executor>: decodes one packet, rebuilds the native argument structures and calls the
//    executor method of the same name (the recorder's executor forwards to vk::CommandBuffer).
//  - Verify (KYTY_CP_RECORDER_VERIFY): Encoder computes VerifyHash::<Command> from its original
//    arguments and stores it with a sequence number; Replay computes the same function from the
//    arguments it passes to the executor and compares. Per command buffer, both sides also fold
//    every packet into a digest that Submit carries and the replayer compares.

#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/queueSubmission.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

namespace Libs::Graphics::CommandStream {

enum class Op : uint16_t {
	Wrap = 0,
	DrainMarker,
	Begin,
	Submit,
	BeginRendering,
	EndRendering,
	PipelineBarrier2,
	PipelineBarrier,
	CopyBuffer,
	CopyBufferToImage,
	CopyImageToBuffer,
	CopyImage,
	FillBuffer,
	BindPipeline,
	BindDescriptorSets,
	PushDescriptorSet,
	PushConstants,
	BindVertexBuffers2,
	BindIndexBuffer,
	SetViewportWithCount,
	SetScissorWithCount,
	SetLineWidth,
	SetBlendConstants,
	SetDepthTestEnable,
	SetDepthWriteEnable,
	SetDepthCompareOp,
	SetDepthBiasEnable,
	SetDepthBias,
	SetStencilTestEnable,
	SetStencilOp,
	SetStencilCompareMask,
	SetStencilWriteMask,
	SetStencilReference,
	SetCullMode,
	SetFrontFace,
	SetDepthBoundsTestEnable,
	SetDepthBounds,
	SetColorWriteEnable,
	SetAttachmentFeedbackLoopEnable,
	Draw,
	DrawIndexed,
	DrawMeshTasks,
	DrawMeshTasksIndirect,
	DrawMeshTasksIndirectCount,
	DrawIndirect,
	DrawIndexedIndirect,
	DrawIndirectCount,
	DrawIndexedIndirectCount,
	Dispatch,
	DispatchIndirect,
	DispatchBase,
	ResetQueryPool,
	BeginQuery,
	EndQuery,
	CopyQueryPoolResults,
	WriteTimestamp2,
	// vkUpdateDescriptorSets of one descriptor set (KYTY_RECORDER_DESCRIPTOR_SETS): a device call,
	// not a command, replayed in stream order before the command that binds the set.
	UpdateDescriptorSets,
	// vkCmdWriteBufferMarkerAMD (KYTY_GPU_BREADCRUMBS, gpuBreadcrumbs.h).
	WriteBufferMarker,
	// Transfer clears (TextureCache::ClearImage).
	ClearColorImage,
	ClearDepthStencilImage,
	// A host memcpy of guest bytes into mapped memory (KYTY_STREAM_COPY_DEFER, BufferCache): not a
	// command; replayed in stream order, so before the Submit of the commands that read it.
	HostCopy,
	// vkCmdInsertDebugUtilsLabelEXT (under --rd / KYTY_TRACE_VK_LABELS: pass and shader names for
	// RenderDoc); the packet carries the label text.
	InsertDebugLabel,
	// KYTY_EMIT_SPLIT (ResolvedDrawPacket below): one resolved draw; the consumer expands it into
	// the draw's state commands and the draw command, deduplicating against its own shadow.
	ResolvedDraw,
	Count,
};

[[nodiscard]] const char* OpName(Op op) noexcept;

// ------------------------------------------------------------------------------------------------
// Packet layout: Header [SiteBlock] [VerifyBlock] payload. Sizes are multiples of 8.

struct Header {
	Op       op    = Op::Wrap;
	uint16_t flags = 0;
	uint32_t size  = 0; // whole packet, bytes
};
static_assert(sizeof(Header) == 8);

inline constexpr uint16_t FlagSite   = 1u << 0u; // SiteBlock follows the header
inline constexpr uint16_t FlagVerify = 1u << 1u; // VerifyBlock follows (after the site block)

// GpuOpProfiler attribution of the producer (opaque here: GpuOpProfiler::Site pointers).
struct SiteBlock {
	const void* site  = nullptr;
	const void* scope = nullptr;
};

struct VerifyBlock {
	uint64_t hash     = 0;
	uint64_t sequence = 0;
};

inline constexpr uint32_t Align8(uint64_t bytes) noexcept {
	return static_cast<uint32_t>((bytes + 7u) & ~uint64_t {7u});
}

// Fixed packet parts. Arrays follow them, each 8-byte aligned.
struct DrainMarkerPacket {
	uint64_t serial = 0;
};
struct BeginPacket {
	vk::CommandBuffer command   = nullptr;
	uint64_t          tick      = 0;
	uint64_t          record_ns = 0;
};
struct SubmitPacket {
	SubmitInfo submit;
	uint64_t   tick                = 0;
	uint64_t   submit_ns           = 0;
	uint64_t   debug_submit        = 0;
	uint64_t   debug_arg4          = 0;
	uint32_t   debug_op            = 0;
	uint32_t   debug_arg0          = 0;
	uint32_t   debug_arg1          = 0;
	uint32_t   debug_arg2          = 0;
	uint32_t   debug_arg3          = 0;
	uint32_t   preserve_completion = 0;
	// Verify: the producer's digest and packet count of this command buffer since its Begin.
	uint64_t cb_digest  = 0;
	uint64_t cb_packets = 0;
};
struct BeginRenderingPacket {
	vk::Rect2D         area {};
	vk::RenderingFlags flags {};
	uint32_t           layers      = 0;
	uint32_t           view_mask   = 0;
	uint32_t           colors      = 0;
	uint32_t           has_depth   = 0;
	uint32_t           has_stencil = 0;
	// Followed by `colors` vk::RenderingAttachmentInfo, then depth, then stencil.
};
struct PipelineBarrier2Packet {
	vk::DependencyFlags flags {};
	uint32_t            memory_count = 0;
	uint32_t            buffer_count = 0;
	uint32_t            image_count  = 0;
};
struct PipelineBarrierPacket {
	vk::PipelineStageFlags src_stages {};
	vk::PipelineStageFlags dst_stages {};
	vk::DependencyFlags    flags {};
	uint32_t               memory_count = 0;
	uint32_t               buffer_count = 0;
	uint32_t               image_count  = 0;
};
struct CopyBufferPacket {
	vk::Buffer source      = nullptr;
	vk::Buffer destination = nullptr;
	uint32_t   count       = 0;
};
struct CopyBufferToImagePacket {
	vk::Buffer      source      = nullptr;
	vk::Image       destination = nullptr;
	vk::ImageLayout layout      = vk::ImageLayout::eUndefined;
	uint32_t        count       = 0;
};
struct CopyImageToBufferPacket {
	vk::Image       source      = nullptr;
	vk::Buffer      destination = nullptr;
	vk::ImageLayout layout      = vk::ImageLayout::eUndefined;
	uint32_t        count       = 0;
};
struct CopyImagePacket {
	vk::Image       source             = nullptr;
	vk::Image       destination        = nullptr;
	vk::ImageLayout source_layout      = vk::ImageLayout::eUndefined;
	vk::ImageLayout destination_layout = vk::ImageLayout::eUndefined;
	uint32_t        count              = 0;
};
struct FillBufferPacket {
	vk::Buffer buffer = nullptr;
	uint64_t   offset = 0;
	uint64_t   size   = 0;
	uint32_t   value  = 0;
};
struct BindPipelinePacket {
	vk::Pipeline          pipeline = nullptr;
	vk::PipelineBindPoint point    = vk::PipelineBindPoint::eGraphics;
};
struct BindDescriptorSetsPacket {
	vk::PipelineLayout    layout        = nullptr;
	vk::PipelineBindPoint point         = vk::PipelineBindPoint::eGraphics;
	uint32_t              first_set     = 0;
	uint32_t              set_count     = 0;
	uint32_t              dynamic_count = 0;
};
struct PushDescriptorSetPacket {
	vk::PipelineLayout    layout      = nullptr;
	vk::PipelineBindPoint point       = vk::PipelineBindPoint::eGraphics;
	uint32_t              set         = 0;
	uint32_t              write_count = 0;
	uint32_t              info_count  = 0; // DescriptorInfo entries after the write records
};
// One vk::WriteDescriptorSet without its pointers; its descriptorCount infos follow in order.
struct DescriptorWriteRecord {
	uint32_t           binding  = 0;
	uint32_t           element  = 0;
	uint32_t           count    = 0;
	vk::DescriptorType type     = vk::DescriptorType::eSampler;
	uint32_t           is_image = 0;
	uint32_t           pad      = 0;
};
// A vk::DescriptorBufferInfo or vk::DescriptorImageInfo (both 24 bytes).
struct DescriptorInfo {
	uint64_t words[3] = {};
};
static_assert(sizeof(vk::DescriptorBufferInfo) == sizeof(DescriptorInfo));
static_assert(sizeof(vk::DescriptorImageInfo) == sizeof(DescriptorInfo));
// Every write targets `set` (the replay sets each dstSet to it); records and infos as for
// PushDescriptorSetPacket.
struct UpdateDescriptorSetsPacket {
	vk::DescriptorSet set         = nullptr;
	uint32_t          write_count = 0;
	uint32_t          info_count  = 0;
};
struct PushConstantsPacket {
	vk::PipelineLayout   layout = nullptr;
	vk::ShaderStageFlags stages {};
	uint32_t             offset = 0;
	uint32_t             size   = 0;
};
struct BindVertexBuffers2Packet {
	uint32_t first       = 0;
	uint32_t count       = 0;
	uint32_t has_sizes   = 0;
	uint32_t has_strides = 0;
};
struct BindIndexBufferPacket {
	vk::Buffer    buffer = nullptr;
	uint64_t      offset = 0;
	vk::IndexType type   = vk::IndexType::eUint16;
};
struct CountPacket {
	uint32_t count = 0;
};
struct FloatPacket {
	float value = 0.0f;
};
struct Float2Packet {
	float values[2] = {};
};
struct Float3Packet {
	float values[3] = {};
};
struct Float4Packet {
	float values[4] = {};
};
struct UintPacket {
	uint32_t value = 0;
};
struct StencilOpPacket {
	vk::StencilFaceFlags faces {};
	vk::StencilOp        fail       = vk::StencilOp::eKeep;
	vk::StencilOp        pass       = vk::StencilOp::eKeep;
	vk::StencilOp        depth_fail = vk::StencilOp::eKeep;
	vk::CompareOp        compare    = vk::CompareOp::eNever;
};
struct StencilValuePacket {
	vk::StencilFaceFlags faces {};
	uint32_t             value = 0;
};
struct DrawPacket {
	uint32_t vertex_count   = 0;
	uint32_t instance_count = 0;
	uint32_t first_vertex   = 0;
	uint32_t first_instance = 0;
};
struct DrawIndexedPacket {
	uint32_t index_count    = 0;
	uint32_t instance_count = 0;
	uint32_t first_index    = 0;
	int32_t  vertex_offset  = 0;
	uint32_t first_instance = 0;
};
struct Groups3Packet {
	uint32_t x = 0;
	uint32_t y = 0;
	uint32_t z = 0;
};
struct DispatchBasePacket {
	uint32_t base_x = 0;
	uint32_t base_y = 0;
	uint32_t base_z = 0;
	uint32_t x      = 0;
	uint32_t y      = 0;
	uint32_t z      = 0;
};
struct IndirectPacket {
	vk::Buffer buffer     = nullptr;
	uint64_t   offset     = 0;
	uint32_t   draw_count = 0;
	uint32_t   stride     = 0;
};
struct IndirectCountPacket {
	vk::Buffer buffer       = nullptr;
	uint64_t   offset       = 0;
	vk::Buffer count_buffer = nullptr;
	uint64_t   count_offset = 0;
	uint32_t   max_count    = 0;
	uint32_t   stride       = 0;
};
struct DispatchIndirectPacket {
	vk::Buffer buffer = nullptr;
	uint64_t   offset = 0;
};
struct QueryRangePacket {
	vk::QueryPool pool  = nullptr;
	uint32_t      first = 0;
	uint32_t      count = 0;
};
struct BeginQueryPacket {
	vk::QueryPool         pool  = nullptr;
	uint32_t              query = 0;
	vk::QueryControlFlags flags {};
};
struct CopyQueryPoolResultsPacket {
	vk::QueryPool        pool        = nullptr;
	vk::Buffer           destination = nullptr;
	uint64_t             offset      = 0;
	uint64_t             stride      = 0;
	uint32_t             first       = 0;
	uint32_t             count       = 0;
	vk::QueryResultFlags flags {};
};
struct WriteTimestampPacket {
	vk::QueryPool pool  = nullptr;
	uint64_t      stage = 0; // VkPipelineStageFlags2
	uint32_t      query = 0;
};
struct HostCopyPacket {
	uint64_t destination = 0; // host pointer (mapped stream ring)
	uint64_t source      = 0; // guest virtual address
	uint64_t size        = 0;
};
struct WriteBufferMarkerPacket {
	vk::Buffer buffer = nullptr;
	uint64_t   offset = 0;
	uint32_t   stage  = 0; // VkPipelineStageFlagBits
	uint32_t   marker = 0;
};
// KYTY_EMIT_SPLIT: a resolved draw. The commit thread resolved every buffer, image and pipeline; the
// packet holds values and handles only (the same handles today's packets carry, so the same
// lifetime rules: deferred deletion, KYTY_DEAD_VIEW_SCRUB). The consumer (Replay) expands it, in
// this order, into:
//   bindVertexBuffers2(0, vertex_count, buffers, offsets, sizes)            (FlagVertex)
//   pushConstants(layout, push_stages, 0, 128, push)                        (FlagPushData, shadowed)
//   pushDescriptorSetKHR(graphics, layout, 0, writes)                       (FlagPush, shadowed)
//   bindDescriptorSets(graphics, layout, 1, {bindless_set})                 (FlagBindless)
//   bindIndexBuffer(index_buffer, index_offset, index_type)                 (FlagIndex)
//   bindPipeline(graphics, pipeline)                                        (FlagBind)
//   pushConstants(layout, clip_stages, clip_offset, clip_bytes, clip)       (clip_bytes != 0)
//   pushConstants(layout, objprim_stages, objprim_offset, 4, &object_id)    (FlagObjprim)
//   drawIndexed(count, instance_count, 0, vertex_offset, first_instance)    (FlagIndexed)
//   or draw(count, instance_count, first_vertex, first_instance)
// KYTY_EMIT_SPLIT_COPIES: the deferred stream copies (HostCopy) the encoder held back since its last
// packet travel in the record and run first, in their order (HostCopyPacket[copy_count] first).
// Arrays after the fixed part, each 8-byte aligned: vk::Buffer[vertex_count],
// vk::DeviceSize[vertex_count] offsets and sizes, DescriptorWriteRecord[write_count],
// DescriptorInfo[info_count], uint32_t[ResolvedDrawPushDwords] (FlagPushData), uint32_t[clip_bytes/4].
// The shadow comparisons the commit thread made for these commands happen on the consumer
// (ResolvedDrawShadow): a push identical to the previous resolved draw's push of this command
// buffer is not recorded. Any other packet that changes that state (a push descriptor update or a
// set-0 bind at the graphics bind point, a push-constant update, Begin) forgets the shadow, and the
// producer sets FlagShadowReset when commands were recorded natively since its last resolved draw
// (a direct window). FlagVerifyOnly: nothing is recorded; the packet is compared with the state the
// preceding packets left (ResolvedDrawModel, verify mode).
inline constexpr uint32_t ResolvedDrawPushDwords = 32;
struct ResolvedDrawPacket {
	static constexpr uint32_t FlagVertex      = 1u << 0u;
	static constexpr uint32_t FlagPushData    = 1u << 1u;
	static constexpr uint32_t FlagPush        = 1u << 2u;
	static constexpr uint32_t FlagKnownMiss   = 1u << 3u; // record the push without comparing or keeping it
	static constexpr uint32_t FlagBindless    = 1u << 4u;
	static constexpr uint32_t FlagIndex       = 1u << 5u;
	static constexpr uint32_t FlagBind        = 1u << 6u;
	static constexpr uint32_t FlagObjprim     = 1u << 7u;
	static constexpr uint32_t FlagIndexed     = 1u << 8u;
	static constexpr uint32_t FlagVerifyOnly  = 1u << 9u;
	static constexpr uint32_t FlagShadowReset = 1u << 10u;
	static constexpr uint32_t FlagPushDedup   = 1u << 11u; // DrawEncode::ShadowFlags::renderer_batch
	static constexpr uint32_t FlagConstDedup  = 1u << 12u; // DrawEncode::ShadowFlags::push_constant_shadow

	vk::PipelineLayout layout       = nullptr;
	vk::Pipeline       pipeline     = nullptr;
	vk::Buffer         index_buffer = nullptr;
	uint64_t           index_offset = 0;
	vk::DescriptorSet  bindless_set = nullptr;
	uint32_t           flags        = 0;
	uint32_t           index_type   = 0; // vk::IndexType
	uint32_t           vertex_count = 0;
	uint32_t           write_count  = 0;
	uint32_t           info_count   = 0;
	uint32_t           push_stages  = 0; // vk::ShaderStageFlags
	uint32_t           clip_offset  = 0;
	uint32_t           clip_bytes   = 0;
	uint32_t           clip_stages  = 0;
	uint32_t           objprim_stages = 0;
	uint32_t           objprim_offset = 0;
	uint32_t           object_id      = 0;
	uint32_t           count          = 0; // indices or vertices
	uint32_t           instance_count = 0;
	int32_t            vertex_offset  = 0;
	uint32_t           first_vertex   = 0;
	uint32_t           first_instance = 0;
	uint32_t           copy_count     = 0; // HostCopyPacket[] first among the arrays (set by the encoder)
};
// The arrays of a resolved draw (producer side; null when the count is 0 or the flag is clear).
struct ResolvedDrawArrays {
	const vk::Buffer*            vertex_buffers = nullptr;
	const vk::DeviceSize*        vertex_offsets = nullptr;
	const vk::DeviceSize*        vertex_sizes   = nullptr;
	const DescriptorWriteRecord* writes         = nullptr;
	const DescriptorInfo*        infos          = nullptr;
	const uint32_t*              push           = nullptr;
	const uint32_t*              clip           = nullptr;
};

// ClearColorImage: value = the vk::ClearColorValue's 16 bytes. ClearDepthStencilImage: value[0] =
// the depth's bits, value[1] = the stencil. vk::ImageSubresourceRange[count] follow.
struct ClearImagePacket {
	vk::Image       image  = nullptr;
	vk::ImageLayout layout = vk::ImageLayout::eUndefined;
	uint32_t        count  = 0;
	uint32_t        value[4] {};
};

// ------------------------------------------------------------------------------------------------
// Verify hashes: one function per native command over its complete argument values (arrays and
// structures field by field; pNext chains must be empty). Encoder and Replay call the same
// function, from the caller's original arguments and from the rebuilt ones respectively.

class Hasher {
public:
	explicit Hasher(Op op) noexcept { Add(static_cast<uint64_t>(op) + 0x51ed270b27ea1a3bull); }
	void Add(uint64_t value) noexcept {
		m_state ^= value + 0x9e3779b97f4a7c15ull + (m_state << 6u) + (m_state >> 2u);
		m_state *= 0xff51afd7ed558ccdull;
		m_state ^= m_state >> 33u;
	}
	template <typename Handle>
	void AddHandle(Handle handle) noexcept {
		using C     = typename Handle::CType;
		const C raw = static_cast<C>(handle);
		static_assert(sizeof(C) == sizeof(uint64_t));
		uint64_t bits = 0;
		std::memcpy(&bits, &raw, sizeof(bits));
		Add(bits);
	}
	template <typename Flags>
	void AddFlags(Flags flags) noexcept {
		using M = typename Flags::MaskType;
		Add(static_cast<uint64_t>(static_cast<M>(flags)));
	}
	template <typename Enum>
	void AddEnum(Enum value) noexcept {
		Add(static_cast<uint64_t>(static_cast<std::underlying_type_t<Enum>>(value)));
	}
	void AddFloat(float value) noexcept { Add(std::bit_cast<uint32_t>(value)); }
	void AddBytes(const void* data, size_t size) noexcept {
		const auto* bytes = static_cast<const uint8_t*>(data);
		Add(size);
		size_t offset = 0;
		for (; offset + 8 <= size; offset += 8) {
			uint64_t word = 0;
			std::memcpy(&word, bytes + offset, 8);
			Add(word);
		}
		if (offset < size) {
			uint64_t word = 0;
			std::memcpy(&word, bytes + offset, size - offset);
			Add(word);
		}
	}
	[[nodiscard]] uint64_t Value() const noexcept { return m_state == 0 ? 1 : m_state; }

private:
	uint64_t m_state = 0x243f6a8885a308d3ull;
};

namespace VerifyHash {
uint64_t Begin(vk::CommandBuffer command, uint64_t tick);
uint64_t Submit(const SubmitPacket& submit);
uint64_t DrainMarker(uint64_t serial);
uint64_t BeginRendering(const vk::RenderingInfo& info);
uint64_t EndRendering();
uint64_t PipelineBarrier2(const vk::DependencyInfo& info);
uint64_t PipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst,
                         vk::DependencyFlags flags, uint32_t memory_count,
                         const vk::MemoryBarrier* memory, uint32_t buffer_count,
                         const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
                         const vk::ImageMemoryBarrier* images);
uint64_t CopyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
                    const vk::BufferCopy* regions);
uint64_t CopyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
                           uint32_t count, const vk::BufferImageCopy* regions);
uint64_t CopyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
                           uint32_t count, const vk::BufferImageCopy* regions);
uint64_t CopyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
                   vk::ImageLayout destination_layout, uint32_t count,
                   const vk::ImageCopy* regions);
uint64_t FillBuffer(vk::Buffer buffer, uint64_t offset, uint64_t size, uint32_t value);
uint64_t BindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline);
uint64_t BindDescriptorSets(vk::PipelineBindPoint point, vk::PipelineLayout layout,
                            uint32_t first_set, uint32_t set_count, const vk::DescriptorSet* sets,
                            uint32_t dynamic_count, const uint32_t* dynamic_offsets);
uint64_t PushDescriptorSet(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t set,
                           uint32_t count, const vk::WriteDescriptorSet* writes);
uint64_t UpdateDescriptorSets(vk::DescriptorSet set, uint32_t count,
                              const vk::WriteDescriptorSet* writes);
uint64_t PushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
                       uint32_t size, const void* data);
uint64_t BindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
                            const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
                            const vk::DeviceSize* strides);
uint64_t BindIndexBuffer(vk::Buffer buffer, uint64_t offset, vk::IndexType type);
uint64_t Viewports(uint32_t count, const vk::Viewport* viewports);
uint64_t Scissors(uint32_t count, const vk::Rect2D* scissors);
uint64_t Value(Op op, uint64_t a, uint64_t b = 0, uint64_t c = 0, uint64_t d = 0, uint64_t e = 0,
               uint64_t f = 0);
uint64_t Floats(Op op, const float* values, uint32_t count);
uint64_t DebugLabel(const char* name, uint32_t size);
// The payload bytes of a ResolvedDraw packet (fixed part and arrays, as encoded).
uint64_t ResolvedDraw(const void* payload, uint64_t size);
uint64_t ColorWriteEnable(uint32_t count, const vk::Bool32* enables);
uint64_t ClearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue& color,
                         uint32_t count, const vk::ImageSubresourceRange* ranges);
uint64_t ClearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
                                const vk::ClearDepthStencilValue& value, uint32_t count,
                                const vk::ImageSubresourceRange* ranges);
} // namespace VerifyHash

// Order-sensitive fold of one packet into a command-buffer digest.
[[nodiscard]] inline uint64_t FoldDigest(uint64_t digest, Op op, uint64_t hash) noexcept {
	Hasher h(op);
	h.Add(digest);
	h.Add(hash);
	return h.Value();
}

// ------------------------------------------------------------------------------------------------
// Ring

struct WaitStats {
	uint64_t spins   = 0; // waits that spun
	uint64_t blocks  = 0; // waits that blocked in the OS
	uint64_t wait_ns = 0; // total time waiting
	uint64_t wakes   = 0; // OS wakes issued to the other side
	uint64_t spin_tsc = 0; // WaitConsumed: TSC ticks spent spinning (KYTY_SPIN_BUDGET counters)
};

// Spin, then (after spin_ns) block. A spin of 0 blocks at once.
struct WaitPolicy {
	uint64_t spin_ns = 30000;
};

class Ring {
public:
	// capacity: a power of two, at least 64 KiB.
	explicit Ring(uint64_t capacity);
	~Ring();
	KYTY_CLASS_NO_COPY(Ring);

	[[nodiscard]] uint64_t Capacity() const noexcept { return m_capacity; }
	[[nodiscard]] uint32_t MaxPacket() const noexcept {
		return static_cast<uint32_t>(m_capacity / 4u);
	}

	// ---- Producer (one thread) ----

	// A contiguous region of `bytes` (8-byte multiple, at most MaxPacket()) at the write
	// position. Emits a Wrap packet first when the region would cross the end, and waits for the
	// consumer when there is no room.
	[[nodiscard]] uint8_t* Reserve(uint32_t bytes, const WaitPolicy& policy, WaitStats& stats);
	// Advances the write position past a region filled after Reserve. Publication is batched:
	// every SetPublishBatch() bytes (0: every packet), and at every Publish()/Kick(); publishing
	// each packet moves the index's cache line to a polling consumer once per packet.
	void SetPublishBatch(uint64_t bytes) noexcept { m_publish_batch = bytes; }
	void Commit(uint32_t bytes) noexcept {
		m_write += bytes;
		if (m_write - m_last_published >= m_publish_batch) {
			Publish();
		}
	}
	// Makes everything committed visible to the consumer (release), without a wake.
	void Publish() noexcept {
		m_last_published = m_write;
		m_published.store(m_write, std::memory_order_release);
	}
	// Publication plus a wake of a parked consumer (seq_cst handshake). Call before the producer
	// waits for anything the consumer does.
	void                   Kick(WaitStats& stats) noexcept;
	[[nodiscard]] uint64_t WritePosition() const noexcept { return m_write; }
	// The packet at a producer position the producer wrote (it stays intact until the producer
	// writes over it): the producer reading back its own packets (KYTY_ENC_OFFLOAD verify).
	[[nodiscard]] const Header* At(uint64_t position) const noexcept {
		return reinterpret_cast<const Header*>(m_data + (position & m_mask));
	}
	// Waits until the consumer has released everything up to `position`.
	void WaitConsumed(uint64_t position, const WaitPolicy& policy, WaitStats& stats);
	[[nodiscard]] uint64_t Consumed() const noexcept {
		return m_consumed.load(std::memory_order_acquire);
	}

	// ---- Consumer (one thread) ----

	// The next packet (Wrap packets included), or nullptr when nothing more is published.
	[[nodiscard]] const Header* Peek() noexcept {
		if (m_read == m_published_cache) {
			m_published_cache = m_published.load(std::memory_order_acquire);
			if (m_read == m_published_cache) {
				return nullptr;
			}
		}
		return reinterpret_cast<const Header*>(m_data + (m_read & m_mask));
	}
	void Advance(uint32_t bytes) noexcept { m_read += bytes; }
	// Publishes the consumed position and wakes a producer waiting for it.
	void                   Release(WaitStats& stats) noexcept;
	[[nodiscard]] uint64_t ReadPosition() const noexcept { return m_read; }
	// Waits until something is published (spin, then park) or `stop` is set. Returns false on
	// stop with nothing left to read.
	bool WaitPublished(const std::atomic<bool>& stop, const WaitPolicy& policy, WaitStats& stats);
	// Wakes a parked consumer unconditionally (shutdown).
	void WakeConsumer() noexcept;

private:
	bool EnsureSpace(uint64_t bytes, const WaitPolicy& policy, WaitStats& stats);

	uint8_t* m_data     = nullptr;
	uint64_t m_capacity = 0;
	uint64_t m_mask     = 0;

	// Each side parks on its own doorbell (a counter the other side increments before notifying),
	// announced through a flag the other side reads after its seq_cst store of the position.
	alignas(64) std::atomic<uint64_t> m_published {0};
	std::atomic<uint32_t> m_consumer_parked {0};
	std::atomic<uint32_t> m_consumer_bell {0};
	alignas(64) std::atomic<uint64_t> m_consumed {0};
	std::atomic<uint32_t> m_producer_waiting {0};
	std::atomic<uint32_t> m_producer_bell {0};
	// Producer-local.
	alignas(64) uint64_t m_write = 0;
	uint64_t m_last_published    = 0;
	uint64_t m_publish_batch     = 4u << 10u;
	uint64_t m_consumed_cache    = 0;
	// Consumer-local.
	alignas(64) uint64_t m_read = 0;
	uint64_t m_published_cache  = 0;
};

// ------------------------------------------------------------------------------------------------
// Encoder (producer side)

class Encoder {
public:
	struct Options {
		bool verify = false;
		// GpuOpProfiler attribution: sites on barrier packets (the only hooks that read the site
		// outside a capture), or on every packet (capture mode).
		bool barrier_sites = false;
		bool all_sites     = false;
		// Returns the producer's current site and scope (GpuOpProfiler::Detail).
		void (*current_site)(const void** site, const void** scope) = nullptr;
		// Called after every committed packet (inline mode replays it at once).
		void (*after_commit)(void* context) = nullptr;
		void* after_commit_context          = nullptr;
		// Spin before blocking when the ring is full.
		WaitPolicy policy;
	};

	// Packets after which a parked consumer is woken: the ends of emission sequences (draws,
	// dispatches) and everything the producer may wait for next (Submit, DrainMarker). Other
	// packets are published without a wake; a spinning consumer sees them at once.
	[[nodiscard]] static constexpr bool KicksConsumer(Op op) noexcept {
		switch (op) {
			case Op::Submit:
			case Op::DrainMarker:
			case Op::Draw:
			case Op::DrawIndexed:
			case Op::DrawMeshTasks:
			case Op::DrawMeshTasksIndirect:
			case Op::DrawMeshTasksIndirectCount:
			case Op::DrawIndirect:
			case Op::DrawIndexedIndirect:
			case Op::DrawIndirectCount:
			case Op::DrawIndexedIndirectCount:
			case Op::Dispatch:
			case Op::DispatchIndirect:
			case Op::DispatchBase:
			case Op::ResolvedDraw: return true;
			default: return false;
		}
	}

	Encoder(Ring& ring, const Options& options): m_ring(ring), m_options(options) {}

	[[nodiscard]] Ring&            GetRing() noexcept { return m_ring; }
	[[nodiscard]] WaitStats&       Stats() noexcept { return m_stats; }
	[[nodiscard]] const WaitStats& Stats() const noexcept { return m_stats; }
	[[nodiscard]] const Options&   GetOptions() const noexcept { return m_options; }
	[[nodiscard]] uint64_t         Packets() const noexcept { return m_packets; }
	[[nodiscard]] uint64_t         Bytes() const noexcept { return m_bytes; }
	// Verify: digest and packet count of the current command buffer so far.
	[[nodiscard]] uint64_t CommandDigest() const noexcept { return m_cb_digest; }
	[[nodiscard]] uint64_t CommandPackets() const noexcept { return m_cb_packets; }

	// KYTY_ENC_OFFLOAD (encodeOffload.h): called once before the next packet is opened (then
	// cleared), so packets encoded elsewhere are spliced in ahead of it.
	void SetOpenHook(void (*hook)(void* context), void* context) noexcept {
		m_open_hook         = hook;
		m_open_hook_context = context;
	}
	[[nodiscard]] bool HasOpenHook() const noexcept { return m_open_hook != nullptr; }
	// Appends the packets another encoder with the same options wrote into `source` from its read
	// position up to `end` (Wrap packets skipped), consuming them there: each packet is copied as it
	// is, and its verify sequence renumbered and folded into this command buffer's digest exactly as
	// Close would. Returns the bytes appended.
	uint64_t Splice(Ring& source, uint64_t end);

	void Begin(vk::CommandBuffer command, uint64_t tick, uint64_t record_ns);
	// `submit` fields other than the digest; the encoder fills the verify digest and count.
	void Submit(const SubmitPacket& submit);
	// Returns the ring position just past the marker.
	uint64_t DrainMarker(uint64_t serial);

	void beginRendering(const vk::RenderingInfo& info);
	void endRendering();
	void pipelineBarrier2(const vk::DependencyInfo& info);
	void pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst,
	                     vk::DependencyFlags flags, uint32_t memory_count,
	                     const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images);
	void copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
	                const vk::BufferCopy* regions);
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       uint32_t count, const vk::BufferImageCopy* regions);
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       uint32_t count, const vk::BufferImageCopy* regions);
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count,
	               const vk::ImageCopy* regions);
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size, uint32_t value);
	void bindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline);
	void bindDescriptorSets(vk::PipelineBindPoint point, vk::PipelineLayout layout,
	                        uint32_t first_set, uint32_t set_count, const vk::DescriptorSet* sets,
	                        uint32_t dynamic_count, const uint32_t* dynamic_offsets);
	void pushDescriptorSetKHR(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t set,
	                          uint32_t count, const vk::WriteDescriptorSet* writes);
	// Every write's dstSet must be `set`.
	void updateDescriptorSets(vk::DescriptorSet set, uint32_t count,
	                          const vk::WriteDescriptorSet* writes);
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
	                   uint32_t size, const void* data);
	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
	                        const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides);
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type);
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports);
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors);
	void setLineWidth(float width);
	void setBlendConstants(const float constants[4]);
	void setDepthTestEnable(vk::Bool32 enable);
	void setDepthWriteEnable(vk::Bool32 enable);
	void setDepthCompareOp(vk::CompareOp op);
	void setDepthBiasEnable(vk::Bool32 enable);
	void setDepthBias(float constant, float clamp, float slope);
	void setStencilTestEnable(vk::Bool32 enable);
	void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
	                  vk::StencilOp depth_fail, vk::CompareOp compare);
	void setStencilCompareMask(vk::StencilFaceFlags faces, uint32_t mask);
	void setStencilWriteMask(vk::StencilFaceFlags faces, uint32_t mask);
	void setStencilReference(vk::StencilFaceFlags faces, uint32_t reference);
	void setCullMode(vk::CullModeFlags mode);
	void setFrontFace(vk::FrontFace face);
	void setDepthBoundsTestEnable(vk::Bool32 enable);
	void setDepthBounds(float min, float max);
	void setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables);
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects);
	void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
	          uint32_t first_instance);
	void drawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
	                 int32_t vertex_offset, uint32_t first_instance);
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z);
	void drawMeshTasksIndirectEXT(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                              uint32_t stride);
	void drawMeshTasksIndirectCountEXT(vk::Buffer buffer, vk::DeviceSize offset,
	                                   vk::Buffer count_buffer, vk::DeviceSize count_offset,
	                                   uint32_t max_count, uint32_t stride);
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                  uint32_t stride);
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                         uint32_t stride);
	void drawIndirectCount(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,
	                       vk::DeviceSize count_offset, uint32_t max_count, uint32_t stride);
	void drawIndexedIndirectCount(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,
	                              vk::DeviceSize count_offset, uint32_t max_count, uint32_t stride);
	void dispatch(uint32_t x, uint32_t y, uint32_t z);
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset);
	void dispatchBase(uint32_t base_x, uint32_t base_y, uint32_t base_z, uint32_t x, uint32_t y,
	                  uint32_t z);
	void resetQueryPool(vk::QueryPool pool, uint32_t first, uint32_t count);
	void beginQuery(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags);
	void endQuery(vk::QueryPool pool, uint32_t query);
	void copyQueryPoolResults(vk::QueryPool pool, uint32_t first, uint32_t count,
	                          vk::Buffer destination, vk::DeviceSize offset, vk::DeviceSize stride,
	                          vk::QueryResultFlags flags);
	void writeTimestamp2(vk::PipelineStageFlags2 stage, vk::QueryPool pool, uint32_t query);
	void hostCopy(void* destination, uint64_t source, uint64_t size);
	void insertDebugLabel(const char* name);
	// KYTY_EMIT_SPLIT: one resolved draw (ResolvedDrawPacket); kicks the consumer like a draw.
	void resolvedDraw(const ResolvedDrawPacket& packet, const ResolvedDrawArrays& arrays);
	// The same in two steps: open writes everything but the infos and the push data (arrays.infos and
	// arrays.push are not read) and returns where they go; the caller fills them, then closes. Nothing
	// else may be encoded in between.
	struct ResolvedDrawSlots {
		DescriptorInfo* infos = nullptr;
		uint32_t*       push  = nullptr; // null without FlagPushData
	};
	[[nodiscard]] ResolvedDrawSlots openResolvedDraw(const ResolvedDrawPacket& packet, const ResolvedDrawArrays& arrays);
	void                            closeResolvedDraw();
	// KYTY_EMIT_SPLIT_COPIES (live, default 0): hostCopy holds its copy back (up to MaxHeldCopies); the
	// next resolved draw carries the held copies, any other packet first writes them as HostCopy
	// packets, so their order against every other packet is unchanged. Whoever waits for the consumer
	// to have executed everything encoded (a drain) flushes first.
	static constexpr uint32_t MaxHeldCopies = 16;
	void                      FlushHostCopies();
	[[nodiscard]] uint32_t    HeldCopies() const noexcept { return m_copy_count; }
	void writeBufferMarkerAMD(vk::PipelineStageFlagBits stage, vk::Buffer buffer, vk::DeviceSize offset,
	                          uint32_t marker);
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color,
	                     uint32_t count, const vk::ImageSubresourceRange* ranges);
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue* value, uint32_t count,
	                            const vk::ImageSubresourceRange* ranges);

private:
	class Writer;
	// Reserves a packet with `payload` bytes after the header blocks.
	Writer Open(Op op, uint64_t payload, bool barrier);
	void   Close(Writer& writer, uint64_t hash);
	// Descriptor writes (PushDescriptorSet, UpdateDescriptorSets): the number of infos they carry
	// (only image and buffer descriptors, no pNext chains), and their records and infos.
	static uint64_t DescriptorInfoCount(uint32_t count, const vk::WriteDescriptorSet* writes);
	static void     PutDescriptorWrites(Writer& writer, uint32_t count,
	                                    const vk::WriteDescriptorSet* writes, uint64_t infos);

	Ring&     m_ring;
	Options   m_options;
	WaitStats m_stats;
	uint64_t  m_sequence   = 0;
	uint64_t  m_packets    = 0;
	uint64_t  m_bytes      = 0;
	uint64_t  m_cb_digest  = 0;
	uint64_t  m_cb_packets = 0;
	uint32_t  m_kick_skipped = 0; // KYTY_RING_KICK_EVERY
	std::array<HostCopyPacket, MaxHeldCopies> m_copies {};
	uint32_t                                  m_copy_count = 0;
	bool                                      m_rd_open    = false;
	uint8_t*                                  m_rd_base    = nullptr;
	uint32_t                                  m_rd_size    = 0;
	uint32_t                                  m_rd_start   = 0;
	VerifyBlock*                              m_rd_verify  = nullptr;
	void EmitHostCopy(const HostCopyPacket& copy);
	void (*m_open_hook)(void* context) = nullptr;
	void* m_open_hook_context          = nullptr;
};

class Encoder::Writer {
public:
	Writer(uint8_t* base, uint32_t size, uint32_t offset, VerifyBlock* verify)
	    : m_base(base), m_size(size), m_offset(offset), m_verify(verify) {}

	template <typename T>
	T& Put(const T& value) {
		static_assert(std::is_trivially_copyable_v<T> && alignof(T) <= 8);
		EXIT_IF(m_offset + sizeof(T) > m_size);
		auto* out = m_base + m_offset;
		std::memcpy(out, &value, sizeof(T));
		m_offset += Align8(sizeof(T));
		return *std::launder(reinterpret_cast<T*>(out));
	}
	template <typename T>
	void PutArray(const T* values, uint64_t count) {
		static_assert(std::is_trivially_copyable_v<T> && alignof(T) <= 8);
		const auto bytes = sizeof(T) * count;
		EXIT_IF(m_offset + bytes > m_size || (count != 0 && values == nullptr));
		if (bytes != 0) {
			std::memcpy(m_base + m_offset, values, bytes);
		}
		m_offset += Align8(bytes);
	}
	[[nodiscard]] uint8_t*     Cursor() noexcept { return m_base + m_offset; }
	void                       Skip(uint64_t bytes) noexcept { m_offset += Align8(bytes); }
	[[nodiscard]] uint32_t     Size() const noexcept { return m_size; }
	[[nodiscard]] uint32_t     Offset() const noexcept { return m_offset; }
	[[nodiscard]] VerifyBlock* Verify() const noexcept { return m_verify; }
	[[nodiscard]] Header*      GetHeader() noexcept { return reinterpret_cast<Header*>(m_base); }

private:
	uint8_t*     m_base   = nullptr;
	uint32_t     m_size   = 0;
	uint32_t     m_offset = 0;
	VerifyBlock* m_verify = nullptr;
};

// ------------------------------------------------------------------------------------------------
// Replay (consumer side)

class Reader {
public:
	Reader(const Header* header, uint32_t offset)
	    : m_base(reinterpret_cast<const uint8_t*>(header)), m_size(header->size), m_offset(offset) {
	}
	template <typename T>
	const T& Get() {
		EXIT_IF(m_offset + sizeof(T) > m_size);
		const auto* value = reinterpret_cast<const T*>(m_base + m_offset);
		m_offset += Align8(sizeof(T));
		return *value;
	}
	template <typename T>
	const T* GetArray(uint64_t count) {
		const auto bytes = sizeof(T) * count;
		EXIT_IF(m_offset + bytes > m_size);
		const auto* values = count != 0 ? reinterpret_cast<const T*>(m_base + m_offset) : nullptr;
		m_offset += Align8(bytes);
		return values;
	}
	[[nodiscard]] uint32_t Offset() const noexcept { return m_offset; }

private:
	const uint8_t* m_base   = nullptr;
	uint32_t       m_size   = 0;
	uint32_t       m_offset = 0;
};

// KYTY_EMIT_SPLIT consumer state (ResolvedDrawPacket). The shadow: what the last resolved draw of
// this command buffer pushed (descriptors at the graphics bind point, set 0; push constants), while
// no other packet changed that state since. The comparisons are DrawEncode::Shadow's (drawEncode.h):
// descriptors equal in layout, write shapes and every info; push constants equal in layout, stages,
// size and bytes.
struct ResolvedDrawShadow {
	bool                               desc_valid   = false;
	vk::PipelineLayout                 desc_layout  = nullptr;
	std::vector<DescriptorWriteRecord> desc_writes;
	std::vector<DescriptorInfo>        desc_infos;
	bool                               const_valid  = false;
	vk::PipelineLayout                 const_layout = nullptr;
	uint32_t                           const_stages = 0;
	std::array<uint32_t, ResolvedDrawPushDwords> const_dwords {};

	void Reset() noexcept {
		desc_valid  = false;
		const_valid = false;
	}
	// Whether a push of these writes/infos with `layout` would only repeat the state in effect.
	[[nodiscard]] bool SameDescriptors(vk::PipelineLayout layout, uint32_t write_count, const DescriptorWriteRecord* writes,
	                                   uint32_t info_count, const DescriptorInfo* infos) const noexcept {
		return desc_valid && desc_layout == layout && desc_writes.size() == write_count &&
		       desc_infos.size() == info_count &&
		       (write_count == 0 || std::memcmp(desc_writes.data(), writes, sizeof(DescriptorWriteRecord) * write_count) == 0) &&
		       (info_count == 0 || std::memcmp(desc_infos.data(), infos, sizeof(DescriptorInfo) * info_count) == 0);
	}
	void KeepDescriptors(vk::PipelineLayout layout, uint32_t write_count, const DescriptorWriteRecord* writes,
	                     uint32_t info_count, const DescriptorInfo* infos) {
		desc_valid  = true;
		desc_layout = layout;
		desc_writes.assign(writes, writes + write_count);
		desc_infos.assign(infos, infos + info_count);
	}
	[[nodiscard]] bool SameConstants(vk::PipelineLayout layout, uint32_t stages, const uint32_t* dwords) const noexcept {
		return const_valid && const_layout == layout && const_stages == stages &&
		       std::memcmp(const_dwords.data(), dwords, sizeof(const_dwords)) == 0;
	}
	void KeepConstants(vk::PipelineLayout layout, uint32_t stages, const uint32_t* dwords) noexcept {
		const_valid  = true;
		const_layout = layout;
		const_stages = stages;
		std::memcpy(const_dwords.data(), dwords, sizeof(const_dwords));
	}
};

// KYTY_EMIT_SPLIT=verify: the state the replayed packets left in the command buffer (graphics bind
// point), kept while `on`, so a verify-only resolved draw placed right after the draw's own packets
// can be compared with it field by field.
struct ResolvedDrawModel {
	enum Field : uint32_t { Pipeline, Vertex, Index, Layout, Writes, Infos, Constants, Draw, FieldCount };
	static constexpr const char* FieldNames[FieldCount] = {"pipeline", "vertex", "index", "layout",
	                                                       "writes",   "infos",  "constants", "draw"};
	struct Descriptor {
		uint32_t           binding = 0;
		uint32_t           element = 0;
		vk::DescriptorType type    = vk::DescriptorType::eSampler;
		DescriptorInfo     info;
	};
	bool on    = false; // set by the first verify-only record
	bool valid = false; // a Begin was replayed since `on`
	vk::Pipeline              pipeline    = nullptr;
	std::vector<vk::Buffer>   vb_buffers;
	std::vector<vk::DeviceSize> vb_offsets;
	std::vector<vk::DeviceSize> vb_sizes;
	bool                      vb_sizes_set = false;
	vk::Buffer                ib_buffer    = nullptr;
	uint64_t                  ib_offset    = 0;
	uint32_t                  ib_type      = 0;
	vk::PipelineLayout        desc_layout  = nullptr;
	std::vector<Descriptor>   descriptors;
	std::array<uint32_t, 64>  constants {};
	uint64_t                  constants_valid = 0; // one bit per dword
	bool                      draw_indexed    = false;
	uint32_t                  draw_args[5]    = {};

	void Begin() {
		valid    = on;
		pipeline = nullptr;
		vb_buffers.clear();
		vb_offsets.clear();
		vb_sizes.clear();
		ib_buffer   = nullptr;
		desc_layout = nullptr;
		descriptors.clear();
		constants_valid = 0;
		draw_indexed    = false;
		std::memset(draw_args, 0, sizeof(draw_args));
	}
	void Push(vk::PipelineLayout layout, uint32_t count, const vk::WriteDescriptorSet* writes) {
		if (layout != desc_layout) {
			descriptors.clear();
			desc_layout = layout;
		}
		for (uint32_t w = 0; w < count; w++) {
			const auto& write = writes[w];
			for (uint32_t j = 0; j < write.descriptorCount; j++) {
				DescriptorInfo info;
				if (write.pImageInfo != nullptr) {
					std::memcpy(&info, write.pImageInfo + j, sizeof(info));
				} else if (write.pBufferInfo != nullptr) {
					std::memcpy(&info, write.pBufferInfo + j, sizeof(info));
				}
				Set(write.dstBinding, write.dstArrayElement + j, write.descriptorType, info);
			}
		}
	}
	void Set(uint32_t binding, uint32_t element, vk::DescriptorType type, const DescriptorInfo& info) {
		for (auto& d: descriptors) {
			if (d.binding == binding && d.element == element) {
				d.type = type;
				d.info = info;
				return;
			}
		}
		descriptors.push_back({binding, element, type, info});
	}
	[[nodiscard]] const Descriptor* Find(uint32_t binding, uint32_t element) const {
		for (const auto& d: descriptors) {
			if (d.binding == binding && d.element == element) {
				return &d;
			}
		}
		return nullptr;
	}
	void SetConstants(uint32_t offset, uint32_t size, const void* data) {
		if ((offset & 3u) != 0 || (size & 3u) != 0 || offset + size > sizeof(constants)) {
			constants_valid = 0;
			return;
		}
		std::memcpy(reinterpret_cast<uint8_t*>(constants.data()) + offset, data, size);
		for (uint32_t d = offset / 4u; d < (offset + size) / 4u; d++) {
			constants_valid |= uint64_t {1} << d;
		}
	}
	void UnsetDescriptors() {
		desc_layout = nullptr;
		descriptors.clear();
	}
};

struct ResolvedDrawStats {
	uint64_t records          = 0; // expanded (not verify-only)
	uint64_t pushes           = 0;
	uint64_t pushes_avoided   = 0;
	uint64_t constants        = 0;
	uint64_t constants_avoided = 0;
	uint64_t shadow_resets    = 0;
	uint64_t verify_records   = 0;
	uint64_t verify_unmodeled = 0;
	uint64_t verify_checks    = 0;
	uint64_t mismatches[ResolvedDrawModel::FieldCount] = {};
	uint64_t reported         = 0;
};

// The consumers' resolved-draw counters, summed over every replaying thread (published in
// increments every 1024 records and at every verify mismatch; read by the reports and live lines).
struct ResolvedDrawTotals {
	uint64_t records = 0, pushes = 0, pushes_avoided = 0, constants = 0, constants_avoided = 0, shadow_resets = 0;
	uint64_t verify_records = 0, verify_checks = 0, verify_unmodeled = 0;
	uint64_t mismatches[ResolvedDrawModel::FieldCount] = {};
};
void                             PublishResolvedDraw(const ResolvedDrawStats& stats, ResolvedDrawStats& published) noexcept;
[[nodiscard]] ResolvedDrawTotals ResolvedDrawPublished() noexcept;

// Consumer-side verification state and argument scratch.
struct ReplayState {
	bool     verify     = false;
	uint64_t sequence   = 0; // last verified sequence number
	uint64_t cb_digest  = 0;
	uint64_t cb_packets = 0;
	uint64_t checks     = 0;
	uint64_t mismatches = 0;
	// First mismatch details (for the report).
	Op                                  mismatch_op       = Op::Wrap;
	uint64_t                            mismatch_sequence = 0;
	uint64_t                            mismatch_expected = 0;
	uint64_t                            mismatch_actual   = 0;
	std::vector<vk::WriteDescriptorSet> writes;
	// KYTY_EMIT_SPLIT (ResolvedDrawPacket).
	ResolvedDrawShadow record_shadow;
	ResolvedDrawModel  record_model;
	ResolvedDrawStats  record_stats;
	ResolvedDrawStats  record_published; // the part of record_stats already in ResolvedDrawPublished
};

// Called on a verify mismatch: kind 0 = argument hash, 1 = sequence, 2 = command-buffer digest.
using MismatchHandler = void (*)(void* context, uint32_t kind, Op op, uint64_t sequence,
                                 uint64_t expected, uint64_t actual);

namespace Detail {
// Parses the header blocks; returns the payload offset.
inline uint32_t PayloadOffset(const Header& header, const SiteBlock** site,
                              const VerifyBlock** verify) {
	uint32_t offset  = sizeof(Header);
	*site            = nullptr;
	*verify          = nullptr;
	const auto* base = reinterpret_cast<const uint8_t*>(&header);
	if ((header.flags & FlagSite) != 0) {
		*site = reinterpret_cast<const SiteBlock*>(base + offset);
		offset += sizeof(SiteBlock);
	}
	if ((header.flags & FlagVerify) != 0) {
		*verify = reinterpret_cast<const VerifyBlock*>(base + offset);
		offset += sizeof(VerifyBlock);
	}
	return offset;
}
} // namespace Detail

// Replays one non-Wrap packet into `exec`. Returns false only for a Submit whose command-buffer
// digest did not verify (the packet itself is still executed).
template <typename Exec>
void Replay(const Header& header, Exec& exec, ReplayState& state, MismatchHandler on_mismatch,
            void* mismatch_context);

} // namespace Libs::Graphics::CommandStream

#include "graphics/host_gpu/renderer/commandStreamReplay.inl"

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSTREAM_H_
