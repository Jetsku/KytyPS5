#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSTREAMLOG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSTREAMLOG_H_

// A Replay executor (commandStream.h) that records each replayed command as (op, VerifyHash of its
// arguments): a canonical form of a packet sequence that ignores packet padding, site blocks and
// verify sequence numbers. Used to compare two encodes of the same commands (the step-2 builder
// verify, KYTY_ENC_OFFLOAD verify) and by the unit tests.

#include "graphics/host_gpu/renderer/commandStream.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace Libs::Graphics::CommandStream {

struct Call {
	Op       op   = Op::Wrap;
	uint64_t hash = 0;
	bool operator==(const Call&) const = default;
};

// Receives replayed calls and hashes their arguments with the shared VerifyHash functions.
struct CallLog {
	std::vector<Call> calls;
	uint32_t          sites  = 0;
	bool              record = true;

	void Push(Op op, uint64_t hash) {
		if (record) {
			calls.push_back({op, hash});
		}
	}

	void Begin(const BeginPacket& p) { Push(Op::Begin, VerifyHash::Begin(p.command, p.tick)); }
	void Submit(const SubmitPacket& p) {
		// The digest fields are verify bookkeeping, not arguments of the submission.
		SubmitPacket copy = p;
		copy.cb_digest    = 0;
		copy.cb_packets   = 0;
		Push(Op::Submit, VerifyHash::Submit(copy));
	}
	void DrainMarker(uint64_t serial) { Push(Op::DrainMarker, VerifyHash::DrainMarker(serial)); }
	void EnterSite(const void*, const void*) { sites++; }
	void LeaveSite() {}

	void beginRendering(const vk::RenderingInfo& info) {
		Push(Op::BeginRendering, VerifyHash::BeginRendering(info));
	}
	void endRendering() { Push(Op::EndRendering, VerifyHash::EndRendering()); }
	void pipelineBarrier2(const vk::DependencyInfo& info) {
		Push(Op::PipelineBarrier2, VerifyHash::PipelineBarrier2(info));
	}
	void pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst,
	                     vk::DependencyFlags flags, uint32_t mc, const vk::MemoryBarrier* m,
	                     uint32_t bc, const vk::BufferMemoryBarrier* b, uint32_t ic,
	                     const vk::ImageMemoryBarrier* i) {
		Push(Op::PipelineBarrier, VerifyHash::PipelineBarrier(src, dst, flags, mc, m, bc, b, ic, i));
	}
	void copyBuffer(vk::Buffer s, vk::Buffer d, uint32_t n, const vk::BufferCopy* r) {
		Push(Op::CopyBuffer, VerifyHash::CopyBuffer(s, d, n, r));
	}
	void copyBufferToImage(vk::Buffer s, vk::Image d, vk::ImageLayout l, uint32_t n,
	                       const vk::BufferImageCopy* r) {
		Push(Op::CopyBufferToImage, VerifyHash::CopyBufferToImage(s, d, l, n, r));
	}
	void copyImageToBuffer(vk::Image s, vk::ImageLayout l, vk::Buffer d, uint32_t n,
	                       const vk::BufferImageCopy* r) {
		Push(Op::CopyImageToBuffer, VerifyHash::CopyImageToBuffer(s, l, d, n, r));
	}
	void copyImage(vk::Image s, vk::ImageLayout sl, vk::Image d, vk::ImageLayout dl, uint32_t n,
	               const vk::ImageCopy* r) {
		Push(Op::CopyImage, VerifyHash::CopyImage(s, sl, d, dl, n, r));
	}
	void fillBuffer(vk::Buffer b, vk::DeviceSize o, vk::DeviceSize s, uint32_t v) {
		Push(Op::FillBuffer, VerifyHash::FillBuffer(b, o, s, v));
	}
	void bindPipeline(vk::PipelineBindPoint p, vk::Pipeline pipeline) {
		Push(Op::BindPipeline, VerifyHash::BindPipeline(p, pipeline));
	}
	void bindDescriptorSets(vk::PipelineBindPoint p, vk::PipelineLayout l, uint32_t f, uint32_t n,
	                        const vk::DescriptorSet* s, uint32_t dn, const uint32_t* d) {
		Push(Op::BindDescriptorSets, VerifyHash::BindDescriptorSets(p, l, f, n, s, dn, d));
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint p, vk::PipelineLayout l, uint32_t s,
	                          uint32_t n, const vk::WriteDescriptorSet* w) {
		Push(Op::PushDescriptorSet, VerifyHash::PushDescriptorSet(p, l, s, n, w));
	}
	void updateDescriptorSets(vk::DescriptorSet s, uint32_t n, const vk::WriteDescriptorSet* w) {
		Push(Op::UpdateDescriptorSets, VerifyHash::UpdateDescriptorSets(s, n, w));
	}
	void pushConstants(vk::PipelineLayout l, vk::ShaderStageFlags st, uint32_t o, uint32_t s,
	                   const void* d) {
		Push(Op::PushConstants, VerifyHash::PushConstants(l, st, o, s, d));
	}
	void bindVertexBuffers2(uint32_t f, uint32_t n, const vk::Buffer* b,
	                        const vk::DeviceSize* o, const vk::DeviceSize* s,
	                        const vk::DeviceSize* st) {
		Push(Op::BindVertexBuffers2, VerifyHash::BindVertexBuffers2(f, n, b, o, s, st));
	}
	void bindIndexBuffer(vk::Buffer b, vk::DeviceSize o, vk::IndexType t) {
		Push(Op::BindIndexBuffer, VerifyHash::BindIndexBuffer(b, o, t));
	}
	void setViewportWithCount(uint32_t n, const vk::Viewport* v) {
		Push(Op::SetViewportWithCount, VerifyHash::Viewports(n, v));
	}
	void setScissorWithCount(uint32_t n, const vk::Rect2D* s) {
		Push(Op::SetScissorWithCount, VerifyHash::Scissors(n, s));
	}
	void setLineWidth(float w) { Push(Op::SetLineWidth, VerifyHash::Floats(Op::SetLineWidth, &w, 1)); }
	void setBlendConstants(const float c[4]) {
		Push(Op::SetBlendConstants, VerifyHash::Floats(Op::SetBlendConstants, c, 4));
	}
	void setDepthTestEnable(vk::Bool32 e) {
		Push(Op::SetDepthTestEnable, VerifyHash::Value(Op::SetDepthTestEnable, e));
	}
	void setDepthWriteEnable(vk::Bool32 e) {
		Push(Op::SetDepthWriteEnable, VerifyHash::Value(Op::SetDepthWriteEnable, e));
	}
	void setDepthCompareOp(vk::CompareOp op) {
		Push(Op::SetDepthCompareOp,
		     VerifyHash::Value(Op::SetDepthCompareOp, static_cast<uint32_t>(op)));
	}
	void setDepthBiasEnable(vk::Bool32 e) {
		Push(Op::SetDepthBiasEnable, VerifyHash::Value(Op::SetDepthBiasEnable, e));
	}
	void setDepthBias(float a, float b, float c) {
		const float v[3] = {a, b, c};
		Push(Op::SetDepthBias, VerifyHash::Floats(Op::SetDepthBias, v, 3));
	}
	void setStencilTestEnable(vk::Bool32 e) {
		Push(Op::SetStencilTestEnable, VerifyHash::Value(Op::SetStencilTestEnable, e));
	}
	void setStencilOp(vk::StencilFaceFlags f, vk::StencilOp a, vk::StencilOp b, vk::StencilOp c,
	                  vk::CompareOp d) {
		Push(Op::SetStencilOp,
		     VerifyHash::Value(Op::SetStencilOp, static_cast<VkStencilFaceFlags>(f),
		                       static_cast<uint64_t>(a), static_cast<uint64_t>(b),
		                       static_cast<uint64_t>(c), static_cast<uint64_t>(d)));
	}
	void setStencilCompareMask(vk::StencilFaceFlags f, uint32_t v) {
		Push(Op::SetStencilCompareMask,
		     VerifyHash::Value(Op::SetStencilCompareMask, static_cast<VkStencilFaceFlags>(f), v));
	}
	void setStencilWriteMask(vk::StencilFaceFlags f, uint32_t v) {
		Push(Op::SetStencilWriteMask,
		     VerifyHash::Value(Op::SetStencilWriteMask, static_cast<VkStencilFaceFlags>(f), v));
	}
	void setStencilReference(vk::StencilFaceFlags f, uint32_t v) {
		Push(Op::SetStencilReference,
		     VerifyHash::Value(Op::SetStencilReference, static_cast<VkStencilFaceFlags>(f), v));
	}
	void setCullMode(vk::CullModeFlags m) {
		Push(Op::SetCullMode, VerifyHash::Value(Op::SetCullMode, static_cast<VkCullModeFlags>(m)));
	}
	void setFrontFace(vk::FrontFace f) {
		Push(Op::SetFrontFace, VerifyHash::Value(Op::SetFrontFace, static_cast<uint32_t>(f)));
	}
	void setDepthBoundsTestEnable(vk::Bool32 e) {
		Push(Op::SetDepthBoundsTestEnable, VerifyHash::Value(Op::SetDepthBoundsTestEnable, e));
	}
	void setDepthBounds(float a, float b) {
		const float v[2] = {a, b};
		Push(Op::SetDepthBounds, VerifyHash::Floats(Op::SetDepthBounds, v, 2));
	}
	void setColorWriteEnableEXT(uint32_t n, const vk::Bool32* e) {
		Push(Op::SetColorWriteEnable, VerifyHash::ColorWriteEnable(n, e));
	}
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags a) {
		Push(Op::SetAttachmentFeedbackLoopEnable,
		     VerifyHash::Value(Op::SetAttachmentFeedbackLoopEnable,
		                       static_cast<VkImageAspectFlags>(a)));
	}
	void draw(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
		Push(Op::Draw, VerifyHash::Value(Op::Draw, a, b, c, d));
	}
	void drawIndexed(uint32_t a, uint32_t b, uint32_t c, int32_t d, uint32_t e) {
		Push(Op::DrawIndexed,
		     VerifyHash::Value(Op::DrawIndexed, a, b, c, static_cast<uint32_t>(d), e));
	}
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) {
		Push(Op::DrawMeshTasks, VerifyHash::Value(Op::DrawMeshTasks, x, y, z));
	}
	static uint64_t Bits(vk::Buffer b) {
		uint64_t   v = 0;
		const auto r = static_cast<VkBuffer>(b);
		std::memcpy(&v, &r, 8);
		return v;
	}
	static uint64_t Bits(vk::QueryPool b) {
		uint64_t   v = 0;
		const auto r = static_cast<VkQueryPool>(b);
		std::memcpy(&v, &r, 8);
		return v;
	}
	void drawMeshTasksIndirectEXT(vk::Buffer b, vk::DeviceSize o, uint32_t n, uint32_t s) {
		Push(Op::DrawMeshTasksIndirect, VerifyHash::Value(Op::DrawMeshTasksIndirect, Bits(b), o, n, s));
	}
	void drawMeshTasksIndirectCountEXT(vk::Buffer b, vk::DeviceSize o, vk::Buffer cb,
	                                   vk::DeviceSize co, uint32_t m, uint32_t s) {
		Push(Op::DrawMeshTasksIndirectCount,
		     VerifyHash::Value(Op::DrawMeshTasksIndirectCount, Bits(b), o, Bits(cb), co, m, s));
	}
	void drawIndirect(vk::Buffer b, vk::DeviceSize o, uint32_t n, uint32_t s) {
		Push(Op::DrawIndirect, VerifyHash::Value(Op::DrawIndirect, Bits(b), o, n, s));
	}
	void drawIndexedIndirect(vk::Buffer b, vk::DeviceSize o, uint32_t n, uint32_t s) {
		Push(Op::DrawIndexedIndirect, VerifyHash::Value(Op::DrawIndexedIndirect, Bits(b), o, n, s));
	}
	void drawIndirectCount(vk::Buffer b, vk::DeviceSize o, vk::Buffer cb, vk::DeviceSize co,
	                       uint32_t m, uint32_t s) {
		Push(Op::DrawIndirectCount,
		     VerifyHash::Value(Op::DrawIndirectCount, Bits(b), o, Bits(cb), co, m, s));
	}
	void drawIndexedIndirectCount(vk::Buffer b, vk::DeviceSize o, vk::Buffer cb,
	                              vk::DeviceSize co, uint32_t m, uint32_t s) {
		Push(Op::DrawIndexedIndirectCount,
		     VerifyHash::Value(Op::DrawIndexedIndirectCount, Bits(b), o, Bits(cb), co, m, s));
	}
	void dispatch(uint32_t x, uint32_t y, uint32_t z) {
		Push(Op::Dispatch, VerifyHash::Value(Op::Dispatch, x, y, z));
	}
	void dispatchBase(uint32_t bx, uint32_t by, uint32_t bz, uint32_t x, uint32_t y, uint32_t z) {
		Push(Op::DispatchBase, VerifyHash::Value(Op::DispatchBase, bx, by, bz, x, y, z));
	}
	void dispatchIndirect(vk::Buffer b, vk::DeviceSize o) {
		Push(Op::DispatchIndirect, VerifyHash::Value(Op::DispatchIndirect, Bits(b), o));
	}
	void resetQueryPool(vk::QueryPool p, uint32_t f, uint32_t n) {
		Push(Op::ResetQueryPool, VerifyHash::Value(Op::ResetQueryPool, Bits(p), f, n));
	}
	void beginQuery(vk::QueryPool p, uint32_t q, vk::QueryControlFlags f) {
		Push(Op::BeginQuery,
		     VerifyHash::Value(Op::BeginQuery, Bits(p), q, static_cast<VkQueryControlFlags>(f)));
	}
	void endQuery(vk::QueryPool p, uint32_t q) {
		Push(Op::EndQuery, VerifyHash::Value(Op::EndQuery, Bits(p), q, 0));
	}
	void copyQueryPoolResults(vk::QueryPool p, uint32_t f, uint32_t n, vk::Buffer d,
	                          vk::DeviceSize o, vk::DeviceSize s, vk::QueryResultFlags fl) {
		Push(Op::CopyQueryPoolResults,
		     VerifyHash::Value(Op::CopyQueryPoolResults, Bits(p), f, n, Bits(d), o,
		                       s ^ (static_cast<uint64_t>(static_cast<VkQueryResultFlags>(fl))
		                            << 40u)));
	}
	void writeTimestamp2(vk::PipelineStageFlags2 st, vk::QueryPool p, uint32_t q) {
		Push(Op::WriteTimestamp2,
		     VerifyHash::Value(Op::WriteTimestamp2, Bits(p), q,
		                       static_cast<uint64_t>(static_cast<VkPipelineStageFlags2>(st))));
	}
	void hostCopy(void* d, uint64_t src, uint64_t n) {
		Push(Op::HostCopy, VerifyHash::Value(Op::HostCopy, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d)), src, n));
	}
	void insertDebugLabel(const char* name) {
		const auto size = static_cast<uint32_t>(std::strlen(name) + 1u);
		Push(Op::InsertDebugLabel, VerifyHash::DebugLabel(name, size));
	}
	void writeBufferMarkerAMD(vk::PipelineStageFlagBits st, vk::Buffer b, vk::DeviceSize o, uint32_t m) {
		Push(Op::WriteBufferMarker,
		     VerifyHash::Value(Op::WriteBufferMarker, Bits(b), o,
		                       static_cast<uint32_t>(static_cast<VkPipelineStageFlagBits>(st)), m));
	}
	void clearColorImage(vk::Image i, vk::ImageLayout l, const vk::ClearColorValue* c, uint32_t n,
	                     const vk::ImageSubresourceRange* r) {
		Push(Op::ClearColorImage, VerifyHash::ClearColorImage(i, l, *c, n, r));
	}
	void clearDepthStencilImage(vk::Image i, vk::ImageLayout l, const vk::ClearDepthStencilValue* v,
	                            uint32_t n, const vk::ImageSubresourceRange* r) {
		Push(Op::ClearDepthStencilImage, VerifyHash::ClearDepthStencilImage(i, l, *v, n, r));
	}
};


// Replays the packets of `ring` from `begin` to `end` (producer positions; Wrap packets skipped)
// into `log`. The caller is the ring's producer, or nothing writes the ring meanwhile.
inline void LogPackets(const Ring& ring, uint64_t begin, uint64_t end, CallLog& log) {
	ReplayState state;
	for (uint64_t position = begin; position < end;) {
		const auto* header = ring.At(position);
		EXIT_IF(header->size == 0);
		if (header->op != Op::Wrap) {
			Replay(*header, log, state, nullptr, nullptr);
		}
		position += header->size;
	}
}

} // namespace Libs::Graphics::CommandStream

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSTREAMLOG_H_
