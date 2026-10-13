// Unit tests for the CP recorder's command stream (graphics/host_gpu/renderer/commandStream.h):
// the SPSC ring, the encoder and Replay, without a Vulkan device.
//  - every packet type round-trips: the executor receives exactly the original arguments
//    (checked through the shared VerifyHash functions, computed independently on both sides);
//  - wrap-around, ring-full back-pressure and drains, single-threaded and threaded;
//  - verify mode detects a corrupted argument, a dropped packet (sequence) and a command buffer
//    whose packets changed (digest).
// `cp_recorder_tests --bench` also prints the producer's encode cost per typical draw.

#include "graphics/host_gpu/renderer/commandStream.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

// commandStream.cpp includes vulkan.hpp's dispatcher declarations; the tests never call Vulkan.
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::CommandStream;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::printf("FAILED: %s\n", what);
		++g_failures;
	}
}

struct Call {
	Op       op   = Op::Wrap;
	uint64_t hash = 0;
	bool operator==(const Call&) const = default;
};

template <typename Handle>
Handle MakeHandle(uint64_t value) {
	using C = typename Handle::CType;
	C raw {};
	static_assert(sizeof(C) == sizeof(uint64_t));
	std::memcpy(&raw, &value, sizeof(raw));
	return Handle(raw);
}

// Receives replayed calls and hashes their arguments with the shared VerifyHash functions.
struct LogExecutor {
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
	void writeBufferMarkerAMD(vk::PipelineStageFlagBits st, vk::Buffer b, vk::DeviceSize o, uint32_t m) {
		Push(Op::WriteBufferMarker,
		     VerifyHash::Value(Op::WriteBufferMarker, Bits(b), o,
		                       static_cast<uint32_t>(static_cast<VkPipelineStageFlagBits>(st)), m));
	}
};

// Discards everything (benchmark consumer).
struct NullExecutor: LogExecutor {
	NullExecutor() { record = false; }
};

struct MismatchLog {
	std::vector<std::array<uint64_t, 4>> entries; // kind, op, sequence, 0
};

void RecordMismatch(void* context, uint32_t kind, Op op, uint64_t sequence, uint64_t, uint64_t) {
	static_cast<MismatchLog*>(context)->entries.push_back(
	    {kind, static_cast<uint64_t>(op), sequence, 0});
}

// Encodes calls with random arguments and records the expected (op, argument hash) sequence.
class Generator {
public:
	explicit Generator(uint32_t seed): m_rng(seed) {}

	uint64_t U64() { return m_rng(); }
	uint32_t U32() { return static_cast<uint32_t>(m_rng()); }
	uint32_t Below(uint32_t n) { return n == 0 ? 0 : U32() % n; }
	float    F32() { return static_cast<float>(U32() % 100000) * 0.125f - 1000.0f; }
	template <typename Handle>
	Handle H() {
		return MakeHandle<Handle>((U64() | 1u) & 0x0000fffffffffff0ull);
	}

	std::vector<Call> expected;

	// One call of `op` (never Wrap/Count/Begin/Submit/DrainMarker).
	void Emit(Encoder& e, Op op, uint32_t array_scale) {
		const uint32_t n = 1 + Below(array_scale);
		switch (op) {
			case Op::BeginRendering: {
				std::vector<vk::RenderingAttachmentInfo> colors(Below(9));
				for (auto& c: colors) {
					c.imageView                   = H<vk::ImageView>();
					c.imageLayout                 = vk::ImageLayout::eColorAttachmentOptimal;
					c.loadOp                      = (U32() & 1) ? vk::AttachmentLoadOp::eClear
					                                            : vk::AttachmentLoadOp::eLoad;
					c.storeOp                     = vk::AttachmentStoreOp::eStore;
					c.clearValue.color.uint32[0] = U32();
					c.clearValue.color.uint32[3] = U32();
				}
				vk::RenderingAttachmentInfo depth {};
				depth.imageView                       = H<vk::ImageView>();
				depth.imageLayout                     = vk::ImageLayout::eDepthAttachmentOptimal;
				depth.clearValue.depthStencil.depth   = F32();
				vk::RenderingAttachmentInfo stencil   = depth;
				stencil.clearValue.depthStencil.stencil = U32();
				vk::RenderingInfo info {};
				info.renderArea.extent    = vk::Extent2D {U32() % 8192, U32() % 8192};
				info.layerCount           = 1 + Below(4);
				info.colorAttachmentCount = static_cast<uint32_t>(colors.size());
				info.pColorAttachments    = colors.data();
				info.pDepthAttachment     = (U32() & 1) ? &depth : nullptr;
				info.pStencilAttachment   = (U32() & 1) ? &stencil : nullptr;
				e.beginRendering(info);
				expected.push_back({op, VerifyHash::BeginRendering(info)});
				break;
			}
			case Op::EndRendering:
				e.endRendering();
				expected.push_back({op, VerifyHash::EndRendering()});
				break;
			case Op::PipelineBarrier2: {
				std::vector<vk::MemoryBarrier2>       memory(Below(2));
				std::vector<vk::BufferMemoryBarrier2> buffers(Below(n + 1));
				std::vector<vk::ImageMemoryBarrier2>  images(Below(n + 1));
				for (auto& m: memory) {
					m.srcStageMask  = vk::PipelineStageFlags2(U64() & 0xffffff);
					m.dstAccessMask = vk::AccessFlags2(U64() & 0xffffff);
				}
				for (auto& b: buffers) {
					b.buffer = H<vk::Buffer>();
					b.offset = U64() & 0xffffff;
					b.size   = VK_WHOLE_SIZE;
				}
				for (auto& i: images) {
					i.image     = H<vk::Image>();
					i.oldLayout = vk::ImageLayout::eGeneral;
					i.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
					i.subresourceRange = {vk::ImageAspectFlagBits::eColor, Below(4), 1 + Below(3),
					                      Below(6), 1};
				}
				vk::DependencyInfo info {};
				info.memoryBarrierCount       = static_cast<uint32_t>(memory.size());
				info.pMemoryBarriers          = memory.data();
				info.bufferMemoryBarrierCount = static_cast<uint32_t>(buffers.size());
				info.pBufferMemoryBarriers    = buffers.data();
				info.imageMemoryBarrierCount  = static_cast<uint32_t>(images.size());
				info.pImageMemoryBarriers     = images.data();
				e.pipelineBarrier2(info);
				expected.push_back({op, VerifyHash::PipelineBarrier2(info)});
				break;
			}
			case Op::PipelineBarrier: {
				std::vector<vk::MemoryBarrier>       memory(Below(2));
				std::vector<vk::BufferMemoryBarrier> buffers(Below(3));
				std::vector<vk::ImageMemoryBarrier>  images(Below(3));
				for (auto& b: buffers) {
					b.buffer = H<vk::Buffer>();
					b.size   = U64() & 0xffff;
				}
				for (auto& i: images) {
					i.image = H<vk::Image>();
				}
				const auto src = vk::PipelineStageFlags(U32() & 0xffff);
				const auto dst = vk::PipelineStageFlags(U32() & 0xffff);
				e.pipelineBarrier(src, dst, {}, static_cast<uint32_t>(memory.size()), memory.data(),
				                  static_cast<uint32_t>(buffers.size()), buffers.data(),
				                  static_cast<uint32_t>(images.size()), images.data());
				expected.push_back(
				    {op, VerifyHash::PipelineBarrier(src, dst, {}, static_cast<uint32_t>(memory.size()),
				                                     memory.data(), static_cast<uint32_t>(buffers.size()),
				                                     buffers.data(), static_cast<uint32_t>(images.size()),
				                                     images.data())});
				break;
			}
			case Op::CopyBuffer: {
				std::vector<vk::BufferCopy> regions(n);
				for (auto& r: regions) {
					r = vk::BufferCopy {U64() & 0xffffff, U64() & 0xffffff, U64() & 0xffff};
				}
				const auto s = H<vk::Buffer>();
				const auto d = H<vk::Buffer>();
				e.copyBuffer(s, d, n, regions.data());
				expected.push_back({op, VerifyHash::CopyBuffer(s, d, n, regions.data())});
				break;
			}
			case Op::CopyBufferToImage:
			case Op::CopyImageToBuffer: {
				std::vector<vk::BufferImageCopy> regions(n);
				for (auto& r: regions) {
					r.bufferOffset     = U64() & 0xffffff;
					r.imageSubresource = {vk::ImageAspectFlagBits::eColor, Below(5), 0, 1};
					r.imageExtent      = vk::Extent3D {1 + Below(512), 1 + Below(512), 1};
				}
				const auto b = H<vk::Buffer>();
				const auto i = H<vk::Image>();
				const auto l = vk::ImageLayout::eTransferDstOptimal;
				if (op == Op::CopyBufferToImage) {
					e.copyBufferToImage(b, i, l, n, regions.data());
					expected.push_back({op, VerifyHash::CopyBufferToImage(b, i, l, n, regions.data())});
				} else {
					e.copyImageToBuffer(i, l, b, n, regions.data());
					expected.push_back({op, VerifyHash::CopyImageToBuffer(i, l, b, n, regions.data())});
				}
				break;
			}
			case Op::CopyImage: {
				std::vector<vk::ImageCopy> regions(n);
				for (auto& r: regions) {
					r.srcSubresource = {vk::ImageAspectFlagBits::eColor, Below(3), 0, 1};
					r.dstSubresource = {vk::ImageAspectFlagBits::eColor, Below(3), 0, 1};
					r.extent         = vk::Extent3D {1 + Below(64), 1 + Below(64), 1};
				}
				const auto s = H<vk::Image>();
				const auto d = H<vk::Image>();
				e.copyImage(s, vk::ImageLayout::eTransferSrcOptimal, d,
				            vk::ImageLayout::eTransferDstOptimal, n, regions.data());
				expected.push_back({op, VerifyHash::CopyImage(s, vk::ImageLayout::eTransferSrcOptimal, d,
				                                              vk::ImageLayout::eTransferDstOptimal, n,
				                                              regions.data())});
				break;
			}
			case Op::FillBuffer: {
				const auto b = H<vk::Buffer>();
				const auto o = U64() & 0xfff0;
				const auto s = U64() & 0xfff0;
				const auto v = U32();
				e.fillBuffer(b, o, s, v);
				expected.push_back({op, VerifyHash::FillBuffer(b, o, s, v)});
				break;
			}
			case Op::BindPipeline: {
				const auto p = H<vk::Pipeline>();
				const auto point =
				    (U32() & 1) ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;
				e.bindPipeline(point, p);
				expected.push_back({op, VerifyHash::BindPipeline(point, p)});
				break;
			}
			case Op::BindDescriptorSets: {
				std::vector<vk::DescriptorSet> sets(n);
				for (auto& s: sets) {
					s = H<vk::DescriptorSet>();
				}
				std::vector<uint32_t> offsets(Below(3));
				for (auto& o: offsets) {
					o = U32();
				}
				const auto l = H<vk::PipelineLayout>();
				e.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, l, 0, n, sets.data(),
				                     static_cast<uint32_t>(offsets.size()), offsets.data());
				expected.push_back({op, VerifyHash::BindDescriptorSets(
				                            vk::PipelineBindPoint::eGraphics, l, 0, n, sets.data(),
				                            static_cast<uint32_t>(offsets.size()), offsets.data())});
				break;
			}
			case Op::PushDescriptorSet:
			case Op::UpdateDescriptorSets: {
				// Mixed image, sampler and buffer bindings with multi-element arrays, as
				// CommitBindings builds them (infos in separate vectors).
				const uint32_t                        writes_n = n;
				std::vector<vk::WriteDescriptorSet>   writes(writes_n);
				std::vector<vk::DescriptorBufferInfo> buffers;
				std::vector<vk::DescriptorImageInfo>  images;
				buffers.reserve(writes_n * 4);
				images.reserve(writes_n * 4);
				// UpdateDescriptorSets: every write targets the same set (CommitDescriptorSet).
				const auto set = op == Op::UpdateDescriptorSets ? H<vk::DescriptorSet>()
				                                                : vk::DescriptorSet {};
				for (uint32_t i = 0; i < writes_n; i++) {
					auto& w           = writes[i];
					w.dstSet          = set;
					w.dstBinding      = U32() % 128;
					w.dstArrayElement = Below(2);
					w.descriptorCount = 1 + Below(4);
					const auto kind   = Below(4);
					if (kind == 0) {
						w.descriptorType = vk::DescriptorType::eStorageBuffer;
						w.pBufferInfo    = buffers.data() + buffers.size();
						for (uint32_t j = 0; j < w.descriptorCount; j++) {
							buffers.push_back({H<vk::Buffer>(), U64() & 0xffff00, U64() & 0xffff});
						}
					} else {
						w.descriptorType = kind == 1   ? vk::DescriptorType::eSampler
						                   : kind == 2 ? vk::DescriptorType::eSampledImage
						                               : vk::DescriptorType::eStorageImage;
						w.pImageInfo     = images.data() + images.size();
						for (uint32_t j = 0; j < w.descriptorCount; j++) {
							images.push_back({kind == 1 ? H<vk::Sampler>() : vk::Sampler {},
							                  kind == 1 ? vk::ImageView {} : H<vk::ImageView>(),
							                  kind == 1 ? vk::ImageLayout::eUndefined
							                            : vk::ImageLayout::eGeneral});
						}
					}
				}
				if (op == Op::UpdateDescriptorSets) {
					e.updateDescriptorSets(set, writes_n, writes.data());
					expected.push_back(
					    {op, VerifyHash::UpdateDescriptorSets(set, writes_n, writes.data())});
				} else {
					const auto l = H<vk::PipelineLayout>();
					e.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, l, 0, writes_n,
					                       writes.data());
					expected.push_back(
					    {op, VerifyHash::PushDescriptorSet(vk::PipelineBindPoint::eGraphics, l, 0,
					                                       writes_n, writes.data())});
				}
				// The encoder copied: scribbling over the caller's arrays changes nothing.
				for (auto& b: buffers) {
					b.offset = 0xdead;
				}
				for (auto& im: images) {
					im.imageLayout = vk::ImageLayout::ePreinitialized;
				}
				break;
			}
			case Op::PushConstants: {
				std::vector<uint8_t> bytes(4 * (1 + Below(64)));
				for (auto& b: bytes) {
					b = static_cast<uint8_t>(U32());
				}
				const auto l      = H<vk::PipelineLayout>();
				const auto stages = vk::ShaderStageFlags(vk::ShaderStageFlagBits::eFragment);
				e.pushConstants(l, stages, 0, static_cast<uint32_t>(bytes.size()), bytes.data());
				expected.push_back({op, VerifyHash::PushConstants(l, stages, 0,
				                                                  static_cast<uint32_t>(bytes.size()),
				                                                  bytes.data())});
				break;
			}
			case Op::BindVertexBuffers2: {
				std::vector<vk::Buffer>     buffers(n);
				std::vector<vk::DeviceSize> offsets(n);
				std::vector<vk::DeviceSize> sizes(n);
				for (uint32_t i = 0; i < n; i++) {
					buffers[i] = H<vk::Buffer>();
					offsets[i] = U64() & 0xffffff;
					sizes[i]   = U64() & 0xffffff;
				}
				const bool with_sizes = (U32() & 1) != 0;
				e.bindVertexBuffers2(0, n, buffers.data(), offsets.data(),
				                     with_sizes ? sizes.data() : nullptr, nullptr);
				expected.push_back({op, VerifyHash::BindVertexBuffers2(
				                            0, n, buffers.data(), offsets.data(),
				                            with_sizes ? sizes.data() : nullptr, nullptr)});
				break;
			}
			case Op::BindIndexBuffer: {
				const auto b = H<vk::Buffer>();
				const auto o = U64() & 0xffffff;
				e.bindIndexBuffer(b, o, vk::IndexType::eUint32);
				expected.push_back({op, VerifyHash::BindIndexBuffer(b, o, vk::IndexType::eUint32)});
				break;
			}
			case Op::SetViewportWithCount: {
				std::vector<vk::Viewport> v(n);
				for (auto& x: v) {
					x = vk::Viewport {F32(), F32(), F32(), F32(), 0.0f, 1.0f};
				}
				e.setViewportWithCount(n, v.data());
				expected.push_back({op, VerifyHash::Viewports(n, v.data())});
				break;
			}
			case Op::SetScissorWithCount: {
				std::vector<vk::Rect2D> s(n);
				for (auto& x: s) {
					x = vk::Rect2D {{static_cast<int32_t>(Below(100)), 0}, {U32() % 4096, 7}};
				}
				e.setScissorWithCount(n, s.data());
				expected.push_back({op, VerifyHash::Scissors(n, s.data())});
				break;
			}
			case Op::SetLineWidth: {
				const float w = F32();
				e.setLineWidth(w);
				expected.push_back({op, VerifyHash::Floats(op, &w, 1)});
				break;
			}
			case Op::SetBlendConstants: {
				const float c[4] = {F32(), F32(), F32(), F32()};
				e.setBlendConstants(c);
				expected.push_back({op, VerifyHash::Floats(op, c, 4)});
				break;
			}
			case Op::SetDepthTestEnable:
			case Op::SetDepthWriteEnable:
			case Op::SetDepthBiasEnable:
			case Op::SetStencilTestEnable:
			case Op::SetDepthBoundsTestEnable: {
				const vk::Bool32 v = U32() & 1;
				if (op == Op::SetDepthTestEnable) e.setDepthTestEnable(v);
				if (op == Op::SetDepthWriteEnable) e.setDepthWriteEnable(v);
				if (op == Op::SetDepthBiasEnable) e.setDepthBiasEnable(v);
				if (op == Op::SetStencilTestEnable) e.setStencilTestEnable(v);
				if (op == Op::SetDepthBoundsTestEnable) e.setDepthBoundsTestEnable(v);
				expected.push_back({op, VerifyHash::Value(op, v)});
				break;
			}
			case Op::SetDepthCompareOp: {
				const auto c = static_cast<vk::CompareOp>(Below(8));
				e.setDepthCompareOp(c);
				expected.push_back({op, VerifyHash::Value(op, static_cast<uint32_t>(c))});
				break;
			}
			case Op::SetDepthBias: {
				const float v[3] = {F32(), F32(), F32()};
				e.setDepthBias(v[0], v[1], v[2]);
				expected.push_back({op, VerifyHash::Floats(op, v, 3)});
				break;
			}
			case Op::SetStencilOp: {
				const auto f = vk::StencilFaceFlags(vk::StencilFaceFlagBits::eFront);
				e.setStencilOp(f, vk::StencilOp::eReplace, vk::StencilOp::eKeep,
				               vk::StencilOp::eIncrementAndWrap, vk::CompareOp::eEqual);
				expected.push_back({op, VerifyHash::Value(op, static_cast<VkStencilFaceFlags>(f),
				                                          static_cast<uint64_t>(vk::StencilOp::eReplace),
				                                          static_cast<uint64_t>(vk::StencilOp::eKeep),
				                                          static_cast<uint64_t>(vk::StencilOp::eIncrementAndWrap),
				                                          static_cast<uint64_t>(vk::CompareOp::eEqual))});
				break;
			}
			case Op::SetStencilCompareMask:
			case Op::SetStencilWriteMask:
			case Op::SetStencilReference: {
				const auto     f = vk::StencilFaceFlags(vk::StencilFaceFlagBits::eBack);
				const uint32_t v = U32() & 0xff;
				if (op == Op::SetStencilCompareMask) e.setStencilCompareMask(f, v);
				if (op == Op::SetStencilWriteMask) e.setStencilWriteMask(f, v);
				if (op == Op::SetStencilReference) e.setStencilReference(f, v);
				expected.push_back({op, VerifyHash::Value(op, static_cast<VkStencilFaceFlags>(f), v)});
				break;
			}
			case Op::SetCullMode: {
				const auto m = vk::CullModeFlags(Below(4));
				e.setCullMode(m);
				expected.push_back({op, VerifyHash::Value(op, static_cast<VkCullModeFlags>(m))});
				break;
			}
			case Op::SetFrontFace: {
				const auto f = static_cast<vk::FrontFace>(Below(2));
				e.setFrontFace(f);
				expected.push_back({op, VerifyHash::Value(op, static_cast<uint32_t>(f))});
				break;
			}
			case Op::SetDepthBounds: {
				const float v[2] = {F32(), F32()};
				e.setDepthBounds(v[0], v[1]);
				expected.push_back({op, VerifyHash::Floats(op, v, 2)});
				break;
			}
			case Op::SetColorWriteEnable: {
				std::vector<vk::Bool32> v(1 + Below(8));
				for (auto& x: v) {
					x = U32() & 1;
				}
				e.setColorWriteEnableEXT(static_cast<uint32_t>(v.size()), v.data());
				expected.push_back({op, VerifyHash::ColorWriteEnable(static_cast<uint32_t>(v.size()),
				                                                     v.data())});
				break;
			}
			case Op::SetAttachmentFeedbackLoopEnable: {
				const auto a = vk::ImageAspectFlags(Below(7));
				e.setAttachmentFeedbackLoopEnableEXT(a);
				expected.push_back({op, VerifyHash::Value(op, static_cast<VkImageAspectFlags>(a))});
				break;
			}
			case Op::Draw: {
				const uint32_t a = U32(), b = U32(), c = U32(), d = U32();
				e.draw(a, b, c, d);
				expected.push_back({op, VerifyHash::Value(op, a, b, c, d)});
				break;
			}
			case Op::DrawIndexed: {
				const uint32_t a = U32(), b = U32(), c = U32(), d = U32();
				const auto     v = static_cast<int32_t>(U32());
				e.drawIndexed(a, b, c, v, d);
				expected.push_back({op, VerifyHash::Value(op, a, b, c, static_cast<uint32_t>(v), d)});
				break;
			}
			case Op::DrawMeshTasks:
			case Op::Dispatch: {
				const uint32_t x = U32(), y = U32(), z = U32();
				if (op == Op::DrawMeshTasks) e.drawMeshTasksEXT(x, y, z);
				if (op == Op::Dispatch) e.dispatch(x, y, z);
				expected.push_back({op, VerifyHash::Value(op, x, y, z)});
				break;
			}
			case Op::DrawMeshTasksIndirect:
			case Op::DrawIndirect:
			case Op::DrawIndexedIndirect: {
				const auto     b = H<vk::Buffer>();
				const auto     o = U64() & 0xfff0;
				const uint32_t c = U32(), s = U32();
				if (op == Op::DrawMeshTasksIndirect) e.drawMeshTasksIndirectEXT(b, o, c, s);
				if (op == Op::DrawIndirect) e.drawIndirect(b, o, c, s);
				if (op == Op::DrawIndexedIndirect) e.drawIndexedIndirect(b, o, c, s);
				expected.push_back({op, VerifyHash::Value(op, LogExecutor::Bits(b), o, c, s)});
				break;
			}
			case Op::DrawMeshTasksIndirectCount:
			case Op::DrawIndirectCount:
			case Op::DrawIndexedIndirectCount: {
				const auto     b  = H<vk::Buffer>();
				const auto     cb = H<vk::Buffer>();
				const auto     o = U64() & 0xfff0, co = U64() & 0xfff0;
				const uint32_t m = U32(), s = U32();
				if (op == Op::DrawMeshTasksIndirectCount) e.drawMeshTasksIndirectCountEXT(b, o, cb, co, m, s);
				if (op == Op::DrawIndirectCount) e.drawIndirectCount(b, o, cb, co, m, s);
				if (op == Op::DrawIndexedIndirectCount) e.drawIndexedIndirectCount(b, o, cb, co, m, s);
				expected.push_back({op, VerifyHash::Value(op, LogExecutor::Bits(b), o,
				                                          LogExecutor::Bits(cb), co, m, s)});
				break;
			}
			case Op::DispatchIndirect: {
				const auto b = H<vk::Buffer>();
				const auto o = U64() & 0xfff0;
				e.dispatchIndirect(b, o);
				expected.push_back({op, VerifyHash::Value(op, LogExecutor::Bits(b), o)});
				break;
			}
			case Op::ResetQueryPool: {
				const auto     p = H<vk::QueryPool>();
				const uint32_t f = U32() % 100, c = 1 + Below(4);
				e.resetQueryPool(p, f, c);
				expected.push_back({op, VerifyHash::Value(op, LogExecutor::Bits(p), f, c)});
				break;
			}
			case Op::BeginQuery: {
				const auto     p = H<vk::QueryPool>();
				const uint32_t q = U32() % 100;
				e.beginQuery(p, q, vk::QueryControlFlagBits::ePrecise);
				expected.push_back({op, VerifyHash::Value(op, LogExecutor::Bits(p), q,
				                                          VK_QUERY_CONTROL_PRECISE_BIT)});
				break;
			}
			case Op::EndQuery: {
				const auto     p = H<vk::QueryPool>();
				const uint32_t q = U32() % 100;
				e.endQuery(p, q);
				expected.push_back({op, VerifyHash::Value(op, LogExecutor::Bits(p), q, 0)});
				break;
			}
			case Op::CopyQueryPoolResults: {
				const auto     p = H<vk::QueryPool>();
				const auto     d = H<vk::Buffer>();
				const uint32_t f = U32() % 100, c = 1 + Below(8);
				const auto     o = U64() & 0xfff0;
				const auto     flags =
				    vk::QueryResultFlags(vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
				e.copyQueryPoolResults(p, f, c, d, o, 8, flags);
				expected.push_back({op, VerifyHash::Value(op, LogExecutor::Bits(p), f, c,
				                                          LogExecutor::Bits(d), o,
				                                          8u ^ (static_cast<uint64_t>(
				                                                    static_cast<VkQueryResultFlags>(flags))
				                                                << 40u))});
				break;
			}
			case Op::WriteTimestamp2: {
				const auto     p = H<vk::QueryPool>();
				const uint32_t q = U32() % 100;
				const auto     stage = (U32() & 1u) != 0 ? vk::PipelineStageFlagBits2::eAllCommands
				                                        : vk::PipelineStageFlagBits2::eTopOfPipe;
				e.writeTimestamp2(stage, p, q);
				expected.push_back(
				    {op, VerifyHash::Value(op, LogExecutor::Bits(p), q,
				                           static_cast<uint64_t>(
				                               static_cast<VkPipelineStageFlags2>(
				                                   vk::PipelineStageFlags2(stage))))});
				break;
			}
			case Op::WriteBufferMarker: {
				const auto     b      = H<vk::Buffer>();
				const auto     o      = U64() & 0xffcu;
				const uint32_t m      = U32();
				const auto     stage  = (U32() & 1u) != 0 ? vk::PipelineStageFlagBits::eTopOfPipe
				                                         : vk::PipelineStageFlagBits::eBottomOfPipe;
				e.writeBufferMarkerAMD(stage, b, o, m);
				expected.push_back({op, VerifyHash::Value(op, LogExecutor::Bits(b), o,
				                                          static_cast<uint32_t>(static_cast<VkPipelineStageFlagBits>(stage)),
				                                          m)});
				break;
			}
			default: break;
		}
	}

	void EmitBegin(Encoder& e, uint64_t tick) {
		const auto c = H<vk::CommandBuffer>();
		e.Begin(c, tick, U64());
		expected.push_back({Op::Begin, VerifyHash::Begin(c, tick)});
	}
	void EmitSubmit(Encoder& e, uint64_t tick) {
		SubmitPacket p;
		p.tick = tick;
		p.submit.AddWait(H<vk::Semaphore>(), U64(), vk::PipelineStageFlagBits::eTransfer);
		p.submit.AddSignal(H<vk::Semaphore>(), tick);
		p.debug_op            = U32();
		p.debug_submit        = U64();
		p.preserve_completion = U32() & 1;
		e.Submit(p);
		expected.push_back({Op::Submit, VerifyHash::Submit(p)});
	}
	uint64_t EmitDrain(Encoder& e, uint64_t serial) {
		const auto end = e.DrainMarker(serial);
		expected.push_back({Op::DrainMarker, VerifyHash::DrainMarker(serial)});
		return end;
	}

private:
	std::mt19937_64 m_rng;
};

constexpr Op kFirstCommand = Op::BeginRendering;

// Replays everything currently in the ring into `exec`.
template <typename Exec>
uint64_t ConsumeAll(Ring& ring, Exec& exec, ReplayState& state, MismatchLog& mismatches,
                    WaitStats& stats) {
	uint64_t packets = 0;
	while (const auto* header = ring.Peek()) {
		if (header->op != Op::Wrap) {
			Replay(*header, exec, state, &RecordMismatch, &mismatches);
			packets++;
		}
		ring.Advance(header->size);
	}
	ring.Release(stats);
	return packets;
}

void TestRoundTrip(bool verify) {
	Ring                  ring(1u << 20u);
	Encoder::Options      options;
	options.verify = verify;
	Encoder     encoder(ring, options);
	Generator   gen(verify ? 7u : 11u);
	LogExecutor exec;
	ReplayState state;
	state.verify = verify;
	MismatchLog mismatches;
	WaitStats   stats;
	uint64_t    tick = 1;
	for (uint32_t round = 0; round < 40; round++) {
		gen.EmitBegin(encoder, tick);
		for (auto op = static_cast<uint32_t>(kFirstCommand); op < static_cast<uint32_t>(Op::Count);
		     op++) {
			gen.Emit(encoder, static_cast<Op>(op), 1 + round);
		}
		gen.EmitDrain(encoder, round);
		gen.EmitSubmit(encoder, tick++);
		ConsumeAll(ring, exec, state, mismatches, stats);
	}
	Check(exec.calls.size() == gen.expected.size(), "round trip: every packet replayed once");
	bool equal = exec.calls.size() == gen.expected.size();
	for (size_t i = 0; equal && i < exec.calls.size(); i++) {
		if (!(exec.calls[i] == gen.expected[i])) {
			std::printf("  first difference at call %zu: %s\n", i, OpName(gen.expected[i].op));
			equal = false;
		}
	}
	Check(equal, "round trip: the executor received exactly the encoded arguments, in order");
	Check(mismatches.entries.empty(), "round trip: no verify mismatch");
	if (verify) {
		Check(state.checks == gen.expected.size(), "round trip: every packet verified");
	}
	Check(ring.Consumed() == ring.WritePosition(), "round trip: everything consumed");
}

void TestWrapAndBackPressure() {
	// A 64 KiB ring and ~1.5 KiB packets: many wraps; the producer runs ahead and waits.
	Ring             ring(64u << 10u);
	Encoder::Options options;
	options.verify         = true;
	options.policy.spin_ns = 0; // block at once when full
	Encoder     encoder(ring, options);
	Generator   gen(3);
	LogExecutor exec;
	ReplayState state;
	state.verify = true;
	MismatchLog       mismatches;
	std::atomic<bool> stop {false};
	std::atomic<bool> done {false};

	std::thread consumer([&] {
		WaitStats  stats;
		WaitPolicy policy;
		policy.spin_ns = 0; // park at once
		for (;;) {
			if (!ring.WaitPublished(stop, policy, stats)) {
				if (stop.load()) {
					break;
				}
				continue;
			}
			while (const auto* header = ring.Peek()) {
				if (header->op != Op::Wrap) {
					Replay(*header, exec, state, &RecordMismatch, &mismatches);
				}
				ring.Advance(header->size);
				ring.Release(stats);
				// A slow consumer keeps the ring full.
				if ((ring.ReadPosition() & 0x3fffu) < 1024u) {
					std::this_thread::sleep_for(std::chrono::microseconds(50));
				}
			}
		}
		done = true;
	});

	uint64_t tick = 1;
	for (uint32_t round = 0; round < 400; round++) {
		gen.EmitBegin(encoder, tick);
		for (uint32_t i = 0; i < 6; i++) {
			gen.Emit(encoder, Op::PushDescriptorSet, 24);
			gen.Emit(encoder, Op::DrawIndexed, 1);
		}
		if ((round % 16) == 0) {
			// Drain: everything encoded so far has been executed afterwards.
			const auto end = gen.EmitDrain(encoder, round);
			ring.WaitConsumed(end, options.policy, encoder.Stats());
			Check(ring.Consumed() >= end, "drain: consumed index reached the marker");
		}
		gen.EmitSubmit(encoder, tick++);
	}
	ring.WaitConsumed(ring.WritePosition(), options.policy, encoder.Stats());
	stop = true;
	ring.WakeConsumer();
	consumer.join();

	Check(done.load(), "back-pressure: consumer finished");
	Check(exec.calls == gen.expected, "back-pressure: all packets in order across wraps");
	Check(mismatches.entries.empty(), "back-pressure: no verify mismatch");
	Check(encoder.Stats().blocks != 0, "back-pressure: the producer waited for ring space");
	Check(ring.WritePosition() > 16u * ring.Capacity(), "back-pressure: the ring wrapped many times");
}

void TestThreadedStream() {
	// Default policies, a busy consumer and a producer of random commands.
	Ring             ring(256u << 10u);
	Encoder::Options options;
	options.verify = true;
	Encoder     encoder(ring, options);
	Generator   gen(99);
	LogExecutor exec;
	ReplayState state;
	state.verify = true;
	MismatchLog       mismatches;
	std::atomic<bool> stop {false};
	std::thread       consumer([&] {
        WaitStats  stats;
        WaitPolicy policy;
        while (!stop.load()) {
            if (!ring.WaitPublished(stop, policy, stats)) {
                continue;
            }
            ConsumeAll(ring, exec, state, mismatches, stats);
        }
        ConsumeAll(ring, exec, state, mismatches, stats);
    });
	uint64_t tick = 1;
	for (uint32_t round = 0; round < 2000; round++) {
		gen.EmitBegin(encoder, tick);
		const auto commands = gen.Below(40);
		for (uint32_t i = 0; i < commands; i++) {
			const auto op = static_cast<Op>(static_cast<uint32_t>(kFirstCommand) +
			                                gen.Below(static_cast<uint32_t>(Op::Count) -
			                                          static_cast<uint32_t>(kFirstCommand)));
			gen.Emit(encoder, op, 8);
		}
		if (gen.Below(8) == 0) {
			const auto end = gen.EmitDrain(encoder, round);
			ring.WaitConsumed(end, options.policy, encoder.Stats());
		}
		gen.EmitSubmit(encoder, tick++);
	}
	ring.WaitConsumed(ring.WritePosition(), options.policy, encoder.Stats());
	stop = true;
	ring.WakeConsumer();
	consumer.join();
	Check(exec.calls == gen.expected, "threaded: all packets in order");
	Check(mismatches.entries.empty(), "threaded: no verify mismatch");
	Check(state.checks == gen.expected.size(), "threaded: every packet verified");
}

void TestVerifyDetectsCorruption() {
	Ring             ring(1u << 20u);
	Encoder::Options options;
	options.verify = true;
	Encoder     encoder(ring, options);
	LogExecutor exec;
	ReplayState state;
	state.verify = true;
	MismatchLog mismatches;
	WaitStats   stats;

	encoder.Begin(MakeHandle<vk::CommandBuffer>(0x1000), 1, 0);
	encoder.draw(3, 1, 0, 0);
	// Flip one bit of the draw's vertex count in the ring (a faulty encode or a stale buffer).
	{
		const auto* begin = ring.Peek();
		auto*       draw  = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(begin) + begin->size);
		auto*       payload = draw + sizeof(Header) + sizeof(VerifyBlock);
		payload[0] ^= 0x40;
	}
	ConsumeAll(ring, exec, state, mismatches, stats);
	Check(mismatches.entries.size() == 1 && mismatches.entries[0][0] == 0 &&
	          mismatches.entries[0][1] == static_cast<uint64_t>(Op::Draw),
	      "verify: a corrupted argument is detected at its packet");

	// A dropped packet: its successor's sequence number and the command-buffer digest disagree.
	mismatches.entries.clear();
	encoder.draw(4, 1, 0, 0);
	encoder.draw(5, 1, 0, 0);
	SubmitPacket submit;
	submit.tick = 1;
	encoder.Submit(submit);
	{
		const auto* dropped = ring.Peek();
		ring.Advance(dropped->size); // skip draw(4)
	}
	ConsumeAll(ring, exec, state, mismatches, stats);
	bool sequence = false;
	bool digest   = false;
	for (const auto& entry: mismatches.entries) {
		sequence |= entry[0] == 1;
		digest |= entry[0] == 2;
	}
	Check(sequence, "verify: a dropped packet breaks the sequence");
	Check(digest, "verify: a dropped packet breaks the command-buffer digest");

	// A duplicated packet (replayed twice): sequence mismatch.
	mismatches.entries.clear();
	encoder.Begin(MakeHandle<vk::CommandBuffer>(0x2000), 2, 0);
	encoder.draw(6, 1, 0, 0);
	{
		const auto* begin = ring.Peek();
		Replay(*begin, exec, state, &RecordMismatch, &mismatches);
		ring.Advance(begin->size);
		const auto* draw = ring.Peek();
		Replay(*draw, exec, state, &RecordMismatch, &mismatches);
		Replay(*draw, exec, state, &RecordMismatch, &mismatches);
		ring.Advance(draw->size);
		ring.Release(stats);
	}
	Check(!mismatches.entries.empty() && mismatches.entries[0][0] == 1,
	      "verify: a duplicated packet breaks the sequence");
}

void TestSites() {
	Ring             ring(1u << 20u);
	Encoder::Options options;
	options.barrier_sites = true;
	static int site_a     = 0;
	static int scope_a    = 0;
	options.current_site  = [](const void** site, const void** scope) {
        *site  = &site_a;
        *scope = &scope_a;
	};
	Encoder     encoder(ring, options);
	LogExecutor exec;
	ReplayState state;
	MismatchLog mismatches;
	WaitStats   stats;
	vk::DependencyInfo info {};
	encoder.pipelineBarrier2(info);
	encoder.draw(1, 1, 0, 0);
	ConsumeAll(ring, exec, state, mismatches, stats);
	Check(exec.sites == 1, "sites: only barrier packets carry the producer's site");
}

void Bench(uint64_t publish_batch) {
	// Producer cost of a typical Sky Garden draw (U52: ~8 packets, ~1 KB) with a consumer thread
	// replaying into a null executor.
	Ring             ring(16u << 20u);
	ring.SetPublishBatch(publish_batch);
	Encoder::Options options;
	Encoder     encoder(ring, options);
	NullExecutor exec;
	ReplayState state;
	MismatchLog mismatches;
	std::atomic<bool> stop {false};
	std::thread       consumer([&] {
        WaitStats  stats;
        WaitPolicy policy;
        while (!stop.load()) {
            if (ring.WaitPublished(stop, policy, stats)) {
                ConsumeAll(ring, exec, state, mismatches, stats);
            }
        }
    });
	std::array<vk::WriteDescriptorSet, 20>   writes {};
	std::array<vk::DescriptorBufferInfo, 20> buffers {};
	for (uint32_t i = 0; i < writes.size(); i++) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &buffers[i];
		buffers[i]                = {MakeHandle<vk::Buffer>(0x1000 + i * 16), i * 256u, 256};
	}
	std::array<uint32_t, 16>     push {};
	std::array<vk::Buffer, 3>    vbs {MakeHandle<vk::Buffer>(0x10), MakeHandle<vk::Buffer>(0x20),
	                                  MakeHandle<vk::Buffer>(0x30)};
	std::array<vk::DeviceSize, 3> offsets {};
	vk::Viewport                  viewport {0, 0, 3840, 2160, 0, 1};
	vk::Rect2D                    scissor {{0, 0}, {3840, 2160}};
	// Timed: the encode loops only. Each repetition fits the ring, so the producer never waits
	// for space; the consumer drains between repetitions (untimed).
	constexpr uint32_t Draws       = 10000;
	constexpr uint32_t Repetitions = 20;
	double             elapsed     = 0;
	for (uint32_t rep = 0; rep < Repetitions; rep++) {
		const auto start = std::chrono::steady_clock::now();
		for (uint32_t i = 0; i < Draws; i++) {
			if ((i % 25) == 0) {
				encoder.Begin(MakeHandle<vk::CommandBuffer>(0x100), i, 0);
			}
			encoder.bindVertexBuffers2(0, 3, vbs.data(), offsets.data(), offsets.data(), nullptr);
			push[0] = i;
			encoder.pushConstants(MakeHandle<vk::PipelineLayout>(0x40),
			                      vk::ShaderStageFlagBits::eVertex, 0, sizeof(push), push.data());
			buffers[3].offset = i;
			encoder.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics,
			                             MakeHandle<vk::PipelineLayout>(0x40), 0, 20, writes.data());
			encoder.bindIndexBuffer(MakeHandle<vk::Buffer>(0x50), i * 64u, vk::IndexType::eUint16);
			encoder.setViewportWithCount(1, &viewport);
			encoder.setScissorWithCount(1, &scissor);
			if ((i % 4) == 0) {
				encoder.bindPipeline(vk::PipelineBindPoint::eGraphics,
				                     MakeHandle<vk::Pipeline>(0x60 + i));
			}
			encoder.drawIndexed(300, 1, 0, 0, 0);
			if ((i % 25) == 24) {
				SubmitPacket submit;
				submit.tick = i;
				encoder.Submit(submit);
			}
		}
		elapsed += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
		               .count();
		ring.WaitConsumed(ring.WritePosition(), options.policy, encoder.Stats());
	}
	const uint64_t total = uint64_t {Draws} * Repetitions;
	stop = true;
	ring.WakeConsumer();
	consumer.join();
	std::printf("bench (publish every %" PRIu64 " B): %.1f ns per draw encoded (%.2f packets, "
	            "%.0f bytes per draw), %" PRIu64 " ring-full waits, %" PRIu64 " wakes\n",
	            publish_batch, elapsed / static_cast<double>(total),
	            static_cast<double>(encoder.Packets()) / static_cast<double>(total),
	            static_cast<double>(encoder.Bytes()) / static_cast<double>(total),
	            encoder.Stats().spins, encoder.Stats().wakes);
}

} // namespace

int main(int argc, char** argv) {
	TestRoundTrip(false);
	TestRoundTrip(true);
	TestWrapAndBackPressure();
	TestThreadedStream();
	TestVerifyDetectsCorruption();
	TestSites();
	if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
		for (uint32_t round = 0; round < 3; round++) {
			Bench(0);
			Bench(4u << 10u);
		}
	}
	if (g_failures != 0) {
		std::printf("cp recorder tests: %d failure(s)\n", g_failures);
		return EXIT_FAILURE;
	}
	std::printf("cp recorder tests passed\n");
	return EXIT_SUCCESS;
}
