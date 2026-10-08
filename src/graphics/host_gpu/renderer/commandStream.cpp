#include "graphics/host_gpu/renderer/commandStream.h"
#include "common/ramStats.h"

#include "common/hangWatchdog.h"

#include <chrono>
#include <cinttypes>
#include <new>
#include <thread>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <immintrin.h>
#endif

namespace Libs::Graphics::CommandStream {

namespace {

inline void CpuRelax() noexcept {
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
	_mm_pause();
#elif defined(__aarch64__)
	__asm__ __volatile__("yield");
#endif
}

inline uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Spins with pause until `done()` or `spin_ns` elapsed; returns whether done.
template <typename Done>
bool Spin(uint64_t spin_ns, Done&& done) {
	if (done()) {
		return true;
	}
	if (spin_ns == 0) {
		return false;
	}
	const auto start = NowNs();
	for (uint32_t i = 1;; i++) {
		CpuRelax();
		if (done()) {
			return true;
		}
		if ((i & 63u) == 0 && NowNs() - start >= spin_ns) {
			return false;
		}
	}
}

// Flattened (pNext-free) structures; a pNext chain would be copied as a dangling pointer.
template <typename T>
void RequireNoChain(const T* values, uint64_t count, const char* what) {
	for (uint64_t i = 0; i < count; i++) {
		if (values[i].pNext != nullptr) {
			EXIT("CommandStream: %s with a pNext chain is not supported\n", what);
		}
	}
}

bool IsImageDescriptor(vk::DescriptorType type) {
	switch (type) {
		case vk::DescriptorType::eSampler:
		case vk::DescriptorType::eCombinedImageSampler:
		case vk::DescriptorType::eSampledImage:
		case vk::DescriptorType::eStorageImage:
		case vk::DescriptorType::eInputAttachment: return true;
		default: return false;
	}
}

bool IsBufferDescriptor(vk::DescriptorType type) {
	switch (type) {
		case vk::DescriptorType::eUniformBuffer:
		case vk::DescriptorType::eStorageBuffer:
		case vk::DescriptorType::eUniformBufferDynamic:
		case vk::DescriptorType::eStorageBufferDynamic: return true;
		default: return false;
	}
}

template <typename T>
constexpr uint64_t Sz() noexcept {
	return Align8(sizeof(T));
}
template <typename T>
constexpr uint64_t Arr(uint64_t count) noexcept {
	return Align8(sizeof(T) * count);
}

// ---- Field-wise hashing of the Vulkan structures (no padding bytes) ----

void AddMemoryBarrier2(Hasher& h, const vk::MemoryBarrier2& b) {
	h.AddFlags(b.srcStageMask);
	h.AddFlags(b.srcAccessMask);
	h.AddFlags(b.dstStageMask);
	h.AddFlags(b.dstAccessMask);
	h.Add(b.pNext == nullptr ? 0 : 1);
}

void AddBufferBarrier2(Hasher& h, const vk::BufferMemoryBarrier2& b) {
	h.AddFlags(b.srcStageMask);
	h.AddFlags(b.srcAccessMask);
	h.AddFlags(b.dstStageMask);
	h.AddFlags(b.dstAccessMask);
	h.Add(b.srcQueueFamilyIndex);
	h.Add(b.dstQueueFamilyIndex);
	h.AddHandle(b.buffer);
	h.Add(b.offset);
	h.Add(b.size);
	h.Add(b.pNext == nullptr ? 0 : 1);
}

void AddRange(Hasher& h, const vk::ImageSubresourceRange& r) {
	h.AddFlags(r.aspectMask);
	h.Add(r.baseMipLevel);
	h.Add(r.levelCount);
	h.Add(r.baseArrayLayer);
	h.Add(r.layerCount);
}

void AddLayers(Hasher& h, const vk::ImageSubresourceLayers& r) {
	h.AddFlags(r.aspectMask);
	h.Add(r.mipLevel);
	h.Add(r.baseArrayLayer);
	h.Add(r.layerCount);
}

void AddImageBarrier2(Hasher& h, const vk::ImageMemoryBarrier2& b) {
	h.AddFlags(b.srcStageMask);
	h.AddFlags(b.srcAccessMask);
	h.AddFlags(b.dstStageMask);
	h.AddFlags(b.dstAccessMask);
	h.AddEnum(b.oldLayout);
	h.AddEnum(b.newLayout);
	h.Add(b.srcQueueFamilyIndex);
	h.Add(b.dstQueueFamilyIndex);
	h.AddHandle(b.image);
	AddRange(h, b.subresourceRange);
	h.Add(b.pNext == nullptr ? 0 : 1);
}

void AddAttachment(Hasher& h, const vk::RenderingAttachmentInfo& a) {
	h.AddHandle(a.imageView);
	h.AddEnum(a.imageLayout);
	h.AddFlags(vk::ResolveModeFlags(a.resolveMode));
	h.AddHandle(a.resolveImageView);
	h.AddEnum(a.resolveImageLayout);
	h.AddEnum(a.loadOp);
	h.AddEnum(a.storeOp);
	h.AddBytes(&a.clearValue, sizeof(a.clearValue));
	h.Add(a.pNext == nullptr ? 0 : 1);
}

void AddBufferImageCopy(Hasher& h, const vk::BufferImageCopy& c) {
	h.Add(c.bufferOffset);
	h.Add(c.bufferRowLength);
	h.Add(c.bufferImageHeight);
	AddLayers(h, c.imageSubresource);
	h.Add(static_cast<uint32_t>(c.imageOffset.x));
	h.Add(static_cast<uint32_t>(c.imageOffset.y));
	h.Add(static_cast<uint32_t>(c.imageOffset.z));
	h.Add(c.imageExtent.width);
	h.Add(c.imageExtent.height);
	h.Add(c.imageExtent.depth);
}

} // namespace

const char* OpName(Op op) noexcept {
	static constexpr const char* names[] = {
	    "Wrap",
	    "DrainMarker",
	    "Begin",
	    "Submit",
	    "BeginRendering",
	    "EndRendering",
	    "PipelineBarrier2",
	    "PipelineBarrier",
	    "CopyBuffer",
	    "CopyBufferToImage",
	    "CopyImageToBuffer",
	    "CopyImage",
	    "FillBuffer",
	    "BindPipeline",
	    "BindDescriptorSets",
	    "PushDescriptorSet",
	    "PushConstants",
	    "BindVertexBuffers2",
	    "BindIndexBuffer",
	    "SetViewportWithCount",
	    "SetScissorWithCount",
	    "SetLineWidth",
	    "SetBlendConstants",
	    "SetDepthTestEnable",
	    "SetDepthWriteEnable",
	    "SetDepthCompareOp",
	    "SetDepthBiasEnable",
	    "SetDepthBias",
	    "SetStencilTestEnable",
	    "SetStencilOp",
	    "SetStencilCompareMask",
	    "SetStencilWriteMask",
	    "SetStencilReference",
	    "SetCullMode",
	    "SetFrontFace",
	    "SetDepthBoundsTestEnable",
	    "SetDepthBounds",
	    "SetColorWriteEnable",
	    "SetAttachmentFeedbackLoopEnable",
	    "Draw",
	    "DrawIndexed",
	    "DrawMeshTasks",
	    "DrawMeshTasksIndirect",
	    "DrawMeshTasksIndirectCount",
	    "DrawIndirect",
	    "DrawIndexedIndirect",
	    "DrawIndirectCount",
	    "DrawIndexedIndirectCount",
	    "Dispatch",
	    "DispatchIndirect",
	    "ResetQueryPool",
	    "BeginQuery",
	    "EndQuery",
	    "CopyQueryPoolResults",
	    "WriteTimestamp2",
	    "UpdateDescriptorSets",
	    "WriteBufferMarker",
	};
	static_assert(std::size(names) == static_cast<size_t>(Op::Count));
	const auto index = static_cast<size_t>(op);
	return index < std::size(names) ? names[index] : "?";
}

// ------------------------------------------------------------------------------------------------
// VerifyHash

namespace VerifyHash {

uint64_t Begin(vk::CommandBuffer command, uint64_t tick) {
	Hasher h(Op::Begin);
	h.AddHandle(command);
	h.Add(tick);
	return h.Value();
}

uint64_t Submit(const SubmitPacket& p) {
	Hasher h(Op::Submit);
	h.Add(p.tick);
	h.Add(p.submit_ns);
	h.Add(p.debug_submit);
	h.Add(p.debug_arg4);
	h.Add(p.debug_op);
	h.Add(p.debug_arg0);
	h.Add(p.debug_arg1);
	h.Add(p.debug_arg2);
	h.Add(p.debug_arg3);
	h.Add(p.preserve_completion);
	h.Add(p.cb_digest);
	h.Add(p.cb_packets);
	const auto& s = p.submit;
	h.Add(s.num_wait_semaphores);
	h.Add(s.num_signal_semaphores);
	for (uint32_t i = 0; i < s.num_wait_semaphores && i < SubmitInfo::MaxSemaphores; i++) {
		h.AddHandle(s.wait_semaphores[i]);
		h.Add(s.wait_ticks[i]);
		h.AddFlags(s.wait_stages[i]);
	}
	for (uint32_t i = 0; i < s.num_signal_semaphores && i < SubmitInfo::MaxSemaphores; i++) {
		h.AddHandle(s.signal_semaphores[i]);
		h.Add(s.signal_ticks[i]);
	}
	return h.Value();
}

uint64_t DrainMarker(uint64_t serial) {
	Hasher h(Op::DrainMarker);
	h.Add(serial);
	return h.Value();
}

uint64_t BeginRendering(const vk::RenderingInfo& info) {
	Hasher h(Op::BeginRendering);
	h.AddFlags(info.flags);
	h.Add(static_cast<uint32_t>(info.renderArea.offset.x));
	h.Add(static_cast<uint32_t>(info.renderArea.offset.y));
	h.Add(info.renderArea.extent.width);
	h.Add(info.renderArea.extent.height);
	h.Add(info.layerCount);
	h.Add(info.viewMask);
	h.Add(info.colorAttachmentCount);
	for (uint32_t i = 0; i < info.colorAttachmentCount; i++) {
		AddAttachment(h, info.pColorAttachments[i]);
	}
	h.Add(info.pDepthAttachment != nullptr ? 1 : 0);
	if (info.pDepthAttachment != nullptr) {
		AddAttachment(h, *info.pDepthAttachment);
	}
	h.Add(info.pStencilAttachment != nullptr ? 1 : 0);
	if (info.pStencilAttachment != nullptr) {
		AddAttachment(h, *info.pStencilAttachment);
	}
	h.Add(info.pNext == nullptr ? 0 : 1);
	return h.Value();
}

uint64_t EndRendering() {
	return Hasher(Op::EndRendering).Value();
}

uint64_t PipelineBarrier2(const vk::DependencyInfo& info) {
	Hasher h(Op::PipelineBarrier2);
	h.AddFlags(info.dependencyFlags);
	h.Add(info.memoryBarrierCount);
	for (uint32_t i = 0; i < info.memoryBarrierCount; i++) {
		AddMemoryBarrier2(h, info.pMemoryBarriers[i]);
	}
	h.Add(info.bufferMemoryBarrierCount);
	for (uint32_t i = 0; i < info.bufferMemoryBarrierCount; i++) {
		AddBufferBarrier2(h, info.pBufferMemoryBarriers[i]);
	}
	h.Add(info.imageMemoryBarrierCount);
	for (uint32_t i = 0; i < info.imageMemoryBarrierCount; i++) {
		AddImageBarrier2(h, info.pImageMemoryBarriers[i]);
	}
	h.Add(info.pNext == nullptr ? 0 : 1);
	return h.Value();
}

uint64_t PipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst,
                         vk::DependencyFlags flags, uint32_t memory_count,
                         const vk::MemoryBarrier* memory, uint32_t buffer_count,
                         const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
                         const vk::ImageMemoryBarrier* images) {
	Hasher h(Op::PipelineBarrier);
	h.AddFlags(src);
	h.AddFlags(dst);
	h.AddFlags(flags);
	h.Add(memory_count);
	for (uint32_t i = 0; i < memory_count; i++) {
		h.AddFlags(memory[i].srcAccessMask);
		h.AddFlags(memory[i].dstAccessMask);
		h.Add(memory[i].pNext == nullptr ? 0 : 1);
	}
	h.Add(buffer_count);
	for (uint32_t i = 0; i < buffer_count; i++) {
		const auto& b = buffers[i];
		h.AddFlags(b.srcAccessMask);
		h.AddFlags(b.dstAccessMask);
		h.Add(b.srcQueueFamilyIndex);
		h.Add(b.dstQueueFamilyIndex);
		h.AddHandle(b.buffer);
		h.Add(b.offset);
		h.Add(b.size);
		h.Add(b.pNext == nullptr ? 0 : 1);
	}
	h.Add(image_count);
	for (uint32_t i = 0; i < image_count; i++) {
		const auto& b = images[i];
		h.AddFlags(b.srcAccessMask);
		h.AddFlags(b.dstAccessMask);
		h.AddEnum(b.oldLayout);
		h.AddEnum(b.newLayout);
		h.Add(b.srcQueueFamilyIndex);
		h.Add(b.dstQueueFamilyIndex);
		h.AddHandle(b.image);
		AddRange(h, b.subresourceRange);
		h.Add(b.pNext == nullptr ? 0 : 1);
	}
	return h.Value();
}

uint64_t CopyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
                    const vk::BufferCopy* regions) {
	Hasher h(Op::CopyBuffer);
	h.AddHandle(source);
	h.AddHandle(destination);
	h.Add(count);
	for (uint32_t i = 0; i < count; i++) {
		h.Add(regions[i].srcOffset);
		h.Add(regions[i].dstOffset);
		h.Add(regions[i].size);
	}
	return h.Value();
}

uint64_t CopyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
                           uint32_t count, const vk::BufferImageCopy* regions) {
	Hasher h(Op::CopyBufferToImage);
	h.AddHandle(source);
	h.AddHandle(destination);
	h.AddEnum(layout);
	h.Add(count);
	for (uint32_t i = 0; i < count; i++) {
		AddBufferImageCopy(h, regions[i]);
	}
	return h.Value();
}

uint64_t CopyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
                           uint32_t count, const vk::BufferImageCopy* regions) {
	Hasher h(Op::CopyImageToBuffer);
	h.AddHandle(source);
	h.AddEnum(layout);
	h.AddHandle(destination);
	h.Add(count);
	for (uint32_t i = 0; i < count; i++) {
		AddBufferImageCopy(h, regions[i]);
	}
	return h.Value();
}

uint64_t CopyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
                   vk::ImageLayout destination_layout, uint32_t count,
                   const vk::ImageCopy* regions) {
	Hasher h(Op::CopyImage);
	h.AddHandle(source);
	h.AddEnum(source_layout);
	h.AddHandle(destination);
	h.AddEnum(destination_layout);
	h.Add(count);
	for (uint32_t i = 0; i < count; i++) {
		const auto& r = regions[i];
		AddLayers(h, r.srcSubresource);
		h.Add(static_cast<uint32_t>(r.srcOffset.x));
		h.Add(static_cast<uint32_t>(r.srcOffset.y));
		h.Add(static_cast<uint32_t>(r.srcOffset.z));
		AddLayers(h, r.dstSubresource);
		h.Add(static_cast<uint32_t>(r.dstOffset.x));
		h.Add(static_cast<uint32_t>(r.dstOffset.y));
		h.Add(static_cast<uint32_t>(r.dstOffset.z));
		h.Add(r.extent.width);
		h.Add(r.extent.height);
		h.Add(r.extent.depth);
	}
	return h.Value();
}

uint64_t FillBuffer(vk::Buffer buffer, uint64_t offset, uint64_t size, uint32_t value) {
	Hasher h(Op::FillBuffer);
	h.AddHandle(buffer);
	h.Add(offset);
	h.Add(size);
	h.Add(value);
	return h.Value();
}

uint64_t BindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline) {
	Hasher h(Op::BindPipeline);
	h.AddEnum(point);
	h.AddHandle(pipeline);
	return h.Value();
}

uint64_t BindDescriptorSets(vk::PipelineBindPoint point, vk::PipelineLayout layout,
                            uint32_t first_set, uint32_t set_count, const vk::DescriptorSet* sets,
                            uint32_t dynamic_count, const uint32_t* dynamic_offsets) {
	Hasher h(Op::BindDescriptorSets);
	h.AddEnum(point);
	h.AddHandle(layout);
	h.Add(first_set);
	h.Add(set_count);
	for (uint32_t i = 0; i < set_count; i++) {
		h.AddHandle(sets[i]);
	}
	h.Add(dynamic_count);
	for (uint32_t i = 0; i < dynamic_count; i++) {
		h.Add(dynamic_offsets[i]);
	}
	return h.Value();
}

// The writes' fields other than dstSet, and their infos.
static void AddDescriptorWrites(Hasher& h, uint32_t count, const vk::WriteDescriptorSet* writes) {
	for (uint32_t i = 0; i < count; i++) {
		const auto& w = writes[i];
		h.Add(w.dstBinding);
		h.Add(w.dstArrayElement);
		h.Add(w.descriptorCount);
		h.AddEnum(w.descriptorType);
		h.Add(w.pNext == nullptr ? 0 : 1);
		if (IsImageDescriptor(w.descriptorType)) {
			for (uint32_t j = 0; j < w.descriptorCount; j++) {
				h.AddHandle(w.pImageInfo[j].sampler);
				h.AddHandle(w.pImageInfo[j].imageView);
				h.AddEnum(w.pImageInfo[j].imageLayout);
			}
		} else if (IsBufferDescriptor(w.descriptorType)) {
			for (uint32_t j = 0; j < w.descriptorCount; j++) {
				h.AddHandle(w.pBufferInfo[j].buffer);
				h.Add(w.pBufferInfo[j].offset);
				h.Add(w.pBufferInfo[j].range);
			}
		} else {
			h.Add(0xdeadu);
		}
	}
}

uint64_t PushDescriptorSet(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t set,
                           uint32_t count, const vk::WriteDescriptorSet* writes) {
	Hasher h(Op::PushDescriptorSet);
	h.AddEnum(point);
	h.AddHandle(layout);
	h.Add(set);
	h.Add(count);
	// dstSet is ignored by vkCmdPushDescriptorSetKHR.
	AddDescriptorWrites(h, count, writes);
	return h.Value();
}

uint64_t UpdateDescriptorSets(vk::DescriptorSet set, uint32_t count,
                              const vk::WriteDescriptorSet* writes) {
	Hasher h(Op::UpdateDescriptorSets);
	h.AddHandle(set);
	h.Add(count);
	for (uint32_t i = 0; i < count; i++) {
		h.AddHandle(writes[i].dstSet);
	}
	AddDescriptorWrites(h, count, writes);
	return h.Value();
}

uint64_t PushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
                       uint32_t size, const void* data) {
	Hasher h(Op::PushConstants);
	h.AddHandle(layout);
	h.AddFlags(stages);
	h.Add(offset);
	h.AddBytes(data, size);
	return h.Value();
}

uint64_t BindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
                            const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
                            const vk::DeviceSize* strides) {
	Hasher h(Op::BindVertexBuffers2);
	h.Add(first);
	h.Add(count);
	h.Add(sizes != nullptr ? 1 : 0);
	h.Add(strides != nullptr ? 1 : 0);
	for (uint32_t i = 0; i < count; i++) {
		h.AddHandle(buffers[i]);
		h.Add(offsets[i]);
		if (sizes != nullptr) {
			h.Add(sizes[i]);
		}
		if (strides != nullptr) {
			h.Add(strides[i]);
		}
	}
	return h.Value();
}

uint64_t BindIndexBuffer(vk::Buffer buffer, uint64_t offset, vk::IndexType type) {
	Hasher h(Op::BindIndexBuffer);
	h.AddHandle(buffer);
	h.Add(offset);
	h.AddEnum(type);
	return h.Value();
}

uint64_t Viewports(uint32_t count, const vk::Viewport* viewports) {
	Hasher h(Op::SetViewportWithCount);
	h.Add(count);
	for (uint32_t i = 0; i < count; i++) {
		const auto& v = viewports[i];
		h.AddFloat(v.x);
		h.AddFloat(v.y);
		h.AddFloat(v.width);
		h.AddFloat(v.height);
		h.AddFloat(v.minDepth);
		h.AddFloat(v.maxDepth);
	}
	return h.Value();
}

uint64_t Scissors(uint32_t count, const vk::Rect2D* scissors) {
	Hasher h(Op::SetScissorWithCount);
	h.Add(count);
	for (uint32_t i = 0; i < count; i++) {
		h.Add(static_cast<uint32_t>(scissors[i].offset.x));
		h.Add(static_cast<uint32_t>(scissors[i].offset.y));
		h.Add(scissors[i].extent.width);
		h.Add(scissors[i].extent.height);
	}
	return h.Value();
}

uint64_t Value(Op op, uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f) {
	Hasher h(op);
	h.Add(a);
	h.Add(b);
	h.Add(c);
	h.Add(d);
	h.Add(e);
	h.Add(f);
	return h.Value();
}

uint64_t Floats(Op op, const float* values, uint32_t count) {
	Hasher h(op);
	h.Add(count);
	for (uint32_t i = 0; i < count; i++) {
		h.AddFloat(values[i]);
	}
	return h.Value();
}

uint64_t ColorWriteEnable(uint32_t count, const vk::Bool32* enables) {
	Hasher h(Op::SetColorWriteEnable);
	h.Add(count);
	for (uint32_t i = 0; i < count; i++) {
		h.Add(enables[i]);
	}
	return h.Value();
}

} // namespace VerifyHash

// ------------------------------------------------------------------------------------------------
// Ring

Ring::Ring(uint64_t capacity) {
	EXIT_IF(capacity < (64u << 10u) || (capacity & (capacity - 1u)) != 0);
	m_capacity = capacity;
	m_mask     = capacity - 1u;
	m_data     = static_cast<uint8_t*>(::operator new(capacity, std::align_val_t {64}));
	Common::RamStats::Range("CP command ring", m_data, capacity);
}

Ring::~Ring() {
	::operator delete(m_data, std::align_val_t {64});
}

bool Ring::EnsureSpace(uint64_t bytes, const WaitPolicy& policy, WaitStats& stats) {
	const auto fits = [&] { return m_write + bytes - m_consumed_cache <= m_capacity; };
	if (fits()) {
		return false;
	}
	m_consumed_cache = m_consumed.load(std::memory_order_acquire);
	if (fits()) {
		return false;
	}
	// The consumer may be parked on packets published without a wake.
	HangWatchdog::Scope wait("recorder-ring-space", reinterpret_cast<uint64_t>(this),
	                         m_write + bytes, m_consumed_cache, 0, m_capacity);
	Kick(stats);
	const auto start = NowNs();
	stats.spins++;
	const bool spun = Spin(policy.spin_ns, [&] {
		m_consumed_cache = m_consumed.load(std::memory_order_acquire);
		return fits();
	});
	if (!spun) {
		stats.blocks++;
		for (;;) {
			const auto bell = m_producer_bell.load(std::memory_order_acquire);
			m_producer_waiting.store(1, std::memory_order_seq_cst);
			m_consumed_cache = m_consumed.load(std::memory_order_seq_cst);
			if (fits()) {
				break;
			}
			m_producer_bell.wait(bell, std::memory_order_acquire);
		}
		m_producer_waiting.store(0, std::memory_order_relaxed);
	}
	stats.wait_ns += NowNs() - start;
	return true;
}

uint8_t* Ring::Reserve(uint32_t bytes, const WaitPolicy& policy, WaitStats& stats) {
	EXIT_IF(bytes == 0 || (bytes & 7u) != 0 || bytes > MaxPacket());
	const auto position = m_write & m_mask;
	if (position + bytes > m_capacity) {
		// Pad to the end with a Wrap packet; the region starts at the beginning.
		const auto pad = m_capacity - position;
		EnsureSpace(pad + bytes, policy, stats);
		auto* wrap  = reinterpret_cast<Header*>(m_data + position);
		wrap->op    = Op::Wrap;
		wrap->flags = 0;
		wrap->size  = static_cast<uint32_t>(pad);
		m_write += pad;
		return m_data;
	}
	EnsureSpace(bytes, policy, stats);
	return m_data + position;
}

void Ring::Kick(WaitStats& stats) noexcept {
	m_last_published = m_write;
	m_published.store(m_write, std::memory_order_seq_cst);
	if (m_consumer_parked.load(std::memory_order_seq_cst) != 0) {
		m_consumer_bell.fetch_add(1, std::memory_order_release);
		m_consumer_bell.notify_one();
		stats.wakes++;
	}
}

void Ring::WaitConsumed(uint64_t position, const WaitPolicy& policy, WaitStats& stats) {
	EXIT_IF(position > m_write);
	const auto done = [&] {
		m_consumed_cache = m_consumed.load(std::memory_order_acquire);
		return m_consumed_cache >= position;
	};
	if (done()) {
		return;
	}
	HangWatchdog::Scope wait("recorder-ring-consumed", reinterpret_cast<uint64_t>(this), position,
	                         m_consumed_cache);
	Kick(stats);
	const auto start = NowNs();
	stats.spins++;
	if (!Spin(policy.spin_ns, done)) {
		stats.blocks++;
		for (;;) {
			const auto bell = m_producer_bell.load(std::memory_order_acquire);
			m_producer_waiting.store(1, std::memory_order_seq_cst);
			m_consumed_cache = m_consumed.load(std::memory_order_seq_cst);
			if (m_consumed_cache >= position) {
				break;
			}
			m_producer_bell.wait(bell, std::memory_order_acquire);
		}
		m_producer_waiting.store(0, std::memory_order_relaxed);
	}
	stats.wait_ns += NowNs() - start;
}

void Ring::Release(WaitStats& stats) noexcept {
	m_consumed.store(m_read, std::memory_order_seq_cst);
	if (m_producer_waiting.load(std::memory_order_seq_cst) != 0) {
		m_producer_bell.fetch_add(1, std::memory_order_release);
		m_producer_bell.notify_one();
		stats.wakes++;
	}
}

bool Ring::WaitPublished(const std::atomic<bool>& stop, const WaitPolicy& policy,
                         WaitStats& stats) {
	const auto available = [&] {
		m_published_cache = m_published.load(std::memory_order_acquire);
		return m_published_cache != m_read;
	};
	if (available()) {
		return true;
	}
	HangWatchdog::Scope wait("recorder-ring-published", reinterpret_cast<uint64_t>(this), m_read,
	                         m_published_cache);
	const auto start = NowNs();
	stats.spins++;
	const bool spun =
	    Spin(policy.spin_ns, [&] { return available() || stop.load(std::memory_order_acquire); });
	if (!spun) {
		stats.blocks++;
		for (;;) {
			const auto bell = m_consumer_bell.load(std::memory_order_acquire);
			m_consumer_parked.store(1, std::memory_order_seq_cst);
			m_published_cache = m_published.load(std::memory_order_seq_cst);
			if (m_published_cache != m_read || stop.load(std::memory_order_seq_cst)) {
				break;
			}
			m_consumer_bell.wait(bell, std::memory_order_acquire);
		}
		m_consumer_parked.store(0, std::memory_order_relaxed);
	}
	stats.wait_ns += NowNs() - start;
	return m_published_cache != m_read;
}

void Ring::WakeConsumer() noexcept {
	m_consumer_bell.fetch_add(1, std::memory_order_seq_cst);
	m_consumer_bell.notify_all();
}

// ------------------------------------------------------------------------------------------------
// Encoder

Encoder::Writer Encoder::Open(Op op, uint64_t payload, bool barrier) {
	const bool     site   = m_options.all_sites || (barrier && m_options.barrier_sites);
	const bool     verify = m_options.verify;
	const uint64_t bytes  = sizeof(Header) + (site ? sizeof(SiteBlock) : 0u) +
	                        (verify ? sizeof(VerifyBlock) : 0u) + Align8(payload);
	if (bytes > m_ring.MaxPacket()) {
		EXIT("CommandStream: %s packet of %" PRIu64 " bytes exceeds the ring's packet limit %u\n",
		     OpName(op), bytes, m_ring.MaxPacket());
	}
	auto* base      = m_ring.Reserve(static_cast<uint32_t>(bytes), m_options.policy, m_stats);
	auto* header    = reinterpret_cast<Header*>(base);
	header->op      = op;
	header->flags   = static_cast<uint16_t>((site ? FlagSite : 0u) | (verify ? FlagVerify : 0u));
	header->size    = static_cast<uint32_t>(bytes);
	uint32_t offset = sizeof(Header);
	if (site) {
		SiteBlock block;
		if (m_options.current_site != nullptr) {
			m_options.current_site(&block.site, &block.scope);
		}
		std::memcpy(base + offset, &block, sizeof(block));
		offset += sizeof(SiteBlock);
	}
	VerifyBlock* verify_block = nullptr;
	if (verify) {
		verify_block = reinterpret_cast<VerifyBlock*>(base + offset);
		offset += sizeof(VerifyBlock);
	}
	return {base, static_cast<uint32_t>(bytes), offset, verify_block};
}

void Encoder::Close(Writer& writer, uint64_t hash) {
	// The payload size computed for Open must match what was written.
	EXIT_IF(writer.Offset() != writer.Size());
	if (auto* verify = writer.Verify(); verify != nullptr) {
		verify->hash     = hash;
		verify->sequence = ++m_sequence;
		m_cb_digest      = FoldDigest(m_cb_digest, writer.GetHeader()->op, hash);
		m_cb_packets++;
	}
	m_packets++;
	m_bytes += writer.Size();
	const auto op = writer.GetHeader()->op;
	m_ring.Commit(writer.Size());
	if (KicksConsumer(op)) {
		m_ring.Kick(m_stats);
	}
	if (m_options.after_commit != nullptr) {
		// Inline replay reads only published packets.
		m_ring.Publish();
		m_options.after_commit(m_options.after_commit_context);
	}
}

void Encoder::Begin(vk::CommandBuffer command, uint64_t tick, uint64_t record_ns) {
	m_cb_digest  = 0;
	m_cb_packets = 0;
	auto w       = Open(Op::Begin, Sz<BeginPacket>(), false);
	w.Put(BeginPacket {command, tick, record_ns});
	Close(w, m_options.verify ? VerifyHash::Begin(command, tick) : 0);
}

void Encoder::Submit(const SubmitPacket& submit) {
	SubmitPacket packet = submit;
	packet.cb_digest    = m_cb_digest;
	packet.cb_packets   = m_cb_packets;
	auto w              = Open(Op::Submit, Sz<SubmitPacket>(), false);
	w.Put(packet);
	Close(w, m_options.verify ? VerifyHash::Submit(packet) : 0);
}

uint64_t Encoder::DrainMarker(uint64_t serial) {
	auto w = Open(Op::DrainMarker, Sz<DrainMarkerPacket>(), false);
	w.Put(DrainMarkerPacket {serial});
	Close(w, m_options.verify ? VerifyHash::DrainMarker(serial) : 0);
	return m_ring.WritePosition();
}

void Encoder::beginRendering(const vk::RenderingInfo& info) {
	EXIT_IF(info.pNext != nullptr ||
	        (info.colorAttachmentCount != 0 && info.pColorAttachments == nullptr));
	RequireNoChain(info.pColorAttachments, info.colorAttachmentCount, "rendering attachment");
	RequireNoChain(info.pDepthAttachment, info.pDepthAttachment != nullptr ? 1 : 0,
	               "depth attachment");
	RequireNoChain(info.pStencilAttachment, info.pStencilAttachment != nullptr ? 1 : 0,
	               "stencil attachment");
	const auto payload =
	    Sz<BeginRenderingPacket>() + Arr<vk::RenderingAttachmentInfo>(info.colorAttachmentCount) +
	    (info.pDepthAttachment != nullptr ? Sz<vk::RenderingAttachmentInfo>() : 0) +
	    (info.pStencilAttachment != nullptr ? Sz<vk::RenderingAttachmentInfo>() : 0);
	auto                 w = Open(Op::BeginRendering, payload, false);
	BeginRenderingPacket p;
	p.area        = info.renderArea;
	p.flags       = info.flags;
	p.layers      = info.layerCount;
	p.view_mask   = info.viewMask;
	p.colors      = info.colorAttachmentCount;
	p.has_depth   = info.pDepthAttachment != nullptr ? 1u : 0u;
	p.has_stencil = info.pStencilAttachment != nullptr ? 1u : 0u;
	w.Put(p);
	w.PutArray(info.pColorAttachments, info.colorAttachmentCount);
	if (info.pDepthAttachment != nullptr) {
		w.Put(*info.pDepthAttachment);
	}
	if (info.pStencilAttachment != nullptr) {
		w.Put(*info.pStencilAttachment);
	}
	Close(w, m_options.verify ? VerifyHash::BeginRendering(info) : 0);
}

void Encoder::endRendering() {
	auto w = Open(Op::EndRendering, 0, false);
	Close(w, m_options.verify ? VerifyHash::EndRendering() : 0);
}

void Encoder::pipelineBarrier2(const vk::DependencyInfo& info) {
	EXIT_IF(info.pNext != nullptr);
	RequireNoChain(info.pMemoryBarriers, info.memoryBarrierCount, "memory barrier");
	RequireNoChain(info.pBufferMemoryBarriers, info.bufferMemoryBarrierCount, "buffer barrier");
	RequireNoChain(info.pImageMemoryBarriers, info.imageMemoryBarrierCount, "image barrier");
	const auto payload = Sz<PipelineBarrier2Packet>() +
	                     Arr<vk::MemoryBarrier2>(info.memoryBarrierCount) +
	                     Arr<vk::BufferMemoryBarrier2>(info.bufferMemoryBarrierCount) +
	                     Arr<vk::ImageMemoryBarrier2>(info.imageMemoryBarrierCount);
	auto       w       = Open(Op::PipelineBarrier2, payload, true);
	w.Put(PipelineBarrier2Packet {info.dependencyFlags, info.memoryBarrierCount,
	                              info.bufferMemoryBarrierCount, info.imageMemoryBarrierCount});
	w.PutArray(info.pMemoryBarriers, info.memoryBarrierCount);
	w.PutArray(info.pBufferMemoryBarriers, info.bufferMemoryBarrierCount);
	w.PutArray(info.pImageMemoryBarriers, info.imageMemoryBarrierCount);
	Close(w, m_options.verify ? VerifyHash::PipelineBarrier2(info) : 0);
}

void Encoder::pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst,
                              vk::DependencyFlags flags, uint32_t memory_count,
                              const vk::MemoryBarrier* memory, uint32_t buffer_count,
                              const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
                              const vk::ImageMemoryBarrier* images) {
	RequireNoChain(memory, memory_count, "memory barrier");
	RequireNoChain(buffers, buffer_count, "buffer barrier");
	RequireNoChain(images, image_count, "image barrier");
	const auto payload = Sz<PipelineBarrierPacket>() + Arr<vk::MemoryBarrier>(memory_count) +
	                     Arr<vk::BufferMemoryBarrier>(buffer_count) +
	                     Arr<vk::ImageMemoryBarrier>(image_count);
	auto       w       = Open(Op::PipelineBarrier, payload, true);
	w.Put(PipelineBarrierPacket {src, dst, flags, memory_count, buffer_count, image_count});
	w.PutArray(memory, memory_count);
	w.PutArray(buffers, buffer_count);
	w.PutArray(images, image_count);
	Close(w, m_options.verify
	             ? VerifyHash::PipelineBarrier(src, dst, flags, memory_count, memory, buffer_count,
	                                           buffers, image_count, images)
	             : 0);
}

void Encoder::copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
                         const vk::BufferCopy* regions) {
	auto w = Open(Op::CopyBuffer, Sz<CopyBufferPacket>() + Arr<vk::BufferCopy>(count), false);
	w.Put(CopyBufferPacket {source, destination, count});
	w.PutArray(regions, count);
	Close(w, m_options.verify ? VerifyHash::CopyBuffer(source, destination, count, regions) : 0);
}

void Encoder::copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
                                uint32_t count, const vk::BufferImageCopy* regions) {
	auto w = Open(Op::CopyBufferToImage,
	              Sz<CopyBufferToImagePacket>() + Arr<vk::BufferImageCopy>(count), false);
	w.Put(CopyBufferToImagePacket {source, destination, layout, count});
	w.PutArray(regions, count);
	Close(w, m_options.verify
	             ? VerifyHash::CopyBufferToImage(source, destination, layout, count, regions)
	             : 0);
}

void Encoder::copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
                                uint32_t count, const vk::BufferImageCopy* regions) {
	auto w = Open(Op::CopyImageToBuffer,
	              Sz<CopyImageToBufferPacket>() + Arr<vk::BufferImageCopy>(count), false);
	w.Put(CopyImageToBufferPacket {source, destination, layout, count});
	w.PutArray(regions, count);
	Close(w, m_options.verify
	             ? VerifyHash::CopyImageToBuffer(source, layout, destination, count, regions)
	             : 0);
}

void Encoder::copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
                        vk::ImageLayout destination_layout, uint32_t count,
                        const vk::ImageCopy* regions) {
	auto w = Open(Op::CopyImage, Sz<CopyImagePacket>() + Arr<vk::ImageCopy>(count), false);
	w.Put(CopyImagePacket {source, destination, source_layout, destination_layout, count});
	w.PutArray(regions, count);
	Close(w, m_options.verify ? VerifyHash::CopyImage(source, source_layout, destination,
	                                                  destination_layout, count, regions)
	                          : 0);
}

void Encoder::fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
                         uint32_t value) {
	auto w = Open(Op::FillBuffer, Sz<FillBufferPacket>(), false);
	w.Put(FillBufferPacket {buffer, offset, size, value});
	Close(w, m_options.verify ? VerifyHash::FillBuffer(buffer, offset, size, value) : 0);
}

void Encoder::bindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline) {
	auto w = Open(Op::BindPipeline, Sz<BindPipelinePacket>(), false);
	w.Put(BindPipelinePacket {pipeline, point});
	Close(w, m_options.verify ? VerifyHash::BindPipeline(point, pipeline) : 0);
}

void Encoder::bindDescriptorSets(vk::PipelineBindPoint point, vk::PipelineLayout layout,
                                 uint32_t first_set, uint32_t set_count,
                                 const vk::DescriptorSet* sets, uint32_t dynamic_count,
                                 const uint32_t* dynamic_offsets) {
	auto w = Open(Op::BindDescriptorSets,
	              Sz<BindDescriptorSetsPacket>() + Arr<vk::DescriptorSet>(set_count) +
	                  Arr<uint32_t>(dynamic_count),
	              false);
	w.Put(BindDescriptorSetsPacket {layout, point, first_set, set_count, dynamic_count});
	w.PutArray(sets, set_count);
	w.PutArray(dynamic_offsets, dynamic_count);
	Close(w, m_options.verify ? VerifyHash::BindDescriptorSets(point, layout, first_set, set_count,
	                                                           sets, dynamic_count, dynamic_offsets)
	                          : 0);
}

uint64_t Encoder::DescriptorInfoCount(uint32_t count, const vk::WriteDescriptorSet* writes) {
	uint64_t infos = 0;
	for (uint32_t i = 0; i < count; i++) {
		const auto& write = writes[i];
		if (write.pNext != nullptr || write.pTexelBufferView != nullptr ||
		    (!IsImageDescriptor(write.descriptorType) &&
		     !IsBufferDescriptor(write.descriptorType))) {
			EXIT("CommandStream: descriptor type %u is not supported\n",
			     static_cast<uint32_t>(write.descriptorType));
		}
		const bool image = IsImageDescriptor(write.descriptorType);
		EXIT_IF(write.descriptorCount != 0 &&
		        (image ? write.pImageInfo == nullptr : write.pBufferInfo == nullptr));
		infos += write.descriptorCount;
	}
	return infos;
}

void Encoder::PutDescriptorWrites(Writer& w, uint32_t count, const vk::WriteDescriptorSet* writes,
                                  uint64_t infos) {
	auto* records = reinterpret_cast<DescriptorWriteRecord*>(w.Cursor());
	w.Skip(sizeof(DescriptorWriteRecord) * count);
	auto* out = w.Cursor();
	w.Skip(sizeof(DescriptorInfo) * infos);
	for (uint32_t i = 0; i < count; i++) {
		const auto&           write = writes[i];
		const bool            image = IsImageDescriptor(write.descriptorType);
		DescriptorWriteRecord record;
		record.binding  = write.dstBinding;
		record.element  = write.dstArrayElement;
		record.count    = write.descriptorCount;
		record.type     = write.descriptorType;
		record.is_image = image ? 1u : 0u;
		std::memcpy(records + i, &record, sizeof(record));
		const auto bytes = sizeof(DescriptorInfo) * write.descriptorCount;
		if (bytes != 0) {
			std::memcpy(out,
			            image ? static_cast<const void*>(write.pImageInfo)
			                  : static_cast<const void*>(write.pBufferInfo),
			            bytes);
			out += bytes;
		}
	}
}

void Encoder::pushDescriptorSetKHR(vk::PipelineBindPoint point, vk::PipelineLayout layout,
                                   uint32_t set, uint32_t count,
                                   const vk::WriteDescriptorSet* writes) {
	const auto infos = DescriptorInfoCount(count, writes);
	auto       w     = Open(Op::PushDescriptorSet,
	                        Sz<PushDescriptorSetPacket>() + Arr<DescriptorWriteRecord>(count) +
	                            Arr<DescriptorInfo>(infos),
	                        false);
	w.Put(PushDescriptorSetPacket {layout, point, set, count, static_cast<uint32_t>(infos)});
	PutDescriptorWrites(w, count, writes, infos);
	Close(w,
	      m_options.verify ? VerifyHash::PushDescriptorSet(point, layout, set, count, writes) : 0);
}

void Encoder::updateDescriptorSets(vk::DescriptorSet set, uint32_t count,
                                   const vk::WriteDescriptorSet* writes) {
	EXIT_IF(set == nullptr);
	for (uint32_t i = 0; i < count; i++) {
		EXIT_IF(writes[i].dstSet != set);
	}
	const auto infos = DescriptorInfoCount(count, writes);
	auto       w     = Open(Op::UpdateDescriptorSets,
	                        Sz<UpdateDescriptorSetsPacket>() + Arr<DescriptorWriteRecord>(count) +
	                            Arr<DescriptorInfo>(infos),
	                        false);
	w.Put(UpdateDescriptorSetsPacket {set, count, static_cast<uint32_t>(infos)});
	PutDescriptorWrites(w, count, writes, infos);
	Close(w, m_options.verify ? VerifyHash::UpdateDescriptorSets(set, count, writes) : 0);
}

void Encoder::pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
                            uint32_t size, const void* data) {
	auto w = Open(Op::PushConstants, Sz<PushConstantsPacket>() + Arr<uint8_t>(size), false);
	w.Put(PushConstantsPacket {layout, stages, offset, size});
	w.PutArray(static_cast<const uint8_t*>(data), size);
	Close(w, m_options.verify ? VerifyHash::PushConstants(layout, stages, offset, size, data) : 0);
}

void Encoder::bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
                                 const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
                                 const vk::DeviceSize* strides) {
	const uint64_t arrays = 2u + (sizes != nullptr ? 1u : 0u) + (strides != nullptr ? 1u : 0u);
	auto           w = Open(Op::BindVertexBuffers2,
	                        Sz<BindVertexBuffers2Packet>() + arrays * Arr<uint64_t>(count), false);
	w.Put(BindVertexBuffers2Packet {first, count, sizes != nullptr ? 1u : 0u,
	                                strides != nullptr ? 1u : 0u});
	w.PutArray(buffers, count);
	w.PutArray(offsets, count);
	if (sizes != nullptr) {
		w.PutArray(sizes, count);
	}
	if (strides != nullptr) {
		w.PutArray(strides, count);
	}
	Close(w, m_options.verify
	             ? VerifyHash::BindVertexBuffers2(first, count, buffers, offsets, sizes, strides)
	             : 0);
}

void Encoder::bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) {
	auto w = Open(Op::BindIndexBuffer, Sz<BindIndexBufferPacket>(), false);
	w.Put(BindIndexBufferPacket {buffer, offset, type});
	Close(w, m_options.verify ? VerifyHash::BindIndexBuffer(buffer, offset, type) : 0);
}

void Encoder::setViewportWithCount(uint32_t count, const vk::Viewport* viewports) {
	auto w = Open(Op::SetViewportWithCount, Sz<CountPacket>() + Arr<vk::Viewport>(count), false);
	w.Put(CountPacket {count});
	w.PutArray(viewports, count);
	Close(w, m_options.verify ? VerifyHash::Viewports(count, viewports) : 0);
}

void Encoder::setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) {
	auto w = Open(Op::SetScissorWithCount, Sz<CountPacket>() + Arr<vk::Rect2D>(count), false);
	w.Put(CountPacket {count});
	w.PutArray(scissors, count);
	Close(w, m_options.verify ? VerifyHash::Scissors(count, scissors) : 0);
}

void Encoder::setLineWidth(float width) {
	auto w = Open(Op::SetLineWidth, Sz<FloatPacket>(), false);
	w.Put(FloatPacket {width});
	Close(w, m_options.verify ? VerifyHash::Floats(Op::SetLineWidth, &width, 1) : 0);
}

void Encoder::setBlendConstants(const float constants[4]) {
	auto         w = Open(Op::SetBlendConstants, Sz<Float4Packet>(), false);
	Float4Packet p;
	std::memcpy(p.values, constants, sizeof(p.values));
	w.Put(p);
	Close(w, m_options.verify ? VerifyHash::Floats(Op::SetBlendConstants, constants, 4) : 0);
}

namespace {
template <typename T>
uint32_t U32(T value) {
	return static_cast<uint32_t>(value);
}
} // namespace

#define KYTY_STREAM_UINT_STATE(method, op, type, raw)                                              \
	void Encoder::method(type value) {                                                             \
		const uint32_t v = raw;                                                                    \
		auto           w = Open(Op::op, Sz<UintPacket>(), false);                                  \
		w.Put(UintPacket {v});                                                                     \
		Close(w, m_options.verify ? VerifyHash::Value(Op::op, v) : 0);                             \
	}
KYTY_STREAM_UINT_STATE(setDepthTestEnable, SetDepthTestEnable, vk::Bool32, value)
KYTY_STREAM_UINT_STATE(setDepthWriteEnable, SetDepthWriteEnable, vk::Bool32, value)
KYTY_STREAM_UINT_STATE(setDepthCompareOp, SetDepthCompareOp, vk::CompareOp, U32(value))
KYTY_STREAM_UINT_STATE(setDepthBiasEnable, SetDepthBiasEnable, vk::Bool32, value)
KYTY_STREAM_UINT_STATE(setStencilTestEnable, SetStencilTestEnable, vk::Bool32, value)
KYTY_STREAM_UINT_STATE(setCullMode, SetCullMode, vk::CullModeFlags,
                       static_cast<VkCullModeFlags>(value))
KYTY_STREAM_UINT_STATE(setFrontFace, SetFrontFace, vk::FrontFace, U32(value))
KYTY_STREAM_UINT_STATE(setDepthBoundsTestEnable, SetDepthBoundsTestEnable, vk::Bool32, value)
KYTY_STREAM_UINT_STATE(setAttachmentFeedbackLoopEnableEXT, SetAttachmentFeedbackLoopEnable,
                       vk::ImageAspectFlags, static_cast<VkImageAspectFlags>(value))
#undef KYTY_STREAM_UINT_STATE

void Encoder::setDepthBias(float constant, float clamp, float slope) {
	auto               w = Open(Op::SetDepthBias, Sz<Float3Packet>(), false);
	const Float3Packet p {{constant, clamp, slope}};
	w.Put(p);
	Close(w, m_options.verify ? VerifyHash::Floats(Op::SetDepthBias, p.values, 3) : 0);
}

void Encoder::setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
                           vk::StencilOp depth_fail, vk::CompareOp compare) {
	auto w = Open(Op::SetStencilOp, Sz<StencilOpPacket>(), false);
	w.Put(StencilOpPacket {faces, fail, pass, depth_fail, compare});
	Close(w,
	      m_options.verify
	          ? VerifyHash::Value(Op::SetStencilOp, static_cast<VkStencilFaceFlags>(faces),
	                              static_cast<uint64_t>(fail), static_cast<uint64_t>(pass),
	                              static_cast<uint64_t>(depth_fail), static_cast<uint64_t>(compare))
	          : 0);
}

#define KYTY_STREAM_STENCIL_VALUE(method, op)                                                      \
	void Encoder::method(vk::StencilFaceFlags faces, uint32_t value) {                             \
		auto w = Open(Op::op, Sz<StencilValuePacket>(), false);                                    \
		w.Put(StencilValuePacket {faces, value});                                                  \
		Close(w, m_options.verify                                                                  \
		             ? VerifyHash::Value(Op::op, static_cast<VkStencilFaceFlags>(faces), value)    \
		             : 0);                                                                         \
	}
KYTY_STREAM_STENCIL_VALUE(setStencilCompareMask, SetStencilCompareMask)
KYTY_STREAM_STENCIL_VALUE(setStencilWriteMask, SetStencilWriteMask)
KYTY_STREAM_STENCIL_VALUE(setStencilReference, SetStencilReference)
#undef KYTY_STREAM_STENCIL_VALUE

void Encoder::setDepthBounds(float min, float max) {
	auto               w = Open(Op::SetDepthBounds, Sz<Float2Packet>(), false);
	const Float2Packet p {{min, max}};
	w.Put(p);
	Close(w, m_options.verify ? VerifyHash::Floats(Op::SetDepthBounds, p.values, 2) : 0);
}

void Encoder::setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) {
	auto w = Open(Op::SetColorWriteEnable, Sz<CountPacket>() + Arr<vk::Bool32>(count), false);
	w.Put(CountPacket {count});
	w.PutArray(enables, count);
	Close(w, m_options.verify ? VerifyHash::ColorWriteEnable(count, enables) : 0);
}

void Encoder::draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                   uint32_t first_instance) {
	auto w = Open(Op::Draw, Sz<DrawPacket>(), false);
	w.Put(DrawPacket {vertex_count, instance_count, first_vertex, first_instance});
	Close(w, m_options.verify ? VerifyHash::Value(Op::Draw, vertex_count, instance_count,
	                                              first_vertex, first_instance)
	                          : 0);
}

void Encoder::drawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                          int32_t vertex_offset, uint32_t first_instance) {
	auto w = Open(Op::DrawIndexed, Sz<DrawIndexedPacket>(), false);
	w.Put(DrawIndexedPacket {index_count, instance_count, first_index, vertex_offset,
	                         first_instance});
	Close(w, m_options.verify
	             ? VerifyHash::Value(Op::DrawIndexed, index_count, instance_count, first_index,
	                                 static_cast<uint32_t>(vertex_offset), first_instance)
	             : 0);
}

void Encoder::drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) {
	auto w = Open(Op::DrawMeshTasks, Sz<Groups3Packet>(), false);
	w.Put(Groups3Packet {x, y, z});
	Close(w, m_options.verify ? VerifyHash::Value(Op::DrawMeshTasks, x, y, z) : 0);
}

namespace {
uint64_t HandleBits(vk::Buffer buffer) {
	uint64_t   bits = 0;
	const auto raw  = static_cast<VkBuffer>(buffer);
	std::memcpy(&bits, &raw, sizeof(bits));
	return bits;
}
uint64_t HandleBits(vk::QueryPool pool) {
	uint64_t   bits = 0;
	const auto raw  = static_cast<VkQueryPool>(pool);
	std::memcpy(&bits, &raw, sizeof(bits));
	return bits;
}
} // namespace

#define KYTY_STREAM_INDIRECT(method, op)                                                           \
	void Encoder::method(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,            \
	                     uint32_t stride) {                                                        \
		auto w = Open(Op::op, Sz<IndirectPacket>(), false);                                        \
		w.Put(IndirectPacket {buffer, offset, draw_count, stride});                                \
		Close(w, m_options.verify                                                                  \
		             ? VerifyHash::Value(Op::op, HandleBits(buffer), offset, draw_count, stride)   \
		             : 0);                                                                         \
	}
KYTY_STREAM_INDIRECT(drawMeshTasksIndirectEXT, DrawMeshTasksIndirect)
KYTY_STREAM_INDIRECT(drawIndirect, DrawIndirect)
KYTY_STREAM_INDIRECT(drawIndexedIndirect, DrawIndexedIndirect)
#undef KYTY_STREAM_INDIRECT

#define KYTY_STREAM_INDIRECT_COUNT(method, op)                                                     \
	void Encoder::method(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,        \
	                     vk::DeviceSize count_offset, uint32_t max_count, uint32_t stride) {       \
		auto w = Open(Op::op, Sz<IndirectCountPacket>(), false);                                   \
		w.Put(                                                                                     \
		    IndirectCountPacket {buffer, offset, count_buffer, count_offset, max_count, stride});  \
		Close(w, m_options.verify ? VerifyHash::Value(Op::op, HandleBits(buffer), offset,          \
		                                              HandleBits(count_buffer), count_offset,      \
		                                              max_count, stride)                           \
		                          : 0);                                                            \
	}
KYTY_STREAM_INDIRECT_COUNT(drawMeshTasksIndirectCountEXT, DrawMeshTasksIndirectCount)
KYTY_STREAM_INDIRECT_COUNT(drawIndirectCount, DrawIndirectCount)
KYTY_STREAM_INDIRECT_COUNT(drawIndexedIndirectCount, DrawIndexedIndirectCount)
#undef KYTY_STREAM_INDIRECT_COUNT

void Encoder::dispatch(uint32_t x, uint32_t y, uint32_t z) {
	auto w = Open(Op::Dispatch, Sz<Groups3Packet>(), false);
	w.Put(Groups3Packet {x, y, z});
	Close(w, m_options.verify ? VerifyHash::Value(Op::Dispatch, x, y, z) : 0);
}

void Encoder::dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) {
	auto w = Open(Op::DispatchIndirect, Sz<DispatchIndirectPacket>(), false);
	w.Put(DispatchIndirectPacket {buffer, offset});
	Close(w, m_options.verify ? VerifyHash::Value(Op::DispatchIndirect, HandleBits(buffer), offset)
	                          : 0);
}

void Encoder::resetQueryPool(vk::QueryPool pool, uint32_t first, uint32_t count) {
	auto w = Open(Op::ResetQueryPool, Sz<QueryRangePacket>(), false);
	w.Put(QueryRangePacket {pool, first, count});
	Close(w, m_options.verify
	             ? VerifyHash::Value(Op::ResetQueryPool, HandleBits(pool), first, count)
	             : 0);
}

void Encoder::beginQuery(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) {
	auto w = Open(Op::BeginQuery, Sz<BeginQueryPacket>(), false);
	w.Put(BeginQueryPacket {pool, query, flags});
	Close(w, m_options.verify ? VerifyHash::Value(Op::BeginQuery, HandleBits(pool), query,
	                                              static_cast<VkQueryControlFlags>(flags))
	                          : 0);
}

void Encoder::endQuery(vk::QueryPool pool, uint32_t query) {
	auto w = Open(Op::EndQuery, Sz<QueryRangePacket>(), false);
	w.Put(QueryRangePacket {pool, query, 0});
	Close(w, m_options.verify ? VerifyHash::Value(Op::EndQuery, HandleBits(pool), query, 0) : 0);
}

void Encoder::copyQueryPoolResults(vk::QueryPool pool, uint32_t first, uint32_t count,
                                   vk::Buffer destination, vk::DeviceSize offset,
                                   vk::DeviceSize stride, vk::QueryResultFlags flags) {
	auto w = Open(Op::CopyQueryPoolResults, Sz<CopyQueryPoolResultsPacket>(), false);
	w.Put(CopyQueryPoolResultsPacket {pool, destination, offset, stride, first, count, flags});
	Close(w,
	      m_options.verify
	          ? VerifyHash::Value(
	                Op::CopyQueryPoolResults, HandleBits(pool), first, count,
	                HandleBits(destination), offset,
	                stride ^ (static_cast<uint64_t>(static_cast<VkQueryResultFlags>(flags)) << 40u))
	          : 0);
}

void Encoder::writeTimestamp2(vk::PipelineStageFlags2 stage, vk::QueryPool pool, uint32_t query) {
	const auto stage_bits = static_cast<uint64_t>(static_cast<VkPipelineStageFlags2>(stage));
	auto       w          = Open(Op::WriteTimestamp2, Sz<WriteTimestampPacket>(), false);
	w.Put(WriteTimestampPacket {pool, stage_bits, query});
	Close(w, m_options.verify
	             ? VerifyHash::Value(Op::WriteTimestamp2, HandleBits(pool), query, stage_bits)
	             : 0);
}

void Encoder::writeBufferMarkerAMD(vk::PipelineStageFlagBits stage, vk::Buffer buffer,
                                   vk::DeviceSize offset, uint32_t marker) {
	const auto stage_bits = static_cast<uint32_t>(static_cast<VkPipelineStageFlagBits>(stage));
	auto       w          = Open(Op::WriteBufferMarker, Sz<WriteBufferMarkerPacket>(), false);
	w.Put(WriteBufferMarkerPacket {buffer, offset, stage_bits, marker});
	Close(w, m_options.verify
	             ? VerifyHash::Value(Op::WriteBufferMarker, HandleBits(buffer), offset, stage_bits, marker)
	             : 0);
}

} // namespace Libs::Graphics::CommandStream
