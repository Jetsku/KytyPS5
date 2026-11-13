// Unit tests for the draw encode (graphics/host_gpu/renderer/drawEncode.h) without a Vulkan device:
//  - Shadow: the comparisons CommandBuffer made (pipeline binds, push constants, push descriptors,
//    descriptor-set binds, epochs) with a command-stream Encoder as the sink, checked through the
//    replayed (op, argument hash) sequence (commandStreamLog.h).

#include "graphics/host_gpu/renderer/commandStreamLog.h"
#include "graphics/host_gpu/renderer/drawEncode.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::CommandStream;
using DrawEncode::Shadow;
using DrawEncode::ShadowFlags;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::printf("FAILED: %s\n", what);
		++g_failures;
	}
}

template <typename Handle>
Handle H(uint64_t value) {
	using C = typename Handle::CType;
	C raw {};
	std::memcpy(&raw, &value, sizeof(raw));
	return Handle(raw);
}

// An encoder on its own ring; Take() returns the calls encoded since the last Take().
struct Capture {
	Ring     ring {1u << 16u};
	Encoder  encoder {ring, Encoder::Options {}};
	uint64_t taken = 0;

	std::vector<Call> Take() {
		CallLog log;
		LogPackets(ring, taken, ring.WritePosition(), log);
		taken = ring.WritePosition();
		ring.Publish();
		while (const auto* header = ring.Peek()) {
			ring.Advance(header->size);
		}
		WaitStats stats;
		ring.Release(stats);
		return log.calls;
	}
};

void TestBindPipeline() {
	Capture     c;
	Shadow      shadow;
	ShadowFlags flags;
	const auto  p1 = H<vk::Pipeline>(0x100);
	const auto  p2 = H<vk::Pipeline>(0x200);
	shadow.BindPipeline(c.encoder, flags, vk::PipelineBindPoint::eGraphics, p1);
	shadow.BindPipeline(c.encoder, flags, vk::PipelineBindPoint::eGraphics, p1);
	shadow.BindPipeline(c.encoder, flags, vk::PipelineBindPoint::eCompute, p1);
	shadow.BindPipeline(c.encoder, flags, vk::PipelineBindPoint::eGraphics, p2);
	const auto calls = c.Take();
	Check(calls.size() == 3, "bind pipeline: the repeat is avoided per bind point");
	Check(shadow.BoundPipeline(vk::PipelineBindPoint::eGraphics) == p2 &&
	          shadow.BoundPipeline(vk::PipelineBindPoint::eCompute) == p1,
	      "bind pipeline: bound per bind point");
	flags.renderer_batch = false;
	shadow.BindPipeline(c.encoder, flags, vk::PipelineBindPoint::eGraphics, p2);
	Check(c.Take().size() == 1, "bind pipeline: no dedup with the renderer batch off");
	shadow.Begin();
	Check(shadow.BoundPipeline(vk::PipelineBindPoint::eGraphics) == nullptr, "bind pipeline: Begin unbinds");
}

void TestPushConstants() {
	Capture     c;
	Shadow      shadow;
	ShadowFlags flags;
	const auto  layout = H<vk::PipelineLayout>(0x300);
	uint32_t    data[8] {1, 2, 3, 4, 5, 6, 7, 8};
	const auto  stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
	shadow.PushConstants(c.encoder, flags, layout, stages, sizeof(data), data);
	shadow.PushConstants(c.encoder, flags, layout, stages, sizeof(data), data);
	Check(c.Take().size() == 1, "push constants: an identical update is avoided");
	data[7] = 9;
	shadow.PushConstants(c.encoder, flags, layout, stages, sizeof(data), data);
	shadow.PushConstants(c.encoder, flags, layout, vk::ShaderStageFlagBits::eVertex, sizeof(data), data);
	shadow.PushConstants(c.encoder, flags, layout, vk::ShaderStageFlagBits::eVertex, 16, data);
	Check(c.Take().size() == 3, "push constants: data, stages and size differences are recorded");
	shadow.InvalidatePushConstants();
	shadow.PushConstants(c.encoder, flags, layout, vk::ShaderStageFlagBits::eVertex, 16, data);
	Check(c.Take().size() == 1, "push constants: recorded again after an invalidation");
	shadow.Begin();
	shadow.PushConstants(c.encoder, flags, layout, vk::ShaderStageFlagBits::eVertex, 16, data);
	Check(c.Take().size() == 1, "push constants: recorded again after Begin");
	flags.push_constant_shadow = false;
	shadow.PushConstants(c.encoder, flags, layout, vk::ShaderStageFlagBits::eVertex, 16, data);
	shadow.PushConstants(c.encoder, flags, layout, vk::ShaderStageFlagBits::eVertex, 16, data);
	Check(c.Take().size() == 2 && !shadow.push_constants.valid, "push constants: no shadow when off");
	// The recorded argument hash is the one of the call itself.
	flags.push_constant_shadow = true;
	shadow.PushConstants(c.encoder, flags, layout, vk::ShaderStageFlagBits::eVertex, 16, data);
	const auto calls = c.Take();
	Check(calls.size() == 1 && calls[0].op == Op::PushConstants &&
	          calls[0].hash == VerifyHash::PushConstants(layout, vk::ShaderStageFlagBits::eVertex, 0, 16, data),
	      "push constants: offset 0 and the caller's bytes");
}

struct Writes {
	std::vector<vk::DescriptorBufferInfo> buffers;
	std::vector<vk::DescriptorImageInfo>  images;
	std::vector<vk::WriteDescriptorSet>   writes;

	Writes(uint64_t buffer, uint64_t offset, uint64_t view) {
		buffers = {{H<vk::Buffer>(buffer), offset, 256}, {H<vk::Buffer>(buffer + 1), 0, 64}};
		images  = {{H<vk::Sampler>(0x77), H<vk::ImageView>(view), vk::ImageLayout::eShaderReadOnlyOptimal}};
		vk::WriteDescriptorSet b {};
		b.dstBinding      = 1;
		b.descriptorCount = 2;
		b.descriptorType  = vk::DescriptorType::eStorageBuffer;
		b.pBufferInfo     = buffers.data();
		vk::WriteDescriptorSet i {};
		i.dstBinding      = 5;
		i.descriptorCount = 1;
		i.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
		i.pImageInfo      = images.data();
		writes            = {b, i};
	}
};

void TestPushDescriptors() {
	Capture     c;
	Shadow      shadow;
	ShadowFlags flags;
	const auto  point  = vk::PipelineBindPoint::eGraphics;
	const auto  layout = H<vk::PipelineLayout>(0x400);
	Writes      a(0x1000, 0, 0x50);
	const auto  e0 = shadow.DescriptorEpoch(point);
	auto        r  = shadow.PushDescriptors(c.encoder, flags, point, layout, 0, 2, a.writes.data(), false);
	Check(r == Shadow::PushMissState && shadow.DescriptorEpoch(point) == e0 + 1, "push: first is a state miss");
	Writes same(0x1000, 0, 0x50);
	r = shadow.PushDescriptors(c.encoder, flags, point, layout, 0, 2, same.writes.data(), false);
	Check(r == Shadow::PushAvoided && shadow.DescriptorEpoch(point) == e0 + 1, "push: identical is avoided");
	Check(c.Take().size() == 1, "push: one packet for the two calls");
	Writes moved(0x1000, 64, 0x50);
	r = shadow.PushDescriptors(c.encoder, flags, point, layout, 0, 2, moved.writes.data(), false);
	Check(r == 0 && shadow.DescriptorEpoch(point) == e0 + 2, "push: a buffer offset change is write 0");
	Writes view(0x1000, 64, 0x51);
	r = shadow.PushDescriptors(c.encoder, flags, point, layout, 0, 2, view.writes.data(), false);
	Check(r == 1, "push: an image view change is write 1");
	r = shadow.PushDescriptors(c.encoder, flags, point, layout, 0, 1, view.writes.data(), false);
	Check(r == Shadow::PushMissShape, "push: another write count is a shape miss");
	r = shadow.PushDescriptors(c.encoder, flags, point, H<vk::PipelineLayout>(0x401), 0, 1, view.writes.data(), false);
	Check(r == Shadow::PushMissState, "push: another layout is a state miss");
	const auto e1 = shadow.DescriptorEpoch(point);
	r = shadow.PushDescriptors(c.encoder, flags, point, layout, 0, 2, view.writes.data(), true);
	Check(r == Shadow::PushMissFresh && shadow.DescriptorEpoch(point) == e1 &&
	          shadow.descriptor_states[0].layout == nullptr,
	      "push: a known miss records without a copy and without an epoch step");
	r = shadow.PushDescriptors(c.encoder, flags, point, layout, 0, 2, view.writes.data(), false);
	Check(r == Shadow::PushMissState, "push: after a known miss nothing is comparable");
	const auto calls = c.Take();
	Check(calls.size() == 6, "push: every miss recorded");
	Check(calls.back().op == Op::PushDescriptorSet &&
	          calls.back().hash == VerifyHash::PushDescriptorSet(point, layout, 0, 2, view.writes.data()),
	      "push: the recorded writes are the caller's");
	// Compute is separate.
	const auto ec = shadow.DescriptorEpoch(vk::PipelineBindPoint::eCompute);
	r = shadow.PushDescriptors(c.encoder, flags, vk::PipelineBindPoint::eCompute, layout, 0, 2, view.writes.data(), false);
	Check(r == Shadow::PushMissState && shadow.DescriptorEpoch(vk::PipelineBindPoint::eCompute) == ec + 1,
	      "push: compute has its own shadow");
	// The renderer batch off: never compared.
	flags.renderer_batch = false;
	r = shadow.PushDescriptors(c.encoder, flags, point, layout, 0, 2, view.writes.data(), false);
	r = shadow.PushDescriptors(c.encoder, flags, point, layout, 0, 2, view.writes.data(), false);
	Check(r == Shadow::PushMissState, "push: no comparison with the renderer batch off");
	shadow.InvalidateDescriptors(point);
	Check(shadow.descriptor_states[0].layout == nullptr, "push: invalidated");
}

void TestBindDescriptorSet() {
	Capture     c;
	Shadow      shadow;
	ShadowFlags flags;
	const auto  point  = vk::PipelineBindPoint::eGraphics;
	const auto  layout = H<vk::PipelineLayout>(0x500);
	const auto  set    = H<vk::DescriptorSet>(0x600);
	const auto  e0     = shadow.DescriptorEpoch(point);
	shadow.BindDescriptorSet(c.encoder, flags, point, layout, set);
	shadow.BindDescriptorSet(c.encoder, flags, point, layout, set);
	Check(c.Take().size() == 1 && shadow.DescriptorEpoch(point) == e0 + 1, "set: a rebind is avoided");
	Writes a(0x2000, 0, 0x60);
	auto   r = shadow.PushDescriptors(c.encoder, flags, point, layout, 0, 2, a.writes.data(), false);
	Check(r == Shadow::PushMissState, "set: a push after a bound set is not comparable");
	shadow.BindDescriptorSet(c.encoder, flags, point, layout, set);
	Check(c.Take().size() == 2, "set: bound again after the push");
	flags.set_reuse = false;
	shadow.BindDescriptorSet(c.encoder, flags, point, layout, set);
	shadow.BindDescriptorSet(c.encoder, flags, point, layout, set);
	Check(c.Take().size() == 2, "set: no reuse when off");
	shadow.Begin();
	Check(shadow.descriptor_states[0].bound_set == nullptr, "set: Begin forgets the bound set");
}

void TestDrawCommands() {
	Capture              c;
	const vk::Buffer     buffers[2] {H<vk::Buffer>(0x10), H<vk::Buffer>(0x11)};
	const vk::DeviceSize offsets[2] {16, 32};
	const vk::DeviceSize sizes[2] {256, 512};
	DrawEncode::EmitVertexBuffers(c.encoder, 2, buffers, offsets, sizes);
	DrawEncode::EmitVertexBuffers(c.encoder, 0, buffers, offsets, sizes);
	DrawEncode::EmitIndexBuffer(c.encoder, H<vk::Buffer>(0x12), 64, vk::IndexType::eUint32);
	DrawEncode::EmitIndexBuffer(c.encoder, nullptr, 0, vk::IndexType::eUint16);
	auto calls = c.Take();
	Check(calls.size() == 2 && calls[0].op == Op::BindVertexBuffers2 &&
	          calls[0].hash == VerifyHash::BindVertexBuffers2(0, 2, buffers, offsets, sizes, nullptr) &&
	          calls[1].hash == VerifyHash::BindIndexBuffer(H<vk::Buffer>(0x12), 64, vk::IndexType::eUint32),
	      "binds: none for no buffers / no index buffer");
	DrawEncode::DirectDraw draw;
	draw.indexed        = true;
	draw.count          = 12;
	draw.instance_count = 2;
	draw.vertex_offset  = -5;
	draw.first_instance = 3;
	DrawEncode::EmitDirectDraw(c.encoder, draw);
	draw.quad_list = true;
	DrawEncode::EmitDirectDraw(c.encoder, draw);
	draw.indexed      = false;
	draw.first_vertex = 100;
	DrawEncode::EmitDirectDraw(c.encoder, draw);
	calls = c.Take();
	Check(calls.size() == 1 + 3 + 3, "draws: one, then one per quad");
	Check(calls[0].hash == VerifyHash::Value(Op::DrawIndexed, 12, 2, 0, static_cast<uint32_t>(-5), 3) &&
	          calls[2].hash == VerifyHash::Value(Op::DrawIndexed, 4, 2, 4, static_cast<uint32_t>(-5), 3) &&
	          calls[6].hash == VerifyHash::Value(Op::Draw, 4, 2, 108, 3),
	      "draws: arguments");
}

} // namespace

int main() {
	TestDrawCommands();
	TestBindPipeline();
	TestPushConstants();
	TestPushDescriptors();
	TestBindDescriptorSet();
	if (g_failures != 0) {
		std::printf("draw_encode_tests: %d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("draw_encode_tests: all passed\n");
	return 0;
}
