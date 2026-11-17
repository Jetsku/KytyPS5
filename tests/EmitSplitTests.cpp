// Unit tests for KYTY_EMIT_SPLIT's resolved draw (graphics/host_gpu/renderer/commandStream.h,
// ResolvedDrawPacket), without a Vulkan device:
//  - encode/expand round trip: a resolved draw replays into exactly the calls the inline encode makes
//    (checked through the shared VerifyHash functions), with and without the CP recorder's verify;
//  - the consumer's shadow: an identical push or push-constant block is not recorded again; every other
//    packet that changes that state, Begin, FlagShadowReset, FlagKnownMiss and the object id forget it;
//  - verify mode: a verify-only record placed after the inline packets of the same draw matches the
//    replayed state (0 mismatches); a changed field is counted under that field.

#include "graphics/host_gpu/renderer/commandStream.h"
#include "graphics/host_gpu/renderer/commandStreamLog.h"
#include "common/liveSwitch.h"

#include <cstdlib>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// commandStream.cpp includes vulkan.hpp's dispatcher declarations; the tests never call Vulkan.
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::CommandStream;
using P = ResolvedDrawPacket;

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

// One draw's inputs, as the commit thread would hold them.
struct Draw {
	vk::PipelineLayout                  layout   = H<vk::PipelineLayout>(0x1000);
	vk::Pipeline                        pipeline = H<vk::Pipeline>(0x2000);
	std::array<vk::Buffer, 2>           vb {H<vk::Buffer>(0x3000), H<vk::Buffer>(0x3100)};
	std::array<vk::DeviceSize, 2>       vo {16, 32};
	std::array<vk::DeviceSize, 2>       vs {4096, 8192};
	std::vector<DescriptorWriteRecord>  writes;
	std::vector<vk::DescriptorBufferInfo> infos;
	std::array<uint32_t, 32>            push {};
	std::array<uint32_t, 8>             clip {};
	uint32_t                            object_id = 7;
	uint32_t                            flags     = P::FlagVertex | P::FlagPushData | P::FlagPush | P::FlagIndex |
	                         P::FlagBind | P::FlagIndexed | P::FlagPushDedup | P::FlagConstDedup;
	bool clip_on = false;

	Draw() {
		DescriptorWriteRecord a;
		a.binding = 0;
		a.count   = 2;
		a.type    = vk::DescriptorType::eStorageBuffer;
		DescriptorWriteRecord b;
		b.binding = 5;
		b.count   = 1;
		b.type    = vk::DescriptorType::eUniformBuffer;
		writes    = {a, b};
		infos     = {{H<vk::Buffer>(0x4000), 0, 256}, {H<vk::Buffer>(0x4100), 64, 128}, {H<vk::Buffer>(0x4200), 0, 512}};
		for (uint32_t i = 0; i < push.size(); i++) {
			push[i] = 0x100 + i;
		}
		for (uint32_t i = 0; i < clip.size(); i++) {
			clip[i] = 0x900 + i;
		}
	}

	[[nodiscard]] P Packet() const {
		P p;
		p.layout         = layout;
		p.pipeline       = pipeline;
		p.index_buffer   = H<vk::Buffer>(0x5000);
		p.index_offset   = 48;
		p.index_type     = static_cast<uint32_t>(vk::IndexType::eUint16);
		p.flags          = flags | (clip_on ? 0u : 0u);
		p.vertex_count   = 2;
		p.write_count    = static_cast<uint32_t>(writes.size());
		p.info_count     = static_cast<uint32_t>(infos.size());
		p.push_stages    = static_cast<uint32_t>(VkShaderStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment));
		p.clip_offset    = 28;
		p.clip_bytes     = clip_on ? 32u : 0u;
		p.clip_stages    = p.push_stages;
		p.objprim_stages = p.push_stages;
		p.objprim_offset = 4 * 20;
		p.object_id      = object_id;
		p.count          = 300;
		p.instance_count = 1;
		p.vertex_offset  = -4;
		p.first_instance = 0;
		return p;
	}
	[[nodiscard]] ResolvedDrawArrays Arrays() const {
		ResolvedDrawArrays a;
		a.vertex_buffers = vb.data();
		a.vertex_offsets = vo.data();
		a.vertex_sizes   = vs.data();
		a.writes         = writes.data();
		a.infos          = reinterpret_cast<const DescriptorInfo*>(infos.data());
		a.push           = push.data();
		a.clip           = clip_on ? clip.data() : nullptr;
		return a;
	}
	[[nodiscard]] std::vector<vk::WriteDescriptorSet> VkWrites() const {
		std::vector<vk::WriteDescriptorSet> out;
		uint32_t                            next = 0;
		for (const auto& r: writes) {
			vk::WriteDescriptorSet w {};
			w.dstBinding      = r.binding;
			w.dstArrayElement = r.element;
			w.descriptorCount = r.count;
			w.descriptorType  = r.type;
			w.pBufferInfo     = infos.data() + next;
			next += r.count;
			out.push_back(w);
		}
		return out;
	}
	// The inline encode of this draw (what ExecutePreparedDraw records without the switch).
	template <typename Sink>
	void Inline(Sink& e, bool push_constants = true, bool push_descriptors = true) const {
		const auto p = Packet();
		e.bindVertexBuffers2(0, 2, vb.data(), vo.data(), vs.data(), nullptr);
		if (push_constants) {
			e.pushConstants(layout, vk::ShaderStageFlags(p.push_stages), 0, 128, push.data());
		}
		if (push_descriptors) {
			const auto w = VkWrites();
			e.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, layout, 0, static_cast<uint32_t>(w.size()), w.data());
		}
		e.bindIndexBuffer(p.index_buffer, p.index_offset, vk::IndexType::eUint16);
		if ((p.flags & P::FlagBind) != 0) {
			e.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		}
		if (clip_on) {
			e.pushConstants(layout, vk::ShaderStageFlags(p.clip_stages), 28, 32, clip.data());
		}
		if ((p.flags & P::FlagObjprim) != 0) {
			e.pushConstants(layout, vk::ShaderStageFlags(p.objprim_stages), p.objprim_offset, 4, &object_id);
		}
		e.drawIndexed(300, 1, 0, -4, 0);
	}
};

// Replays everything in the ring.
void Consume(Ring& ring, CallLog& log, ReplayState& state) {
	WaitStats stats;
	while (const auto* header = ring.Peek()) {
		if (header->op != Op::Wrap) {
			Replay(*header, log, state, nullptr, nullptr);
		}
		ring.Advance(header->size);
	}
	ring.Release(stats);
}

void TestRoundTrip(bool verify) {
	Ring             ring(1u << 20u);
	Encoder::Options options;
	options.verify = verify;
	Encoder     encoder(ring, options);
	ReplayState state;
	state.verify = verify;
	CallLog log;
	CallLog expected;
	for (uint32_t variant = 0; variant < 4; variant++) {
		Draw d;
		d.clip_on = (variant & 1u) != 0;
		if ((variant & 2u) != 0) {
			d.flags |= P::FlagObjprim;
			d.flags &= ~P::FlagIndexed;
		}
		encoder.Begin(H<vk::CommandBuffer>(0x77), variant + 1, 0);
		expected.Begin(BeginPacket {H<vk::CommandBuffer>(0x77), variant + 1, 0});
		encoder.resolvedDraw(d.Packet(), d.Arrays());
		if ((d.flags & P::FlagIndexed) != 0) {
			d.Inline(expected);
		} else {
			// Non-indexed: the same commands with draw() at the end.
			auto p = d.Packet();
			expected.bindVertexBuffers2(0, 2, d.vb.data(), d.vo.data(), d.vs.data(), nullptr);
			expected.pushConstants(d.layout, vk::ShaderStageFlags(p.push_stages), 0, 128, d.push.data());
			const auto w = d.VkWrites();
			expected.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, d.layout, 0, static_cast<uint32_t>(w.size()), w.data());
			expected.bindIndexBuffer(p.index_buffer, p.index_offset, vk::IndexType::eUint16);
			expected.bindPipeline(vk::PipelineBindPoint::eGraphics, d.pipeline);
			if (d.clip_on) {
				expected.pushConstants(d.layout, vk::ShaderStageFlags(p.clip_stages), 28, 32, d.clip.data());
			}
			expected.pushConstants(d.layout, vk::ShaderStageFlags(p.objprim_stages), p.objprim_offset, 4, &d.object_id);
			expected.draw(300, 1, 0, 0);
		}
		Consume(ring, log, state);
	}
	Check(log.calls == expected.calls, verify ? "round trip (recorder verify): expansion equals the inline calls"
	                                           : "round trip: expansion equals the inline calls");
	Check(state.mismatches == 0, "round trip: no recorder verify mismatch");
	if (verify) {
		Check(state.checks == 8, "round trip: every packet hash-checked");
	}
	Check(state.record_stats.records == 4, "round trip: four records expanded");
}

uint64_t Count(const CallLog& log, Op op) {
	uint64_t n = 0;
	for (const auto& c: log.calls) {
		n += c.op == op ? 1u : 0u;
	}
	return n;
}

void TestShadow() {
	Ring             ring(1u << 20u);
	Encoder::Options options;
	Encoder          encoder(ring, options);
	ReplayState      state;
	CallLog          log;
	Draw             d;
	const auto       run = [&](auto&& body) {
        log.calls.clear();
        body();
        Consume(ring, log, state);
	};
	encoder.Begin(H<vk::CommandBuffer>(1), 1, 0);
	run([&] {
		encoder.resolvedDraw(d.Packet(), d.Arrays());
		encoder.resolvedDraw(d.Packet(), d.Arrays());
	});
	Check(Count(log, Op::PushDescriptorSet) == 1, "shadow: identical second push not recorded");
	Check(Count(log, Op::PushConstants) == 1, "shadow: identical second constant block not recorded");
	Check(Count(log, Op::DrawIndexed) == 2, "shadow: both draws recorded");

	// A different buffer info: recorded.
	run([&] {
		Draw e;
		e.infos[1].offset = 128;
		encoder.resolvedDraw(e.Packet(), e.Arrays());
	});
	Check(Count(log, Op::PushDescriptorSet) == 1, "shadow: changed info pushed");
	Check(Count(log, Op::PushConstants) == 0, "shadow: unchanged constants still avoided");

	// A foreign push descriptor update and push-constant update between two identical records.
	run([&] {
		encoder.resolvedDraw(d.Packet(), d.Arrays());
		const auto w = d.VkWrites();
		encoder.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, d.layout, 0, 1, w.data());
		encoder.pushConstants(d.layout, vk::ShaderStageFlagBits::eFragment, 0, 4, d.push.data());
		encoder.resolvedDraw(d.Packet(), d.Arrays());
	});
	Check(Count(log, Op::PushDescriptorSet) == 3, "shadow: a foreign push forgets the descriptor shadow");
	Check(Count(log, Op::PushConstants) == 2, "shadow: a foreign constant update forgets the constant shadow");

	// A compute push does not disturb the graphics descriptor shadow; a set-1 bind neither.
	run([&] {
		encoder.resolvedDraw(d.Packet(), d.Arrays());
		const auto w = d.VkWrites();
		encoder.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, d.layout, 0, 1, w.data());
		const auto set = H<vk::DescriptorSet>(0x99);
		encoder.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, d.layout, 1, 1, &set, 0, nullptr);
		encoder.resolvedDraw(d.Packet(), d.Arrays());
	});
	// Only the compute push itself: both resolved draws repeat the kept graphics push.
	Check(Count(log, Op::PushDescriptorSet) == 1, "shadow: compute push and set-1 bind keep the graphics shadow");
	// A set-0 bind forgets it.
	run([&] {
		const auto set = H<vk::DescriptorSet>(0x98);
		encoder.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, d.layout, 0, 1, &set, 0, nullptr);
		encoder.resolvedDraw(d.Packet(), d.Arrays());
	});
	Check(Count(log, Op::PushDescriptorSet) == 1, "shadow: a set-0 bind forgets the descriptor shadow");

	// Begin, FlagShadowReset, FlagKnownMiss, another layout, the object id.
	run([&] {
		encoder.Submit(SubmitPacket {});
		encoder.Begin(H<vk::CommandBuffer>(2), 2, 0);
		encoder.resolvedDraw(d.Packet(), d.Arrays());
	});
	Check(Count(log, Op::PushDescriptorSet) == 1 && Count(log, Op::PushConstants) == 1, "shadow: Begin forgets both");
	run([&] {
		auto p = d.Packet();
		p.flags |= P::FlagShadowReset;
		encoder.resolvedDraw(p, d.Arrays());
	});
	Check(Count(log, Op::PushDescriptorSet) == 1 && Count(log, Op::PushConstants) == 1,
	      "shadow: FlagShadowReset forgets both");
	run([&] {
		auto p = d.Packet();
		p.flags |= P::FlagKnownMiss;
		encoder.resolvedDraw(p, d.Arrays());
		encoder.resolvedDraw(d.Packet(), d.Arrays());
	});
	Check(Count(log, Op::PushDescriptorSet) == 2, "shadow: a known miss is pushed and not kept");
	run([&] {
		auto p   = d.Packet();
		p.layout = H<vk::PipelineLayout>(0x1001);
		encoder.resolvedDraw(p, d.Arrays());
	});
	Check(Count(log, Op::PushDescriptorSet) == 1 && Count(log, Op::PushConstants) == 1,
	      "shadow: another layout is pushed");
	run([&] {
		auto p = d.Packet();
		p.flags |= P::FlagObjprim;
		encoder.resolvedDraw(p, d.Arrays());
		encoder.resolvedDraw(d.Packet(), d.Arrays());
	});
	// Two blocks (the layout changed back, then the object id forgot the block) and one id.
	Check(Count(log, Op::PushConstants) == 3, "shadow: the object id forgets the constant block");
	// Dedup off.
	run([&] {
		auto p = d.Packet();
		p.flags &= ~(P::FlagPushDedup | P::FlagConstDedup);
		encoder.resolvedDraw(p, d.Arrays());
		encoder.resolvedDraw(p, d.Arrays());
	});
	Check(Count(log, Op::PushDescriptorSet) == 2 && Count(log, Op::PushConstants) == 2,
	      "shadow: without the dedup switches every push is recorded");
}

void TestVerifyModel() {
	Ring             ring(1u << 20u);
	Encoder::Options options;
	Encoder          encoder(ring, options);
	ReplayState      state;
	CallLog          log;
	Draw             d;
	d.clip_on = true;
	d.flags |= P::FlagObjprim;
	const auto verify_packet = [&](P p) {
		p.flags |= P::FlagVerifyOnly;
		return p;
	};
	// The first verify-only record turns the model on; it counts from the next Begin.
	encoder.Begin(H<vk::CommandBuffer>(1), 1, 0);
	d.Inline(encoder);
	encoder.resolvedDraw(verify_packet(d.Packet()), d.Arrays());
	Consume(ring, log, state);
	Check(state.record_stats.verify_unmodeled == 1, "verify: the first record is unmodeled");
	Check(Count(log, Op::DrawIndexed) == 1, "verify: a verify-only record records nothing");

	encoder.Submit(SubmitPacket {});
	encoder.Begin(H<vk::CommandBuffer>(2), 2, 0);
	// A full push with more bindings first (a run's first draw), then the partial draw.
	{
		const auto w = d.VkWrites();
		std::vector<vk::WriteDescriptorSet> full(w.begin(), w.end());
		vk::DescriptorImageInfo             image {H<vk::Sampler>(0x10), H<vk::ImageView>(0x11), vk::ImageLayout::eGeneral};
		vk::WriteDescriptorSet              extra {};
		extra.dstBinding      = 9;
		extra.descriptorCount = 1;
		extra.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
		extra.pImageInfo      = &image;
		full.push_back(extra);
		encoder.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, d.layout, 0, static_cast<uint32_t>(full.size()),
		                             full.data());
	}
	d.Inline(encoder);
	encoder.resolvedDraw(verify_packet(d.Packet()), d.Arrays());
	// The second draw of the run: the inline shadow avoided the block and the push (identical).
	d.Inline(encoder, false, false);
	encoder.resolvedDraw(verify_packet(d.Packet()), d.Arrays());
	Consume(ring, log, state);
	const auto& s       = state.record_stats;
	uint64_t    mismatch = 0;
	for (const auto m: s.mismatches) {
		mismatch += m;
	}
	Check(s.verify_checks == 2, "verify: two records checked");
	Check(mismatch == 0, "verify: matching records have no mismatch");

	// Each changed field is counted under its own name.
	const auto expect_field = [&](ResolvedDrawModel::Field field, auto&& change) {
		auto  before = state.record_stats.mismatches[field];
		Draw  e      = d;
		P     p      = e.Packet();
		change(e, p);
		d.Inline(encoder, false, false);
		encoder.resolvedDraw(verify_packet(p), e.Arrays());
		Consume(ring, log, state);
		Check(state.record_stats.mismatches[field] == before + 1, ResolvedDrawModel::FieldNames[field]);
	};
	expect_field(ResolvedDrawModel::Infos, [](Draw& e, P&) { e.infos[2].range = 8; });
	expect_field(ResolvedDrawModel::Writes, [](Draw& e, P&) { e.writes[1].binding = 6; });
	expect_field(ResolvedDrawModel::Constants, [](Draw& e, P&) { e.push[3] ^= 1u; });
	expect_field(ResolvedDrawModel::Constants, [](Draw&, P& p) { p.object_id = 8; });
	expect_field(ResolvedDrawModel::Constants, [](Draw& e, P&) { e.clip[0] ^= 1u; });
	expect_field(ResolvedDrawModel::Vertex, [](Draw& e, P&) { e.vo[1] = 40; });
	expect_field(ResolvedDrawModel::Index, [](Draw&, P& p) { p.index_offset = 0; });
	expect_field(ResolvedDrawModel::Pipeline, [](Draw&, P& p) { p.pipeline = H<vk::Pipeline>(0x2001); });
	expect_field(ResolvedDrawModel::Layout, [](Draw&, P& p) { p.layout = H<vk::PipelineLayout>(0x1002); });
	expect_field(ResolvedDrawModel::Draw, [](Draw&, P& p) { p.count = 301; });
	expect_field(ResolvedDrawModel::Draw, [](Draw&, P& p) { p.flags &= ~P::FlagIndexed; });
}

// KYTY_EMIT_SPLIT_COPIES: held copies travel in the next record (executed first) or are written before any
// other packet; the replayed order equals the unheld one.
void TestHeldCopies(bool hold) {
	Ring             ring(1u << 20u);
	Encoder::Options options;
	options.verify = true;
	Encoder     encoder(ring, options);
	ReplayState state;
	state.verify = true;
	CallLog log;
	Draw    d;
	uint8_t dst[64] {};
	encoder.Begin(H<vk::CommandBuffer>(1), 1, 0);
	encoder.hostCopy(dst, 0x1000, 16);
	encoder.hostCopy(dst + 16, 0x2000, 16);
	Check(encoder.HeldCopies() == (hold ? 2u : 0u), "copies: held only with the switch");
	encoder.resolvedDraw(d.Packet(), d.Arrays());
	Check(encoder.HeldCopies() == 0, "copies: the record takes the held copies");
	encoder.hostCopy(dst + 32, 0x3000, 8);
	encoder.endRendering(); // another packet writes the held copy first
	encoder.hostCopy(dst + 40, 0x4000, 8);
	encoder.FlushHostCopies(); // a drain
	ring.Publish();            // (non-kicking packets are published in batches)
	Consume(ring, log, state);
	std::vector<Op> ops;
	for (const auto& c: log.calls) {
		ops.push_back(c.op);
	}
	const std::vector<Op> expect = {Op::Begin,       Op::HostCopy,        Op::HostCopy, Op::BindVertexBuffers2,
	                                Op::PushConstants, Op::PushDescriptorSet, Op::BindIndexBuffer, Op::BindPipeline,
	                                Op::DrawIndexed, Op::HostCopy,        Op::EndRendering, Op::HostCopy};
	Check(ops == expect, hold ? "copies (held): order equals the unheld order" : "copies (unheld): order");
	if (ops != expect) {
		for (const auto op: ops) {
			std::printf("  replayed %s\n", OpName(op));
		}
	}
	Check(state.mismatches == 0, "copies: no recorder verify mismatch");
}

void TestPublished() {
	ResolvedDrawStats stats;
	ResolvedDrawStats published;
	const auto        before = ResolvedDrawPublished();
	stats.records            = 5;
	stats.mismatches[3]      = 2;
	PublishResolvedDraw(stats, published);
	stats.records = 9;
	PublishResolvedDraw(stats, published);
	const auto after = ResolvedDrawPublished();
	Check(after.records - before.records == 9 && after.mismatches[3] - before.mismatches[3] == 2,
	      "published totals add increments");
}

} // namespace

int main() {
	TestRoundTrip(false);
	TestRoundTrip(true);
	TestShadow();
	TestVerifyModel();
	TestPublished();
	TestHeldCopies(false);
	_putenv("KYTY_EMIT_SPLIT_COPIES=1");
	Check(Live::ReloadFromEnvironment("KYTY_EMIT_SPLIT_COPIES"), "copies: switch registered");
	TestHeldCopies(true);
	if (g_failures == 0) {
		std::printf("emit_split_tests: all passed\n");
	}
	return g_failures == 0 ? 0 : 1;
}
