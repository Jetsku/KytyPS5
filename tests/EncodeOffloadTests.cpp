// Unit tests for the encode offload hand-over (graphics/host_gpu/renderer/encodeOffload.h) and the
// command-stream splice (Encoder::Splice, Encoder::SetOpenHook), without a Vulkan device:
//  - jobs posted in order run on the encode thread and their packets reach the target stream in
//    posting order, equal (as replayed (op, argument hash) pairs) to a serial encode of the same
//    calls; with verify blocks the target's sequence numbers and command-buffer digest verify on
//    replay exactly as for a serial encode;
//  - the open hook settles pending jobs before the target's next packet;
//  - many jobs in flight, the job limit, wrap-around of both rings, and TakeOldest (verify mode).

#include "graphics/host_gpu/renderer/commandStreamLog.h"
#include "graphics/host_gpu/renderer/encodeOffload.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::CommandStream;
using EncodeOffload::Engine;

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

// A synthetic draw encode: what a continuation's job records (vertex buffers, push constants, push
// descriptors, index buffer, pipeline bind, draw), with arguments from a seed.
struct Job {
	uint32_t seed = 0;
	template <typename Sink>
	void Run(Sink& sink) const {
		std::mt19937 rng(seed);
		const auto   n = 1u + rng() % 4u;
		vk::Buffer     buffers[4];
		vk::DeviceSize offsets[4];
		vk::DeviceSize sizes[4];
		for (uint32_t i = 0; i < n; i++) {
			buffers[i] = H<vk::Buffer>(0x1000 + rng() % 64);
			offsets[i] = rng() % 4096;
			sizes[i]   = 64 + rng() % 4096;
		}
		sink.bindVertexBuffers2(0, n, buffers, offsets, sizes, nullptr);
		uint32_t push[32];
		for (auto& d: push) {
			d = rng();
		}
		const uint32_t push_size = 4u * (1u + rng() % 32u);
		sink.pushConstants(H<vk::PipelineLayout>(0x2000), vk::ShaderStageFlagBits::eVertex, 0, push_size, push);
		std::vector<vk::DescriptorBufferInfo> infos(1u + rng() % 24u);
		for (auto& info: infos) {
			info = {H<vk::Buffer>(0x3000 + rng() % 16), rng() % 65536, 256};
		}
		vk::WriteDescriptorSet write {};
		write.dstBinding      = rng() % 8;
		write.descriptorCount = static_cast<uint32_t>(infos.size());
		write.descriptorType  = vk::DescriptorType::eStorageBuffer;
		write.pBufferInfo     = infos.data();
		sink.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, H<vk::PipelineLayout>(0x2000), 0, 1, &write);
		sink.bindIndexBuffer(H<vk::Buffer>(0x4000), rng() % 1024, vk::IndexType::eUint16);
		if ((rng() & 3u) == 0) {
			sink.bindPipeline(vk::PipelineBindPoint::eGraphics, H<vk::Pipeline>(0x5000 + rng() % 4));
		}
		sink.drawIndexed(rng() % 3000, 1, 0, static_cast<int32_t>(rng() % 100000), 0);
	}
	static void RunOn(void* job, Encoder& out) { static_cast<const Job*>(job)->Run(out); }
};

// Replays the target ring with verification and logs the calls.
struct Replayed {
	CallLog     log;
	ReplayState state;
};

void ReplayAll(Ring& ring, Replayed& out) {
	ring.Publish();
	while (const auto* header = ring.Peek()) {
		if (header->op != Op::Wrap) {
			Replay(*header, out.log, out.state, nullptr, nullptr);
		}
		ring.Advance(header->size);
	}
	WaitStats stats;
	ring.Release(stats);
}

void TestSpliceMatchesSerial(bool verify, uint32_t jobs, uint32_t batch, uint64_t ring_bytes) {
	Encoder::Options options;
	options.verify = verify;
	Engine engine;
	engine.Configure(options);

	Ring    target_ring(ring_bytes);
	Encoder target(target_ring, options);
	Ring    serial_ring(ring_bytes);
	Encoder serial(serial_ring, options);

	std::vector<Job> work(jobs);
	for (uint32_t i = 0; i < jobs; i++) {
		work[i].seed = 1234u + i * 7u;
	}
	Replayed got;
	Replayed want;
	target.Begin(H<vk::CommandBuffer>(0x9000), 1, 0);
	serial.Begin(H<vk::CommandBuffer>(0x9000), 1, 0);
	for (uint32_t i = 0; i < jobs; i++) {
		// Every few jobs the commit thread records a packet of its own (the hook settles first).
		if (!engine.Post(&work[i], Job::RunOn)) {
			engine.Settle(target);
			Check(engine.Post(&work[i], Job::RunOn), "post after settle");
		}
		work[i].Run(serial);
		if ((i % batch) == batch - 1) {
			// Arm the hook as the renderer does: the next packet of the target settles.
			struct Ctx {
				Engine*  engine;
				Encoder* target;
			} ctx {&engine, &target};
			target.SetOpenHook([](void* c) {
				auto* x = static_cast<Ctx*>(c);
				x->engine->Settle(*x->target);
			}, &ctx);
			target.fillBuffer(H<vk::Buffer>(0x7000), i, 4, i);
			Check(!target.HasOpenHook(), "hook cleared after it ran");
			serial.fillBuffer(H<vk::Buffer>(0x7000), i, 4, i);
		}
		// Keep the rings drained so a small ring wraps many times.
		if (engine.Pending() == 0) {
			ReplayAll(target_ring, got);
		}
		ReplayAll(serial_ring, want);
	}
	engine.Settle(target);
	SubmitPacket submit {};
	submit.tick = 1;
	target.Submit(submit);
	serial.Submit(submit);
	ReplayAll(target_ring, got);
	ReplayAll(serial_ring, want);
	Check(got.log.calls == want.log.calls, "spliced calls equal the serial encode");
	if (verify) {
		Check(got.state.checks > 0 && got.state.mismatches == 0, "verify: sequence, hashes and digest hold");
	}
	Check(engine.Pending() == 0, "nothing pending at the end");
}

void TestJobLimit() {
	Encoder::Options options;
	Engine           engine;
	engine.Configure(options);
	Ring             ring(1u << 20u);
	Encoder          target(ring, options);
	std::vector<Job> work(Engine::MaxJobs + 1);
	uint32_t         posted = 0;
	for (auto& job: work) {
		posted += engine.Post(&job, Job::RunOn) ? 1u : 0u;
	}
	Check(posted == Engine::MaxJobs, "the job limit");
	const auto bytes = engine.Settle(target);
	Check(bytes > 0 && engine.Pending() == 0, "settle splices every pending job");
	Check(engine.Settle(target) == 0, "an empty settle is free");
}

void TestTakeOldest() {
	Encoder::Options options;
	Engine           engine;
	engine.Configure(options);
	Job a {11};
	Job b {12};
	Check(engine.Post(&a, Job::RunOn) && engine.Post(&b, Job::RunOn), "post two");
	CallLog first;
	engine.TakeOldest(first);
	Ring    ring(1u << 16u);
	Encoder serial(ring, options);
	a.Run(serial);
	Replayed want;
	ReplayAll(ring, want);
	Check(first.calls == want.log.calls, "TakeOldest logs the oldest job");
	Check(engine.Pending() == 1, "the newer job stays pending");
	Ring    target_ring(1u << 16u);
	Encoder target(target_ring, options);
	engine.Settle(target);
	Replayed got;
	ReplayAll(target_ring, got);
	b.Run(serial);
	Replayed want_b;
	ReplayAll(ring, want_b);
	Check(got.log.calls == want_b.log.calls, "settle after TakeOldest splices the rest");
}

} // namespace

int main() {
	TestSpliceMatchesSerial(false, 2000, 5, 1u << 20u);
	TestSpliceMatchesSerial(true, 2000, 5, 1u << 20u);
	TestSpliceMatchesSerial(true, 20000, 37, 64u << 10u);
	TestSpliceMatchesSerial(false, 5000, 1, 64u << 10u);
	TestJobLimit();
	TestTakeOldest();
	if (g_failures != 0) {
		std::printf("encode_offload_tests: %d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("encode_offload_tests: all passed\n");
	return 0;
}
