// Unit tests for the P3 op stream (graphics/guest_gpu/command_processor/cpOps.h), without a
// Vulkan device:
//  - every op kind round-trips through OpStream: header (kind, sequence, packet position, packets
//    hash, inline data size, verify hash) and payload and inline data bytes;
//  - wrap-around with random record sizes up to the largest record, single-threaded;
//  - a producer and a consumer thread with ring back-pressure;
//  - the op table (payload sizes, lockstep kinds, names).

#include "graphics/guest_gpu/command_processor/cpOps.h"
#include "graphics/guest_gpu/command_processor/cpSequencer.h"
#include "graphics/guest_gpu/command_processor/submitDefer.h"
#include "graphics/host_gpu/gpuTouchedPages.h"

#include <array>
#include <atomic>
#include <memory>
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
using namespace Libs::Graphics::CpSeq;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::printf("FAILED: %s\n", what);
		++g_failures;
	}
}

constexpr uint32_t KindCount = static_cast<uint32_t>(OpKind::Count);

// Inline data is carried by these kinds only (the front's WRITE_DATA dwords and constant RAM).
bool CarriesData(OpKind kind) {
	return kind == OpKind::WriteData || kind == OpKind::DumpConstRam;
}

struct Expected {
	OpKind               kind         = OpKind::Count;
	uint64_t             sequence     = 0;
	uint64_t             packet       = 0;
	uint64_t             packets_hash = 0;
	bool                 verify       = false;
	std::vector<uint8_t> payload;
	std::vector<uint8_t> data;
};

Expected MakeOp(std::mt19937_64& random, uint64_t sequence, uint32_t max_data) {
	Expected op;
	op.kind         = static_cast<OpKind>(random() % KindCount);
	op.sequence     = sequence;
	op.packet       = random();
	op.packets_hash = random();
	op.verify       = (random() & 1u) != 0;
	op.payload.resize(PayloadSize(op.kind));
	for (auto& byte: op.payload) {
		byte = static_cast<uint8_t>(random());
	}
	if (CarriesData(op.kind) && max_data != 0) {
		op.data.resize(random() % (max_data + 1u));
		for (auto& byte: op.data) {
			byte = static_cast<uint8_t>(random());
		}
	}
	return op;
}

void EmitOp(OpStream& stream, const Expected& op) {
	const auto sequence =
	    stream.Emit(op.kind, op.payload.data(), static_cast<uint32_t>(op.payload.size()),
	                op.data.empty() ? nullptr : op.data.data(),
	                static_cast<uint32_t>(op.data.size()), op.packet, op.packets_hash, op.verify);
	Check(sequence == op.sequence, "Emit returns the op's sequence number");
}

bool Matches(const OpView& view, const Expected& op) {
	const auto* header = view.header;
	bool        ok     = header->kind == op.kind && header->sequence == op.sequence &&
	          header->packet == op.packet && header->packets_hash == op.packets_hash &&
	          header->data_size == op.data.size() &&
	          ((header->flags & FlagVerify) != 0) == op.verify &&
	          header->ring.op == CommandStream::Op::Count &&
	          header->ring.size == RecordSize(static_cast<uint32_t>(op.payload.size()),
	                                          static_cast<uint32_t>(op.data.size()));
	ok = ok && std::memcmp(view.payload, op.payload.data(), op.payload.size()) == 0;
	ok = ok && (op.data.empty() || std::memcmp(view.data, op.data.data(), op.data.size()) == 0);
	if (op.verify) {
		ok = ok && header->verify_hash == HashOp(op.kind, view.payload,
		                                         static_cast<uint32_t>(op.payload.size()),
		                                         view.data,
		                                         static_cast<uint32_t>(op.data.size()));
	}
	return ok;
}

void TestTable() {
	for (uint32_t index = 0; index < KindCount; index++) {
		const auto kind = static_cast<OpKind>(index);
		Check(PayloadSize(kind) != 0 && PayloadSize(kind) % 8 == 0, "payload sizes");
		Check(std::strcmp(OpKindName(kind), "?") != 0, "op kind names");
	}
	Check(PayloadSize(OpKind::Count) == 0, "no payload for Count");
	Check(PayloadSize(OpKind::DrawIndirect) == sizeof(DrawIndirectOp) &&
	          PayloadSize(OpKind::DrawIndirectMulti) == sizeof(DrawIndirectOp) &&
	          PayloadSize(OpKind::WriteData) == sizeof(WriteDataOp) &&
	          PayloadSize(OpKind::ReadCheck) == sizeof(ReadCheckOp),
	      "payload size table");
	Check(IsLockstep(OpKind::WaitRegMem) && IsLockstep(OpKind::WaitFlipDone) &&
	          IsLockstep(OpKind::Predication) && IsLockstep(OpKind::CondExec) &&
	          IsLockstep(OpKind::Branch) && !IsLockstep(OpKind::DrawIndex) &&
	          !IsLockstep(OpKind::EndOfPipe) && !IsLockstep(OpKind::ReadCheck),
	      "lockstep kinds");
	Check(RecordSize(sizeof(WriteDataOp), 5) == sizeof(OpHeader) + sizeof(WriteDataOp) + 8,
	      "record sizes pad inline data to 8 bytes");
	// The hash covers the kind, the payload and the data.
	const WriteDataOp write {0x1000, 2, 0};
	const std::array<uint32_t, 2> dwords {1, 2};
	const auto base = HashOp(OpKind::WriteData, &write, sizeof(write), dwords.data(), 8);
	auto       other = dwords;
	other[1]         = 3;
	Check(base != HashOp(OpKind::WriteData, &write, sizeof(write), other.data(), 8) &&
	          base != HashOp(OpKind::WriteData, &write, sizeof(write), dwords.data(), 4) &&
	          base != HashOp(OpKind::ReferenceClock, &write, sizeof(write), dwords.data(), 8),
	      "op hash covers kind, payload and data");
}

// Every kind, emitted and consumed one by one (the inline protocol).
void TestRoundTrip() {
	OpStream        stream(512u << 10u);
	std::mt19937_64 random(1234);
	uint64_t        sequence = 0;
	for (uint32_t index = 0; index < KindCount * 64u; index++) {
		auto op = MakeOp(random, sequence++, 256);
		op.kind = static_cast<OpKind>(index % KindCount);
		op.payload.resize(PayloadSize(op.kind));
		if (!CarriesData(op.kind)) {
			op.data.clear();
		}
		EmitOp(stream, op);
		OpView view;
		Check(stream.Peek(view), "an emitted op is published");
		Check(Matches(view, op), "op round trip");
		stream.Pop();
		Check(!stream.Peek(view), "nothing after the consumed op");
	}
	// The largest WRITE_DATA (a 14-bit dword count) fits the default ring.
	Expected large;
	large.kind     = OpKind::WriteData;
	large.sequence = sequence++;
	large.verify   = true;
	large.payload.resize(sizeof(WriteDataOp));
	large.data.resize(16381u * 4u, 0x5a);
	Check(RecordSize(sizeof(WriteDataOp), 16381u * 4u) <= stream.MaxRecord(),
	      "the largest WRITE_DATA fits a record");
	EmitOp(stream, large);
	OpView view;
	Check(stream.Peek(view) && Matches(view, large), "largest op round trip");
	stream.Pop();
	Check(stream.Emitted() == sequence, "sequence numbers are consecutive");
}

// Several ops in flight, records of random size, many wraps of a small ring.
void TestWrap() {
	OpStream        stream(64u << 10u);
	std::mt19937_64 random(99);
	std::vector<Expected> pending;
	uint64_t              sequence = 0;
	uint64_t              consumed = 0;
	const uint32_t        max_data = stream.MaxRecord() - static_cast<uint32_t>(sizeof(OpHeader)) -
	                          static_cast<uint32_t>(sizeof(WriteDataOp)) - 8u;
	for (uint32_t round = 0; round < 4000; round++) {
		// At most 2 records of up to MaxRecord (a quarter of the ring) plus a wrap's padding
		// fit, so the single thread never waits for itself.
		const auto burst = 1u + static_cast<uint32_t>(random() % 2u);
		for (uint32_t i = 0; i < burst; i++) {
			auto op = MakeOp(random, sequence++, (random() % 8u) == 0 ? max_data : 512u);
			EmitOp(stream, op);
			pending.push_back(std::move(op));
		}
		while (!pending.empty()) {
			OpView view;
			if (!stream.Peek(view)) {
				Check(false, "a pending op is published");
				break;
			}
			Check(Matches(view, pending.front()), "op round trip across wraps");
			stream.Pop();
			pending.erase(pending.begin());
			consumed++;
		}
	}
	Check(consumed == sequence && stream.Bytes() > (64u << 10u) * 8u, "the ring wrapped");
}

// A producer thread against a consumer thread: back-pressure and publication order.
void TestThreaded() {
	OpStream          stream(64u << 10u);
	constexpr uint64_t Count = 20000;
	std::thread       producer([&] {
        std::mt19937_64 random(7);
        for (uint64_t index = 0; index < Count; index++) {
            EmitOp(stream, MakeOp(random, index, 2048));
        }
    });
	std::mt19937_64 random(7);
	uint64_t        received = 0;
	bool            ok       = true;
	while (received < Count) {
		OpView view;
		if (!stream.Peek(view)) {
			std::this_thread::yield();
			continue;
		}
		ok = ok && Matches(view, MakeOp(random, received, 2048));
		stream.Pop();
		received++;
	}
	producer.join();
	Check(ok, "threaded ops arrive intact and in order");
}

// The sticky GPU-touched page bitmap (gpuTouchedPages.h): page granularity, leaf boundaries, the
// universe flag, and marks racing checks (a mark finished before a check starts is seen).
void TestTouchedPages() {
	using GpuTouched::Pages;
	auto pages = std::make_unique<Pages>();
	constexpr uint64_t base = 0x0000000203000000ull;
	Check(!pages->AnyTouched(base, base + 0x100000), "a fresh bitmap is clean");
	pages->Mark(base + 0x1234, base + 0x1238);
	Check(pages->AnyTouched(base + 0x1000, base + 0x1001) &&
	          pages->AnyTouched(base + 0x1fff, base + 0x2000) &&
	          !pages->AnyTouched(base + 0x2000, base + 0x3000) &&
	          !pages->AnyTouched(base, base + 0x1000) &&
	          pages->AnyTouched(base, base + 0x100000),
	      "a mark covers its whole 4 KiB page and nothing else");
	// A range across a 64 GiB leaf boundary.
	constexpr uint64_t leaf = uint64_t {1} << GpuTouched::LeafShift;
	pages->Mark(4u * leaf - 0x10, 4u * leaf + 0x10);
	Check(pages->AnyTouched(4u * leaf - 1u, 4u * leaf) &&
	          pages->AnyTouched(4u * leaf, 4u * leaf + 1u) &&
	          !pages->AnyTouched(4u * leaf + 0x1000, 4u * leaf + 0x2000) &&
	          !pages->AnyTouched(4u * leaf - 0x2000, 4u * leaf - 0x1000),
	      "marks across a leaf boundary");
	// A large range: every page of it, word boundaries included.
	pages->Mark(base + 0x400000, base + 0x400000 + 257u * 0x1000u);
	bool all = true;
	for (uint64_t page = 0; page < 257u; page++) {
		const auto address = base + 0x400000 + page * 0x1000u;
		all = all && pages->AnyTouched(address, address + 1u);
	}
	Check(all && !pages->AnyTouched(base + 0x400000 + 257u * 0x1000u,
	                                base + 0x400000 + 258u * 0x1000u),
	      "a multi-word range marks exactly its pages");
	Check(!pages->Universe() && !pages->AnyTouched(0x7000000000ull, 0x7000001000ull),
	      "no universe yet");
	pages->Mark(0, UINT64_MAX);
	Check(pages->Universe() && pages->AnyTouched(0x7000000000ull, 0x7000001000ull),
	      "an unknown range makes every range touched");
	pages->ResetForTest();
	Check(!pages->Universe() && !pages->AnyTouched(base, base + 0x1000000),
	      "reset for the next check");

	// Two threads: one marks pages in order and publishes how far it got; the other checks that
	// every page published as marked reads as touched.
	std::atomic<uint64_t> published {0};
	std::atomic<bool>     seen_all {true};
	constexpr uint64_t    count = 20000;
	std::thread           marker([&] {
        for (uint64_t page = 0; page < count; page++) {
            const auto address = base + page * 0x1000u;
            pages->Mark(address, address + 8u);
            published.store(page + 1u, std::memory_order_seq_cst);
        }
    });
	for (uint64_t checked = 0; checked < count;) {
		const auto limit = published.load(std::memory_order_seq_cst);
		for (; checked < limit; checked++) {
			const auto address = base + checked * 0x1000u;
			if (!pages->AnyTouched(address, address + 0x1000u)) {
				seen_all.store(false);
			}
		}
	}
	marker.join();
	Check(seen_all.load(), "a mark finished before a check is seen by it");
}

// The register snapshot ring of thread mode (cpSequencer.h): in-order release, capacity.
void TestSnapshotRing() {
	CpSeq::SnapshotRing ring(4);
	std::array<uint32_t, 4> taken {};
	for (auto& index: taken) {
		Check(ring.HasFree(), "free entries while fewer than capacity are in use");
		index = ring.Acquire();
	}
	Check(!ring.HasFree(), "full at capacity");
	Check(taken == std::array<uint32_t, 4> {0, 1, 2, 3}, "entries in order");
	ring.Release();
	Check(ring.HasFree() && ring.Acquire() == 0, "a released entry is reused in order");
	Check(!ring.HasFree(), "full again");
	// Threads: the producer writes an entry's value, the consumer reads it back in order.
	constexpr uint32_t    count = 5000;
	std::atomic<uint32_t> produced {0};
	CpSeq::SnapshotRing   shared(8);
	std::thread           producer([&] {
        for (uint32_t i = 0; i < count; i++) {
            while (!shared.HasFree()) {
                std::this_thread::yield();
            }
            const auto index = shared.Acquire();
            shared.Entry(index).context.SetPsInControl(i);
            produced.store(i + 1u, std::memory_order_release);
        }
    });
	bool ordered = true;
	for (uint32_t i = 0; i < count; i++) {
		while (produced.load(std::memory_order_acquire) <= i) {
			std::this_thread::yield();
		}
		ordered = ordered && shared.Entry(i % 8u).context.GetShaderRegisters().ps_in_control == i;
		shared.Release();
	}
	producer.join();
	Check(ordered, "snapshots arrive intact and in order");
}

// KYTY_PASS_SUBMIT_DEFER (submitDefer.h): optional submissions and slice yields inside a
// rendering instance wait for its end; a graphics submission that completes inside one is carried
// to the next graphics slice; nothing deferred is ever dropped.
void TestSubmitDefer() {
	namespace D = SubmitDefer;
	{
		D::State s;
		Check(!D::Defer(s, true, false, false), "defer: no instance, submit now");
		Check(!D::Defer(s, false, true, false), "defer: switch off, submit now");
		Check(!D::Defer(s, false, true, true), "defer: switch off, yield now");
		Check(!s.submit && !s.yield, "defer: nothing pending after immediate requests");
	}
	{
		D::State s;
		Check(D::Defer(s, true, true, false), "defer: optional submit inside an instance waits");
		auto take = D::AtBoundary(s, true);
		Check(!take.submit && !take.yield, "defer: still inside the instance, nothing due");
		take = D::AtBoundary(s, false);
		Check(take.submit && !take.yield, "defer: due at the first boundary outside the instance");
		D::Submitted(s);
		take = D::AtBoundary(s, false);
		Check(!take.submit && !take.yield, "defer: a submission clears the deferred one");
	}
	{
		D::State s;
		Check(D::Defer(s, true, true, true), "yield: slice yield inside an instance waits");
		auto take = D::AtBoundary(s, false);
		Check(take.yield && !take.submit, "yield: due outside the instance");
		take = D::AtBoundary(s, false);
		Check(!take.yield, "yield: taken once");
		Check(D::Defer(s, true, true, true), "yield: deferred again");
		D::SliceStart(s, 7);
		Check(!D::AtBoundary(s, false).yield, "yield: a new slice drops the old slice's yield");
	}
	{
		D::State s;
		Check(!D::Carry(s, true, false, 5), "carry: no instance, the submission ends normally");
		Check(!D::Carry(s, false, true, 5), "carry: switch off, the submission ends normally");
		Check(!D::CarryPending(s, 5), "carry: nothing carried");
		Check(D::Carry(s, true, true, 5), "carry: completed inside an instance, recording kept");
		Check(D::CarryPending(s, 5), "carry: pending while the recording is unsubmitted");
		D::SliceStart(s, 5);
		auto take = D::AtBoundary(s, true);
		Check(!take.submit, "carry: next slice continues the instance");
		take = D::AtBoundary(s, false);
		Check(take.submit, "carry: next slice submits it outside the instance");
		D::Submitted(s);
		Check(!D::CarryPending(s, 6), "carry: submitted");
	}
	{
		D::State s;
		Check(D::Carry(s, true, true, 9), "carry: kept");
		Check(!D::CarryPending(s, 10), "carry: another queue's submission took the recording");
		D::SliceStart(s, 10);
		Check(!D::AtBoundary(s, false).submit, "carry: nothing left to submit after it was taken");
	}
	{
		// Switching the live switch off never strands deferred work.
		D::State s;
		Check(D::Defer(s, true, true, false) && D::Carry(s, true, true, 3), "live off: deferred");
		Check(!D::Defer(s, false, true, false), "live off: new requests submit now");
		Check(D::CarryPending(s, 3) && D::AtBoundary(s, false).submit, "live off: old work still due");
	}
}

} // namespace

int main() {
	TestTable();
	TestRoundTrip();
	TestWrap();
	TestThreaded();
	TestTouchedPages();
	TestSnapshotRing();
	TestSubmitDefer();
	if (g_failures != 0) {
		std::printf("cp sequencer tests: %d failure(s)\n", g_failures);
		return EXIT_FAILURE;
	}
	std::printf("cp sequencer tests passed\n");
	return EXIT_SUCCESS;
}
