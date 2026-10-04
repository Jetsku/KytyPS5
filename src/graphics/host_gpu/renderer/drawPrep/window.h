#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_WINDOW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_WINDOW_H_

#include "common/ramStats.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

// Draw-prep S6: the preparation window, a bounded single-producer ring whose published slots
// are claimed by any number of preparing threads (SPMC) and retired by the producer in order.
//
// Slot life cycle (seq = the slot's position in the stream, never reused):
//   producer: Reserve -> fill the payload -> Publish            state = {seq, Published}
//   worker:   TryClaim (CAS Published -> Claimed) -> prepare -> Complete   {seq, Done}
//   producer: TryClaimHead (CAS Published -> Claimed: prepares the head itself) or wait for
//             Done -> commit -> Retire                           state = Free, head advances
// Every state word carries its seq, so a slow worker holding an old position can never claim
// the reused slot of a later position (no ABA). The release store of Published (and of Done)
// publishes the payload written before it; the acquiring CAS (or load) makes it visible.
//
// P3b (KYTY_CP_SEQ=1): the producer's two roles may run on two threads. The sequencer reserves
// and publishes; the resolver claims the head, commits and retires. The head is then read by the
// publishing thread (Full), so it is atomic, stored and loaded seq_cst (the sequencer parks on
// it, cpSequencer.h).
namespace Libs::Graphics::DrawPrep {

template <typename Payload>
class Window {
public:
	explicit Window(uint32_t capacity) {
		uint32_t size = 2;
		while (size < capacity && size < (1u << 16u)) {
			size <<= 1u;
		}
		m_mask  = size - 1u;
		m_slots = std::vector<Slot>(size);
		Common::RamStats::Range("draw preparation window", m_slots.data(),
		                       m_slots.capacity() * sizeof(Slot));
	}

	[[nodiscard]] uint32_t Capacity() const noexcept { return m_mask + 1u; }

	// Producer.
	[[nodiscard]] uint64_t Head() const noexcept { return m_head.load(std::memory_order_seq_cst); }
	[[nodiscard]] uint64_t Tail() const noexcept { return m_published.load(std::memory_order_relaxed); }
	[[nodiscard]] bool     Empty() const noexcept { return Head() == Tail(); }
	[[nodiscard]] bool     Full() const noexcept { return Tail() - Head() > m_mask; }
	[[nodiscard]] uint64_t Occupancy() const noexcept { return Tail() - Head(); }

	// The payload of the next position; valid until Publish. Requires !Full().
	[[nodiscard]] Payload& Reserve() noexcept { return m_slots[Tail() & m_mask].payload; }

	void Publish() noexcept {
		const auto seq = Tail();
		m_slots[seq & m_mask].state.store(Encode(seq, State::Published), std::memory_order_release);
		m_published.store(seq + 1u, std::memory_order_seq_cst);
	}

	[[nodiscard]] Payload& HeadPayload() noexcept { return m_slots[Head() & m_mask].payload; }
	// The payload of a published position in [Head(), Tail()) (P3c: a speculative slot's key).
	[[nodiscard]] const Payload& PayloadAt(uint64_t seq) const noexcept {
		return m_slots[seq & m_mask].payload;
	}

	// Claims the head for the producer when no worker has; true when it did.
	[[nodiscard]] bool TryClaimHead() noexcept {
		const auto head     = Head();
		auto       expected = Encode(head, State::Published);
		return m_slots[head & m_mask].state.compare_exchange_strong(
		    expected, Encode(head, State::Claimed), std::memory_order_acq_rel);
	}

	// The head was claimed by a worker that has finished.
	[[nodiscard]] bool HeadDone() const noexcept {
		const auto head = Head();
		return m_slots[head & m_mask].state.load(std::memory_order_acquire) ==
		       Encode(head, State::Done);
	}

	// Retires the head after its commit.
	void Retire() noexcept {
		const auto head = Head();
		m_slots[head & m_mask].state.store(Encode(head, State::Free), std::memory_order_relaxed);
		m_head.store(head + 1u, std::memory_order_seq_cst);
	}

	// Workers.
	[[nodiscard]] bool HasClaimable() const noexcept {
		return m_next_claim.load(std::memory_order_acquire) <
		       m_published.load(std::memory_order_seq_cst);
	}

	// Published slots no thread has claimed yet. The claim counter is read first: it never passes
	// the published count, so the difference cannot underflow. Any thread.
	[[nodiscard]] uint64_t Unclaimed() const noexcept {
		const auto claimed = m_next_claim.load(std::memory_order_acquire);
		return m_published.load(std::memory_order_seq_cst) - claimed;
	}

	// Claims the oldest published slot no one has claimed. Null when there is none.
	[[nodiscard]] Payload* TryClaim(uint64_t& seq) noexcept {
		for (;;) {
			auto       next      = m_next_claim.load(std::memory_order_acquire);
			const auto published = m_published.load(std::memory_order_acquire);
			if (next >= published) {
				return nullptr;
			}
			if (!m_next_claim.compare_exchange_weak(next, next + 1u, std::memory_order_acq_rel)) {
				continue;
			}
			auto& slot     = m_slots[next & m_mask];
			auto  expected = Encode(next, State::Published);
			if (slot.state.compare_exchange_strong(expected, Encode(next, State::Claimed),
			                                       std::memory_order_acq_rel)) {
				seq = next;
				return &slot.payload;
			}
			// The producer prepared (and possibly retired) it already.
		}
	}

	void Complete(uint64_t seq) noexcept {
		m_slots[seq & m_mask].state.store(Encode(seq, State::Done), std::memory_order_release);
	}

private:
	enum class State : uint64_t { Free = 0, Published = 1, Claimed = 2, Done = 3 };

	static constexpr uint64_t Encode(uint64_t seq, State state) noexcept {
		return (seq << 2u) | static_cast<uint64_t>(state);
	}

	struct Slot {
		std::atomic<uint64_t> state {0};
		Payload               payload {};
	};

	std::vector<Slot>     m_slots;
	uint32_t              m_mask = 0;
	// The committing thread writes it; a separate publishing thread (P3b) reads it.
	alignas(64) std::atomic<uint64_t> m_head {0};
	alignas(64) std::atomic<uint64_t> m_published {0};
	alignas(64) std::atomic<uint64_t> m_next_claim {0};
};

} // namespace Libs::Graphics::DrawPrep

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_WINDOW_H_
