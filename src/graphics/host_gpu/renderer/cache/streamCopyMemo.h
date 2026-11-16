#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMCOPYMEMO_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMCOPYMEMO_H_

#include <array>
#include <cstddef>
#include <cstdint>

namespace Libs::Graphics {

// KYTY_STREAM_COPY_REUSE (BufferCache::ObtainBufferNow; based on Senaxx/KytyPS5 8bdb5b43f): a small
// read binding over CPU-dirty, not GPU-dirty bytes is copied into the stream buffer at every
// binding, though draw after draw binds the same constants. Within one command buffer (tick) the
// stream buffer keeps a copy (its wrap never overwrites data of a tick the GPU has not finished),
// and the guest can only change the bytes the GPU reads from a submitted command buffer under a
// packet that writes or waits on memory: those end the reuse (NewEpoch). So the same range in the
// same tick and epoch takes the first copy's offset.
//
// A direct-mapped table of the last copies; one thread (the command processor's) uses it.
class StreamCopyMemo {
public:
	static constexpr size_t Slots = 1024;

	[[nodiscard]] static constexpr size_t SlotOf(uint64_t vaddr, uint64_t size) noexcept {
		return static_cast<size_t>((((vaddr >> 4u) * 0x9E3779B97F4A7C15ull) >> 54u) ^ (size & 1023u)) &
		       (Slots - 1u);
	}

	// The stream offset of the copy of [vaddr, vaddr + size) made in this tick and epoch, if any.
	[[nodiscard]] bool Find(uint64_t vaddr, uint64_t size, uint64_t tick, uint64_t& offset) const noexcept {
		const auto& entry = m_entries[SlotOf(vaddr, size)];
		if (entry.vaddr != vaddr || entry.size != size || entry.tick != tick || entry.epoch != m_epoch) {
			return false;
		}
		offset = entry.offset;
		return true;
	}

	void Note(uint64_t vaddr, uint64_t size, uint64_t tick, uint64_t offset) noexcept {
		m_entries[SlotOf(vaddr, size)] = {vaddr, size, tick, m_epoch, offset};
	}

	// A packet that may write or wait on guest memory, or a new submission slice: no copy before it
	// is reused after it.
	void NewEpoch() noexcept { m_epoch++; }

	[[nodiscard]] uint64_t Epoch() const noexcept { return m_epoch; }

private:
	struct Entry {
		uint64_t vaddr  = 0;
		uint64_t size   = 0;
		uint64_t tick   = 0;
		uint64_t epoch  = 0; // never the memo's (it starts at 1): an unused entry never matches
		uint64_t offset = 0;
	};

	std::array<Entry, Slots> m_entries {};
	uint64_t                 m_epoch = 1;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMCOPYMEMO_H_ */
