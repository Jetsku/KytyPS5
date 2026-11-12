#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACECANDIDATES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACECANDIDATES_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

// KYTY_INPLACE (inplaceHeap.h): which 4 KiB guest pages are bound in place. A page becomes a
// candidate when it took guest write faults in `promote_frames` of the last 8 frames (a ring the
// game cycles through faults once per cycle, not every frame); it stops being one when a GPU
// writer or a command-processor write covers it (Clear), and must then fault again. Lookups are lock-free (a two-level bitmap over the 40-bit guest
// address space, 64 MiB of addresses per lazily allocated chunk) and so are fault histories (one
// packed atomic word per page): the fault handler takes no lock here.
namespace Libs::Graphics {

class InplaceCandidates final {
public:
	explicit InplaceCandidates(uint32_t promote_frames);
	~InplaceCandidates();
	InplaceCandidates(const InplaceCandidates&)            = delete;
	InplaceCandidates& operator=(const InplaceCandidates&) = delete;

	// Any thread. True when this fault made the page a candidate.
	bool NoteFault(uint64_t vaddr) noexcept;
	// Once per guest flip.
	void AdvanceFrame() noexcept;
	[[nodiscard]] bool All(uint64_t vaddr, uint64_t size) const noexcept;
	[[nodiscard]] bool Any(uint64_t vaddr, uint64_t size) const noexcept;
	// Returns the number of candidate pages cleared.
	uint64_t Clear(uint64_t vaddr, uint64_t size) noexcept;
	[[nodiscard]] uint64_t Count() const noexcept { return m_count.load(std::memory_order_relaxed); }
	[[nodiscard]] uint32_t Frame() const noexcept { return m_frame.load(std::memory_order_relaxed); }

	static constexpr uint64_t PageBits  = 12;
	static constexpr uint64_t ChunkBits = 26;
	// All() refuses ranges of more pages than this (not the per-frame data this is for).
	static constexpr uint64_t MaxPages = 1024;

private:
	static constexpr uint64_t ChunkPages = uint64_t {1} << (ChunkBits - PageBits);
	static constexpr uint64_t ChunkCount = uint64_t {1} << (40 - ChunkBits);
	struct Chunk {
		std::array<std::atomic<uint64_t>, ChunkPages / 64> bits {};
		// Per page: last faulting frame (high 24 bits) and history (low 8 bits, bit i: a fault in
		// frame (last - i)); 0: never faulted.
		std::array<std::atomic<uint32_t>, ChunkPages> history {};
	};
	[[nodiscard]] Chunk* GetChunk(uint64_t page) noexcept;
	bool Set(uint64_t page) noexcept;
	[[nodiscard]] bool Test(uint64_t page) const noexcept;

	uint32_t                                  m_promote_frames;
	std::unique_ptr<std::atomic<Chunk*>[]>    m_chunks;
	std::mutex                                m_chunk_mutex;
	std::atomic<uint32_t>                     m_frame {1};
	std::atomic<uint64_t>                     m_count {0};
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACECANDIDATES_H_ */
