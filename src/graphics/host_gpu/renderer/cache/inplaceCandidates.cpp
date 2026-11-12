#include "graphics/host_gpu/renderer/cache/inplaceCandidates.h"

#include <algorithm>
#include <bit>

namespace Libs::Graphics {

InplaceCandidates::InplaceCandidates(uint32_t promote_frames)
    : m_promote_frames(std::max<uint32_t>(promote_frames, 1)),
      m_chunks(std::make_unique<std::atomic<Chunk*>[]>(ChunkCount)) {
	for (uint64_t i = 0; i < ChunkCount; i++) {
		m_chunks[i].store(nullptr, std::memory_order_relaxed);
	}
}

InplaceCandidates::~InplaceCandidates() {
	for (uint64_t i = 0; i < ChunkCount; i++) {
		delete m_chunks[i].load(std::memory_order_relaxed);
	}
}

InplaceCandidates::Chunk* InplaceCandidates::GetChunk(uint64_t page) noexcept {
	const uint64_t chunk_index = page >> (ChunkBits - PageBits);
	if (chunk_index >= ChunkCount) {
		return nullptr;
	}
	auto* chunk = m_chunks[chunk_index].load(std::memory_order_acquire);
	if (chunk == nullptr) {
		std::scoped_lock lock {m_chunk_mutex}; // once per 64 MiB of guest addresses
		chunk = m_chunks[chunk_index].load(std::memory_order_acquire);
		if (chunk == nullptr) {
			chunk = new Chunk();
			m_chunks[chunk_index].store(chunk, std::memory_order_release);
		}
	}
	return chunk;
}

bool InplaceCandidates::Set(uint64_t page) noexcept {
	auto* chunk = GetChunk(page);
	if (chunk == nullptr) {
		return false;
	}
	const uint64_t bit  = page & (ChunkPages - 1);
	const uint64_t mask = uint64_t {1} << (bit & 63u);
	if ((chunk->bits[bit >> 6u].fetch_or(mask, std::memory_order_acq_rel) & mask) == 0) {
		m_count.fetch_add(1, std::memory_order_relaxed);
		return true;
	}
	return false;
}

bool InplaceCandidates::Test(uint64_t page) const noexcept {
	const uint64_t chunk_index = page >> (ChunkBits - PageBits);
	if (chunk_index >= ChunkCount) {
		return false;
	}
	const auto* chunk = m_chunks[chunk_index].load(std::memory_order_acquire);
	if (chunk == nullptr) {
		return false;
	}
	const uint64_t bit = page & (ChunkPages - 1);
	return ((chunk->bits[bit >> 6u].load(std::memory_order_acquire) >> (bit & 63u)) & 1u) != 0;
}

bool InplaceCandidates::All(uint64_t vaddr, uint64_t size) const noexcept {
	if (size == 0 || Count() == 0) {
		return false;
	}
	const uint64_t first = vaddr >> PageBits;
	const uint64_t last  = (vaddr + size - 1) >> PageBits;
	if (last < first || last - first >= MaxPages) {
		return false;
	}
	for (uint64_t page = first; page <= last; page++) {
		if (!Test(page)) {
			return false;
		}
	}
	return true;
}

bool InplaceCandidates::Any(uint64_t vaddr, uint64_t size) const noexcept {
	if (size == 0 || Count() == 0) {
		return false;
	}
	const uint64_t first = vaddr >> PageBits;
	const uint64_t last  = (vaddr + size - 1) >> PageBits;
	for (uint64_t page = first; page <= last && page >= first; page++) {
		const uint64_t chunk_index = page >> (ChunkBits - PageBits);
		if (chunk_index >= ChunkCount) {
			return false;
		}
		if (m_chunks[chunk_index].load(std::memory_order_acquire) == nullptr) {
			page = ((chunk_index + 1) << (ChunkBits - PageBits)) - 1; // skip the absent chunk
			continue;
		}
		if (Test(page)) {
			return true;
		}
	}
	return false;
}

uint64_t InplaceCandidates::Clear(uint64_t vaddr, uint64_t size) noexcept {
	if (size == 0 || Count() == 0) {
		return 0;
	}
	const uint64_t first   = vaddr >> PageBits;
	const uint64_t last    = (vaddr + size - 1) >> PageBits;
	uint64_t       cleared = 0;
	for (uint64_t page = first; page <= last && page >= first; page++) {
		const uint64_t chunk_index = page >> (ChunkBits - PageBits);
		if (chunk_index >= ChunkCount) {
			break;
		}
		auto* chunk = m_chunks[chunk_index].load(std::memory_order_acquire);
		if (chunk == nullptr) {
			page = ((chunk_index + 1) << (ChunkBits - PageBits)) - 1;
			continue;
		}
		const uint64_t bit  = page & (ChunkPages - 1);
		const uint64_t mask = uint64_t {1} << (bit & 63u);
		if ((chunk->bits[bit >> 6u].fetch_and(~mask, std::memory_order_acq_rel) & mask) != 0) {
			m_count.fetch_sub(1, std::memory_order_relaxed);
			// A cleared page must fault again before it returns.
			chunk->history[bit].store(0, std::memory_order_relaxed);
			cleared++;
		}
	}
	return cleared;
}

bool InplaceCandidates::NoteFault(uint64_t vaddr) noexcept {
	const uint64_t page  = vaddr >> PageBits;
	auto*          chunk = GetChunk(page);
	if (chunk == nullptr) {
		return false;
	}
	const uint32_t frame = m_frame.load(std::memory_order_relaxed) & 0xffffffu;
	auto&          word  = chunk->history[page & (ChunkPages - 1)];
	uint32_t       old   = word.load(std::memory_order_relaxed);
	uint32_t       next  = 0;
	do {
		const uint32_t last = old >> 8u;
		if (old != 0 && last == frame) {
			return false; // this frame counted already
		}
		// Bit i of the history: a fault in frame (frame - i). Rings the game cycles through every
		// few frames fault once per cycle, so the rule counts faulting frames in a window of 8.
		const uint32_t age     = old == 0 ? 32u : ((frame - last) & 0xffffffu);
		const uint32_t history = age >= 8u ? 0u : ((old & 0xffu) << age) & 0xffu;
		next                   = (frame << 8u) | history | 1u;
	} while (!word.compare_exchange_weak(old, next, std::memory_order_relaxed));
	if (static_cast<uint32_t>(std::popcount(next & 0xffu)) < m_promote_frames) {
		return false;
	}
	return Set(page);
}

void InplaceCandidates::AdvanceFrame() noexcept {
	m_frame.fetch_add(1, std::memory_order_relaxed);
}

} // namespace Libs::Graphics
