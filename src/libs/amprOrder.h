#ifndef EMULATOR_SRC_LIBS_AMPRORDER_H_
#define EMULATOR_SRC_LIBS_AMPRORDER_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

// AMM submission order across the priority queues (after IDXTRI's Wolverine fix). The queues run
// independently (a wait holds only its own priority), so an unmap queued behind a wait can run
// after a map submitted later at the same address, and destroy it. Each AMM-mapped range
// remembers the submission that mapped it; an unmap leaves out the parts a later submission
// mapped, the result submission order gives.
namespace Libs::LibAmpr::AmmOrder {

inline std::atomic<uint64_t> g_next_seq {0};
inline std::mutex            g_mutex;
// Mapped ranges: start -> (end, submission sequence of the map).
inline std::map<uint64_t, std::pair<uint64_t, uint64_t>> g_ranges;

inline uint64_t NextSeq() {
	return g_next_seq.fetch_add(1, std::memory_order_relaxed) + 1;
}

// Removes [start, end) from the tracked ranges, splitting the ones it cuts. g_mutex held.
inline void EraseLocked(uint64_t start, uint64_t end) {
	auto it = g_ranges.lower_bound(start);
	if (it != g_ranges.begin() && std::prev(it)->second.first > start) {
		--it;
	}
	while (it != g_ranges.end() && it->first < end) {
		const auto r_start = it->first;
		const auto r_end   = it->second.first;
		const auto r_seq   = it->second.second;
		it                 = g_ranges.erase(it);
		if (r_start < start) {
			g_ranges.emplace(r_start, std::make_pair(start, r_seq));
		}
		if (r_end > end) {
			it = g_ranges.emplace(end, std::make_pair(r_end, r_seq)).first;
			++it;
		}
	}
}

inline void Mapped(uint64_t va, uint64_t size, uint64_t seq) {
	if (seq == 0 || size == 0) {
		return;
	}
	std::scoped_lock lock(g_mutex);
	EraseLocked(va, va + size);
	g_ranges.emplace(va, std::make_pair(va + size, seq));
}

// The parts of [va, va + size) an unmap from submission seq may remove: all of it except what a
// later submission mapped. Those parts stop being tracked.
inline std::vector<std::pair<uint64_t, uint64_t>> UnmapPieces(uint64_t va, uint64_t size, uint64_t seq,
                                                       uint64_t* kept_bytes) {
	std::vector<std::pair<uint64_t, uint64_t>> pieces;
	*kept_bytes = 0;
	std::scoped_lock lock(g_mutex);
	const auto       end    = va + size;
	uint64_t         cursor = va;
	auto             it     = g_ranges.lower_bound(va);
	if (it != g_ranges.begin() && std::prev(it)->second.first > va) {
		--it;
	}
	std::vector<std::pair<uint64_t, uint64_t>> newer;
	for (; it != g_ranges.end() && it->first < end; ++it) {
		if (seq != 0 && it->second.second > seq) {
			newer.emplace_back(std::max(it->first, va), std::min(it->second.first, end));
		}
	}
	for (const auto& [n_start, n_end]: newer) {
		if (n_start > cursor) {
			pieces.emplace_back(cursor, n_start - cursor);
		}
		*kept_bytes += n_end - n_start;
		cursor = n_end;
	}
	if (cursor < end) {
		pieces.emplace_back(cursor, end - cursor);
	}
	for (const auto& [p_start, p_size]: pieces) {
		EraseLocked(p_start, p_start + p_size);
	}
	return pieces;
}

} // namespace Libs::LibAmpr::AmmOrder

#endif // EMULATOR_SRC_LIBS_AMPRORDER_H_
