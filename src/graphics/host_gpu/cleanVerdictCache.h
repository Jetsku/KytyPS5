#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_CLEANVERDICTCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_CLEANVERDICTCACHE_H_

#include "common/onceValue.h"
#include "graphics/host_gpu/coherenceLog.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

// Generation-tagged "clean for a GPU-backing read" verdicts for 4 KiB guest pages.
//
// A page is clean when no byte of it is GPU-dirty in the buffer cache, covered by a pending
// backing publication, or owned by a GPU-modified image. Only that verdict is cached; guest
// bytes are always copied fresh from the backing store.
//
// Invariants:
// 1. Every transition that can make a clean page non-clean, or that changes which copy of a
//    page is authoritative, calls Invalidate() before the new state becomes observable.
// 2. A verdict is tagged with the generation loaded BEFORE its predicates were evaluated, so a
//    transition racing the evaluation always leaves the stored verdict stale.
// 3. Only complete 4 KiB pages are proven clean, so a later sub-page dirty range still bumps.
// 4. Tables are per thread; only the generation counter is shared.
namespace Libs::Graphics::CleanVerdict {

// KYTY_CLEAN_VERDICT_CACHE=0 disables the verdict cache and the backing mapping cache.
inline bool Enabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_CLEAN_VERDICT_CACHE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}));
	return enabled;
}

constexpr uint64_t PAGE_BITS = 12;
constexpr uint64_t PAGE_SIZE = uint64_t {1} << PAGE_BITS;
// Reads spanning more pages keep the exact range queries instead of probing every page.
constexpr uint64_t MAX_CACHED_PAGES = 4;

// The verdict generation is the coherence log's generation (coherenceLog.h). It starts at 1:
// zero-initialized table entries never match.
[[nodiscard]] inline uint64_t Generation() noexcept {
	return Coherence::Generation();
}

// The RMW is sequenced before the caller's state change, so any thread that later observes
// that change (through the lock or atomic that publishes it) also observes the new generation.
// Without a range the transition is logged as touching all memory.
inline void Invalidate() noexcept {
	Coherence::Append(Coherence::Universe, Coherence::Source::Universe);
}

// The same, logging the guest range [address, address + size) the transition affects.
inline void Invalidate(uint64_t address, uint64_t size, Coherence::Source source) noexcept {
	Coherence::Append(address, size, source);
}

class Table {
public:
	enum class State : uint8_t { Unknown, Clean, Dirty };

	[[nodiscard]] State Lookup(uint64_t page, uint64_t generation) const noexcept {
		const auto& entry = m_entries[Slot(page)];
		return entry.generation == generation && entry.page == page ? entry.state
		                                                            : State::Unknown;
	}

	void Store(uint64_t page, uint64_t generation, State state) noexcept {
		m_entries[Slot(page)] = {page, generation, state};
	}

private:
	struct Entry {
		uint64_t page       = 0;
		uint64_t generation = 0;
		State    state      = State::Unknown;
	};

	static size_t Slot(uint64_t page) noexcept {
		return static_cast<size_t>((page ^ (page >> 8u)) % ENTRIES);
	}

	static constexpr size_t      ENTRIES = 256;
	std::array<Entry, ENTRIES> m_entries {};
};

struct QueryResult {
	bool     clean  = false;
	bool     hit    = false; // every page was answered from the table
	uint32_t stores = 0;
};

// is_clean(vaddr, size) runs the exact, uncached predicates. A page whose verdict is not
// cached is checked as a whole: when clean it proves every sub-range clean. A page that has
// GPU-owned bytes elsewhere is remembered as Dirty, which only skips that whole-page probe;
// the requested range is then checked exactly, so a stale Dirty verdict is merely slower.
template <typename IsClean>
[[nodiscard]] QueryResult Query(Table& table, uint64_t vaddr, uint64_t size, IsClean&& is_clean) {
	static_assert(std::is_invocable_r_v<bool, IsClean&, uint64_t, uint64_t>);
	QueryResult result {};
	const auto  first = vaddr >> PAGE_BITS;
	const auto  last  = (vaddr + size - 1) >> PAGE_BITS;
	if (size == 0 || first == 0 || last - first >= MAX_CACHED_PAGES) {
		result.clean = is_clean(vaddr, size);
		return result;
	}
	const auto generation = Generation();
	bool       all_hit    = true;
	for (auto page = first; page <= last; page++) {
		const auto state = table.Lookup(page, generation);
		if (state == Table::State::Clean) {
			continue;
		}
		all_hit = false;
		if (state == Table::State::Unknown && is_clean(page << PAGE_BITS, PAGE_SIZE)) {
			table.Store(page, generation, Table::State::Clean);
			result.stores++;
			continue;
		}
		if (state == Table::State::Unknown) {
			table.Store(page, generation, Table::State::Dirty);
		}
		result.clean = is_clean(vaddr, size);
		return result;
	}
	result.clean = true;
	result.hit   = all_hit;
	return result;
}

} // namespace Libs::Graphics::CleanVerdict

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_CLEANVERDICTCACHE_H_
