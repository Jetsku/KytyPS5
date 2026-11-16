#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUTOUCHEDPAGES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUTOUCHEDPAGES_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// P3b (KYTY_CP_SEQ=1, Profiling/analysis/P3-SEQUENCER.md §2.5 (c)): a sticky page bitmap of the
// guest address space, one bit per 4 KiB page, set for every page a GPU-side transition ever
// touched: the coherence-log sources that can make the exact clean predicate report a range
// unclean (buffer GPU-dirty ranges, tracker GPU marks, GPU-modified images, backing
// publications). Bits are set inside Coherence::Log::Append, before the transition becomes
// observable, and never cleared.
//
// The sequencer thread reads guest memory (command bytes, register-indirect pairs) directly only
// when none of the pages was ever touched: then the backing held the authoritative bytes when it
// looked, and reading it never faults. Otherwise the resolver reads them in order (lockstep).
//
// Two levels: 4096 leaves of 64 GiB (48-bit addresses), each 2 MiB of bits, allocated on first
// mark and never freed. A mark of an unknown range sets the universe flag: every read is then
// done in lockstep. Marks and checks are lock-free (seq_cst), from any thread.
namespace Libs::Graphics::GpuTouched {

inline constexpr uint32_t PageShift     = 12u;
inline constexpr uint32_t LeafShift     = 36u; // 64 GiB per leaf
inline constexpr uint32_t LeafCount     = 1u << (48u - LeafShift);

// One bit per 2^Shift bytes (Pages: 4 KiB; the KYTY_CLEAN_UNTOUCHED_DIAG fine map: 1 KiB).
template <uint32_t Shift>
class PagesT {
	static constexpr uint32_t PageShift    = Shift;
	static constexpr uint64_t PagesPerLeaf = uint64_t {1} << (LeafShift - PageShift);
	static constexpr uint64_t WordsPerLeaf = PagesPerLeaf / 64u;

public:
	PagesT() = default;
	~PagesT() {
		for (auto& leaf: m_leaves) {
			delete[] leaf.load(std::memory_order_relaxed);
		}
	}
	PagesT(const PagesT&)            = delete;
	PagesT& operator=(const PagesT&) = delete;

	// Marks every page of [begin, end). end = UINT64_MAX or a range beyond 48 bits: universe.
	void Mark(uint64_t begin, uint64_t end) noexcept {
		if (begin >= end) {
			return;
		}
		if (end > (uint64_t {1} << 48u)) {
			m_universe.store(true, std::memory_order_seq_cst);
			return;
		}
		auto       page = begin >> PageShift;
		const auto last = (end - 1u) >> PageShift;
		while (page <= last) {
			auto*      words = Leaf(page >> (LeafShift - PageShift), true);
			const auto first = page & (PagesPerLeaf - 1u);
			// Pages of this leaf up to `last`.
			const auto count = std::min<uint64_t>(last - page + 1u, PagesPerLeaf - first);
			SetBits(words, first, count);
			page += count;
		}
	}

	// Whether any page of [begin, end) was ever marked (or the universe flag is set).
	[[nodiscard]] bool AnyTouched(uint64_t begin, uint64_t end) const noexcept {
		if (m_universe.load(std::memory_order_seq_cst)) {
			return true;
		}
		if (begin >= end) {
			return false;
		}
		if (end > (uint64_t {1} << 48u)) {
			return true;
		}
		auto       page = begin >> PageShift;
		const auto last = (end - 1u) >> PageShift;
		while (page <= last) {
			const auto* words = m_leaves[page >> (LeafShift - PageShift)].load(
			    std::memory_order_seq_cst);
			const auto first = page & (PagesPerLeaf - 1u);
			const auto count = std::min<uint64_t>(last - page + 1u, PagesPerLeaf - first);
			if (words != nullptr && AnyBits(words, first, count)) {
				return true;
			}
			page += count;
		}
		return false;
	}

	[[nodiscard]] bool Universe() const noexcept {
		return m_universe.load(std::memory_order_seq_cst);
	}

	// Tests only.
	void ResetForTest() noexcept {
		for (auto& leaf: m_leaves) {
			if (auto* words = leaf.load(std::memory_order_relaxed); words != nullptr) {
				for (uint64_t i = 0; i < WordsPerLeaf; i++) {
					words[i].store(0, std::memory_order_relaxed);
				}
			}
		}
		m_universe.store(false, std::memory_order_seq_cst);
	}

private:
	using Word = std::atomic<uint64_t>;

	Word* Leaf(uint64_t index, bool create) noexcept {
		auto* words = m_leaves[index].load(std::memory_order_seq_cst);
		if (words != nullptr || !create) {
			return words;
		}
		auto* fresh    = new Word[WordsPerLeaf] {};
		Word* expected = nullptr;
		if (m_leaves[index].compare_exchange_strong(expected, fresh, std::memory_order_seq_cst)) {
			return fresh;
		}
		delete[] fresh;
		return expected;
	}

	static void SetBits(Word* words, uint64_t first, uint64_t count) noexcept {
		while (count != 0) {
			const auto bit   = first & 63u;
			const auto take  = std::min<uint64_t>(count, 64u - bit);
			const auto mask  = (take == 64u ? ~uint64_t {0} : ((uint64_t {1} << take) - 1u)) << bit;
			auto&      word  = words[first >> 6u];
			if ((word.load(std::memory_order_relaxed) & mask) != mask) {
				word.fetch_or(mask, std::memory_order_seq_cst);
			}
			first += take;
			count -= take;
		}
	}

	static bool AnyBits(const Word* words, uint64_t first, uint64_t count) noexcept {
		while (count != 0) {
			const auto bit  = first & 63u;
			const auto take = std::min<uint64_t>(count, 64u - bit);
			const auto mask = (take == 64u ? ~uint64_t {0} : ((uint64_t {1} << take) - 1u)) << bit;
			if ((words[first >> 6u].load(std::memory_order_seq_cst) & mask) != 0) {
				return true;
			}
			first += take;
			count -= take;
		}
		return false;
	}

	std::array<std::atomic<Word*>, LeafCount> m_leaves {};
	std::atomic<bool>                        m_universe {false};
};

using Pages = PagesT<PageShift>;

// The process-wide bitmap.
inline Pages g_pages;

// KYTY_CLEAN_UNTOUCHED_DIAG=1 (diagnostic, default off, read once): which source touched the pages
// that make a certificate "touched" (one 4 KiB map per marking source), and whether a 1 KiB map
// would have called it untouched. Nothing is allocated unless it is on.
enum class DiagSource : uint32_t { BufferDirtyAdd, TrackerGpuMark, ImageGpuModified, PublicationBegin, Count };
[[nodiscard]] inline bool DiagEnabled() noexcept {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CLEAN_UNTOUCHED_DIAG");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}
inline std::array<Pages, static_cast<size_t>(DiagSource::Count)> g_diag_pages;

// KYTY_CLEAN_UNTOUCHED_FINE (drawPrep.cpp, live): the same sticky marks at 1 KiB, maintained
// alongside g_pages from the first transition on (marks are rare next to queries), so the
// clean-untouched verdicts can switch between the two maps at any time. Both are exact.
inline PagesT<10> g_fine_pages;

// The clean-untouched verdict's map: 1 KiB marks when `fine`, else 4 KiB.
[[nodiscard]] inline bool AnyTouched(bool fine, uint64_t begin, uint64_t end) noexcept {
	return fine ? g_fine_pages.AnyTouched(begin, end) : g_pages.AnyTouched(begin, end);
}

// Maintained from the first transition on when the sequencer thread is configured
// (KYTY_CP_SEQ=1, read once here so that no mark before the sequencer starts is missed).
[[nodiscard]] inline bool Enabled() noexcept {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CP_SEQ");
		return value != nullptr && (std::strcmp(value, "1") == 0 || std::strcmp(value, "thread") == 0);
	}();
	return enabled;
}

// Coherence::Log::Append: the sources whose transitions can make a range unclean.
// `source`: the DiagSource of an unclean source, or DiagSource::Count for the others.
inline void NoteTransition(uint64_t begin, uint64_t end, DiagSource source) noexcept {
	if (source != DiagSource::Count && Enabled()) {
		g_pages.Mark(begin, end);
		g_fine_pages.Mark(begin, end);
		if (DiagEnabled()) {
			g_diag_pages[static_cast<size_t>(source)].Mark(begin, end);
		}
	}
}

} // namespace Libs::Graphics::GpuTouched

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUTOUCHEDPAGES_H_
