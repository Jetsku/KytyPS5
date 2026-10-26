#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_COHERENCELOG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_COHERENCELOG_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <thread>

#include "graphics/host_gpu/gpuTouchedPages.h"

// Draw-prep S4: the coherence generation and its range log.
//
// Every transition that can change whether a guest range is clean for a GPU-backing read, or
// which copy of it is authoritative (buffer GPU-dirty ranges, backing publications, image GPU
// ownership, tracker GPU bits, GPU map/unmap), appends one entry {generation, [begin, end),
// source} BEFORE its state change becomes observable (the same point that retires cached clean
// verdicts, see cleanVerdictCache.h). Emulator writes of backing bytes outside a publication
// (occlusion and LOD-statistics results) append AFTER the bytes are written. A transition whose
// range is unknown appends a universe entry.
//
// The generation is the clean-verdict generation: CleanVerdict::Invalidate() appends here.
// Entries are always recorded (LogReadersEnabled).
//
// Readers (the draw-prep certificate check on the GPU thread) ask whether any entry in the
// generation interval (g0, g1] intersects a set of ranges. The log is a ring: an interval that
// is older than the ring, or an entry whose writer has not finished yet, answers Unknown, which
// callers treat like a conflict. Answers are therefore conservative, never optimistic.
namespace Libs::Graphics::Coherence {

enum class Source : uint8_t {
	Universe,
	BufferDirtyAdd,
	BufferDirtySubtract,
	PublicationBegin,
	PublicationEnd,
	ContentRevisions,
	ImageRegister,
	ImageUnregister,
	ImageGpuModified,
	ImageGpuClear,
	TrackerGpuMark,
	TrackerGpuUnmark,
	TrackerDownload,
	TrackerReadbackUnmark,
	MapMemory,
	UnmapMemory,
	OcclusionWrite,
	LodStatsWrite,
	CpWrite,
	Test,
	Count,
};

// Half-open guest range [begin, end).
struct Range {
	uint64_t begin = 0;
	uint64_t end   = 0;

	[[nodiscard]] bool Intersects(const Range& other) const noexcept {
		return begin < other.end && other.begin < end;
	}
	bool operator==(const Range&) const = default;
};

inline constexpr Range Universe {0, UINT64_MAX};

[[nodiscard]] constexpr Range MakeRange(uint64_t address, uint64_t size) noexcept {
	if (size == 0) {
		return {address, address};
	}
	return {address, size > UINT64_MAX - address ? UINT64_MAX : address + size};
}

enum class CheckResult : uint8_t {
	Clean,    // no entry in the interval intersects the ranges
	Conflict, // an entry intersects (conflict_source says which kind)
	Overflow, // the interval is older than the ring
	Unknown,  // an entry was being written or recycled while it was read
};

struct CheckOutcome {
	CheckResult result          = CheckResult::Clean;
	Source      conflict_source = Source::Universe;
	uint64_t    entries         = 0; // entries examined
};

class Log {
public:
	static constexpr uint64_t Capacity = uint64_t {1} << 14u;
	// The counter starts at 1, so the first appended entry is generation 2.
	static constexpr uint64_t FirstEntryGeneration = 2;

	// Starts at 1: zero-initialized verdict table entries and zero sequence words never match.
	[[nodiscard]] uint64_t Generation() const noexcept {
		return m_generation.load(std::memory_order_acquire);
	}

	// Claims the next generation and records [range) for it. The seq_cst RMW is sequenced before
	// the caller's state change, so any thread that later observes that change (through the lock
	// or atomic publishing it) also observes this generation.
	uint64_t Append(Range range, Source source) noexcept {
		// KYTY_CP_SEQ=1: the sticky GPU-touched pages (gpuTouchedPages.h), set before the
		// transition becomes observable, for the transitions that can make a range unclean.
		using GpuTouched::DiagSource;
		GpuTouched::NoteTransition(range.begin, range.end,
		                           source == Source::BufferDirtyAdd     ? DiagSource::BufferDirtyAdd
		                           : source == Source::TrackerGpuMark   ? DiagSource::TrackerGpuMark
		                           : source == Source::ImageGpuModified ? DiagSource::ImageGpuModified
		                           : source == Source::PublicationBegin ? DiagSource::PublicationBegin
		                                                                : DiagSource::Count);
		const auto generation = m_generation.fetch_add(1, std::memory_order_seq_cst) + 1u;
		auto&      entry      = m_entries[generation & (Capacity - 1u)];
		// Slots are written in turn: the writer of this lap waits until the previous lap's writer
		// of the slot has published (the slot starts at 0), so two writers never interleave on
		// one slot and a reader can never accept a mix of two payloads.
		const auto previous =
		    generation >= Capacity + FirstEntryGeneration ? generation - Capacity : 0u;
		for (uint32_t spins = 0; entry.sequence.load(std::memory_order_acquire) != previous;
		     spins++) {
			if (spins >= 64u) {
				std::this_thread::yield();
			}
		}
		// Seqlock write: invalidate, then the payload, then publish the generation.
		entry.sequence.store(0, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
		entry.begin.store(range.begin, std::memory_order_relaxed);
		entry.end.store(range.end, std::memory_order_relaxed);
		entry.source.store(static_cast<uint8_t>(source), std::memory_order_relaxed);
		entry.sequence.store(generation, std::memory_order_release);
		return generation;
	}

	// Claims the next generation without recording an entry. For a log that is never read; must
	// not be mixed with Append on the same log (Append waits for the previous lap's entry).
	uint64_t Bump() noexcept { return m_generation.fetch_add(1, std::memory_order_seq_cst) + 1u; }

	// Reads the entry of `generation`. False when it is not (or no longer) readable.
	[[nodiscard]] bool Read(uint64_t generation, Range& range, Source& source) const noexcept {
		const auto& entry = m_entries[generation & (Capacity - 1u)];
		const auto  first = entry.sequence.load(std::memory_order_acquire);
		if (first != generation) {
			return false;
		}
		range.begin = entry.begin.load(std::memory_order_relaxed);
		range.end   = entry.end.load(std::memory_order_relaxed);
		source      = static_cast<Source>(entry.source.load(std::memory_order_relaxed));
		std::atomic_thread_fence(std::memory_order_acquire);
		if (entry.sequence.load(std::memory_order_relaxed) != generation) {
			return false;
		}
		// The slot is reused only by generation + Capacity, which is claimed (the counter
		// passes it) before its writer touches the slot.
		return m_generation.load(std::memory_order_seq_cst) - generation < Capacity;
	}

	// Whether any entry with a generation in (from, to] intersects one of `ranges`, which must be
	// sorted by begin and non-overlapping. An unfinished entry is waited for briefly. `hull`: test
	// each entry against [first begin, last end) before searching the ranges (same answer; the
	// search touches the range list only for entries inside the hull).
	[[nodiscard]] CheckOutcome Check(uint64_t from, uint64_t to, std::span<const Range> ranges,
	                                 bool hull = false) const noexcept {
		CheckOutcome outcome;
		if (to <= from) {
			return outcome;
		}
		if (to - from >= Capacity || Generation() - from >= Capacity) {
			outcome.result = CheckResult::Overflow;
			return outcome;
		}
		for (auto generation = from + 1u; generation <= to; generation++) {
			Range  range;
			Source source = Source::Universe;
			bool   read   = false;
			for (uint32_t attempt = 0; attempt < 64u && !(read = Read(generation, range, source));
			     attempt++) {
				if (Generation() - generation >= Capacity) {
					break;
				}
			}
			outcome.entries++;
			if (!read) {
				outcome.result = CheckResult::Unknown;
				return outcome;
			}
			if (hull && (ranges.empty() || range.end <= ranges.front().begin ||
			             range.begin >= ranges.back().end)) {
				continue;
			}
			if (IntersectsAny(range, ranges)) {
				outcome.result          = CheckResult::Conflict;
				outcome.conflict_source = source;
				return outcome;
			}
		}
		return outcome;
	}

	// `ranges` sorted by begin, non-overlapping.
	[[nodiscard]] static bool IntersectsAny(Range range, std::span<const Range> ranges) noexcept {
		if (range.begin >= range.end) {
			return false;
		}
		// First range whose end is past range.begin.
		const auto it = std::lower_bound(ranges.begin(), ranges.end(), range.begin,
		                                 [](const Range& r, uint64_t value) { return r.end <= value; });
		return it != ranges.end() && it->begin < range.end;
	}

private:
	struct Entry {
		std::atomic<uint64_t> sequence {0};
		std::atomic<uint64_t> begin {0};
		std::atomic<uint64_t> end {0};
		std::atomic<uint8_t>  source {0};
	};

	alignas(64) std::atomic<uint64_t> m_generation {1};
	std::array<Entry, Capacity> m_entries {};
};

// KYTY_CERT_LOG_SUMMARY (drawPrep.cpp): a single-thread summary of a log, for a reader that asks
// many Checks whose intervals overlap (the draw-prep commits: every commit walks the transitions
// since its preparation began, so with a deep preparation window each entry was read by dozens of
// commits). The reader ingests every entry once, in generation order, into a sparse table holding,
// per 64 KiB block of guest address space, the newest generation of an entry touching it (entries
// wider than WideBlocks blocks, and universe entries, raise one "wide" generation instead). Then
// "no entry in (from, to] intersects these ranges" holds when every entry up to `to` was ingested
// without a gap since before `from`, the wide generation is <= from and every block the ranges
// touch holds a generation <= from: an entry in the interval intersecting a range would have raised
// one of them above `from`. Otherwise (a block or the wide generation is newer, an entry not yet
// readable, a gap) Clean() answers false and the caller runs the exact Check.
class Summary {
public:
	static constexpr uint32_t BlockShift    = 16u;
	static constexpr uint32_t LeafShift     = 36u; // 64 GiB per leaf
	static constexpr uint64_t BlocksPerLeaf = uint64_t {1} << (LeafShift - BlockShift);
	static constexpr uint64_t WideBlocks    = 4096; // 256 MiB

	explicit Summary(const Log& log) noexcept: m_log(log) {}
	~Summary() {
		for (auto* leaf: m_leaves) {
			std::free(leaf);
		}
	}
	Summary(const Summary&)            = delete;
	Summary& operator=(const Summary&) = delete;

	// True: no entry with a generation in (from, to] intersects `ranges` (sorted, non-overlapping).
	// False: unknown (the caller decides with Log::Check).
	[[nodiscard]] bool Clean(uint64_t from, uint64_t to, std::span<const Range> ranges) noexcept {
		if (to <= from) {
			return true;
		}
		if (!m_started) {
			// Entries before this point were never ingested: only later intervals qualify.
			m_started    = true;
			m_ingested   = m_log.Generation();
			m_valid_from = m_ingested;
		}
		if (from < m_valid_from || !Ingest(to) || m_wide > from) {
			return false;
		}
		for (const auto& range: ranges) {
			if (range.begin >= range.end) {
				continue;
			}
			if (range.end > (uint64_t {1} << 48u)) {
				return false;
			}
			for (auto block = range.begin >> BlockShift; block <= (range.end - 1u) >> BlockShift; block++) {
				const auto* leaf = m_leaves[block >> (LeafShift - BlockShift)];
				if (leaf != nullptr && leaf[block & (BlocksPerLeaf - 1u)] > from) {
					return false;
				}
			}
		}
		return true;
	}

	[[nodiscard]] uint64_t Ingested() const noexcept { return m_ingested; }

private:
	// Ingests the entries up to `to`; false when one is not readable yet (or no longer: then the
	// summary restarts after the gap).
	bool Ingest(uint64_t to) noexcept {
		while (m_ingested < to) {
			const auto generation = m_ingested + 1u;
			Range      range;
			Source     source = Source::Universe;
			if (!m_log.Read(generation, range, source)) {
				if (m_log.Generation() - generation >= Log::Capacity) {
					// Recycled before it was ingested: nothing before the current end can be
					// vouched for any more.
					m_ingested   = m_log.Generation();
					m_valid_from = m_ingested;
				}
				return false;
			}
			Note(range, generation);
			m_ingested = generation;
		}
		return true;
	}

	void Note(Range range, uint64_t generation) noexcept {
		if (range.begin >= range.end) {
			return;
		}
		const auto first = range.begin >> BlockShift;
		if (range.end > (uint64_t {1} << 48u) || ((range.end - 1u) >> BlockShift) - first >= WideBlocks) {
			m_wide = generation;
			return;
		}
		for (auto block = first; block <= (range.end - 1u) >> BlockShift; block++) {
			auto*& leaf = m_leaves[block >> (LeafShift - BlockShift)];
			if (leaf == nullptr) {
				leaf = static_cast<uint64_t*>(std::calloc(BlocksPerLeaf, sizeof(uint64_t)));
				if (leaf == nullptr) {
					m_wide = generation; // no table: treat as touching everything
					return;
				}
			}
			leaf[block & (BlocksPerLeaf - 1u)] = generation; // generations only grow
		}
	}

	const Log&                                         m_log;
	bool                                               m_started    = false;
	uint64_t                                           m_ingested   = 0;
	uint64_t                                           m_valid_from = 0;
	uint64_t                                           m_wide       = 0;
	std::array<uint64_t*, 1u << (48u - LeafShift)>    m_leaves {};
};

// The process-wide log (one per coherence domain: the guest GPU).
inline Log g_log;

// Transitions always record their entries. The log-mode draw-prep certificate reads them (the
// default, with draw prep on by default), and so does the value certificate's audit. A log that
// only bumped its generation answers Unknown for every interval containing a transition, which
// fails every such certificate: in U51 this switch still required KYTY_DRAW_PREP=parallel in the
// environment after draw prep became the default, and 170-210 draws per flip fell back to the
// serial preparation. Recording costs a few stores per transition; nothing else depends on the
// environment, so the log cannot disagree with DrawPrep::GetMode() or GetCertMode() again.
[[nodiscard]] constexpr bool LogReadersEnabled() noexcept {
	return true;
}

[[nodiscard]] inline uint64_t Generation() noexcept {
	return g_log.Generation();
}

inline uint64_t Append(Range range, Source source) noexcept {
	return g_log.Append(range, source);
}

inline uint64_t Append(uint64_t address, uint64_t size, Source source) noexcept {
	return Append(MakeRange(address, size), source);
}

// Emulator writes of backing bytes outside a publication change content, not cleanliness; only
// the log-mode certificate needs them (they also move the generation the clean-verdict cache
// shares). Call after the bytes are written.
inline void NoteContentWrite(uint64_t address, uint64_t size, Source source) noexcept {
	Append(address, size, source);
}

} // namespace Libs::Graphics::Coherence

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_COHERENCELOG_H_
