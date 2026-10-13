#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_READSET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_READSET_H_

#include "graphics/host_gpu/coherenceLog.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>
#include <xxhash.h>

// Draw-prep S4: the read set of one speculative draw preparation, and its certificate check.
//
// A preparation reads guest memory only through LibKernel::Memory::TryReadGpuCleanBacking. While
// a Recorder is active on the preparing thread, that function routes every read here: the bytes
// are copied from the backing view and recorded (address, size, bytes). Any read that cannot be
// served (not provably clean, unmapped, read-set overflow) fails the whole preparation.
//
// At commit, on the GPU thread, Validate() re-reads every coalesced range with the coherent
// clean-backing read and compares it with the recorded bytes. Correctness argument: the
// preparation is a deterministic function of (register snapshot, shader map generation, the
// values returned by its reads, append-only program-cache state). If every range is clean for a
// backing read now and holds the recorded bytes, then the serial path run now would issue the same
// first read (same registers), get the same value, and by induction issue the same reads in the
// same order and compute the same outputs. Guest bytes that changed and changed back in between
// are indistinguishable from unchanged ones, which is exactly what the serial path would observe.
//
// Digest reads (RecordDigest): a read whose bytes the preparation only hashes, with the XXH3-64
// the serial path uses as the identity of those bytes (headerless shader code: the program key).
// It is certified by that digest instead of its bytes: Validate() re-reads the range with the same
// clean read and compares digests. Equal digests make the serial path compute the same identity
// from the bytes it would read now, and nothing else of the preparation depends on those bytes.
namespace Libs::Graphics::DrawPrep {

enum class ReadFailure : uint8_t {
	None,
	Unclean,      // not provably clean (exact predicate on the GPU thread, hint on workers)
	Backing,      // no backing translation
	Overflow,     // too many reads or bytes to certify
	Inconsistent, // two reads of the same byte returned different values
	Uncertified,  // the preparation needed a read or a decision outside the certificate
};

enum class ValidateResult : uint8_t {
	Ok,
	Unclean, // a range is not clean for a backing read now
	Changed, // a range is clean but its bytes differ from the recorded ones
};

class ReadSet {
public:
	static constexpr uint32_t MaxReads = 2048;
	// Room for the code of both stages (headerless shaders are hashed through the recorder, see
	// shader.cpp) besides the metadata and resource reads.
	static constexpr uint32_t MaxBytes = 256u * 1024u;
	static constexpr uint32_t MaxDigests = 64;
	static constexpr uint64_t PageSize = 4096;

	void Reset() noexcept {
		m_reads.clear();
		m_bytes.clear();
		m_ranges.clear();
		m_range_offsets.clear();
		m_merged.clear();
		m_digests.clear();
		m_digest_ranges.clear();
		m_failure  = ReadFailure::None;
		m_finished = false;
	}

	// Records that the `size` bytes read at `address` have the XXH3-64 digest `digest` (see the
	// header comment). False (and a failure) when over the limits.
	bool RecordDigest(uint64_t address, uint64_t size, uint64_t digest) {
		if (m_failure != ReadFailure::None) {
			return false;
		}
		if (size == 0) {
			return true;
		}
		if (m_digests.size() >= MaxDigests || address > UINT64_MAX - size) {
			Fail(ReadFailure::Overflow);
			return false;
		}
		m_digests.push_back({address, size, digest});
		m_digest_ranges.push_back({address, address + size});
		return true;
	}
	// The ranges certified by digest (in record order; they may overlap byte ranges).
	[[nodiscard]] std::span<const Coherence::Range> DigestRanges() const noexcept {
		return m_digest_ranges;
	}

	// Records `size` bytes read at `address`. False (and a failure) when over the limits.
	bool Record(uint64_t address, const void* data, uint64_t size) {
		if (m_failure != ReadFailure::None) {
			return false;
		}
		if (size == 0) {
			return true;
		}
		if (m_reads.size() >= MaxReads || size > MaxBytes - m_bytes.size() ||
		    address > UINT64_MAX - size) {
			Fail(ReadFailure::Overflow);
			return false;
		}
		const auto offset = static_cast<uint32_t>(m_bytes.size());
		m_bytes.resize(m_bytes.size() + size);
		std::memcpy(m_bytes.data() + offset, data, size);
		// KYTY_DRAW_PREP_DIRECT_READS: a read that starts where the previous one ended (and not at
		// a page start, where Finish keeps ranges apart) extends it; Finish would merge the two
		// into the same range, so the certificate is the same.
		if (m_coalesce && !m_reads.empty()) {
			auto& last = m_reads.back();
			if (last.address + last.size == address && last.offset + last.size == offset &&
			    (address & (PageSize - 1u)) != 0) {
				last.size += static_cast<uint32_t>(size);
				return true;
			}
		}
		m_reads.push_back({address, static_cast<uint32_t>(size), offset});
		return true;
	}
	// KYTY_DRAW_PREP_DIRECT_READS: whether Record extends contiguous reads (set per preparation).
	void SetCoalesce(bool coalesce) noexcept { m_coalesce = coalesce; }

	void Fail(ReadFailure failure) noexcept {
		if (m_failure == ReadFailure::None) {
			m_failure = failure;
		}
	}
	[[nodiscard]] ReadFailure Failure() const noexcept { return m_failure; }
	[[nodiscard]] bool        Failed() const noexcept { return m_failure != ReadFailure::None; }
	[[nodiscard]] size_t      ReadCount() const noexcept { return m_reads.size(); }
	[[nodiscard]] size_t      ByteCount() const noexcept { return m_bytes.size(); }

	// Coalesces the recorded reads into sorted, non-overlapping ranges. Overlapping reads merge,
	// and so do touching ones unless they meet at a 4 KiB boundary (two guest mappings can meet
	// there, and one read across it could fail where the two separate reads succeeded).
	// Overlapping reads must agree on every shared byte, otherwise the preparation observed a
	// concurrent change and fails (Inconsistent).
	bool Finish() {
		if (m_failure != ReadFailure::None) {
			return false;
		}
		m_order.resize(m_reads.size());
		std::iota(m_order.begin(), m_order.end(), 0u);
		const auto before = [this](uint32_t a, uint32_t b) {
			return m_reads[a].address < m_reads[b].address ||
			       (m_reads[a].address == m_reads[b].address && m_reads[a].size > m_reads[b].size);
		};
		// Reads often come in address order: then the identity order is the sorted one.
		if (!std::is_sorted(m_order.begin(), m_order.end(), before)) {
			std::sort(m_order.begin(), m_order.end(), before);
		}
		m_ranges.clear();
		m_range_offsets.clear();
		m_merged.clear();
		for (const auto index: m_order) {
			const auto& read  = m_reads[index];
			const auto* bytes = m_bytes.data() + read.offset;
			const auto  end   = read.address + read.size;
			if (!m_ranges.empty() &&
			    (read.address < m_ranges.back().end ||
			     (read.address == m_ranges.back().end && (read.address & (PageSize - 1u)) != 0))) {
				auto&      range       = m_ranges.back();
				const auto base        = m_range_offsets.back();
				const auto overlap_end = std::min(end, range.end);
				if (overlap_end > read.address &&
				    std::memcmp(m_merged.data() + base + (read.address - range.begin), bytes,
				                overlap_end - read.address) != 0) {
					Fail(ReadFailure::Inconsistent);
					return false;
				}
				if (end > range.end) {
					const auto tail = end - range.end;
					m_merged.insert(m_merged.end(), bytes + (range.end - read.address),
					                bytes + (range.end - read.address) + tail);
					range.end = end;
				}
				continue;
			}
			m_ranges.push_back({read.address, end});
			m_range_offsets.push_back(static_cast<uint32_t>(m_merged.size()));
			m_merged.insert(m_merged.end(), bytes, bytes + read.size);
		}
		m_finished = true;
		return true;
	}

	[[nodiscard]] bool Finished() const noexcept { return m_finished; }
	// Sorted, non-overlapping; valid after a successful Finish().
	[[nodiscard]] std::span<const Coherence::Range> Ranges() const noexcept { return m_ranges; }

	// Every range the certificate covers (byte ranges and digest ranges), sorted by begin, with
	// overlapping and touching ranges merged: the form Coherence::Log::Check requires. Without
	// digest ranges it is Ranges() itself. A pure function of a finished read set, so the
	// preparing thread builds it once (KYTY_DRAW_PREP_CERT_RANGES) instead of every commit.
	// Only for the coherence-log check: touching ranges merge across 4 KiB boundaries here, which
	// a clean-read verdict must not do (AllClean keeps the separate lists).
	void BuildCertificate(std::vector<Coherence::Range>& out) const {
		out.assign(m_ranges.begin(), m_ranges.end());
		if (m_digest_ranges.empty()) {
			return;
		}
		out.insert(out.end(), m_digest_ranges.begin(), m_digest_ranges.end());
		std::sort(out.begin(), out.end(), [](const Coherence::Range& a, const Coherence::Range& b) {
			return a.begin < b.begin;
		});
		size_t merged = 0;
		for (const auto& range: out) {
			if (merged != 0 && range.begin <= out[merged - 1].end) {
				out[merged - 1].end = std::max(out[merged - 1].end, range.end);
			} else {
				out[merged++] = range;
			}
		}
		out.resize(merged);
	}
	[[nodiscard]] std::span<const uint8_t> RangeBytes(size_t index) const noexcept {
		const auto& range = m_ranges[index];
		return {m_merged.data() + m_range_offsets[index], range.end - range.begin};
	}

	// read(address, destination, size) -> bool: the coherent clean-backing read of the caller.
	template <typename Read>
	[[nodiscard]] ValidateResult Validate(Read&& read, std::vector<uint8_t>& scratch) const {
		static_assert(std::is_invocable_r_v<bool, Read&, uint64_t, void*, uint64_t>);
		for (size_t index = 0; index < m_ranges.size(); index++) {
			const auto bytes = RangeBytes(index);
			scratch.resize(bytes.size());
			if (!read(m_ranges[index].begin, scratch.data(), static_cast<uint64_t>(bytes.size()))) {
				return ValidateResult::Unclean;
			}
			if (std::memcmp(scratch.data(), bytes.data(), bytes.size()) != 0) {
				return ValidateResult::Changed;
			}
		}
		for (const auto& digest: m_digests) {
			scratch.resize(digest.size);
			if (!read(digest.address, scratch.data(), digest.size)) {
				return ValidateResult::Unclean;
			}
			if (XXH3_64bits(scratch.data(), digest.size) != digest.digest) {
				return ValidateResult::Changed;
			}
		}
		return ValidateResult::Ok;
	}

	// Validate without copying the bytes out (KYTY_BACKING_INPLACE): compare(address, expected,
	// size) -> ValidateResult for one byte range (Ok: clean and equal to `expected`; Unclean: not
	// clean for a backing read, or no backing; Changed: clean but different), and hash(address,
	// size, digest) -> bool (false: Unclean) for a digest range. Same checks in the same order as
	// Validate, so the same result for the same guest bytes.
	template <typename Compare, typename Hash>
	[[nodiscard]] ValidateResult ValidateInPlace(Compare&& compare, Hash&& hash) const {
		static_assert(std::is_invocable_r_v<ValidateResult, Compare&, uint64_t, const uint8_t*,
		                                    uint64_t>);
		static_assert(std::is_invocable_r_v<bool, Hash&, uint64_t, uint64_t, uint64_t&>);
		for (size_t index = 0; index < m_ranges.size(); index++) {
			const auto bytes  = RangeBytes(index);
			const auto result = compare(m_ranges[index].begin, bytes.data(),
			                            static_cast<uint64_t>(bytes.size()));
			if (result != ValidateResult::Ok) {
				return result;
			}
		}
		for (const auto& digest: m_digests) {
			uint64_t value = 0;
			if (!hash(digest.address, digest.size, value)) {
				return ValidateResult::Unclean;
			}
			if (value != digest.digest) {
				return ValidateResult::Changed;
			}
		}
		return ValidateResult::Ok;
	}

	// is_clean(address, size) -> bool: the clean verdict alone (no bytes compared).
	template <typename IsClean>
	[[nodiscard]] bool AllClean(IsClean&& is_clean) const {
		static_assert(std::is_invocable_r_v<bool, IsClean&, uint64_t, uint64_t>);
		for (const auto& range: m_ranges) {
			if (!is_clean(range.begin, range.end - range.begin)) {
				return false;
			}
		}
		for (const auto& range: m_digest_ranges) {
			if (!is_clean(range.begin, range.end - range.begin)) {
				return false;
			}
		}
		return true;
	}

private:
	struct Read {
		uint64_t address = 0;
		uint32_t size    = 0;
		uint32_t offset  = 0;
	};
	struct Digest {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t digest  = 0;
	};

	std::vector<Read>             m_reads;
	std::vector<uint8_t>          m_bytes;
	std::vector<uint32_t>         m_order;
	std::vector<Coherence::Range> m_ranges;
	std::vector<uint32_t>         m_range_offsets;
	std::vector<uint8_t>          m_merged;
	std::vector<Digest>           m_digests;
	std::vector<Coherence::Range> m_digest_ranges;
	ReadFailure                   m_failure  = ReadFailure::None;
	bool                          m_finished = false;
	bool                          m_coalesce = false;
};

// The read routing of a preparing thread. `exact`: the thread is the GPU thread and uses the
// exact clean predicate; otherwise (DrawPrep workers) a conservative thread-safe hint gates reads
// and the certificate is checked at commit.
struct Recorder {
	ReadSet* reads = nullptr;
	bool     exact = false;
	// KYTY_DRAW_PREP_DIRECT_READS (kernel/memory.cpp): the last guest page this preparation read
	// whole-page clean by the workers' hint, its host backing bytes and the backing map generation
	// they were found under. Reads inside it copy from there (and are recorded as usual).
	uint64_t       direct_page       = UINT64_MAX;
	const uint8_t* direct_host       = nullptr;
	uint64_t       direct_generation = 0;
};

inline thread_local Recorder* t_recorder = nullptr;

// KYTY_DRAW_PREP_DIRECT_READS (drawPrep.cpp, kernel/memory.cpp): 0 off, 1 on, 2 verify, 3 exit.
[[nodiscard]] int  DirectReadsMode();
[[nodiscard]] bool DirectReadsEnabled();

// Set on DrawPrep worker threads, which must never reach the scheduler or the caches
// (CommandScheduler::CheckActive stops the emulator if one does).
inline thread_local bool t_worker_thread = false;

[[nodiscard]] inline bool IsWorkerThread() noexcept {
	return t_worker_thread;
}

[[nodiscard]] inline Recorder* ActiveRecorder() noexcept {
	return t_recorder;
}

// True while a draw preparation runs on this thread: guest reads must not fall back to the
// guest mapping (they fail the preparation instead).
[[nodiscard]] inline bool Speculative() noexcept {
	return t_recorder != nullptr;
}

// A preparation is running on this thread and has already failed: callers stop early instead of
// continuing on possibly stale or missing inputs (the result is discarded anyway).
[[nodiscard]] inline bool SpeculativeFailed() noexcept {
	return t_recorder != nullptr && t_recorder->reads->Failed();
}

// Diagnostics: what the preparing thread's current guest reads are for (a string literal, or
// null), reported with reads refused as not provably clean (HangTrace unclean.csv).
inline thread_local const char* t_read_purpose = nullptr;

[[nodiscard]] inline const char* ReadPurpose() noexcept {
	return t_read_purpose;
}

class ScopedReadPurpose {
public:
	explicit ScopedReadPurpose(const char* purpose) noexcept: m_previous(t_read_purpose) {
		t_read_purpose = purpose;
	}
	~ScopedReadPurpose() { t_read_purpose = m_previous; }
	ScopedReadPurpose(const ScopedReadPurpose&)            = delete;
	ScopedReadPurpose& operator=(const ScopedReadPurpose&) = delete;

	void Set(const char* purpose) noexcept { t_read_purpose = purpose; }

private:
	const char* m_previous;
};

// Marks the active preparation as failed (no-op outside a preparation).
inline void FailActive(ReadFailure failure) noexcept {
	if (t_recorder != nullptr) {
		t_recorder->reads->Fail(failure);
	}
}

class RecordScope {
public:
	explicit RecordScope(Recorder& recorder) noexcept: m_previous(t_recorder) {
		t_recorder = &recorder;
	}
	~RecordScope() { t_recorder = m_previous; }
	RecordScope(const RecordScope&)            = delete;
	RecordScope& operator=(const RecordScope&) = delete;

private:
	Recorder* m_previous;
};

} // namespace Libs::Graphics::DrawPrep

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_READSET_H_
