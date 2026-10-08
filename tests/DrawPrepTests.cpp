// Draw-prep S4-S6 unit tests: the coherence log, read-set coalescing and certificate checks,
// and (S6) the preparation window ring, worker parking and the producer's work stealing.
// --measure-worker-gate [reps]: the parking/stealing configuration sweep (not a test).
#include "graphics/host_gpu/coherenceLog.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/renderer/drawPrep/packetClass.h"
#include "graphics/host_gpu/renderer/drawPrep/readSet.h"
#include "graphics/host_gpu/renderer/drawPrep/window.h"
#include "graphics/host_gpu/renderer/drawPrep/workerGate.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "FAILED: %s\n", text);
		g_failures++;
	}
}

// ---------------------------------------------------------------------------------------------
// CoherenceLog

void TestLogEmptyIntervalIsClean() {
	auto  log_owner = std::make_unique<Coherence::Log>();
	auto& log       = *log_owner;
	const auto     g = log.Generation();
	Check(g == 1, "log generation starts at 1");
	const std::array<Coherence::Range, 1> ranges {{{0x1000, 0x2000}}};
	Check(log.Check(g, g, ranges).result == Coherence::CheckResult::Clean,
	      "empty interval is clean");
	Check(log.Check(g + 5, g, ranges).result == Coherence::CheckResult::Clean,
	      "reversed interval is clean");
}

void TestLogIntersection() {
	auto  log_owner = std::make_unique<Coherence::Log>();
	auto& log       = *log_owner;
	const auto     g0 = log.Generation();
	log.Append(Coherence::MakeRange(0x5000, 0x100), Coherence::Source::BufferDirtyAdd);
	log.Append(Coherence::MakeRange(0x9000, 0x10), Coherence::Source::ImageGpuModified);
	const auto g1 = log.Generation();
	Check(g1 == g0 + 2, "each append claims one generation");

	const std::array<Coherence::Range, 2> disjoint {{{0x1000, 0x5000}, {0x5100, 0x9000}}};
	auto outcome = log.Check(g0, g1, disjoint);
	Check(outcome.result == Coherence::CheckResult::Clean, "touching ranges do not intersect");
	Check(outcome.entries == 2, "every entry in the interval is examined");

	const std::array<Coherence::Range, 1> hit {{{0x900f, 0x9011}}};
	outcome = log.Check(g0, g1, hit);
	Check(outcome.result == Coherence::CheckResult::Conflict &&
	          outcome.conflict_source == Coherence::Source::ImageGpuModified,
	      "last byte overlap is a conflict with its source");

	// Only the second entry is newer than g0 + 1.
	const std::array<Coherence::Range, 1> first_only {{{0x5000, 0x5001}}};
	Check(log.Check(g0 + 1, g1, first_only).result == Coherence::CheckResult::Clean,
	      "entries at or before the certificate generation are ignored");
	Check(log.Check(g0, g1, first_only).result == Coherence::CheckResult::Conflict,
	      "an entry after the certificate generation conflicts");

	log.Append(Coherence::Universe, Coherence::Source::Universe);
	Check(log.Check(g1, log.Generation(), disjoint).result == Coherence::CheckResult::Conflict,
	      "a universe entry conflicts with everything");
	const std::array<Coherence::Range, 0> none {};
	Check(log.Check(g1, log.Generation(), none).result == Coherence::CheckResult::Clean,
	      "an empty read set never conflicts");
}

void TestLogBumpWithoutReaders() {
	// A log nobody reads only counts generations (Bump); its entries are never readable, and a
	// Check over such an interval is conservative (Unknown), never Clean.
	auto       log = std::make_unique<Coherence::Log>();
	const auto g0  = log->Generation();
	Check(log->Bump() == g0 + 1 && log->Generation() == g0 + 1, "bump claims one generation");
	Coherence::Range  range;
	Coherence::Source source {};
	Check(!log->Read(g0 + 1, range, source), "a bumped generation has no entry");
	const std::array<Coherence::Range, 1> ranges {{{0x1000, 0x2000}}};
	Check(log->Check(g0, log->Generation(), ranges).result == Coherence::CheckResult::Unknown,
	      "an interval of bumped generations is not certified clean");
}

// The process-wide log records every transition whatever the environment. Draw prep and its
// log-mode certificate are the defaults; when the log only bumped its generation unless
// KYTY_DRAW_PREP=parallel was spelled out, every certificate with a transition in its interval
// failed as Unknown (U51: 170-210 extra serial preparations per flip). No variable is set here.
void TestGlobalLogAlwaysRecords() {
	static_assert(Coherence::LogReadersEnabled(), "the global coherence log must always record");
	const auto added = Coherence::MakeRange(0x7000, 0x40);
	const auto g0    = Coherence::Generation();
	const auto g1    = Coherence::Append(added, Coherence::Source::Test);
	Check(g1 == g0 + 1, "a global append claims one generation");
	Coherence::Range  range;
	Coherence::Source source {};
	Check(Coherence::g_log.Read(g1, range, source) && range == added &&
	          source == Coherence::Source::Test,
	      "the global log recorded the transition");
	const std::array<Coherence::Range, 1> touched {{{0x7010, 0x7020}}};
	const std::array<Coherence::Range, 1> elsewhere {{{0x8000, 0x9000}}};
	Check(Coherence::g_log.Check(g0, Coherence::Generation(), touched).result ==
	          Coherence::CheckResult::Conflict,
	      "a recorded transition conflicts with the ranges it touches");
	Check(Coherence::g_log.Check(g0, Coherence::Generation(), elsewhere).result ==
	          Coherence::CheckResult::Clean,
	      "and with no other range");
	const auto g2 = Coherence::Generation();
	Coherence::NoteContentWrite(0x7100, 8, Coherence::Source::CpWrite);
	Check(Coherence::Generation() == g2 + 1 && Coherence::g_log.Read(g2 + 1, range, source) &&
	          source == Coherence::Source::CpWrite,
	      "emulator content writes are recorded too");
}

void TestLogEmptyRangeNeverIntersects() {
	auto  log_owner = std::make_unique<Coherence::Log>();
	auto& log       = *log_owner;
	const auto     g0 = log.Generation();
	log.Append(Coherence::MakeRange(0x2000, 0), Coherence::Source::Test);
	const std::array<Coherence::Range, 1> ranges {{{0x1000, 0x3000}}};
	Check(log.Check(g0, log.Generation(), ranges).result == Coherence::CheckResult::Clean,
	      "a zero-sized entry does not intersect");
	Check(Coherence::MakeRange(UINT64_MAX - 4, 100).end == UINT64_MAX,
	      "range end saturates instead of wrapping");
}

void TestLogOverflow() {
	auto       log = std::make_unique<Coherence::Log>();
	const auto g0  = log->Generation();
	for (uint64_t i = 0; i < Coherence::Log::Capacity; i++) {
		log->Append(Coherence::MakeRange(0x100000 + i * 16, 16), Coherence::Source::Test);
	}
	const std::array<Coherence::Range, 1> ranges {{{0x10, 0x20}}};
	Check(log->Check(g0, log->Generation(), ranges).result == Coherence::CheckResult::Overflow,
	      "an interval as long as the ring overflows");
	// A recent interval is still exact after the ring wrapped.
	const auto g1 = log->Generation();
	log->Append(Coherence::MakeRange(0x10, 1), Coherence::Source::Test);
	Check(log->Check(g1, log->Generation(), ranges).result == Coherence::CheckResult::Conflict,
	      "entries after a wrap are exact");
	Coherence::Range  range;
	Coherence::Source source {};
	Check(!log->Read(g0 + 1, range, source), "a recycled slot is not readable as an old generation");
	Check(log->Read(log->Generation(), range, source) && range.begin == 0x10,
	      "the newest entry is readable");
}

// Every entry thread t appends is [((t + 1) << 20) + 8 i, +8) with Source::Test: a torn read
// (payload words from two appends) breaks this shape.
bool WellFormedEntry(Coherence::Range range, Coherence::Source source, uint32_t threads) {
	const auto window = range.begin >> 20u;
	return source == Coherence::Source::Test && range.end == range.begin + 8u &&
	       (range.begin & 7u) == 0u && window >= 1u && window <= threads &&
	       ((range.begin - (window << 20u)) >> 3u) < (1u << 17u);
}

// Appenders on several threads while readers read recent generations concurrently. Readers must
// only ever accept complete entries; with `laps` > 1 the ring wraps several times, so slots are
// rewritten while being read.
void RunConcurrentLog(Coherence::Log& log, uint32_t threads_count, uint32_t per_thread,
                      std::atomic<uint32_t>& torn, std::atomic<uint64_t>& reads_ok) {
	std::atomic<bool>        done {false};
	std::vector<std::thread> readers;
	for (uint32_t r = 0; r < 2; r++) {
		readers.emplace_back([&, r] {
			uint64_t salt = 0x9e3779b97f4a7c15ull * (r + 1u);
			while (!done.load(std::memory_order_acquire)) {
				const auto newest = log.Generation();
				salt ^= salt << 13u;
				salt ^= salt >> 7u;
				salt ^= salt << 17u;
				const auto back = salt % 64u;
				if (newest < Coherence::Log::FirstEntryGeneration + back) {
					continue;
				}
				Coherence::Range  range;
				Coherence::Source source {};
				if (log.Read(newest - back, range, source)) {
					reads_ok.fetch_add(1, std::memory_order_relaxed);
					if (!WellFormedEntry(range, source, threads_count)) {
						torn.fetch_add(1, std::memory_order_relaxed);
					}
				}
			}
		});
	}
	std::vector<std::thread> writers;
	for (uint32_t t = 0; t < threads_count; t++) {
		writers.emplace_back([&, t] {
			const uint64_t base = (uint64_t {t} + 1u) << 20u;
			for (uint32_t i = 0; i < per_thread; i++) {
				log.Append(Coherence::MakeRange(base + uint64_t {i} * 8u, 8),
				           Coherence::Source::Test);
			}
		});
	}
	for (auto& thread: writers) {
		thread.join();
	}
	done.store(true, std::memory_order_release);
	for (auto& thread: readers) {
		thread.join();
	}
}

void TestLogConcurrentAppends() {
	auto                    log       = std::make_unique<Coherence::Log>();
	constexpr uint32_t      Threads   = 8;
	constexpr uint32_t      PerThread = 1000;
	const auto              g0        = log->Generation();
	std::atomic<uint32_t>   torn {0};
	std::atomic<uint64_t>   reads_ok {0};
	RunConcurrentLog(*log, Threads, PerThread, torn, reads_ok);
	Check(torn.load() == 0, "concurrent readers never accept a torn entry");
	const auto g1 = log->Generation();
	Check(g1 - g0 == Threads * PerThread, "concurrent appends claim distinct generations");
	// Afterwards every entry is readable and falls in exactly one window.
	std::array<uint32_t, Threads> per_window {};
	bool                          all_read = true;
	for (auto g = g0 + 1; g <= g1; g++) {
		Coherence::Range  range;
		Coherence::Source source {};
		if (!log->Read(g, range, source) || range.end - range.begin != 8) {
			all_read = false;
			continue;
		}
		const auto window = (range.begin >> 20u) - 1u;
		if (window < Threads) {
			per_window[window]++;
		}
	}
	Check(all_read, "every concurrently appended entry is readable and intact");
	bool balanced = true;
	for (const auto count: per_window) {
		balanced &= count == PerThread;
	}
	Check(balanced, "every thread's entries are present exactly once");
	for (uint32_t t = 0; t < Threads; t++) {
		const uint64_t base = (uint64_t {t} + 1u) << 20u;
		const std::array<Coherence::Range, 1> probe {{{base + 8u * 500u, base + 8u * 500u + 1u}}};
		Check(log->Check(g0, g1, probe).result == Coherence::CheckResult::Conflict,
		      "a concurrently appended entry is found by Check");
	}
}

void TestLogConcurrentWrap() {
	// About five laps of the ring: slots are rewritten in turn while readers read them.
	auto                  log        = std::make_unique<Coherence::Log>();
	constexpr uint32_t    Threads    = 8;
	constexpr uint32_t    PerThread  = static_cast<uint32_t>(Coherence::Log::Capacity * 5u / 8u);
	const auto            g0         = log->Generation();
	std::atomic<uint32_t> torn {0};
	std::atomic<uint64_t> reads_ok {0};
	RunConcurrentLog(*log, Threads, PerThread, torn, reads_ok);
	Check(torn.load() == 0, "readers never accept a torn entry while the ring wraps");
	const auto g1 = log->Generation();
	Check(g1 - g0 == uint64_t {Threads} * PerThread, "wrapping appends claim distinct generations");
	bool last_lap = true;
	for (auto g = g1 - Coherence::Log::Capacity + 1u; g <= g1; g++) {
		Coherence::Range  range;
		Coherence::Source source {};
		last_lap &= log->Read(g, range, source) && WellFormedEntry(range, source, Threads);
	}
	Check(last_lap, "the last lap of a wrapped ring is readable and intact");
	Check(log->Check(g0, g1, std::array<Coherence::Range, 1> {{{0, 1}}}).result ==
	          Coherence::CheckResult::Overflow,
	      "an interval longer than the ring overflows");
}

void TestIntersectsAny() {
	const std::array<Coherence::Range, 3> ranges {{{10, 20}, {30, 40}, {50, 60}}};
	Check(!Coherence::Log::IntersectsAny({20, 30}, ranges), "gap between ranges");
	Check(Coherence::Log::IntersectsAny({19, 21}, ranges), "overlaps the first range end");
	Check(Coherence::Log::IntersectsAny({0, 100}, ranges), "covers all");
	Check(!Coherence::Log::IntersectsAny({60, 70}, ranges), "after the last range");
	Check(!Coherence::Log::IntersectsAny({0, 10}, ranges), "before the first range");
	Check(Coherence::Log::IntersectsAny({59, 60}, ranges), "last byte");
	Check(!Coherence::Log::IntersectsAny({35, 35}, ranges), "empty query");
}

// ---------------------------------------------------------------------------------------------
// ReadSet

struct FakeMemory {
	std::array<uint8_t, 4096> bytes {};
	uint64_t                  base          = 0x10000;
	uint64_t                  dirty_begin   = 0;
	uint64_t                  dirty_end     = 0;

	bool Read(uint64_t address, void* data, uint64_t size) const {
		if (address < base || address + size > base + bytes.size()) {
			return false;
		}
		if (address < dirty_end && dirty_begin < address + size) {
			return false;
		}
		std::memcpy(data, bytes.data() + (address - base), size);
		return true;
	}
};

void RecordFrom(DrawPrep::ReadSet& set, const FakeMemory& memory, uint64_t address,
                uint64_t size) {
	std::vector<uint8_t> data(size);
	Check(memory.Read(address, data.data(), size), "fake memory read");
	Check(set.Record(address, data.data(), size), "record");
}

void TestReadSetCoalesces() {
	FakeMemory memory;
	for (size_t i = 0; i < memory.bytes.size(); i++) {
		memory.bytes[i] = static_cast<uint8_t>(i * 7u + 3u);
	}
	DrawPrep::ReadSet set;
	RecordFrom(set, memory, 0x10100, 16);
	RecordFrom(set, memory, 0x10000, 8);
	RecordFrom(set, memory, 0x10108, 16); // overlaps the first read
	RecordFrom(set, memory, 0x10008, 4);  // touches the second read
	RecordFrom(set, memory, 0x10104, 4);  // contained in the first read
	RecordFrom(set, memory, 0x10800, 4);
	Check(set.Finish(), "consistent reads finish");
	const auto ranges = set.Ranges();
	Check(ranges.size() == 3, "reads coalesce into three ranges");
	Check(ranges.size() == 3 && ranges[0] == Coherence::Range {0x10000, 0x1000c} &&
	          ranges[1] == Coherence::Range {0x10100, 0x10118} &&
	          ranges[2] == Coherence::Range {0x10800, 0x10804},
	      "coalesced range bounds");
	bool bytes_match = true;
	for (size_t i = 0; i < ranges.size(); i++) {
		const auto bytes = set.RangeBytes(i);
		bytes_match &= std::memcmp(bytes.data(), memory.bytes.data() + (ranges[i].begin - memory.base),
		                           bytes.size()) == 0;
	}
	Check(bytes_match, "coalesced bytes equal memory");

	std::vector<uint8_t> scratch;
	const auto read = [&](uint64_t address, void* data, uint64_t size) {
		return memory.Read(address, data, size);
	};
	Check(set.Validate(read, scratch) == DrawPrep::ValidateResult::Ok,
	      "unchanged clean memory validates");
	memory.bytes[0x110] ^= 0xffu;
	Check(set.Validate(read, scratch) == DrawPrep::ValidateResult::Changed,
	      "a changed byte fails validation");
	memory.bytes[0x110] ^= 0xffu;
	memory.bytes[0x200] ^= 0xffu; // outside every range
	Check(set.Validate(read, scratch) == DrawPrep::ValidateResult::Ok,
	      "a change outside the read set is irrelevant");
	memory.dirty_begin = 0x10802;
	memory.dirty_end   = 0x10803;
	Check(set.Validate(read, scratch) == DrawPrep::ValidateResult::Unclean,
	      "an unclean byte in a range fails validation");
	memory.dirty_begin = memory.dirty_end = 0;
	Check(set.AllClean([&](uint64_t address, uint64_t size) {
		      std::vector<uint8_t> tmp(size);
		      return memory.Read(address, tmp.data(), size);
	      }),
	      "AllClean on clean memory");
}

void TestReadSetPageBoundary() {
	DrawPrep::ReadSet set;
	const std::array<uint8_t, 16> bytes {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	// Touching at a 4 KiB boundary: two ranges (two mappings may meet there).
	Check(set.Record(0x20ff8, bytes.data(), 8), "record below the boundary");
	Check(set.Record(0x21000, bytes.data() + 8, 8), "record above the boundary");
	// Touching inside a page: one range.
	Check(set.Record(0x21008, bytes.data(), 4), "record touching inside the page");
	// One read across a boundary stays one range.
	Check(set.Record(0x22ffc, bytes.data(), 8), "record across the next boundary");
	Check(set.Finish(), "page-boundary reads finish");
	const auto ranges = set.Ranges();
	Check(ranges.size() == 3 && ranges[0] == Coherence::Range {0x20ff8, 0x21000} &&
	          ranges[1] == Coherence::Range {0x21000, 0x2100c} &&
	          ranges[2] == Coherence::Range {0x22ffc, 0x23004},
	      "ranges split only at touching page boundaries");
	Check(set.RangeBytes(1).size() == 12 && set.RangeBytes(1)[8] == 1,
	      "bytes of a range merged inside a page");
}

void TestReadSetInconsistent() {
	DrawPrep::ReadSet set;
	const uint32_t    a = 0x11111111u;
	const uint32_t    b = 0x22222222u;
	Check(set.Record(0x2000, &a, 4), "record a");
	Check(set.Record(0x2000, &b, 4), "record b");
	Check(!set.Finish(), "two different values at one address do not finish");
	Check(set.Failure() == DrawPrep::ReadFailure::Inconsistent, "inconsistent failure reason");

	DrawPrep::ReadSet partial;
	const std::array<uint8_t, 8> wide {1, 2, 3, 4, 5, 6, 7, 8};
	const std::array<uint8_t, 2> narrow {4, 9};
	Check(partial.Record(0x3000, wide.data(), wide.size()), "record wide");
	Check(partial.Record(0x3003, narrow.data(), narrow.size()), "record narrow");
	Check(!partial.Finish(), "a partially overlapping read that disagrees fails");
}

void TestReadSetLimits() {
	DrawPrep::ReadSet set;
	std::vector<uint8_t> big(DrawPrep::ReadSet::MaxBytes);
	Check(set.Record(0x100000, big.data(), big.size()), "a read up to the byte limit fits");
	const uint8_t one = 1;
	Check(!set.Record(0x200000, &one, 1), "one byte over the limit is refused");
	Check(set.Failure() == DrawPrep::ReadFailure::Overflow, "overflow reason");
	Check(!set.Finish(), "a failed set does not finish");
	set.Reset();
	Check(!set.Failed() && set.ReadCount() == 0, "reset clears the failure");
	for (uint32_t i = 0; i < DrawPrep::ReadSet::MaxReads; i++) {
		Check(set.Record(0x1000 + uint64_t {i} * 16u, &one, 1), "read count within limit");
	}
	Check(!set.Record(0x900000, &one, 1), "one read over the count limit is refused");
	set.Reset();
	Check(set.Record(0x1000, &one, 0), "a zero-sized read is ignored");
	Check(set.Finish() && set.Ranges().empty(), "an empty set finishes with no ranges");
	std::vector<uint8_t> scratch;
	Check(set.Validate([](uint64_t, void*, uint64_t) { return false; }, scratch) ==
	          DrawPrep::ValidateResult::Ok,
	      "an empty set validates without reading");
}

// Digest reads (KYTY_DRAW_PREP_CODE_DIGEST): certified by XXH3-64 instead of bytes, alongside byte
// reads of the same memory.
void TestReadSetDigests() {
	FakeMemory memory;
	for (size_t i = 0; i < memory.bytes.size(); i++) {
		memory.bytes[i] = static_cast<uint8_t>(i * 11u + 5u);
	}
	const auto digest_of = [&](uint64_t address, uint64_t size) {
		return XXH3_64bits(memory.bytes.data() + (address - memory.base), size);
	};
	DrawPrep::ReadSet set;
	RecordFrom(set, memory, 0x10000, 8); // a header probe inside the digested code
	Check(set.RecordDigest(0x10000, 0x900, digest_of(0x10000, 0x900)), "record a digest");
	Check(set.RecordDigest(0x10000, 0, 0), "an empty digest read is ignored");
	Check(set.Finish(), "byte and digest reads finish");
	Check(set.Ranges().size() == 1 && set.DigestRanges().size() == 1 &&
	          set.DigestRanges()[0] == Coherence::Range {0x10000, 0x10900} &&
	          set.ByteCount() == 8,
	      "a digest read keeps its range and no bytes");

	std::vector<uint8_t> scratch;
	const auto read = [&](uint64_t address, void* data, uint64_t size) {
		return memory.Read(address, data, size);
	};
	Check(set.Validate(read, scratch) == DrawPrep::ValidateResult::Ok,
	      "unchanged digested memory validates");
	memory.bytes[0x800] ^= 0x40u; // inside the digest only
	Check(set.Validate(read, scratch) == DrawPrep::ValidateResult::Changed,
	      "a change inside a digest range fails validation");
	memory.bytes[0x800] ^= 0x40u;
	Check(set.Validate(read, scratch) == DrawPrep::ValidateResult::Ok,
	      "restored digested memory validates");
	memory.dirty_begin = 0x10880;
	memory.dirty_end   = 0x10881;
	Check(set.Validate(read, scratch) == DrawPrep::ValidateResult::Unclean,
	      "an unclean byte in a digest range fails validation");
	Check(!set.AllClean([&](uint64_t address, uint64_t size) {
		      std::vector<uint8_t> tmp(size);
		      return memory.Read(address, tmp.data(), size);
	      }),
	      "AllClean covers digest ranges");
	memory.dirty_begin = memory.dirty_end = 0;

	set.Reset();
	Check(set.DigestRanges().empty(), "reset clears digests");
	for (uint32_t i = 0; i < DrawPrep::ReadSet::MaxDigests; i++) {
		Check(set.RecordDigest(0x10000 + uint64_t {i} * 16u, 16, 0), "digest count within limit");
	}
	Check(!set.RecordDigest(0x11000, 16, 0) &&
	          set.Failure() == DrawPrep::ReadFailure::Overflow,
	      "one digest over the limit is refused");
}

// The commit-time certificate-range function of drawPrep.cpp (CertificateRanges, before
// KYTY_DRAW_PREP_CERT_RANGES), reproduced as the reference for ReadSet::BuildCertificate.
std::vector<Coherence::Range> ReferenceCertificateRanges(const DrawPrep::ReadSet& reads) {
	const auto digests = reads.DigestRanges();
	if (digests.empty()) {
		return {reads.Ranges().begin(), reads.Ranges().end()};
	}
	std::vector<Coherence::Range> scratch(reads.Ranges().begin(), reads.Ranges().end());
	scratch.insert(scratch.end(), digests.begin(), digests.end());
	std::sort(scratch.begin(), scratch.end(),
	          [](const Coherence::Range& a, const Coherence::Range& b) { return a.begin < b.begin; });
	size_t merged = 0;
	for (const auto& range: scratch) {
		if (merged != 0 && range.begin <= scratch[merged - 1].end) {
			scratch[merged - 1].end = std::max(scratch[merged - 1].end, range.end);
		} else {
			scratch[merged++] = range;
		}
	}
	scratch.resize(merged);
	return scratch;
}

// KYTY_DRAW_PREP_CERT_RANGES: ReadSet::BuildCertificate (built by the preparing thread) equals the
// commit-time list for every read set, is sorted and non-overlapping (Coherence::Log::Check's
// precondition), and covers exactly the bytes of the byte and digest ranges.
void TestReadSetCertificate() {
	std::vector<Coherence::Range> built;
	const std::array<uint8_t, 16> bytes {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

	// No digests: the byte ranges themselves, split at a touching page boundary as Finish keeps
	// them.
	DrawPrep::ReadSet plain;
	Check(plain.Record(0x20ff8, bytes.data(), 8) && plain.Record(0x21000, bytes.data() + 8, 8) &&
	          plain.Record(0x30000, bytes.data(), 4),
	      "record plain reads");
	Check(plain.Finish(), "plain reads finish");
	plain.BuildCertificate(built);
	Check(built == ReferenceCertificateRanges(plain) && built.size() == 3 &&
	          built[0] == Coherence::Range {0x20ff8, 0x21000} &&
	          built[1] == Coherence::Range {0x21000, 0x21008},
	      "without digests the certificate is the byte ranges");

	// Digests overlapping, touching and apart from byte ranges, recorded out of order.
	DrawPrep::ReadSet mixed;
	Check(mixed.Record(0x40000, bytes.data(), 8) && mixed.Record(0x40100, bytes.data(), 8),
	      "record mixed reads");
	Check(mixed.RecordDigest(0x40108, 0x10, 1) && mixed.RecordDigest(0x3ff00, 0x100, 2) &&
	          mixed.RecordDigest(0x50000, 0x20, 3) && mixed.RecordDigest(0x40004, 0x8, 4),
	      "record mixed digests");
	Check(mixed.Finish(), "mixed reads finish");
	mixed.BuildCertificate(built);
	Check(built == ReferenceCertificateRanges(mixed) && built.size() == 3 &&
	          built[0] == Coherence::Range {0x3ff00, 0x4000c} &&
	          built[1] == Coherence::Range {0x40100, 0x40118} &&
	          built[2] == Coherence::Range {0x50000, 0x50020},
	      "digests merge with touching and overlapping byte ranges");

	// Random read sets against the reference and a byte-coverage oracle.
	std::mt19937_64 rng(0x5eed2026u);
	constexpr uint64_t Base   = 0x100000;
	constexpr uint64_t Window = 0x6000; // six pages, so reads meet at page boundaries
	std::vector<uint8_t> memory(Window);
	for (auto& byte: memory) {
		byte = static_cast<uint8_t>(rng());
	}
	bool equal = true;
	bool sorted = true;
	bool covered = true;
	for (int iteration = 0; iteration < 2000; iteration++) {
		DrawPrep::ReadSet set;
		std::vector<uint8_t> expected(Window, 0);
		const auto reads = rng() % 24u;
		for (uint64_t i = 0; i < reads; i++) {
			const auto size    = 1u + rng() % 96u;
			const auto offset  = rng() % (Window - size);
			(void)set.Record(Base + offset, memory.data() + offset, size);
			std::fill(expected.begin() + static_cast<ptrdiff_t>(offset),
			          expected.begin() + static_cast<ptrdiff_t>(offset + size), uint8_t {1});
		}
		const auto digests = rng() % 5u;
		for (uint64_t i = 0; i < digests; i++) {
			const auto size   = 1u + rng() % 0x800u;
			const auto offset = rng() % (Window - size);
			(void)set.RecordDigest(Base + offset, size, rng());
			std::fill(expected.begin() + static_cast<ptrdiff_t>(offset),
			          expected.begin() + static_cast<ptrdiff_t>(offset + size), uint8_t {1});
		}
		if (!set.Finish()) {
			continue; // cannot happen: every read comes from one memory image
		}
		set.BuildCertificate(built);
		equal &= built == ReferenceCertificateRanges(set);
		std::vector<uint8_t> coverage(Window, 0);
		for (size_t i = 0; i < built.size(); i++) {
			sorted &= built[i].begin < built[i].end;
			if (i != 0) {
				sorted &= built[i - 1].end <= built[i].begin;
			}
			for (auto address = built[i].begin; address < built[i].end; address++) {
				coverage[address - Base] = 1;
			}
		}
		covered &= coverage == expected;
	}
	Check(equal, "BuildCertificate equals the commit-time list on random read sets");
	Check(sorted, "the certificate ranges are sorted, non-empty and non-overlapping");
	Check(covered, "the certificate ranges cover exactly the read and digest bytes");

	// The output vector keeps no stale entries from an earlier, longer certificate.
	DrawPrep::ReadSet one;
	Check(one.Record(0x60000, bytes.data(), 4) && one.Finish(), "record one read");
	one.BuildCertificate(built);
	Check(built.size() == 1 && built[0] == Coherence::Range {0x60000, 0x60004},
	      "a rebuilt certificate replaces the previous one");
}

// ReadSet::ValidateInPlace (KYTY_BACKING_INPLACE) returns what Validate returns for the same
// memory: the byte ranges first, then the digests, the first failing range deciding.
void TestReadSetValidateInPlace() {
	FakeMemory memory;
	for (size_t i = 0; i < memory.bytes.size(); i++) {
		memory.bytes[i] = static_cast<uint8_t>(i * 7u + 3u);
	}
	DrawPrep::ReadSet set;
	RecordFrom(set, memory, 0x10000, 64);
	RecordFrom(set, memory, 0x10200, 16);
	RecordFrom(set, memory, 0x107f8, 16);
	Check(set.RecordDigest(0x10a00, 0x400, XXH3_64bits(memory.bytes.data() + 0xa00, 0x400)),
	      "record a digest");
	Check(set.Finish() && set.Ranges().size() == 3, "reads finish as three ranges");

	std::vector<uint8_t> scratch;
	const auto read = [&](uint64_t address, void* data, uint64_t size) {
		return memory.Read(address, data, size);
	};
	uint32_t compares = 0;
	uint32_t hashes   = 0;
	const auto compare = [&](uint64_t address, const uint8_t* expected, uint64_t size) {
		compares++;
		std::vector<uint8_t> bytes(size);
		if (!memory.Read(address, bytes.data(), size)) {
			return DrawPrep::ValidateResult::Unclean;
		}
		return std::memcmp(bytes.data(), expected, size) == 0 ? DrawPrep::ValidateResult::Ok
		                                                     : DrawPrep::ValidateResult::Changed;
	};
	const auto hash = [&](uint64_t address, uint64_t size, uint64_t& digest) {
		hashes++;
		std::vector<uint8_t> bytes(size);
		if (!memory.Read(address, bytes.data(), size)) {
			return false;
		}
		digest = XXH3_64bits(bytes.data(), size);
		return true;
	};
	const auto same = [&](const char* message) {
		const auto copied = set.Validate(read, scratch);
		Check(set.ValidateInPlace(compare, hash) == copied, message);
		return copied;
	};
	Check(same("unchanged memory") == DrawPrep::ValidateResult::Ok && compares == 3 && hashes == 1,
	      "unchanged memory validates in place, every range once");
	for (const size_t offset: {size_t {0x10}, size_t {0x205}, size_t {0x7ff}, size_t {0x800},
	                           size_t {0xc00}}) {
		memory.bytes[offset] ^= 0x10u;
		Check(same("a changed byte") == DrawPrep::ValidateResult::Changed,
		      "a changed byte fails validation in place");
		memory.bytes[offset] ^= 0x10u;
	}
	memory.bytes[0x900] ^= 0x10u; // in no certified range
	Check(same("an uncertified byte") == DrawPrep::ValidateResult::Ok,
	      "a change outside the certificate failed validation in place");
	memory.bytes[0x900] ^= 0x10u;
	// Unclean first: the byte ranges are checked before the (changed) digest.
	memory.bytes[0xc00] ^= 0x10u;
	memory.dirty_begin = 0x10204;
	memory.dirty_end   = 0x10205;
	Check(same("unclean before changed") == DrawPrep::ValidateResult::Unclean,
	      "an unclean byte range decides before a changed digest in place");
	memory.dirty_begin = 0x10b00;
	memory.dirty_end   = 0x10b01;
	Check(same("unclean digest") == DrawPrep::ValidateResult::Unclean,
	      "an unclean digest range fails validation in place");
	memory.dirty_begin = memory.dirty_end = 0;
	memory.bytes[0xc00] ^= 0x10u;
	Check(same("restored memory") == DrawPrep::ValidateResult::Ok, "restored memory validates");
}

void TestRecordScopeNests() {
	DrawPrep::ReadSet  outer_set;
	DrawPrep::ReadSet  inner_set;
	DrawPrep::Recorder outer {&outer_set, true};
	DrawPrep::Recorder inner {&inner_set, false};
	Check(!DrawPrep::Speculative(), "no recorder by default");
	{
		DrawPrep::RecordScope a(outer);
		Check(DrawPrep::ActiveRecorder() == &outer, "outer recorder active");
		{
			DrawPrep::RecordScope b(inner);
			Check(DrawPrep::ActiveRecorder() == &inner, "inner recorder active");
			DrawPrep::FailActive(DrawPrep::ReadFailure::Uncertified);
		}
		Check(DrawPrep::ActiveRecorder() == &outer, "outer recorder restored");
	}
	Check(!DrawPrep::Speculative(), "recorder cleared");
	Check(inner_set.Failure() == DrawPrep::ReadFailure::Uncertified && !outer_set.Failed(),
	      "FailActive marks only the active set");
	std::thread([] { Check(!DrawPrep::Speculative(), "recorders are per thread"); }).join();
}

// ---------------------------------------------------------------------------------------------
// Window (S6)

struct Item {
	uint64_t input  = 0;
	uint64_t output = 0;
	uint32_t prepared_by = 0; // 0 = producer, k = worker k, StolenBy = producer while waiting
	uint32_t preparations = 0;
	// Pipeline workloads: the slot's preparation cost, and the producer's cost of committing it.
	uint32_t prepare_ns = 0;
	uint32_t commit_ns  = 0;
};

constexpr uint32_t StolenBy = 0xffffu;

uint64_t Work(uint64_t value) {
	// Deterministic, a little expensive, so workers and the producer race for slots.
	uint64_t x = value * 0x9e3779b97f4a7c15ull + 1u;
	for (int i = 0; i < 64; i++) {
		x ^= x >> 29u;
		x *= 0xbf58476d1ce4e5b9ull;
	}
	return x;
}

void TestWindowSingleThread() {
	DrawPrep::Window<Item> window(3);
	Check(window.Capacity() == 4, "capacity rounds up to a power of two");
	Check(window.Empty() && !window.Full(), "new window is empty");
	for (uint64_t i = 0; i < 4; i++) {
		window.Reserve().input = i;
		window.Publish();
	}
	Check(window.Full() && window.Occupancy() == 4, "window fills to capacity");
	uint64_t seq  = 0;
	auto*    item = window.TryClaim(seq);
	Check(item != nullptr && seq == 0 && item->input == 0, "a worker claims the oldest slot");
	Check(!window.TryClaimHead(), "the producer cannot claim a worker's slot");
	Check(!window.HeadDone(), "a claimed slot is not done");
	window.Complete(seq);
	Check(window.HeadDone(), "a completed head is done");
	window.Retire();
	Check(window.TryClaimHead(), "the producer claims an unclaimed head");
	window.Retire();
	auto* third = window.TryClaim(seq);
	Check(third != nullptr && seq == 2, "workers skip positions the producer took");
	window.Complete(seq);
	Check(window.HeadDone(), "the third slot is done");
	window.Retire();
	Check(window.TryClaimHead(), "the last slot is claimable by the producer");
	window.Retire();
	Check(window.Empty(), "window drained");
	Check(window.TryClaim(seq) == nullptr, "nothing to claim in an empty window");
	// Reuse after wrap: the slot of position 4 is the slot of position 0.
	window.Reserve().input = 42;
	window.Publish();
	auto* wrapped = window.TryClaim(seq);
	Check(wrapped != nullptr && seq == 4 && wrapped->input == 42, "wrapped slot is claimable");
	window.Complete(seq);
	window.Retire();
}

void TestWindowConcurrent(uint32_t capacity, uint32_t workers, uint64_t items) {
	DrawPrep::Window<Item> window(capacity);
	std::atomic<bool>      stop {false};
	std::vector<std::thread> threads;
	for (uint32_t w = 0; w < workers; w++) {
		threads.emplace_back([&, w] {
			while (!stop.load(std::memory_order_acquire)) {
				uint64_t seq  = 0;
				auto*    item = window.TryClaim(seq);
				if (item == nullptr) {
					std::this_thread::yield();
					continue;
				}
				item->output      = Work(item->input);
				item->prepared_by = w + 1u;
				item->preparations++;
				window.Complete(seq);
			}
		});
	}
	uint64_t next_input   = 0;
	uint64_t next_retire  = 0;
	uint64_t self         = 0;
	uint64_t by_workers   = 0;
	bool     order_ok     = true;
	bool     result_ok    = true;
	const auto retire_head = [&] {
		auto& item = window.HeadPayload();
		if (window.TryClaimHead()) {
			item.output      = Work(item.input);
			item.prepared_by = 0;
			item.preparations++;
			self++;
		} else {
			while (!window.HeadDone()) {
				std::this_thread::yield();
			}
			by_workers++;
		}
		order_ok &= item.input == next_retire;
		result_ok &= item.output == Work(item.input) && item.preparations == 1;
		next_retire++;
		window.Retire();
	};
	while (next_input < items) {
		if (window.Full()) {
			retire_head();
		}
		auto& item        = window.Reserve();
		item.input        = next_input++;
		item.output       = 0;
		item.preparations = 0;
		window.Publish();
		// Occasionally drain like a fence does.
		if (next_input % 97u == 0u) {
			while (!window.Empty()) {
				retire_head();
			}
		}
	}
	while (!window.Empty()) {
		retire_head();
	}
	stop.store(true, std::memory_order_release);
	for (auto& thread: threads) {
		thread.join();
	}
	Check(order_ok, "items retire in publication order");
	Check(result_ok, "every item is prepared exactly once with the right result");
	Check(next_retire == items && self + by_workers == items, "every item retires once");
	// How many items the workers (rather than the producer) prepared depends on scheduling.
	std::printf("  window %u x %u workers: %llu of %llu prepared by workers\n", capacity, workers,
	            static_cast<unsigned long long>(by_workers), static_cast<unsigned long long>(items));
}

// ---------------------------------------------------------------------------------------------
// Worker parking (KYTY_DRAW_PREP_HOT) and work stealing (KYTY_DRAW_PREP_STEAL): WorkerGate, the
// production worker loop and the producer's AwaitHead

void TestWorkerGateBasics() {
	DrawPrep::WorkerGate gate(6, 2, 8);
	Check(gate.Hot(0) && gate.Hot(1) && !gate.Hot(2) && !gate.Hot(5), "first two workers are hot");
	Check(gate.HotCount() == 2 && gate.HasCold() && gate.WakeBacklog() == 8, "gate shape");
	Check(!gate.MaybeWakeCold(100), "no wake without a sleeping cold worker");
	DrawPrep::WorkerGate all(6, 6, 8);
	Check(!all.HasCold() && all.Hot(5), "hot >= workers keeps every worker hot");
	DrawPrep::WorkerGate clamp(6, 40, 0);
	Check(clamp.HotCount() == 6 && clamp.WakeBacklog() == 1, "values clamp");
	int  backlog_reads = 0;
	auto backlog       = [&] {
        backlog_reads++;
        return uint64_t {100};
	};
	for (int i = 0; i < 64; i++) {
		(void)all.OnPublish(backlog);
	}
	Check(backlog_reads == 0, "an all-hot gate never reads the backlog");
	for (int i = 0; i < 64; i++) {
		(void)gate.OnPublish(backlog);
	}
	Check(backlog_reads == 8, "the producer reads the backlog every WakeBacklog-th publish");
}

// AwaitHead step by step: this thread is the producer and also plays the workers holding slots.
void TestAwaitHead() {
	DrawPrep::WorkerGate gate(2, 2, 8); // every worker hot: AwaitHead never wakes a cold one
	int                  cold_wakes   = 0;
	const auto           on_cold_wake = [&] { cold_wakes++; };
	const auto           publish      = [](DrawPrep::Window<Item>& window, uint64_t count) {
        for (uint64_t i = 0; i < count; i++) {
            auto& item = window.Reserve();
            item       = Item {};
            item.input = window.Tail();
            window.Publish();
        }
	};
	// Retires everything in order; a head that is neither done nor claimable fails.
	const auto retire_all = [](DrawPrep::Window<Item>& window) {
		bool ok = true;
		while (!window.Empty()) {
			if (!window.HeadDone()) {
				ok &= window.TryClaimHead();
			}
			window.Retire();
		}
		return ok;
	};
	const DrawPrep::StealPolicy never {};
	const DrawPrep::StealPolicy always {1, 0};

	{
		DrawPrep::Window<Item> window(8);
		publish(window, 5);
		uint64_t head = 0;
		Check(window.TryClaim(head) != nullptr && head == 0, "a worker holds the head");
		std::vector<uint64_t> stolen;
		int                   spins = 0;
		const auto            stats = DrawPrep::AwaitHead(
            gate, window, always,
            [&](Item& item, uint64_t seq) {
                Check(seq > window.Head() && item.input == seq,
                      "a steal takes a later slot, never the head");
                item.preparations++;
                item.prepared_by = StolenBy;
                stolen.push_back(seq);
                window.Complete(seq);
                if (stolen.size() == 2) {
                    window.Complete(head); // the worker finishes the head meanwhile
                }
            },
            [&](uint32_t, uint64_t) { spins++; }, on_cold_wake);
		Check(stats.stolen == 2 && stolen == std::vector<uint64_t> {1, 2} && spins == 0,
		      "the producer steals the oldest unclaimed slots until the head is done");
		uint64_t seq  = 0;
		auto*    next = window.TryClaim(seq);
		Check(next != nullptr && seq == 3 && next->preparations == 0,
		      "slots the producer did not steal stay claimable by workers");
		window.Complete(seq);
		Check(retire_all(window), "stolen slots retire in order like worker slots");
	}
	{
		DrawPrep::Window<Item> window(8);
		publish(window, 3);
		uint64_t head = 0;
		Check(window.TryClaim(head) != nullptr, "a worker holds the head");
		int        prepared = 0;
		int        spins    = 0;
		const auto stats    = DrawPrep::AwaitHead(
            gate, window, never, [&](Item&, uint64_t) { prepared++; },
            [&](uint32_t, uint64_t) {
                if (++spins == 100) {
                    window.Complete(head);
                }
            },
            on_cold_wake);
		uint64_t seq = 0;
		Check(stats.stolen == 0 && prepared == 0 && spins == 100 && window.HeadDone() &&
		          window.TryClaim(seq) != nullptr && seq == 1,
		      "without stealing the producer spins until the head is done and claims nothing");
		window.Complete(seq);
		Check(retire_all(window), "the window drains");
	}
	{
		DrawPrep::Window<Item> window(4);
		publish(window, 2);
		uint64_t head   = 0;
		uint64_t second = 0;
		Check(window.TryClaim(head) != nullptr && window.TryClaim(second) != nullptr,
		      "workers hold every slot");
		int        prepared = 0;
		int        spins    = 0;
		const auto stats    = DrawPrep::AwaitHead(
            gate, window, always, [&](Item&, uint64_t) { prepared++; },
            [&](uint32_t, uint64_t) {
                spins++;
                window.Complete(head);
            },
            on_cold_wake);
		Check(stats.stolen == 0 && prepared == 0 && spins == 1,
		      "with nothing left to claim the producer only spins");
		window.Complete(second);
		Check(retire_all(window), "the window drains");
	}
	{
		// The claim counter lags behind a head the producer prepared itself: claims skip it.
		DrawPrep::Window<Item> window(4);
		publish(window, 1);
		Check(window.TryClaimHead(), "the producer prepares an unclaimed head itself");
		window.Retire();
		publish(window, 3);
		uint64_t head = 0;
		Check(window.TryClaim(head) != nullptr && head == 1,
		      "a worker's claim skips the position the producer took");
		std::vector<uint64_t> stolen;
		const auto            stats = DrawPrep::AwaitHead(
            gate, window, always,
            [&](Item&, uint64_t seq) {
                stolen.push_back(seq);
                window.Complete(seq);
            },
            [&](uint32_t, uint64_t) { window.Complete(head); }, on_cold_wake);
		Check(stats.stolen == 2 && stolen == std::vector<uint64_t> {2, 3},
		      "steals follow the claim order");
		Check(retire_all(window), "the window drains");
	}
	{
		// A minimum backlog: the producer leaves the last unclaimed slot to the workers.
		DrawPrep::Window<Item> window(8);
		publish(window, 4);
		uint64_t head = 0;
		Check(window.TryClaim(head) != nullptr, "a worker holds the head");
		std::vector<uint64_t> stolen;
		const auto            stats = DrawPrep::AwaitHead(
            gate, window, DrawPrep::StealPolicy {2, 0},
            [&](Item&, uint64_t seq) {
                stolen.push_back(seq);
                window.Complete(seq);
            },
            [&](uint32_t, uint64_t) { window.Complete(head); }, on_cold_wake);
		uint64_t seq = 0;
		Check(stats.stolen == 2 && stolen == std::vector<uint64_t> {1, 2} &&
		          window.TryClaim(seq) != nullptr && seq == 3,
		      "the producer steals only while at least the minimum backlog waits");
		window.Complete(seq);
		Check(retire_all(window), "the window drains");
	}
	{
		// A delay: a head done before it passes is never stolen from; a longer one is.
		DrawPrep::Window<Item> window(8);
		publish(window, 3);
		uint64_t head = 0;
		Check(window.TryClaim(head) != nullptr, "a worker holds the head");
		int        prepared = 0;
		const auto quick    = DrawPrep::AwaitHead(
            gate, window, DrawPrep::StealPolicy {1, 60'000'000'000ull},
            [&](Item&, uint64_t) { prepared++; },
            [&](uint32_t spins, uint64_t) {
                if (spins == 50) {
                    window.Complete(head);
                }
            },
            on_cold_wake);
		Check(quick.stolen == 0 && prepared == 0, "no steal before the delay");
		window.Retire();
		uint64_t next = 0;
		Check(window.TryClaim(next) != nullptr && next == 1, "a worker holds the next head");
		const auto begin = std::chrono::steady_clock::now();
		std::vector<uint64_t> stolen;
		uint32_t              looks = 0;
		const auto            slow  = DrawPrep::AwaitHead(
            gate, window, DrawPrep::StealPolicy {1, 200'000},
            [&](Item&, uint64_t seq) {
                stolen.push_back(seq);
                window.Complete(seq);
                window.Complete(next); // the worker finishes after the steal
            },
            [&](uint32_t spins, uint64_t) {
                looks = spins + 1;
                DrawPrep::WorkerRelax();
            },
            on_cold_wake);
		const auto waited = std::chrono::steady_clock::now() - begin;
		Check(slow.stolen == 1 && stolen == std::vector<uint64_t> {2} && looks > 1 &&
		          waited >= std::chrono::microseconds(200),
		      "the producer steals once the delay has passed");
		Check(retire_all(window), "the window drains");
	}
	Check(cold_wakes == 0, "an all-hot gate never wakes a cold worker");
}

// ---------------------------------------------------------------------------------------------
// Pipelines: Engine's producer and workers on synthetic workloads

void BusyNs(uint64_t ns) {
	const auto end = std::chrono::steady_clock::now() + std::chrono::nanoseconds(ns);
	while (std::chrono::steady_clock::now() < end) {
	}
}

// Engine's parallel-mode configuration (the KYTY_DRAW_PREP_* switches).
// The defaults are Engine's.
struct PipelineConfig {
	uint32_t              workers      = 6;
	uint32_t              hot          = 2;
	uint32_t              wake_backlog = 2;
	uint64_t              hot_spin_ns  = 200000; // KYTY_DRAW_PREP_SPIN_US default
	uint64_t              cold_spin_ns = 50000;  // KYTY_DRAW_PREP_COLD_SPIN_US default
	DrawPrep::StealPolicy steal {};              // KYTY_DRAW_PREP_STEAL default: never
	uint32_t              window       = 32;
	bool                  cold_token   = false;  // KYTY_DRAW_PREP_COLD_TOKEN
	// A worker stalls `hiccup_ns` inside this share (per mille) of its preparations: preempted by
	// a guest thread, or late to wake, while it holds the slot. The producer never stalls.
	uint32_t hiccup_per_mille = 0;
	uint32_t hiccup_ns        = 300000;
};

// One draw of a workload. Its slot costs `prepare_ns` on whichever thread prepares it and
// `commit_ns` on the producer when it is committed. After publishing it the producer works
// `after_ns` (parsing up to the next draw). A fence after it drains the window; the producer then
// works `fence_ns` (packets without draws) and blocks `idle_us` (waiting for the next submission;
// 0: not at all).
struct DrawSpec {
	uint32_t prepare_ns = 0;
	uint32_t commit_ns  = 0;
	uint32_t after_ns   = 0;
	uint32_t fence_ns   = 0;
	uint32_t idle_us    = 0;
	bool     fence      = false;
};

struct PipelineResult {
	uint64_t self_prepared = 0; // unclaimed heads the producer prepared itself
	uint64_t commit_waits  = 0; // heads a worker still held when the producer needed them
	uint64_t steals        = 0; // slots the producer prepared meanwhile (AwaitHead)
	uint64_t cold_wakes    = 0;
	uint64_t cold_prepared = 0; // slots prepared by cold workers (index >= hot)
	double   wall_ms       = 0; // the producer's whole run: the command processor's critical path
	double   spin_ms       = 0; // of which spinning on held heads
	double   worker_cpu_ms = 0; // user + kernel time of all workers (Windows)
	double   useful_ms     = 0; // preparation work the workers did
	bool     ok            = true;
};

// Runs a workload the way Engine runs it:
// - The producer publishes the draws, retires the head when the window is full, and drains the
//   window at fences.
// - It prepares an unclaimed head itself. It awaits a head a worker still holds with the
//   production AwaitHead, stealing when config.steal is set.
// - The workers run the production worker loop.
// Every retired slot must be the next in order, carry the right output, and have been prepared
// exactly once.
PipelineResult RunPipeline(const PipelineConfig& config, const std::vector<DrawSpec>& draws) {
	DrawPrep::Window<Item>   window(config.window);
	DrawPrep::WorkerGate     gate(config.workers, config.hot, config.wake_backlog,
	                              config.cold_token);
	std::atomic<bool>        stop {false};
	std::atomic<uint64_t>    useful_ns {0};
	std::atomic<uint64_t>    worker_cold_wakes {0};
	std::vector<std::thread> threads;
	for (uint32_t w = 0; w < config.workers; w++) {
		threads.emplace_back([&, w] {
			std::mt19937 stalls(1000u + w);
			DrawPrep::RunPreparationWorker(
			    gate, window, w, config.hot_spin_ns, config.cold_spin_ns, stop,
			    [&](Item& item, uint64_t seq) {
				    const bool stall = config.hiccup_per_mille != 0 &&
				                       stalls() % 1000u < config.hiccup_per_mille;
				    BusyNs(item.prepare_ns + (stall ? config.hiccup_ns : 0u));
				    item.output      = Work(item.input);
				    item.prepared_by = w + 1u;
				    item.preparations++;
				    useful_ns.fetch_add(item.prepare_ns, std::memory_order_relaxed);
				    window.Complete(seq);
			    },
			    [&] { worker_cold_wakes.fetch_add(1, std::memory_order_relaxed); });
		});
	}
	PipelineResult result;
	uint64_t       next_retire  = 0;
	uint64_t       spin_ns      = 0;
	const auto     prepare_here = [](Item& item, uint32_t by) {
        BusyNs(item.prepare_ns);
        item.output      = Work(item.input);
        item.prepared_by = by;
        item.preparations++;
	};
	const auto retire_head = [&] {
		auto& item = window.HeadPayload();
		if (window.TryClaimHead()) {
			prepare_here(item, 0);
			result.self_prepared++;
		} else if (!window.HeadDone()) {
			result.commit_waits++;
			const auto stats = DrawPrep::AwaitHead(
			    gate, window, config.steal,
			    [&](Item& other, uint64_t seq) {
				    result.ok &= seq > window.Head();
				    prepare_here(other, StolenBy);
				    window.Complete(seq);
			    },
			    [&](uint32_t spins, uint64_t spin_start) {
				    DrawPrep::WorkerRelax();
				    if ((spins & 1023u) == 1023u &&
				        DrawPrep::WorkerNowNs() - spin_start > 10'000'000'000ull) {
					    std::fprintf(stderr, "FAILED: slot %llu was never completed\n",
					                 static_cast<unsigned long long>(window.Head()));
					    std::_Exit(3); // a hung protocol; going on would race the worker
				    }
			    },
			    [] {});
			result.steals += stats.stolen;
			spin_ns += stats.spin_ns;
		}
		result.ok &= item.input == next_retire && item.output == Work(item.input) &&
		             item.preparations == 1;
		if (item.prepared_by > config.hot && item.prepared_by != StolenBy) {
			result.cold_prepared++;
		}
		BusyNs(item.commit_ns);
		next_retire++;
		window.Retire();
	};
	const auto start      = std::chrono::steady_clock::now();
	uint64_t   next_input = 0;
	for (const auto& draw: draws) {
		if (window.Full()) {
			retire_head();
		}
		auto& item      = window.Reserve();
		item            = Item {};
		item.input      = next_input++;
		item.prepare_ns = draw.prepare_ns;
		item.commit_ns  = draw.commit_ns;
		window.Publish();
		(void)gate.OnPublish([&] { return window.Unclaimed(); });
		BusyNs(draw.after_ns);
		if (draw.fence) {
			while (!window.Empty()) {
				retire_head();
			}
			BusyNs(draw.fence_ns);
			if (draw.idle_us != 0) {
				std::this_thread::sleep_for(std::chrono::microseconds(draw.idle_us));
			}
		}
	}
	while (!window.Empty()) {
		retire_head();
	}
	result.wall_ms =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	result.spin_ms = static_cast<double>(spin_ns) / 1e6;
	stop.store(true, std::memory_order_seq_cst);
	gate.WakeAll();
#if defined(_WIN32)
	for (auto& thread: threads) {
		FILETIME creation {};
		FILETIME exit {};
		FILETIME kernel {};
		FILETIME user {};
		if (GetThreadTimes(static_cast<HANDLE>(thread.native_handle()), &creation, &exit, &kernel,
		                   &user) != 0) {
			const auto to_ms = [](FILETIME t) {
				return static_cast<double>((uint64_t {t.dwHighDateTime} << 32u) | t.dwLowDateTime) /
				       10000.0;
			};
			result.worker_cpu_ms += to_ms(kernel) + to_ms(user);
		}
	}
#endif
	for (auto& thread: threads) {
		thread.join();
	}
	result.useful_ms  = static_cast<double>(useful_ns.load()) / 1e6;
	result.cold_wakes = gate.ColdWakes(); // by workers, by the producer and by its steals
	result.ok &= next_retire == draws.size() && worker_cold_wakes.load() <= result.cold_wakes;
	return result;
}

// The earlier workload: a fixed preparation cost, `gap_ns` of producer work after every publish,
// a drain every `drain_every` slots. With `random_bursts` the gap is 4x or a quarter of it, and
// the producer blocks up to 2 ms after every drain, long enough for workers to park.
std::vector<DrawSpec> UniformShape(uint64_t items, uint32_t prepare_ns, uint32_t gap_ns,
                                   uint32_t drain_every, bool random_bursts) {
	std::mt19937          rng(12345);
	std::vector<DrawSpec> draws(items);
	for (uint64_t i = 0; i < items; i++) {
		auto& draw      = draws[i];
		draw.prepare_ns = prepare_ns;
		draw.after_ns   = !random_bursts ? gap_ns : rng() % 4 == 0 ? gap_ns * 4 : gap_ns / 4;
		draw.fence      = (i + 1) % drain_every == 0;
		draw.idle_us    = draw.fence && random_bursts ? static_cast<uint32_t>(rng() % 2000) : 0;
	}
	return draws;
}

// Bursty stress with short spins, with and without stealing: nothing hangs, every slot is
// prepared exactly once, in order, and cold workers are woken when the backlog grows.
void TestWorkerGateStress() {
	for (uint32_t hot: {1u, 2u}) {
		for (const bool steal: {false, true}) {
			PipelineConfig config;
			config.hot          = hot;
			config.wake_backlog = 4;
			config.hot_spin_ns  = 20000;
			config.cold_spin_ns = 0;
			config.steal        = steal ? DrawPrep::StealPolicy {1, 0} : DrawPrep::StealPolicy {};
			const auto r = RunPipeline(config, UniformShape(60000, 2000, 2000, 97, true));
			Check(r.ok, "gated pipeline prepares every slot once, in order");
			Check(r.cold_wakes > 0, "cold workers are woken by a backlog");
			Check(steal || r.steals == 0, "the producer never steals without KYTY_DRAW_PREP_STEAL");
		}
	}
}

// KYTY_DRAW_PREP_COLD_TOKEN: a wake given while its cold sleeper was already leaving the park (the
// race that left the default protocol's pending flag set for good) stays outstanding and is taken
// by the next cold worker to park, which returns to claiming instead of sleeping; afterwards wakes
// work again. Single thread, the interleaving forced through the park predicate; a park that
// slept anyway is released by WakeAll and reported.
void TestColdTokenStranded() {
	DrawPrep::WorkerGate gate(2, 1, 2, true);
	Check(gate.ColdTokenMode() && !gate.ColdWakeOutstanding(), "token mode starts without a wake");
	std::atomic<bool> stop {false};
	bool              woke_inside = false;
	// The sleeper is announced when the predicate runs: a waker sees it and gives the token, and
	// the predicate then finds the backlog high, so the sleeper leaves without taking it.
	gate.ParkCold(
	    [&] {
		    woke_inside = gate.MaybeWakeCold(8);
		    return true;
	    },
	    stop);
	Check(woke_inside && gate.ColdWakeOutstanding() && gate.ColdSleepers() == 0,
	      "a wake given while the sleeper leaves stays outstanding");
	Check(!gate.MaybeWakeCold(8), "no second wake while one is outstanding");
	// The next park takes the stranded token at once (the predicate says nothing to claim).
	std::atomic<bool> returned {false};
	std::thread       parker([&] {
        gate.ParkCold([] { return false; }, stop);
        returned = true;
    });
	for (int i = 0; i < 2000 && !returned.load(); i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	const bool took = returned.load();
	if (!took) {
		stop = true;
		gate.WakeAll();
	}
	parker.join();
	Check(took, "the next cold park takes the stranded wake instead of sleeping");
	Check(stop.load() || !gate.ColdWakeOutstanding(), "the stranded wake is consumed");
	// A real sleeper is woken again afterwards.
	std::atomic<bool> woken {false};
	std::thread       sleeper([&] {
        gate.ParkCold([] { return false; }, stop);
        woken = true;
    });
	bool gave = false;
	for (int i = 0; i < 2000 && !gave; i++) {
		gave = gate.MaybeWakeCold(8); // succeeds once the sleeper has announced itself
		if (!gave) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	for (int i = 0; i < 2000 && !woken.load(); i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	const bool ok = gave && woken.load();
	if (!ok) {
		stop = true;
		gate.WakeAll();
	}
	sleeper.join();
	Check(ok, "a sleeping cold worker is woken after a stranded wake was consumed");
}

// Cold workers keep being woken for a whole bursty run in token mode: with one hot worker and
// preparations twice as long as the producer's gap, the cold workers must prepare a large share
// of the slots. The default protocol is run too and its share printed (it can stop waking cold
// workers after the race above).
void TestColdTokenStress() {
	for (const bool token: {false, true}) {
		PipelineConfig config;
		config.workers      = 6;
		config.hot          = 1;
		config.wake_backlog = 2;
		config.hot_spin_ns  = 20000;
		config.cold_spin_ns = 5000;
		config.cold_token   = token;
		const auto draws    = UniformShape(60000, 4000, 2000, 97, true);
		const auto r        = RunPipeline(config, draws);
		Check(r.ok, "token-mode pipeline prepares every slot once, in order");
		std::printf("  cold wake protocol %s: %llu cold wakes, %llu of %zu slots prepared by cold "
		            "workers, %llu held heads\n",
		            token ? "token" : "default", static_cast<unsigned long long>(r.cold_wakes),
		            static_cast<unsigned long long>(r.cold_prepared), draws.size(),
		            static_cast<unsigned long long>(r.commit_waits));
		if (token) {
			Check(r.cold_prepared * 5 > draws.size(),
			      "cold workers prepare a fifth of the slots or more (they keep waking)");
		}
	}
}

// KYTY_DRAW_PREP_STEAL under stress: random preparation costs and fences, windows of 4 to 32
// slots, one to six workers and several steal policies, so that heads are often held while later
// slots wait, and the producer's claims race the workers'. Every slot is prepared exactly once
// (by a worker, by the producer as an unclaimed head, or stolen) and retired in order.
void TestStealStress() {
	struct Case {
		uint32_t              window;
		uint32_t              workers;
		uint32_t              hot;
		DrawPrep::StealPolicy steal;
	};
	uint64_t steals = 0;
	for (const auto& c: {Case {4, 1, 1, {1, 0}}, Case {4, 4, 1, {1, 0}}, Case {8, 2, 2, {2, 0}},
	                     Case {32, 1, 1, {1, 0}}, Case {32, 1, 1, {3, 2000}},
	                     Case {32, 6, 2, {1, 1000}}, Case {32, 6, 6, {1, 0}}}) {
		PipelineConfig config;
		config.window       = c.window;
		config.workers      = c.workers;
		config.hot          = c.hot;
		config.wake_backlog = 2;
		config.hot_spin_ns  = 5000;
		config.cold_spin_ns = 0;
		config.steal        = c.steal;
		std::mt19937          rng(c.window * 131u + c.workers * 7u + c.hot);
		std::vector<DrawSpec> draws(40000);
		for (auto& draw: draws) {
			draw.prepare_ns = rng() % 4 == 0 ? 3000 + rng() % 5000 : rng() % 1500;
			draw.commit_ns  = rng() % 600;
			draw.after_ns   = rng() % 300;
			draw.fence      = rng() % 24 == 0;
			draw.idle_us    = draw.fence && rng() % 32 == 0 ? rng() % 500 : 0;
		}
		const auto r = RunPipeline(config, draws);
		Check(r.ok, "a stealing pipeline prepares every slot once and retires them in order");
		steals += r.steals;
		std::printf("  steal stress: window %u, %u workers (%u hot), steal %u after %llu ns: %llu "
		            "steals, %llu held heads, %llu self-prepared\n",
		            c.window, c.workers, c.hot, c.steal.min_unclaimed,
		            static_cast<unsigned long long>(c.steal.after_ns),
		            static_cast<unsigned long long>(r.steals),
		            static_cast<unsigned long long>(r.commit_waits),
		            static_cast<unsigned long long>(r.self_prepared));
	}
	Check(steals > 0, "the producer steals while workers hold heads");
}

// ---------------------------------------------------------------------------------------------
// --measure-worker-gate [reps]: parking and stealing configurations on frame-shaped workloads

// Frames of fence-delimited draw segments. Per frame, the number of segments in each
// draws-per-fence bucket follows a scene's U54 Tracy counters (FrameEvent DrawPrepFenceDraws*,
// per flip). The fences without draws are producer work between segments.
struct FrameShape {
	const char*           name;
	std::array<double, 7> segments;     // per frame: 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64+ draws
	uint32_t              long_segment; // mean draws of a 64+ segment
	uint32_t              gap_ns;       // mean producer work between segments
	uint32_t              parse_ns;     // producer work per draw before the next one
	uint32_t              commit_ns;    // producer work per committed draw
	uint32_t              prepare_min_ns;
	uint32_t              prepare_max_ns;
	uint32_t              short_prepare_ns; // draws of 1-3-draw segments (clears, copies)
	uint32_t              burst_draws;      // the first draws after the fence of a 4+ segment...
	uint32_t              burst_min_ns;     // ...cost this much to prepare
	uint32_t              burst_max_ns;
	uint32_t              frame_idle_us; // after each frame: waiting for the next submission
};

// Sky Garden start view (U54): 4,955 draws and about 2,460 fences without draws per flip, Prepare
// 9.4 us on average, the command processor about 9 us per draw.
const FrameShape kSkyGardenFrame {"sky-garden start", {48.0, 4.0, 4.6, 2.2, 1.2, 3.0, 10.0},
                                  468, 50000, 1000, 7000, 4000, 15000, 2000, 0, 0, 0, 1000};
// Heavy frames: the slide's segments (U54, per flip). Each segment of 4+ draws starts with a
// burst of eight expensive draws (30-45 us); the others cost 8-16 us, about the slide's 15 us
// average overall.
const FrameShape kHeavyBurstFrame {"heavy burst (slide)", {49.3, 1.1, 6.5, 1.7, 1.6, 1.4, 2.3},
                                   407, 48000, 1500, 8000, 8000, 16000, 2000, 8, 30000, 45000,
                                   1500};

std::vector<DrawSpec> MakeFrames(const FrameShape& shape, uint32_t frames, uint32_t seed) {
	static constexpr std::array<std::array<uint32_t, 2>, 6> bounds {
	    {{1, 1}, {2, 3}, {4, 7}, {8, 15}, {16, 31}, {32, 63}}};
	std::mt19937 rng(seed);
	const auto   uniform = [&](uint32_t low, uint32_t high) {
        return high > low ? low + static_cast<uint32_t>(rng() % (high - low + 1u)) : low;
	};
	std::vector<DrawSpec> draws;
	for (uint32_t frame = 0; frame < frames; frame++) {
		std::vector<uint32_t> sizes;
		for (size_t bucket = 0; bucket < shape.segments.size(); bucket++) {
			const auto expected = shape.segments[bucket];
			auto       count    = static_cast<uint32_t>(expected);
			if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < expected - count) {
				count++;
			}
			for (uint32_t i = 0; i < count; i++) {
				sizes.push_back(bucket < bounds.size()
				                    ? uniform(bounds[bucket][0], bounds[bucket][1])
				                    : uniform(shape.long_segment / 2u, shape.long_segment * 3u / 2u));
			}
		}
		std::shuffle(sizes.begin(), sizes.end(), rng);
		for (const auto size: sizes) {
			for (uint32_t i = 0; i < size; i++) {
				DrawSpec draw;
				draw.prepare_ns = size < 4                ? shape.short_prepare_ns
				                  : i < shape.burst_draws ? uniform(shape.burst_min_ns, shape.burst_max_ns)
				                                          : uniform(shape.prepare_min_ns, shape.prepare_max_ns);
				draw.commit_ns  = shape.commit_ns;
				draw.after_ns   = shape.parse_ns;
				draws.push_back(draw);
			}
			draws.back().fence    = true;
			draws.back().fence_ns = uniform(shape.gap_ns / 2u, shape.gap_ns * 3u / 2u);
		}
		draws.back().idle_us = shape.frame_idle_us;
	}
	return draws;
}

template <typename T, typename Get>
double Median(const std::vector<T>& runs, Get&& get) {
	std::vector<double> values;
	for (const auto& run: runs) {
		values.push_back(static_cast<double>(get(run)));
	}
	std::sort(values.begin(), values.end());
	return values.empty() ? 0.0 : values[values.size() / 2];
}

// Sweeps KYTY_DRAW_PREP_HOT 2/3/4 x KYTY_DRAW_PREP_WAKE_BACKLOG 2/4/8 x steal policies (off; at
// once while 1 or 2 slots are unclaimed; the same after a 5 us wait), with six workers and the
// default spins, plus the U52 all-hot reference.
// Workloads: the Sky Garden start, heavy bursts, and heavy bursts on workers that now and then
// stall for 300 us (a preempted or late-waking worker holding its slot).
// Runs are interleaved across configurations, and medians over `reps` runs are printed. The
// producer's wall time is the command processor's critical path; worker CPU is what parking saves.
// `only`: a comma-separated list of hot/backlog/steal/after_us configurations to run instead of
// the sweep (e.g. "2/8/0/0,2/2/1/0"; the first one is the reference of the summary).
void MeasureWorkerGate(uint32_t reps, const char* only) {
	struct Workload {
		const char*           name;
		uint32_t              frames;
		std::vector<DrawSpec> draws;
		uint32_t              hiccup_per_mille;
	};
	std::vector<Workload> workloads;
	workloads.push_back({kSkyGardenFrame.name, 4, MakeFrames(kSkyGardenFrame, 4, 1), 0});
	workloads.push_back({kHeavyBurstFrame.name, 16, MakeFrames(kHeavyBurstFrame, 16, 2), 0});
	workloads.push_back({"heavy burst, workers stall 300 us on 0.3% of slots", 16,
	                     MakeFrames(kHeavyBurstFrame, 16, 3), 3});

	struct GateConfig {
		uint32_t              hot;
		uint32_t              backlog;
		DrawPrep::StealPolicy steal;
		const char*           note;
	};
	std::vector<GateConfig> configs;
	if (only != nullptr) {
		for (const char* p = only; *p != '\0';) {
			std::array<uint32_t, 4> fields {};
			for (size_t i = 0; i < fields.size(); i++) {
				char* end = nullptr;
				fields[i] = static_cast<uint32_t>(std::strtoul(p, &end, 10));
				const char expected = i + 1 < fields.size() ? '/' : ',';
				if (end == p || (*end != expected && !(expected == ',' && *end == '\0'))) {
					std::fprintf(stderr, "bad configuration list at '%s'\n", p);
					g_failures++;
					return;
				}
				p = *end == '\0' ? end : end + 1;
			}
			configs.push_back({fields[0], fields[1], {fields[2], uint64_t {fields[3]} * 1000u},
			                   configs.empty() ? "reference" : ""});
		}
	} else {
		configs.push_back({6, 8, {}, "all hot (U52)"});
		const std::array<DrawPrep::StealPolicy, 5> policies {
		    {{0, 0}, {1, 0}, {2, 0}, {1, 5000}, {2, 5000}}};
		for (const auto& steal: policies) {
			for (const uint32_t hot: {2u, 3u, 4u}) {
				for (const uint32_t backlog: {2u, 4u, 8u}) {
					configs.push_back({hot, backlog, steal,
					                   steal.min_unclaimed == 0 && hot == 2 && backlog == 8
					                       ? "U54 default"
					                       : ""});
				}
			}
		}
	}
	const auto steal_name = [](const DrawPrep::StealPolicy& steal) {
		char text[32];
		if (steal.min_unclaimed == 0) {
			std::snprintf(text, sizeof(text), "off");
		} else {
			std::snprintf(text, sizeof(text), "%u@%lluus", steal.min_unclaimed,
			              static_cast<unsigned long long>(steal.after_ns / 1000u));
		}
		return std::string(text);
	};
	std::vector<std::vector<double>> walls(configs.size());
	size_t                           u54 = 0; // the summary's reference
	for (size_t i = 0; i < configs.size(); i++) {
		if (std::strcmp(configs[i].note, "U54 default") == 0) {
			u54 = i;
		}
	}
	for (const auto& workload: workloads) {
		uint64_t prepare_ns = 0;
		for (const auto& draw: workload.draws) {
			prepare_ns += draw.prepare_ns;
		}
		std::vector<std::vector<PipelineResult>> results(configs.size());
		for (uint32_t rep = 0; rep < reps; rep++) {
			for (size_t i = 0; i < configs.size(); i++) {
				PipelineConfig config;
				config.hot              = configs[i].hot;
				config.wake_backlog     = configs[i].backlog;
				config.steal            = configs[i].steal;
				config.hiccup_per_mille = workload.hiccup_per_mille;
				results[i].push_back(RunPipeline(config, workload.draws));
				Check(results[i].back().ok, "measured pipeline is correct");
			}
		}
		std::printf("\n%s: %zu draws in %u frames, %.0f ms of Prepare, %u runs each (medians)\n",
		            workload.name, workload.draws.size(), workload.frames,
		            static_cast<double>(prepare_ns) / 1e6, reps);
		std::printf("  hot backlog steal    | wall ms (min-max)      ms/frame | spin ms | steals | "
		            "self | held | cold wakes | worker CPU ms\n");
		for (size_t i = 0; i < configs.size(); i++) {
			const auto& runs = results[i];
			const auto  wall = Median(runs, [](const PipelineResult& r) { return r.wall_ms; });
			double      low  = runs.front().wall_ms;
			double      high = low;
			for (const auto& r: runs) {
				low  = std::min(low, r.wall_ms);
				high = std::max(high, r.wall_ms);
			}
			walls[i].push_back(wall);
			std::printf("  %3u %7u %-8s | %7.1f (%5.1f-%5.1f) %8.2f | %7.2f | %6.0f | %4.0f | %4.0f | "
			            "%10.0f | %13.0f  %s\n",
			            configs[i].hot, configs[i].backlog, steal_name(configs[i].steal).c_str(), wall,
			            low, high, wall / workload.frames,
			            Median(runs, [](const PipelineResult& r) { return r.spin_ms; }),
			            Median(runs, [](const PipelineResult& r) { return r.steals; }),
			            Median(runs, [](const PipelineResult& r) { return r.self_prepared; }),
			            Median(runs, [](const PipelineResult& r) { return r.commit_waits; }),
			            Median(runs, [](const PipelineResult& r) { return r.cold_wakes; }),
			            Median(runs, [](const PipelineResult& r) { return r.worker_cpu_ms; }),
			            configs[i].note);
		}
	}
	// Each configuration's wall time against the reference, per workload and as a geometric mean.
	std::printf("\nwall time against %s (%%), per workload and geometric mean:\n",
	            configs[u54].note);
	std::vector<std::pair<double, size_t>> order;
	for (size_t i = 0; i < configs.size(); i++) {
		double log_sum = 0;
		for (size_t w = 0; w < workloads.size(); w++) {
			log_sum += std::log(walls[i][w] / walls[u54][w]);
		}
		order.emplace_back(std::exp(log_sum / static_cast<double>(workloads.size())), i);
	}
	std::sort(order.begin(), order.end());
	for (const auto& [mean, i]: order) {
		std::printf("  hot %u backlog %u steal %-8s:", configs[i].hot, configs[i].backlog,
		            steal_name(configs[i].steal).c_str());
		for (size_t w = 0; w < workloads.size(); w++) {
			std::printf(" %+6.2f", (walls[i][w] / walls[u54][w] - 1.0) * 100.0);
		}
		std::printf("  | mean %+6.2f  %s\n", (mean - 1.0) * 100.0, configs[i].note);
	}
}

// ---------------------------------------------------------------------------------------------
// Packet classification (S0/S6)

void TestPacketClassification() {
	using DrawPrep::ClassifyPacket;
	using DrawPrep::PacketClass;
	namespace Pm4 = Libs::Graphics::Pm4;
	const std::array<uint32_t, 4> no_body {};
	const auto classify = [&](uint32_t len, uint32_t op, uint32_t r = 0,
	                          const uint32_t* body = nullptr) {
		return ClassifyPacket(KYTY_PM4(len, op, r), body != nullptr ? body : no_body.data(), len);
	};
	const std::array<uint32_t, 11> safe_ops {
	    Pm4::IT_SET_CONTEXT_REG, Pm4::IT_SET_SH_REG,        Pm4::IT_SET_UCONFIG_REG,
	    Pm4::IT_SET_UCONFIG_REG_INDEX, Pm4::IT_INDEX_TYPE,  Pm4::IT_INDEX_BASE,
	    Pm4::IT_INDEX_BUFFER_SIZE, Pm4::IT_NUM_INSTANCES,   Pm4::IT_SET_BASE,
	    Pm4::IT_CLEAR_STATE,       Pm4::IT_PFP_SYNC_ME};
	for (const auto op: safe_ops) {
		Check(classify(3, op) == PacketClass::WindowSafe, "register/state packet is window-safe");
	}
	const std::array<uint32_t, 4> draw_ops {Pm4::IT_DRAW_INDEX_2, Pm4::IT_DRAW_INDEX_OFFSET_2,
	                                        Pm4::IT_DRAW_INDEX_AUTO,
	                                        Pm4::IT_DISPATCH_DRAW_PREAMBLE};
	for (const auto op: draw_ops) {
		Check(classify(5, op) == PacketClass::Draw, "direct draw packet is a draw");
	}
	const std::array<uint32_t, 27> fence_ops {
	    Pm4::IT_SET_CONTEXT_REG_INDIRECT, Pm4::IT_SET_SH_REG_INDIRECT,
	    Pm4::IT_SET_UCONFIG_REG_INDIRECT, Pm4::IT_DRAW_INDIRECT,
	    Pm4::IT_DRAW_INDEX_INDIRECT,      Pm4::IT_DRAW_INDIRECT_MULTI,
	    Pm4::IT_DRAW_INDEX_INDIRECT_MULTI, Pm4::IT_DISPATCH_DIRECT,
	    Pm4::IT_DISPATCH_INDIRECT,        Pm4::IT_WRITE_DATA,
	    Pm4::IT_WAIT_REG_MEM,             Pm4::IT_WAIT_REG_MEM_64,
	    Pm4::IT_EVENT_WRITE,              Pm4::IT_EVENT_WRITE_EOP,
	    Pm4::IT_EVENT_WRITE_EOS,          Pm4::IT_DMA_DATA,
	    Pm4::IT_ACQUIRE_MEM,              Pm4::IT_COPY_DATA,
	    Pm4::IT_COND_EXEC,                Pm4::IT_SET_PREDICATION,
	    Pm4::IT_WRITE_CONST_RAM,          Pm4::IT_DUMP_CONST_RAM,
	    Pm4::IT_INCREMENT_CE_COUNTER,     Pm4::IT_INCREMENT_DE_COUNTER,
	    Pm4::IT_WAIT_ON_CE_COUNTER,       Pm4::IT_WAIT_ON_DE_COUNTER_DIFF,
	    Pm4::IT_GET_LOD_STATS};
	for (const auto op: fence_ops) {
		Check(classify(5, op) == PacketClass::Fence, "packet with side effects is a fence");
	}
	Check(classify(5, Pm4::IT_REWIND) == PacketClass::Fence, "rewind is a fence");
	// Every opcode outside the two short lists is a fence (unknown opcodes included).
	uint32_t non_fences = 0;
	for (uint32_t op = 0; op < 256; op++) {
		if (op == Pm4::IT_NOP || op == Pm4::IT_INDIRECT_BUFFER) {
			continue;
		}
		non_fences += classify(3, op) != PacketClass::Fence ? 1u : 0u;
	}
	Check(non_fences == safe_ops.size() + draw_ops.size(),
	      "only the listed opcodes avoid draining the window");
	// Indirect buffers: plain call/chain vs conditional branch.
	Check(classify(4, Pm4::IT_INDIRECT_BUFFER) == PacketClass::WindowSafe,
	      "indirect buffer call is window-safe");
	Check(classify(14, Pm4::IT_INDIRECT_BUFFER) == PacketClass::Fence,
	      "conditional indirect branch is a fence");
	// NOPs, markers and custom R codes.
	Check(classify(4, Pm4::IT_NOP) == PacketClass::WindowSafe, "plain NOP is window-safe");
	const std::array<uint32_t, 3> marker_safe {0x68750000u, 0x68750004u, 0x6875000du};
	for (const auto marker: marker_safe) {
		const std::array<uint32_t, 4> body {marker, 0, 0, 0};
		Check(classify(4, Pm4::IT_NOP, Pm4::R_ZERO, body.data()) == PacketClass::WindowSafe,
		      "user-data marker is window-safe");
	}
	const std::array<uint32_t, 3> marker_flip {0x68750777u, 0x68750778u, 0x68750781u};
	for (const auto marker: marker_flip) {
		const std::array<uint32_t, 4> body {marker, 0, 0, 0};
		Check(classify(4, Pm4::IT_NOP, Pm4::R_ZERO, body.data()) == PacketClass::Fence,
		      "flip marker is a fence");
	}
	for (const auto r: {Pm4::R_CONTEXT_STATE, Pm4::R_PUSH_MARKER, Pm4::R_POP_MARKER}) {
		Check(classify(3, Pm4::IT_NOP, r) == PacketClass::WindowSafe,
		      "context state and markers are window-safe");
	}
	for (const auto r: {Pm4::R_RELEASE_MEM, Pm4::R_FLIP, Pm4::R_ACQUIRE_MEM,
	                    Pm4::R_WAIT_FLIP_DONE, Pm4::R_DISPATCH_RESET}) {
		Check(classify(8, Pm4::IT_NOP, r) == PacketClass::Fence, "custom operation is a fence");
	}
}

void TestFenceKinds() {
	using DrawPrep::ClassifyFence;
	using DrawPrep::FenceKind;
	namespace Pm4 = Libs::Graphics::Pm4;
	const auto kind = [](uint32_t op, uint32_t r = 0) { return ClassifyFence(KYTY_PM4(5, op, r)); };
	Check(kind(Pm4::IT_SET_SH_REG_INDIRECT) == FenceKind::RegIndirect &&
	          kind(Pm4::IT_SET_CONTEXT_REG_INDIRECT) == FenceKind::RegIndirect &&
	          kind(Pm4::IT_SET_UCONFIG_REG_INDIRECT) == FenceKind::RegIndirect,
	      "register-indirect fences");
	Check(kind(Pm4::IT_EVENT_WRITE) == FenceKind::EventWrite &&
	          kind(Pm4::IT_EVENT_WRITE_EOS) == FenceKind::EventWrite &&
	          kind(Pm4::IT_EVENT_WRITE_EOP) == FenceKind::EndOfPipe &&
	          kind(Pm4::IT_RELEASE_MEM) == FenceKind::EndOfPipe &&
	          kind(Pm4::IT_NOP, Pm4::R_RELEASE_MEM) == FenceKind::EndOfPipe,
	      "event and end-of-pipe fences");
	Check(kind(Pm4::IT_ACQUIRE_MEM) == FenceKind::AcquireMem &&
	          kind(Pm4::IT_NOP, Pm4::R_ACQUIRE_MEM) == FenceKind::AcquireMem &&
	          kind(Pm4::IT_WAIT_REG_MEM) == FenceKind::Wait &&
	          kind(Pm4::IT_COND_EXEC) == FenceKind::Wait &&
	          kind(Pm4::IT_NOP, Pm4::R_WAIT_FLIP_DONE) == FenceKind::Wait,
	      "acquire and wait fences");
	Check(kind(Pm4::IT_WRITE_DATA) == FenceKind::DataWrite &&
	          kind(Pm4::IT_DMA_DATA) == FenceKind::DataWrite &&
	          kind(Pm4::IT_NOP, Pm4::R_WRITE_DATA) == FenceKind::DataWrite &&
	          kind(Pm4::IT_WRITE_CONST_RAM) == FenceKind::ConstantEngine &&
	          kind(Pm4::IT_INCREMENT_DE_COUNTER) == FenceKind::ConstantEngine,
	      "data-write and constant-engine fences");
	Check(kind(Pm4::IT_DISPATCH_DIRECT) == FenceKind::Dispatch &&
	          kind(Pm4::IT_NOP, Pm4::R_DISPATCH_RESET) == FenceKind::Dispatch &&
	          kind(Pm4::IT_DRAW_INDEX_INDIRECT) == FenceKind::IndirectDraw &&
	          kind(Pm4::IT_CONTEXT_CONTROL) == FenceKind::ContextControl &&
	          kind(Pm4::IT_NOP, Pm4::R_FLIP) == FenceKind::Marker &&
	          kind(Pm4::IT_NOP, Pm4::R_ZERO) == FenceKind::Marker &&
	          kind(Pm4::IT_GET_LOD_STATS) == FenceKind::Other &&
	          kind(Pm4::IT_REWIND) == FenceKind::Other,
	      "dispatch, indirect-draw, context-control, marker and other fences");
}

// KYTY_SYNC_EPOCH: every fence advances the synchronization epoch except register loads from
// memory; window-safe packets and direct draws never do.
void TestSyncEpochPackets() {
	using DrawPrep::AdvancesSyncEpoch;
	namespace Pm4 = Libs::Graphics::Pm4;
	const std::array<uint32_t, 4> no_body {};
	const auto advances = [&](uint32_t len, uint32_t op, uint32_t r = 0) {
		return AdvancesSyncEpoch(KYTY_PM4(len, op, r), no_body.data(), len);
	};
	for (const auto op: {Pm4::IT_WAIT_REG_MEM, Pm4::IT_WAIT_REG_MEM_64, Pm4::IT_ACQUIRE_MEM,
	                     Pm4::IT_SURFACE_SYNC, Pm4::IT_WRITE_DATA, Pm4::IT_DMA_DATA,
	                     Pm4::IT_COPY_DATA, Pm4::IT_EVENT_WRITE, Pm4::IT_EVENT_WRITE_EOP,
	                     Pm4::IT_EVENT_WRITE_EOS, Pm4::IT_RELEASE_MEM, Pm4::IT_DUMP_CONST_RAM,
	                     Pm4::IT_DISPATCH_DIRECT, Pm4::IT_DRAW_INDEX_INDIRECT,
	                     Pm4::IT_GET_LOD_STATS, Pm4::IT_REWIND}) {
		Check(advances(5, op), "a synchronization or memory-writing packet advances the epoch");
	}
	for (const auto r: {Pm4::R_RELEASE_MEM, Pm4::R_ACQUIRE_MEM, Pm4::R_WRITE_DATA,
	                    Pm4::R_WAIT_FLIP_DONE, Pm4::R_FLIP}) {
		Check(advances(8, Pm4::IT_NOP, r), "a custom synchronization operation advances the epoch");
	}
	for (const auto op: {Pm4::IT_SET_SH_REG_INDIRECT, Pm4::IT_SET_CONTEXT_REG_INDIRECT,
	                     Pm4::IT_SET_UCONFIG_REG_INDIRECT}) {
		Check(!advances(5, op), "a register load from memory does not advance the epoch");
	}
	for (const auto op: {Pm4::IT_SET_SH_REG, Pm4::IT_SET_CONTEXT_REG, Pm4::IT_INDEX_BASE,
	                     Pm4::IT_DRAW_INDEX_2, Pm4::IT_DRAW_INDEX_AUTO}) {
		Check(!advances(5, op), "a register write or a direct draw does not advance the epoch");
	}
	Check(!advances(4, Pm4::IT_INDIRECT_BUFFER) && advances(14, Pm4::IT_INDIRECT_BUFFER),
	      "an indirect call does not advance the epoch, a conditional branch on memory does");
}

void TestRegisterIndirectPairs() {
	using DrawPrep::RegisterIndirectPairs;
	using DrawPrep::RegisterIndirectRange;
	namespace Pm4 = Libs::Graphics::Pm4;
	// As pm4Handlers.cpp decodes it: dword0 address low (4-byte aligned), dword1 high, dword3 low
	// 14 bits the pair count.
	const std::array<uint32_t, 4> body {0x12345677u, 0x2u, 0xdeadbeefu, 0xffffc003u};
	for (const auto op: {Pm4::IT_SET_SH_REG_INDIRECT, Pm4::IT_SET_UCONFIG_REG_INDIRECT,
	                     Pm4::IT_SET_CONTEXT_REG_INDIRECT}) {
		RegisterIndirectRange range;
		Check(RegisterIndirectPairs(KYTY_PM4(5, op, 0), body.data(), 5, range) &&
		          range.address == 0x212345674ull && range.size == 3u * 8u,
		      "register-indirect pairs decode");
		Check(!RegisterIndirectPairs(KYTY_PM4(5, op, 0), body.data(), 4, range),
		      "a truncated register-indirect packet is refused");
		Check(!RegisterIndirectPairs(KYTY_PM4(6, op, 0), body.data(), 6, range),
		      "an unexpected register-indirect length is refused");
	}
	const std::array<uint32_t, 4> empty {0x1000u, 0u, 0u, 0xffffc000u};
	RegisterIndirectRange range;
	Check(RegisterIndirectPairs(KYTY_PM4(5, Pm4::IT_SET_SH_REG_INDIRECT, 0), empty.data(), 5,
	                            range) &&
	          range.size == 0,
	      "a register-indirect packet without pairs has an empty range");
	Check(!RegisterIndirectPairs(KYTY_PM4(5, Pm4::IT_SET_SH_REG, 0), body.data(), 5, range) &&
	          !RegisterIndirectPairs(KYTY_PM4(5, Pm4::IT_WRITE_DATA, 0), body.data(), 5, range),
	      "other packets have no register-indirect range");
}

} // namespace

// ---------------------------------------------------------------------------------------------
// CpGaps::Ledger (KYTY_CP_GAP_STATS): exclusive time per category

void TestGapLedgerExclusive() {
	using CpGaps::Cat;
	CpGaps::Ledger ledger;
	ledger.Enter(Cat::Slice, 0);
	ledger.Enter(Cat::Resolver, 10);
	ledger.Enter(Cat::DrawOp, 20);
	ledger.Enter(Cat::Commit, 25);
	ledger.Exit(65); // commit 40
	ledger.Exit(70); // draw op 50 - 40 = 10
	ledger.Enter(Cat::DispatchOp, 80);
	ledger.Exit(100); // 20
	ledger.Exit(110); // resolver 100 - 50 - 20 = 30
	ledger.Exit(120); // slice 120 - 100 = 20
	Check(ledger.Ticks(Cat::Commit) == 40, "gap ledger: commit time");
	Check(ledger.Ticks(Cat::DrawOp) == 10, "gap ledger: draw op time outside its commit");
	Check(ledger.Ticks(Cat::DispatchOp) == 20, "gap ledger: dispatch time");
	Check(ledger.Ticks(Cat::Resolver) == 30, "gap ledger: resolver time without its ops");
	Check(ledger.Ticks(Cat::Slice) == 20, "gap ledger: slice time without the resolver");
	uint64_t total = 0;
	for (size_t i = 0; i < CpGaps::CatCount; i++) {
		total += ledger.Ticks(static_cast<Cat>(i));
	}
	Check(total == 120, "gap ledger: categories add up to the outermost scope");
	Check(ledger.Calls(Cat::Commit) == 1 && ledger.Calls(Cat::DrawOp) == 1, "gap ledger: calls");
	Check(ledger.Depth() == 0, "gap ledger: balanced");
}

void TestGapLedgerPhasesAndNested() {
	using CpGaps::Cat;
	CpGaps::Ledger ledger;
	ledger.Enter(Cat::DispatchOp, 0);
	ledger.Switch(Cat::DispatchProgram, 5); // 5 dispatch op
	ledger.Enter(Cat::Protect, 8);
	ledger.Exit(10);                       // 2 protect
	ledger.Switch(Cat::DispatchBind, 15);  // program 15 - 5 - 2 = 8
	ledger.AddNested(Cat::Fault, 3);       // 3 fault (measured elsewhere)
	ledger.Switch(Cat::DispatchEmit, 30);  // bind 15 - 3 = 12
	ledger.Exit(40);                       // emit 10
	Check(ledger.Ticks(Cat::DispatchOp) == 5, "gap ledger: phase before the first switch");
	Check(ledger.Ticks(Cat::DispatchProgram) == 8, "gap ledger: phase without its nested scope");
	Check(ledger.Ticks(Cat::Protect) == 2, "gap ledger: nested scope inside a phase");
	Check(ledger.Ticks(Cat::DispatchBind) == 12, "gap ledger: phase without nested time");
	Check(ledger.Ticks(Cat::Fault) == 3, "gap ledger: nested time");
	Check(ledger.Ticks(Cat::DispatchEmit) == 10, "gap ledger: last phase");
	// A nested scope inside a commit is reported as in-commit time; the commit itself is not.
	CpGaps::Ledger commit;
	commit.Enter(Cat::DrawOp, 0);
	commit.Enter(Cat::HeadWait, 1);
	commit.Exit(4);
	commit.Enter(Cat::Commit, 4);
	commit.Enter(Cat::GpuWaitDrain, 10);
	commit.Exit(30);
	commit.AddNested(Cat::Protect, 5);
	commit.Exit(50);
	commit.Exit(52);
	Check(commit.Ticks(Cat::Commit) == 21 && commit.InCommit(Cat::Commit) == 0,
	      "gap ledger: commit time without nested waits");
	Check(commit.Ticks(Cat::GpuWaitDrain) == 20 && commit.InCommit(Cat::GpuWaitDrain) == 20,
	      "gap ledger: GPU wait inside a commit");
	Check(commit.InCommit(Cat::Protect) == 5, "gap ledger: nested time inside a commit");
	Check(commit.Ticks(Cat::HeadWait) == 3 && commit.InCommit(Cat::HeadWait) == 0,
	      "gap ledger: head wait outside the commit");
	Check(commit.Ticks(Cat::DrawOp) == 1 + 2, "gap ledger: draw op around head and commit");
}

void TestGapLedgerFlushAndOverflow() {
	using CpGaps::Cat;
	CpGaps::Ledger ledger;
	ledger.Enter(Cat::Slice, 0);
	ledger.Enter(Cat::Idle, 10);
	ledger.Flush(100); // a report while idle
	Check(ledger.Ticks(Cat::Slice) == 10 && ledger.Ticks(Cat::Idle) == 90,
	      "gap ledger: flush accounts the open scopes up to the report");
	ledger.ResetTotals();
	ledger.Exit(150);
	ledger.Exit(160);
	Check(ledger.Ticks(Cat::Idle) == 50 && ledger.Ticks(Cat::Slice) == 10,
	      "gap ledger: open scopes continue after a flush");
	CpGaps::Ledger deep;
	for (uint32_t i = 0; i < CpGaps::Ledger::MaxDepth + 3; i++) {
		deep.Enter(Cat::OtherOp, i);
	}
	Check(deep.Overflows() == 3, "gap ledger: overflow counted");
	for (uint32_t i = 0; i < CpGaps::Ledger::MaxDepth + 3; i++) {
		deep.Exit(100 + i);
	}
	Check(deep.Depth() == 0, "gap ledger: balanced after overflow");
	deep.Exit(200); // unbalanced exit is ignored
	Check(deep.Depth() == 0, "gap ledger: extra exit ignored");
	// Without an attached thread state the scopes do nothing (the diagnostic is off in tests).
	{
		const CpGaps::Scope scope(Cat::Commit);
		CpGaps::Phase(Cat::DispatchEmit);
		CpGaps::NestedNs(Cat::Fault, 10);
	}
	Check(CpGaps::t_state == nullptr, "gap ledger: no state on other threads");
}

int main(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "--measure-worker-gate") == 0) {
		const auto reps = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 5ul;
		MeasureWorkerGate(static_cast<uint32_t>(std::clamp(reps, 1ul, 100ul)),
		                  argc > 3 ? argv[3] : nullptr);
		return g_failures == 0 ? 0 : 1;
	}
	TestLogEmptyIntervalIsClean();
	TestLogIntersection();
	TestLogEmptyRangeNeverIntersects();
	TestLogBumpWithoutReaders();
	TestGlobalLogAlwaysRecords();
	TestLogOverflow();
	TestLogConcurrentAppends();
	TestLogConcurrentWrap();
	TestIntersectsAny();
	TestReadSetCoalesces();
	TestReadSetPageBoundary();
	TestReadSetInconsistent();
	TestReadSetLimits();
	TestReadSetDigests();
	TestReadSetCertificate();
	TestReadSetValidateInPlace();
	TestRecordScopeNests();
	TestPacketClassification();
	TestFenceKinds();
	TestSyncEpochPackets();
	TestRegisterIndirectPairs();
	TestWindowSingleThread();
	TestWindowConcurrent(4, 8, 200000);  // tiny window: constant wrap-around and races
	TestWindowConcurrent(32, 6, 200000); // the default shape
	TestWindowConcurrent(32, 1, 50000);
	TestWorkerGateBasics();
	TestAwaitHead();
	TestWorkerGateStress();
	TestColdTokenStranded();
	TestColdTokenStress();
	TestStealStress();
	TestGapLedgerExclusive();
	TestGapLedgerPhasesAndNested();
	TestGapLedgerFlushAndOverflow();
	if (g_failures != 0) {
		std::fprintf(stderr, "DrawPrepTests: %d failure(s)\n", g_failures);
		return 1;
	}
	std::puts("DrawPrepTests: all cases passed");
	return 0;
}
