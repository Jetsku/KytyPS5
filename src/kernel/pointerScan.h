#ifndef KYTY_KERNEL_POINTER_SCAN_H_
#define KYTY_KERNEL_POINTER_SCAN_H_

// Post-mortem search of memory for 8-byte values that look like a faulting GPU address.
//
// VK_EXT_device_fault reports the address a shader read or wrote when the GPU was lost, rounded to a
// page. The invalid address is usually a pointer the shader loaded from memory, so finding where that
// value sits in guest memory points at the buffer (and the draw) that fed it. A value matches when its
// low 48 bits fall in [low, high]: the GPU reports 48-bit addresses, and a 0xDEADBEEFDEADBEEF fill has
// the low 48 bits 0xBEEFDEADBEEF.
//
// Only the matching logic lives here (no OS or Vulkan types), so it is testable on its own. The caller
// walks the memory and hands over blocks; GuestBackingStore::ScanForAddressRange (memoryAddressSpace.inc)
// does that for the guest mappings.

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace Libs::LibKernel::Memory {

class PointerScan {
public:
	static constexpr uint64_t kAddressMask = 0x0000FFFFFFFFFFFFull;
	static constexpr size_t   kMaxRuns     = 4096;

	// `low` and `high` are inclusive and below 2^48.
	PointerScan(uint64_t low, uint64_t high): m_low(low), m_span(high - low) {}

	[[nodiscard]] bool Matches(uint64_t value) const { return ((value & kAddressMask) - m_low) <= m_span; }

	// A block of 8-byte values that starts at byte `offset` of the scanned space. A quick pass over the
	// whole block rejects the common case (no match) without a branch per value.
	void AddBlock(const uint64_t* words, size_t count, uint64_t offset) {
		uint64_t any = 0;
		for (size_t i = 0; i < count; i++) {
			any |= static_cast<uint64_t>(Matches(words[i]));
		}
		if (any == 0) {
			return;
		}
		for (size_t i = 0; i < count; i++) {
			if (Matches(words[i])) {
				Record(offset + i * sizeof(uint64_t), words[i]);
			}
		}
	}

	void NoteScanned(uint64_t bytes) { m_scanned_bytes += bytes; }

	// A consecutive stretch of equal matching values (a filled region shows up as one run).
	struct Run {
		uint64_t offset  = 0;
		uint64_t qwords  = 0;
		uint64_t value   = 0;
	};

	[[nodiscard]] uint64_t           Hits() const { return m_hits; }
	[[nodiscard]] uint64_t           ScannedBytes() const { return m_scanned_bytes; }
	[[nodiscard]] uint64_t           DroppedRuns() const { return m_dropped_runs; }
	[[nodiscard]] const std::vector<Run>& Runs() const { return m_runs; }

	// The runs to print: the `limit` longest, then ordered by offset.
	[[nodiscard]] std::vector<Run> TopRuns(size_t limit) const {
		std::vector<Run> top = m_runs;
		std::stable_sort(top.begin(), top.end(), [](const Run& a, const Run& b) { return a.qwords > b.qwords; });
		if (top.size() > limit) {
			top.resize(limit);
		}
		std::sort(top.begin(), top.end(), [](const Run& a, const Run& b) { return a.offset < b.offset; });
		return top;
	}

	// `describe(offset)` may add where an offset lives (for example the guest address); it can return "".
	template <typename Describe>
	[[nodiscard]] std::string Format(Describe&& describe, size_t limit = 12) const {
		char line[320];
		std::string text;
		std::snprintf(line, sizeof(line),
		              "looked for 8-byte values with low 48 bits in 0x%016" PRIx64 "..0x%016" PRIx64 "\n", m_low,
		              m_low + m_span);
		text += line;
		std::snprintf(line, sizeof(line), "hits: %" PRIu64 " values in %zu runs%s\n", m_hits, m_runs.size(),
		              m_dropped_runs != 0 ? " (more runs than recorded)" : "");
		text += line;
		for (const Run& run: TopRuns(limit)) {
			std::snprintf(line, sizeof(line), "  at 0x%" PRIx64 "%s: %" PRIu64 " x 0x%016" PRIx64 " (%" PRIu64 " bytes)\n",
			              run.offset, describe(run.offset).c_str(), run.qwords, run.value,
			              run.qwords * sizeof(uint64_t));
			text += line;
		}
		return text;
	}

private:
	void Record(uint64_t offset, uint64_t value) {
		m_hits++;
		if (!m_runs.empty()) {
			Run& last = m_runs.back();
			if (last.value == value && last.offset + last.qwords * sizeof(uint64_t) == offset) {
				last.qwords++;
				return;
			}
		}
		if (m_runs.size() >= kMaxRuns) {
			m_dropped_runs++;
			return;
		}
		m_runs.push_back({offset, 1, value});
	}

	uint64_t         m_low;
	uint64_t         m_span;
	uint64_t         m_hits          = 0;
	uint64_t         m_scanned_bytes = 0;
	uint64_t         m_dropped_runs  = 0;
	std::vector<Run> m_runs;
};

} // namespace Libs::LibKernel::Memory

#endif
