#ifndef KYTY_RENDERER_OCCLUSION_PAIRS_H_
#define KYTY_RENDERER_OCCLUSION_PAIRS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Libs::Graphics {

// Classifies ZPASS dumps (EVENT_WRITE ZPASS_DONE) into begin/end dumps of interleaved pairs.
// A begin dump at A writes the 16 per-DB counters to A + db * 16; its end dump is at A + 8 and
// fills the other qword of every 16-byte slot. A is only 8-byte aligned: Astro Bot allocates its
// query slots 8 bytes apart, so a pair may start at A % 16 == 8 (its end then sits at
// A % 16 == 0). The pair is therefore identified by address, not by the low address bits: a dump
// at X ends the pair begun at X - 8 if that pair is open, and begins a new pair otherwise.
// Open pairs older than MaxAgeDumps dumps are dropped (a begin whose end never came must not
// turn a later begin 8 bytes after it into an end).
class OcclusionDumpPairs {
public:
	enum class Kind : uint8_t { Begin, RepeatedBegin, End };

	static constexpr size_t   MaxOpen     = 256;
	static constexpr uint64_t MaxAgeDumps = uint64_t {1} << 16u;

	Kind Observe(uint64_t address) {
		const auto serial = m_serial++;
		ExpireBefore(serial);
		const auto end = Find(address - 8u);
		if (end != m_open.end()) {
			m_open.erase(end);
			return Kind::End;
		}
		if (Find(address) != m_open.end()) {
			// A repeated begin keeps its pair open (the later end still differs against the
			// newest begin).
			return Kind::RepeatedBegin;
		}
		if (m_open.size() >= MaxOpen) {
			m_open.erase(m_open.begin());
			++m_dropped;
		}
		m_open.push_back({address, serial});
		return Kind::Begin;
	}

	[[nodiscard]] bool   Empty() const noexcept { return m_open.empty(); }
	[[nodiscard]] size_t OpenCount() const noexcept { return m_open.size(); }
	// Open pairs dropped (too many open, or too old) since construction.
	[[nodiscard]] uint64_t Dropped() const noexcept { return m_dropped; }
	void                   Clear() noexcept { m_open.clear(); }

private:
	struct Open {
		uint64_t address;
		uint64_t serial;
	};
	std::vector<Open>::iterator Find(uint64_t address) {
		return std::find_if(m_open.begin(), m_open.end(),
		                    [address](const Open& open) { return open.address == address; });
	}
	void ExpireBefore(uint64_t serial) {
		if (serial < MaxAgeDumps || m_open.empty() || m_open.front().serial + MaxAgeDumps > serial) {
			return;
		}
		const auto old = std::remove_if(m_open.begin(), m_open.end(), [serial](const Open& open) {
			return open.serial + MaxAgeDumps <= serial;
		});
		m_dropped += static_cast<uint64_t>(m_open.end() - old);
		m_open.erase(old, m_open.end());
	}
	std::vector<Open> m_open; // in begin order (oldest first)
	uint64_t          m_serial  = 0;
	uint64_t          m_dropped = 0;
};

} // namespace Libs::Graphics

#endif
