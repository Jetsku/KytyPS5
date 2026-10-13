#ifndef KYTY_RENDERER_OCCLUSION_QUERY_STATE_H_
#define KYTY_RENDERER_OCCLUSION_QUERY_STATE_H_

#include <cstdint>

namespace Libs::Graphics {

// The occlusion counter's query bookkeeping (OcclusionCounter): whether a query is open, and how
// many ended queries wait in the pool for their reduction. Queries use the pool indices in order;
// the query begun next uses index Pending(). Kept apart from the Vulkan recording so the recovery
// paths below can be tested without a GPU.
class OcclusionQueryState {
public:
	static constexpr uint32_t NoQuery = UINT32_MAX;

	explicit constexpr OcclusionQueryState(uint32_t capacity) noexcept: m_capacity(capacity) {}

	[[nodiscard]] bool     Active() const noexcept { return m_active; }
	[[nodiscard]] uint32_t Pending() const noexcept { return m_pending; }
	[[nodiscard]] uint32_t Capacity() const noexcept { return m_capacity; }
	// No index left for another query: the pending ones must be reduced first.
	[[nodiscard]] bool Full() const noexcept { return m_pending >= m_capacity; }
	[[nodiscard]] bool CanBegin() const noexcept { return !m_active && m_pending < m_capacity; }

	// Opens the query at index Pending() (the caller checked CanBegin()) and returns that index.
	uint32_t Begin() noexcept {
		m_active = true;
		return m_pending;
	}

	// Closes the open query and returns its index, or NoQuery when none is open. The state changes
	// before the caller records the end, so an End reached again while that is recorded (a
	// re-entrant end of the rendering instance) finds nothing open and cannot end a second index.
	uint32_t End() noexcept {
		if (!m_active) {
			return NoQuery;
		}
		m_active = false;
		return m_pending++;
	}

	// The pending queries were reduced (their indices may be reset and reused).
	void Reduced() noexcept { m_pending = 0; }

	// What a scope about to be prepared (outside rendering) finds left over: a query still open
	// (its end was skipped) and/or a full pool. Ends the open query in the state; the caller records
	// that end, then reduces the pool when `reduce` is set.
	struct Recovery {
		uint32_t stale_query = NoQuery; // the open query just ended, or NoQuery
		bool     reduce      = false;   // the pool is full: reduce before the next begin
	};
	[[nodiscard]] Recovery BeforePrepare() noexcept {
		Recovery recovery;
		recovery.stale_query = End();
		recovery.reduce      = Full();
		return recovery;
	}

private:
	uint32_t m_capacity;
	uint32_t m_pending = 0;
	bool     m_active  = false;
};

} // namespace Libs::Graphics

#endif
