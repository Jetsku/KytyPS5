#ifndef EMULATOR_SRC_COMMON_SLOTVECTOR_H_
#define EMULATOR_SRC_COMMON_SLOTVECTOR_H_

#include "common/assert.h"
#include "common/abi.h"

#include <atomic>
#include <compare>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace Common {

struct SlotId {
	static constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();

	constexpr SlotId() noexcept = default;
	constexpr SlotId(uint32_t value) noexcept: index(value), generation(1) {}
	constexpr SlotId(uint32_t value, uint32_t slot_generation) noexcept
	    : index(value), generation(slot_generation) {}

	[[nodiscard]] constexpr explicit operator bool() const noexcept { return index != INVALID_INDEX; }
	constexpr auto operator<=>(const SlotId&) const noexcept = default;

	uint32_t index = INVALID_INDEX;
	uint32_t generation = 0;
};

// KYTY_CP_COMMIT=slots (graphics/host_gpu/renderer/cpCommit.h sets it once): SlotVector lookups
// read a dense per-slot record (state word and value pointer) instead of the slot itself, whose
// generation and optional flag sit behind the (large) value in its deque node (MSVC's deque holds
// one such element per block). A lookup then touches a small record array that stays cached, not
// a separate cache line of every slot. The records are always maintained; only the lookups
// switch, so turning it on at any time is exact. Records live in fixed chunks that never move, so
// a lookup is exactly as safe as the deque one wherever that one is (no reallocation to race).
inline std::atomic<bool> g_slot_vector_dense {false};

// Stable-address slot storage for cache resources that are intentionally non-movable.
template <typename T>
class SlotVector {
public:
	SlotVector() = default;
	KYTY_CLASS_NO_COPY(SlotVector);

	[[nodiscard]] T& operator[](SlotId id) noexcept {
		if (g_slot_vector_dense.load(std::memory_order_relaxed)) {
			EXIT_IF(!dense_allocated(id));
			return *dense_record(id.index).value;
		}
		EXIT_IF(!is_allocated(id));
		return *m_values[id.index].value;
	}

	[[nodiscard]] const T& operator[](SlotId id) const noexcept {
		if (g_slot_vector_dense.load(std::memory_order_relaxed)) {
			EXIT_IF(!dense_allocated(id));
			return *dense_record(id.index).value;
		}
		EXIT_IF(!is_allocated(id));
		return *m_values[id.index].value;
	}

	[[nodiscard]] T* try_get(SlotId id) noexcept {
		if (g_slot_vector_dense.load(std::memory_order_relaxed)) {
			return dense_allocated(id) ? dense_record(id.index).value : nullptr;
		}
		return is_allocated(id) ? &*m_values[id.index].value : nullptr;
	}

	[[nodiscard]] const T* try_get(SlotId id) const noexcept {
		if (g_slot_vector_dense.load(std::memory_order_relaxed)) {
			return dense_allocated(id) ? dense_record(id.index).value : nullptr;
		}
		return is_allocated(id) ? &*m_values[id.index].value : nullptr;
	}

	[[nodiscard]] bool is_allocated(SlotId id) const noexcept {
		if (g_slot_vector_dense.load(std::memory_order_relaxed)) {
			return dense_allocated(id);
		}
		return id && id.index < m_values.size() &&
		       m_values[id.index].generation == id.generation && m_values[id.index].value.has_value();
	}

	template <typename... Args>
	[[nodiscard]] SlotId insert(Args&&... args) {
		uint32_t index = 0;
		if (m_free_list.empty()) {
			index = static_cast<uint32_t>(m_values.size());
			// The record first: a throw below leaves it describing an unallocated slot.
			dense_push_dead(index);
			m_values.emplace_back();
			m_values.back().value.emplace(std::forward<Args>(args)...);
		} else {
			index = m_free_list.back();
			m_free_list.pop_back();
			EXIT_IF(m_values[index].value.has_value());
			m_values[index].value.emplace(std::forward<Args>(args)...);
		}
		auto& slot          = m_values[index];
		dense_record(index) = {LiveState(slot.generation), &*slot.value};
		++m_size;
		return SlotId {index, slot.generation};
	}

	void erase(SlotId id) noexcept {
		EXIT_IF(!is_allocated(id));
		auto& slot = m_values[id.index];
		// Same order as the slot itself: while the value is destroyed, both lookups still find it.
		slot.value.reset();
		if (++slot.generation == 0) {
			slot.generation = 1;
		}
		dense_record(id.index) = {DeadState(slot.generation), nullptr};
		m_free_list.push_back(id.index);
		--m_size;
	}

	[[nodiscard]] size_t size() const noexcept { return m_size; }
	[[nodiscard]] size_t capacity() const noexcept { return m_values.size(); }

	template <typename F>
	void ForEach(F&& fn) {
		for (uint32_t index = 0; index < m_values.size(); ++index) {
			if (m_values[index].value) {
				fn(SlotId {index, m_values[index].generation}, *m_values[index].value);
			}
		}
	}

	template <typename F>
	void ForEach(F&& fn) const {
		for (uint32_t index = 0; index < m_values.size(); ++index) {
			if (m_values[index].value) {
				fn(SlotId {index, m_values[index].generation}, *m_values[index].value);
			}
		}
	}

private:
	struct Slot {
		std::optional<T> value;
		uint32_t         generation = 1;
	};
	// Per slot: the generation with bit 32 set while a value lives there, and the value's address.
	struct DenseRecord {
		uint64_t state = 0;
		T*       value = nullptr;
	};
	[[nodiscard]] static constexpr uint64_t LiveState(uint32_t generation) noexcept {
		return (uint64_t {1} << 32u) | generation;
	}
	[[nodiscard]] static constexpr uint64_t DeadState(uint32_t generation) noexcept {
		return generation;
	}
	static constexpr uint32_t DenseChunkBits = 10;
	static constexpr uint32_t DenseChunkSize = 1u << DenseChunkBits;
	static constexpr uint32_t DenseMaxChunks = 1u << 12; // 4M slots
	[[nodiscard]] DenseRecord& dense_record(uint32_t index) noexcept {
		return m_dense_chunks[index >> DenseChunkBits][index & (DenseChunkSize - 1u)];
	}
	[[nodiscard]] const DenseRecord& dense_record(uint32_t index) const noexcept {
		return m_dense_chunks[index >> DenseChunkBits][index & (DenseChunkSize - 1u)];
	}
	// An invalid id's index (INVALID_INDEX) is never below the size.
	[[nodiscard]] bool dense_allocated(SlotId id) const noexcept {
		return id.index < m_dense_size && dense_record(id.index).state == LiveState(id.generation);
	}
	void dense_push_dead(uint32_t index) {
		EXIT_IF(index != m_dense_size);
		const auto chunk = index >> DenseChunkBits;
		EXIT_IF(chunk >= DenseMaxChunks);
		if (m_dense_chunks == nullptr) {
			m_dense_chunks = std::make_unique<std::unique_ptr<DenseRecord[]>[]>(DenseMaxChunks);
		}
		if (m_dense_chunks[chunk] == nullptr) {
			m_dense_chunks[chunk] = std::make_unique<DenseRecord[]>(DenseChunkSize);
		}
		dense_record(index) = {DeadState(1), nullptr};
		m_dense_size        = index + 1u;
	}

	std::deque<Slot>                                 m_values;
	std::unique_ptr<std::unique_ptr<DenseRecord[]>[]> m_dense_chunks;
	uint32_t                                         m_dense_size = 0;
	std::vector<uint32_t>                            m_free_list;
	size_t                                           m_size = 0;
};

} // namespace Common

template <>
struct std::hash<Common::SlotId> {
	[[nodiscard]] size_t operator()(Common::SlotId id) const noexcept {
		return std::hash<uint64_t> {}((static_cast<uint64_t>(id.generation) << 32u) | id.index);
	}
};

#endif // EMULATOR_SRC_COMMON_SLOTVECTOR_H_
