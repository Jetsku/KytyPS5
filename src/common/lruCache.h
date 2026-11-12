#ifndef EMULATOR_SRC_COMMON_LRUCACHE_H_
#define EMULATOR_SRC_COMMON_LRUCACHE_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace Common {

template <typename Object, typename Tick>
class LeastRecentlyUsedCache {
	struct Item {
		Object object {};
		Tick   tick {};
		Item*  next = nullptr;
		Item*  prev = nullptr;
		uint64_t generation = 0;
		bool linked = false;
	};

	// Items in fixed chunks that never move (Item pointers link them): one indexed load per access.
	// (A std::deque of 40-byte items holds one item per block in MSVC's library, so every access
	// was a block-map lookup and another heap line; chenxiao07/KytyPS5 02041c814.)
	class Items {
	public:
		static constexpr size_t ChunkBits = 10;
		static constexpr size_t ChunkSize = size_t {1} << ChunkBits;

		[[nodiscard]] Item& operator[](size_t id) noexcept {
			return m_chunks[id >> ChunkBits][id & (ChunkSize - 1)];
		}
		[[nodiscard]] const Item& operator[](size_t id) const noexcept {
			return m_chunks[id >> ChunkBits][id & (ChunkSize - 1)];
		}
		[[nodiscard]] size_t size() const noexcept { return m_size; }
		[[nodiscard]] bool   empty() const noexcept { return m_size == 0; }
		void                 emplace_back() {
			if ((m_size >> ChunkBits) == m_chunks.size()) {
				m_chunks.push_back(std::make_unique<Item[]>(ChunkSize));
			}
			m_size++;
		}

	private:
		std::vector<std::unique_ptr<Item[]>> m_chunks;
		size_t                               m_size = 0;
	};

public:
	struct Cursor {
		size_t id = static_cast<size_t>(-1);
		uint64_t generation = 0;
	};

	[[nodiscard]] size_t Insert(Object object, Tick tick) {
		const auto id   = Build();
		auto&      item = m_items[id];
		item.object     = std::move(object);
		item.tick       = tick;
		++item.generation;
		item.linked     = true;
		Attach(item);
		return id;
	}

	void Touch(size_t id, Tick tick) {
		auto& item = m_items[id];
		if (item.tick >= tick) {
			return;
		}
		item.tick = tick;
		if (&item != m_last) {
			Detach(item);
			Attach(item);
		}
	}

	// The item's tick (Insert and Touch set it) and whether it is linked (Free unlinks it). Read by
	// TextureCache's LRU mirror checks (KYTY_IMAGE_LRU_SKIP_VERIFY).
	[[nodiscard]] Tick TickOf(size_t id) const { return m_items[id].tick; }
	[[nodiscard]] bool Linked(size_t id) const { return id < m_items.size() && m_items[id].linked; }

	void Free(size_t id) {
		auto& item = m_items[id];
		Detach(item);
		item.next = nullptr;
		item.prev = nullptr;
		item.linked = false;
		m_free.push_back(id);
	}

	template <typename Function>
	void ForEachItemBelow(Tick tick, Function&& function) {
		constexpr bool ReturnsBool = std::is_same_v<std::invoke_result_t<Function, Object>, bool>;
		for (auto* item = m_first; item != nullptr;) {
			if (item->tick > tick) {
				return;
			}
			auto* next = item->next;
			if constexpr (ReturnsBool) {
				if (function(item->object)) {
					return;
				}
			} else {
				function(item->object);
			}
			item = next;
		}
	}

	// Rotate through physical slots rather than links: touching the saved boundary
	// item can move its LRU link to the tail, but must not hide intervening entries.
	// Each visit reads the currently linked generation and applies the age cutoff;
	// no pointers or objects from a previous scan survive slot deletion/reuse.
	// The callback must not mutate this cache while the scan is in progress.
	template <typename Function>
	void ScanItemsBelow(Tick tick, Cursor& cursor, size_t limit, Function&& function) {
		if (m_items.empty()) {
			cursor = {};
			return;
		}
		auto index = cursor.id < m_items.size() ? cursor.id : size_t {0};
		const auto count = limit < m_items.size() ? limit : m_items.size();
		for (size_t visited = 0; visited < count; ++visited) {
			auto& item = m_items[index];
			if (item.linked && item.tick <= tick) {
				function(item.object);
			}
			index = index + 1 == m_items.size() ? 0 : index + 1;
		}
		cursor = Cursor {index, m_items[index].generation};
	}

private:
	[[nodiscard]] size_t Build() {
		if (m_free.empty()) {
			const auto id = m_items.size();
			m_items.emplace_back();
			return id;
		}
		const auto id = m_free.front();
		m_free.pop_front();
		return id;
	}

	void Attach(Item& item) {
		if (m_first == nullptr) {
			m_first = &item;
		}
		if (m_last == nullptr) {
			m_last = &item;
			return;
		}
		item.prev    = m_last;
		m_last->next = &item;
		item.next    = nullptr;
		m_last       = &item;
	}

	void Detach(Item& item) {
		if (item.prev != nullptr) {
			item.prev->next = item.next;
		}
		if (item.next != nullptr) {
			item.next->prev = item.prev;
		}
		if (m_first == &item) {
			m_first = item.next;
		}
		if (m_last == &item) {
			m_last = item.prev;
		}
	}

	Items              m_items;
	std::deque<size_t> m_free;
	Item*              m_first = nullptr;
	Item*              m_last  = nullptr;
};

} // namespace Common

#endif // EMULATOR_SRC_COMMON_LRUCACHE_H_
