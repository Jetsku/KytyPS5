#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXTURESETCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXTURESETCACHE_H_

#include <algorithm>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <xxhash.h>

// KYTY_XFRAME_REUSE (drawPrep/xframeReuse.h): the cross-frame store of resolved stage texture sets.
//
// A stage's texture set is what RenderExecutor::PrepareBindings resolved for one program and its
// T# words: the bindings (texture-cache image, final description, TextureBindingMemo entry tag).
// PreparedBindings keeps the current set and three earlier ones (texture_history); a set that
// falls out of that history is kept here, under the hash of its program and words, and given
// back when a later draw (usually of a later frame) binds the same program with the same words.
// Sets are moved (their vectors swapped), never copied: a slot's displaced vectors become the
// storage the caller resolves into. Nothing here says a set is still valid: the caller
// revalidates every binding exactly as for its current set (TextureBindingMemo::TryRepeatResolve)
// and resolves it again otherwise.
//
// Program: the program pointer type (compiled programs have stable addresses for the pipeline
// cache's lifetime); Word: a trivially copyable T# value whose == compares its bytes; Binding:
// the resolved binding type.
namespace Libs::Graphics {

template <typename Program, typename Word, typename Binding>
class TextureSetCache {
public:
	explicit TextureSetCache(uint32_t slots = 0) { Resize(slots); }

	// slots: rounded up to a power of two (0: no storage, every lookup misses).
	void Resize(uint32_t slots) {
		uint32_t size = slots == 0 ? 0u : 1u;
		while (size < slots) {
			size <<= 1u;
		}
		m_slots.clear();
		m_slots.resize(size);
		m_used = 0;
	}

	[[nodiscard]] static uint64_t Hash(const Program* program, std::span<const Word> words) {
		const auto seed = reinterpret_cast<uint64_t>(program) * 0x9e3779b97f4a7c15ull + words.size();
		return XXH3_64bits_withSeed(words.data(), words.size_bytes(), seed);
	}

	// Moves the set (program, words, bindings) into its slot. The slot's previous vectors come
	// back in `words` and `bindings` (contents unspecified: storage for the caller) and `program`
	// becomes null. A null program stores nothing.
	void Put(const Program*& program, std::vector<Word>& words, std::vector<Binding>& bindings) {
		if (program == nullptr || m_slots.empty()) {
			return;
		}
		auto& slot = m_slots[Hash(program, words) & (m_slots.size() - 1)];
		m_used += slot.program == nullptr ? 1u : 0u;
		m_evictions += slot.program != nullptr ? 1u : 0u;
		std::swap(slot.program, program);
		slot.words.swap(words);
		slot.bindings.swap(bindings);
		program = nullptr;
	}

	// The stored set of `program` with exactly `words`: true when found; it is then moved into
	// (out_program, out_words, out_bindings), whose previous vectors stay in the emptied slot.
	// Otherwise nothing changes.
	[[nodiscard]] bool Take(const Program* program, std::span<const Word> words, uint64_t hash,
	                        const Program*& out_program, std::vector<Word>& out_words,
	                        std::vector<Binding>& out_bindings) {
		if (m_slots.empty()) {
			return false;
		}
		auto& slot = m_slots[hash & (m_slots.size() - 1)];
		if (slot.program != program || slot.words.size() != words.size() ||
		    !std::equal(slot.words.begin(), slot.words.end(), words.begin())) {
			return false;
		}
		out_program  = slot.program;
		slot.program = nullptr;
		slot.words.swap(out_words);
		slot.bindings.swap(out_bindings);
		m_used--;
		return true;
	}

	[[nodiscard]] uint32_t Slots() const noexcept { return static_cast<uint32_t>(m_slots.size()); }
	[[nodiscard]] uint32_t Used() const noexcept { return m_used; }
	[[nodiscard]] uint64_t Evictions() const noexcept { return m_evictions; }

private:
	struct Slot {
		const Program*       program = nullptr;
		std::vector<Word>    words;
		std::vector<Binding> bindings;
	};
	std::vector<Slot> m_slots;
	uint32_t          m_used      = 0;
	uint64_t          m_evictions = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXTURESETCACHE_H_
