#include "graphics/host_gpu/renderer/drawPrep/xframeReuse.h"
#include "graphics/host_gpu/renderer/pipeline/textureSetCache.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// KYTY_XFRAME_REUSE (drawPrep/xframeReuse.h): the cross-frame texture-set store moves sets, never
// copies them, finds a set only for its own program and exact words, and gives the caller storage
// vectors back.
namespace {

using Libs::Graphics::TextureSetCache;
namespace XFrame = Libs::Graphics::XFrameReuse;

struct Program {
	int id = 0;
};
struct Word {
	std::array<uint32_t, 8> dwords {};
	uint32_t                dword_count = 0;
	bool operator==(const Word& other) const {
		return dword_count == other.dword_count && dwords == other.dwords;
	}
};
struct Binding {
	uint32_t image = 0;
	uint64_t tag   = 0;
};
using Cache = TextureSetCache<Program, Word, Binding>;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "TextureSetCacheTests: failed: %s\n", text);
		std::abort();
	}
}

std::vector<Word> Words(uint32_t seed, uint32_t count) {
	std::vector<Word> words(count);
	for (uint32_t i = 0; i < count; i++) {
		words[i].dword_count = 8;
		for (uint32_t d = 0; d < 8; d++) {
			words[i].dwords[d] = seed * 131u + i * 17u + d;
		}
	}
	return words;
}

std::vector<Binding> Bindings(uint32_t seed, uint32_t count) {
	std::vector<Binding> bindings(count);
	for (uint32_t i = 0; i < count; i++) {
		bindings[i] = {seed * 10 + i, seed * 1000 + i};
	}
	return bindings;
}

void TestParse() {
	Check(XFrame::ParseMode(nullptr) == XFrame::Mode::Off, "unset is off");
	Check(XFrame::ParseMode("0") == XFrame::Mode::Off, "0 is off");
	Check(XFrame::ParseMode("1") == XFrame::Mode::On, "1 is on");
	Check(XFrame::ParseMode("verify") == XFrame::Mode::Verify, "verify");
	Check(XFrame::ParseMode("exit") == XFrame::Mode::Exit, "exit");
}

void TestRoundTrip() {
	Cache          cache(64);
	Program        a {1};
	Program        b {2};
	const Program* program  = &a;
	auto           words    = Words(1, 5);
	auto           bindings = Bindings(1, 5);
	const auto*    data     = bindings.data();
	cache.Put(program, words, bindings);
	Check(program == nullptr, "a stored set leaves no program behind");
	Check(cache.Used() == 1, "one slot used");

	const Program*       out_program = nullptr;
	std::vector<Word>    out_words;
	std::vector<Binding> out_bindings;
	const auto           query = Words(1, 5);
	Check(!cache.Take(&b, query, Cache::Hash(&b, query), out_program, out_words, out_bindings),
	      "another program with the same words is not found");
	auto other = Words(1, 5);
	other[3].dwords[7] ^= 1;
	Check(!cache.Take(&a, other, Cache::Hash(&a, other), out_program, out_words, out_bindings),
	      "one different dword is not found");
	Check(!cache.Take(&a, Words(1, 4), Cache::Hash(&a, Words(1, 4)), out_program, out_words,
	                  out_bindings),
	      "a prefix of the words is not found");
	Check(cache.Take(&a, query, Cache::Hash(&a, query), out_program, out_words, out_bindings),
	      "the same program and words are found");
	Check(out_program == &a && out_words == query && out_bindings.size() == 5 &&
	          out_bindings[4].tag == 1004,
	      "the stored set comes back whole");
	Check(out_bindings.data() == data, "the set was moved, not copied");
	Check(cache.Used() == 0, "the slot is empty after the take");
	Check(!cache.Take(&a, query, Cache::Hash(&a, query), out_program, out_words, out_bindings),
	      "a taken set is gone");
}

void TestEvictionGivesStorage() {
	Cache          cache(1); // one slot: every put evicts
	Program        a {1};
	Program        b {2};
	const Program* program  = &a;
	auto           words    = Words(1, 3);
	auto           bindings = Bindings(1, 3);
	cache.Put(program, words, bindings);
	program           = &b;
	auto words_b      = Words(2, 2);
	auto bindings_b   = Bindings(2, 2);
	cache.Put(program, words_b, bindings_b);
	Check(program == nullptr && bindings_b.size() == 3 && words_b.size() == 3,
	      "the evicted set's vectors come back as storage");
	Check(cache.Evictions() == 1 && cache.Used() == 1, "one eviction, one slot used");
	const Program*       out_program = nullptr;
	std::vector<Word>    out_words;
	std::vector<Binding> out_bindings;
	const auto           qa = Words(1, 3);
	Check(!cache.Take(&a, qa, Cache::Hash(&a, qa), out_program, out_words, out_bindings),
	      "the evicted set is not found");
	const auto qb = Words(2, 2);
	Check(cache.Take(&b, qb, Cache::Hash(&b, qb), out_program, out_words, out_bindings) &&
	          out_bindings.size() == 2 && out_bindings[1].tag == 2001,
	      "the newer set is found");
}

void TestNullAndEmpty() {
	Cache          empty(0);
	Program        a {1};
	const Program* program  = &a;
	auto           words    = Words(1, 2);
	auto           bindings = Bindings(1, 2);
	empty.Put(program, words, bindings);
	Check(program == &a && bindings.size() == 2, "a store without slots keeps nothing");
	Cache          cache(8);
	const Program* none = nullptr;
	cache.Put(none, words, bindings);
	Check(cache.Used() == 0 && bindings.size() == 2, "a null program stores nothing");
	// A set without textures (a vertex stage) is a valid key.
	program = &a;
	std::vector<Word>    no_words;
	std::vector<Binding> no_bindings;
	cache.Put(program, no_words, no_bindings);
	const Program*       out_program = nullptr;
	std::vector<Word>    out_words   = Words(9, 1);
	std::vector<Binding> out_bindings;
	Check(cache.Take(&a, std::vector<Word> {}, Cache::Hash(&a, std::vector<Word> {}), out_program,
	                 out_words, out_bindings) &&
	          out_words.empty(),
	      "an empty set round-trips");
}

void TestManySets() {
	Cache               cache(4096);
	std::vector<Program> programs(64);
	for (int i = 0; i < 64; i++) {
		programs[i].id = i;
	}
	uint32_t stored = 0;
	for (uint32_t s = 0; s < 1000; s++) {
		const Program* program  = &programs[s % 64];
		auto           words    = Words(s, 1 + s % 12);
		auto           bindings = Bindings(s, 1 + s % 12);
		cache.Put(program, words, bindings);
		stored++;
	}
	uint32_t found = 0;
	for (uint32_t s = 0; s < 1000; s++) {
		const Program*       out_program = nullptr;
		std::vector<Word>    out_words;
		std::vector<Binding> out_bindings;
		const auto           words = Words(s, 1 + s % 12);
		if (cache.Take(&programs[s % 64], words, Cache::Hash(&programs[s % 64], words), out_program,
		               out_words, out_bindings)) {
			Check(out_bindings.size() == 1 + s % 12 && out_bindings[0].tag == s * 1000ull,
			      "a found set is its own");
			found++;
		}
	}
	Check(found + cache.Evictions() == stored, "every set is found once unless evicted");
	Check(found > 800, "a store four times larger than the sets keeps most of them");
}

} // namespace

int main() {
	TestParse();
	TestRoundTrip();
	TestEvictionGivesStorage();
	TestNullAndEmpty();
	TestManySets();
	std::printf("TextureSetCacheTests: passed\n");
	return 0;
}
