#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/texelImageLookup.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// KYTY_TEXEL_IMAGE_LOOKUP (cache/texelImageLookup.h): FindImageFromRange's first-page query finds
// the same images that start at the address, in the same order, as the walk of the whole range.
namespace {

using Libs::Graphics::TexelImageLookup::Mode;
namespace Lookup = Libs::Graphics::TexelImageLookup;

using PageOwners = Libs::Graphics::InlinePageOwnerList<uint32_t, 16>;
using OwnerTable = Libs::Graphics::MultiLevelPageTable<PageOwners, 20, 40, 10>;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "TexelImageLookupTests: failed: %s\n", text);
		std::abort();
	}
}

struct Image {
	uint64_t address    = 0;
	uint64_t size       = 0;
	uint32_t query_mark = 0;
};

// Registers an image on every page of its range, as TextureCache::RegisterImage does.
void Register(OwnerTable& table, std::vector<Image>& images, uint64_t address, uint64_t size) {
	const auto id = static_cast<uint32_t>(images.size());
	images.push_back({address, size, 0});
	const uint64_t last = (address + size - 1) >> OwnerTable::kPageBits;
	for (uint64_t page = address >> OwnerTable::kPageBits; page <= last; page++) {
		table.GetOrCreate(page).push_back(id);
	}
}

// FindImagesInRegion (byte overlap, each image once, in walk order), then FindImageFromRange's
// filter: the images that start at the address.
std::vector<uint32_t> StartingAt(const OwnerTable& table, std::vector<Image>& images,
                                 uint64_t address, uint64_t query_size, uint32_t mark) {
	std::vector<uint32_t> result;
	Lookup::ForEachOwnerList(table, address, query_size, [&](const PageOwners& owners) {
		owners.ForEach([&](uint32_t id) {
			auto& image = images[id];
			if (image.query_mark == mark) {
				return;
			}
			image.query_mark = mark;
			if (image.address < address + query_size && address < image.address + image.size &&
			    image.address == address) {
				result.push_back(id);
			}
		});
	});
	return result;
}

void TestParse() {
	Check(Lookup::ParseMode(nullptr) == Mode::Page, "unset is page");
	Check(Lookup::ParseMode("") == Mode::Page, "empty is page");
	Check(Lookup::ParseMode("page") == Mode::Page, "page");
	Check(Lookup::ParseMode("range") == Mode::Range, "range");
	Check(Lookup::ParseMode("0") == Mode::Range, "0 is the old range walk");
	Check(Lookup::ParseMode("verify") == Mode::Verify, "verify");
	Check(Lookup::ParseMode("exit") == Mode::Exit, "exit");
}

void TestSizes() {
	Check(Lookup::QuerySize(Mode::Page, 0x400000) == 1, "page mode queries one byte");
	Check(Lookup::QuerySize(Mode::Range, 0x400000) == 0x400000, "range mode queries the range");
	Check(Lookup::PagesVisited(0, 0, 20) == 0, "empty range visits nothing");
	Check(Lookup::PagesVisited(0xfffff, 2, 20) == 2, "a range across a boundary visits two pages");
	Check(Lookup::PagesVisited(0x100000, 0x400000, 20) == 4, "4 MiB aligned visits four pages");
	Check(Lookup::PagesVisited(0x100010, 0x400000, 20) == 5, "4 MiB unaligned visits five pages");
}

void TestSameCandidates() {
	std::mt19937_64    random(12345);
	OwnerTable         table;
	std::vector<Image> images;
	std::vector<uint64_t> starts;
	// Images of 4 KiB .. 8 MiB at 256-byte aligned addresses in a 64 MiB window; many share a
	// start (aliases of one surface) or start mid-page or inside another image.
	for (int i = 0; i < 4000; i++) {
		uint64_t address = 0x10000000 + ((random() % (64ull << 20)) & ~0xffull);
		if (!starts.empty() && random() % 3 == 0) {
			address = starts[random() % starts.size()];
		}
		const uint64_t size = 0x1000 + (random() % (8ull << 20));
		Register(table, images, address, size);
		starts.push_back(address);
	}
	uint32_t mark = 0;
	uint32_t found = 0;
	for (int i = 0; i < 20000; i++) {
		uint64_t address = starts[random() % starts.size()];
		if (random() % 4 == 0) {
			address += (random() % 0x200000) & ~0xfull; // usually no image starts there
		}
		const uint64_t size  = 1 + (random() % (16ull << 20));
		const auto     range = StartingAt(table, images, address, size, ++mark);
		const auto     page  = StartingAt(table, images, address,
		                                  Lookup::QuerySize(Mode::Page, size), ++mark);
		Check(range == page, "the first-page query finds the same images in the same order");
		found += range.empty() ? 0u : 1u;
	}
	Check(found > 1000, "the test exercises queries that find images");
}

} // namespace

int main() {
	TestParse();
	TestSizes();
	TestSameCandidates();
	std::printf("TexelImageLookupTests: passed\n");
	return 0;
}
