#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_TEXELIMAGELOOKUP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_TEXELIMAGELOOKUP_H_

#include <cstdint>
#include <cstring>

// KYTY_TEXEL_IMAGE_LOOKUP (idea from Senaxx's fork): which pages TextureCache::FindImageFromRange
// asks FindImagesInRegion about.
//
// FindImageFromRange accepts only images whose guest range starts at the address (its callers:
// SynchronizeBufferFromImage for every read-only texel buffer of every draw, and the buffer
// cache's copy check). A registered image is listed on every page of its resident range (`live`,
// which starts at the image's address), and it overlaps the first byte of that range. So the
// images that can qualify are all found by a query of that one byte, on the first page, in the
// same order the walk of the whole range finds them first (the walk visits the first page
// first, and visits each image once). The rest of the range's pages (1 MiB each) only add
// candidates that are rejected.
//  - page (default): query the first byte only.
//  - range: query every page of the range, as before.
//  - verify: run both; a different result is counted and logged (the range answer is used).
//  - exit: as verify, and stop at the first difference.
// A live switch (common/liveSwitch.h): each call picks one of equivalent queries.
// KYTY_TEXEL_IMAGE_LOOKUP_STATS=1 (or verify/exit, or KYTY_CP_COMMIT_STATS): one console line
// every 10 s, "TexelImageLookup 10s: ...", with the calls and the pages a range walk visits.
namespace Libs::Graphics::TexelImageLookup {

enum class Mode : uint8_t { Page, Range, Verify, Exit };

[[nodiscard]] inline Mode ParseMode(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "page") == 0 ||
	    std::strcmp(value, "1") == 0) {
		return Mode::Page;
	}
	if (std::strcmp(value, "range") == 0 || std::strcmp(value, "0") == 0) {
		return Mode::Range;
	}
	if (std::strcmp(value, "exit") == 0) {
		return Mode::Exit;
	}
	return std::strcmp(value, "verify") == 0 ? Mode::Verify : Mode::Page;
}

// The size of the region FindImageFromRange queries for [address, address + size).
[[nodiscard]] constexpr uint64_t QuerySize(Mode mode, uint64_t size) {
	return mode == Mode::Page ? (size != 0 ? 1u : 0u) : size;
}

// The pages a walk of [address, address + size) visits (page_bits: the page table's), 0 when
// empty.
[[nodiscard]] constexpr uint64_t PagesVisited(uint64_t address, uint64_t size, uint32_t page_bits) {
	return size == 0 ? 0 : ((address + size - 1) >> page_bits) - (address >> page_bits) + 1;
}

// GPU thread: the 10-second console line (textureCache.cpp).
void PrintSummary();

// The page walk of FindImagesInRegion: each page of [address, address + size) that has an owner
// list, in address order (`size` non-zero and validated by the caller).
template <typename Table, typename Func>
void ForEachOwnerList(const Table& table, uint64_t address, uint64_t size, Func&& func) {
	const uint64_t page_end = (address + size - 1) >> Table::kPageBits;
	for (uint64_t page = address >> Table::kPageBits; page <= page_end; ++page) {
		if (const auto* owners = table.Find(page); owners != nullptr) {
			func(*owners);
		}
	}
}

} // namespace Libs::Graphics::TexelImageLookup

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_TEXELIMAGELOOKUP_H_
