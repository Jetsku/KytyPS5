// Tile addressing against PS5 hardware references.
//
// The other tiling tests compare the CPU tiler with the GPU tiler; these compare tile.cpp with
// maps that PS5 hardware was observed to use. The references come from the PS5_Vulkan project
// (https://github.com/mihawk-99/PS5_Vulkan, docs/HARDWARE_FINDINGS.md), which measured them on a
// retail console; each table names the run it comes from. The per-bit form below is AddrLib's
// GFX10 64 KiB swizzle equation for the element size: an address bit set by each coordinate bit.
// Only facts are taken from that project (no code); it is GPL-3.0-or-later and this one is
// GPL-2.0-only.

#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/tile.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::TileBlockFamily;
using Libs::Graphics::TileBlockLayout;
namespace Prospero = Libs::Graphics::Prospero;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "TileHardwareReferenceTests: failed: %s\n", text);
		std::abort();
	}
}

struct ReferenceMap {
	const char*     name;
	TileBlockFamily family;
	uint32_t        bytes_per_element;
	uint32_t        tile_width;
	uint32_t        tile_height;
	// Address bits set by x bit i and y bit i inside one 64 KiB tile.
	const uint32_t* x_bits;
	const uint32_t* y_bits;
	// Coordinate bits flipped by the parity of the tile's row (x) and column (y).
	uint32_t        x_twist_by_row;
	uint32_t        y_twist_by_column;
	const char*     evidence;
};

// 64KB_R_X colour, one sample.
constexpr uint32_t RT1_X[] = {0x0001, 0x0002, 0x0004, 0x0140, 0x0200, 0x0800, 0x2400, 0x8000};
constexpr uint32_t RT1_Y[] = {0x0010, 0x0008, 0x0020, 0x0100, 0x0280, 0x0400, 0x1800, 0x4000};
constexpr uint32_t RT2_X[] = {0x0002, 0x0004, 0x0008, 0x0180, 0x0200, 0x0800, 0x2400, 0x8000};
constexpr uint32_t RT2_Y[] = {0x0010, 0x0020, 0x0040, 0x0100, 0x1200, 0x0400, 0x4800};
constexpr uint32_t RT4_X[] = {0x0004, 0x0008, 0x0080, 0x0100, 0x2200, 0x0800, 0x8400};
constexpr uint32_t RT4_Y[] = {0x0010, 0x0020, 0x0040, 0x1100, 0x0200, 0x0400, 0x4800};
constexpr uint32_t RT8_X[] = {0x0008, 0x0020, 0x0040, 0x2100, 0x0200, 0x0800, 0x8400};
constexpr uint32_t RT8_Y[] = {0x0010, 0x0080, 0x1000, 0x0100, 0x4200, 0x0400};
constexpr uint32_t RT16_X[] = {0x0010, 0x0040, 0x2000, 0x0100, 0x8200, 0x0800};
constexpr uint32_t RT16_Y[] = {0x0020, 0x0080, 0x1000, 0x4100, 0x0200, 0x0400};

// 64KB_Z_X depth and stencil.
constexpr uint32_t Z1_X[] = {0x0001, 0x0004, 0x0010, 0x0140, 0x0200, 0x0800, 0x2400, 0x8000};
constexpr uint32_t Z1_Y[] = {0x0002, 0x0008, 0x0020, 0x0100, 0x0280, 0x0400, 0x1800, 0x4000};
constexpr uint32_t Z2_X[] = {0x0002, 0x0008, 0x0020, 0x0180, 0x0200, 0x0800, 0x2400, 0x8000};
constexpr uint32_t Z2_Y[] = {0x0004, 0x0010, 0x0040, 0x0100, 0x1200, 0x0400, 0x4800};
constexpr uint32_t Z4_X[] = {0x0004, 0x0010, 0x0040, 0x0100, 0x2200, 0x0800, 0x8400};
constexpr uint32_t Z4_Y[] = {0x0008, 0x0020, 0x0080, 0x1100, 0x0200, 0x0400, 0x4800};

constexpr ReferenceMap REFERENCE_MAPS[] = {
    {"64KB_R_X 1B", TileBlockFamily::RenderTarget64KB, 1, 256, 256, RT1_X, RT1_Y, 0, 0,
     "console run pid 117"},
    // No two-byte format is a colour attachment on PS5 (pid 244), so this row is AddrLib's only.
    {"64KB_R_X 2B", TileBlockFamily::RenderTarget64KB, 2, 256, 128, RT2_X, RT2_Y, 0, 0,
     "AddrLib only"},
    {"64KB_R_X 4B", TileBlockFamily::RenderTarget64KB, 4, 128, 128, RT4_X, RT4_Y, 0, 0,
     "commit 79b5ace"},
    {"64KB_R_X 8B", TileBlockFamily::RenderTarget64KB, 8, 128, 64, RT8_X, RT8_Y, 32, 0,
     "console run pid 337, round R91"},
    {"64KB_R_X 16B", TileBlockFamily::RenderTarget64KB, 16, 64, 64, RT16_X, RT16_Y, 32, 32,
     "console runs pid 113-118"},
    {"64KB_Z_X 1B", TileBlockFamily::Depth64KB, 1, 256, 256, Z1_X, Z1_Y, 0, 0,
     "console run pid 192 (stencil plane)"},
    {"64KB_Z_X 2B", TileBlockFamily::Depth64KB, 2, 256, 128, Z2_X, Z2_Y, 0, 0,
     "console run pid 129 (D16)"},
    {"64KB_Z_X 4B", TileBlockFamily::Depth64KB, 4, 128, 128, Z4_X, Z4_Y, 0, 0,
     "c5-depth battery (D32F)"},
};

uint32_t Log2(uint32_t value) {
	uint32_t result = 0;
	while ((1u << result) < value) {
		result++;
	}
	return result;
}

uint64_t ReferenceAddress(const ReferenceMap& map, uint32_t x, uint32_t y, uint32_t columns) {
	const uint32_t column = x / map.tile_width;
	const uint32_t row    = y / map.tile_height;
	uint32_t       in_x   = x % map.tile_width;
	uint32_t       in_y   = y % map.tile_height;
	in_x ^= (row & 1u) * map.x_twist_by_row;
	in_y ^= (column & 1u) * map.y_twist_by_column;

	uint32_t local = 0;
	for (uint32_t bit = 0; bit < Log2(map.tile_width); bit++) {
		if (((in_x >> bit) & 1u) != 0) local ^= map.x_bits[bit];
	}
	for (uint32_t bit = 0; bit < Log2(map.tile_height); bit++) {
		if (((in_y >> bit) & 1u) != 0) local ^= map.y_bits[bit];
	}
	return (static_cast<uint64_t>(row) * columns + column) * 0x10000u + local;
}

// The composition tile.cpp's consumers use: the texel's offset inside its block XORed with the
// block's own XOR, after the blocks in row-major order.
bool KytyAddress(const TileBlockLayout& block, uint32_t x, uint32_t y, uint32_t columns,
                 uint64_t& address) {
	const uint32_t block_x = x / block.block_width;
	const uint32_t block_y = y / block.block_height;
	uint32_t       local   = 0;
	uint32_t       block_xor = 0;
	if (!Libs::Graphics::TileGetBlockOffset(block, x % block.block_width, y % block.block_height, 0,
	                                        local) ||
	    !Libs::Graphics::TileGetBlockXor(block, block_x, block_y, block_xor)) {
		return false;
	}
	address = (static_cast<uint64_t>(block_y) * columns + block_x) * block.block_size +
	          (local ^ block_xor);
	return true;
}

void TestBlockMaps() {
	for (const auto& map: REFERENCE_MAPS) {
		TileBlockLayout block {};
		Check(Libs::Graphics::TileGetBlockLayout(map.family, map.bytes_per_element, block),
		      "block layout lookup failed");
		Check(block.block_size == 0x10000u && block.block_width == map.tile_width &&
		          block.block_height == map.tile_height && block.block_depth == 1,
		      "block dimensions differ from the hardware tile");

		// Four tiles in each direction exercise both parities of both twists.
		constexpr uint32_t Tiles  = 4;
		const uint32_t     width  = map.tile_width * Tiles;
		const uint32_t     height = map.tile_height * Tiles;
		for (uint32_t y = 0; y < height; y++) {
			for (uint32_t x = 0; x < width; x++) {
				uint64_t address = 0;
				Check(KytyAddress(block, x, y, Tiles, address), "tile.cpp address lookup failed");
				const uint64_t expected = ReferenceAddress(map, x, y, Tiles);
				if (address != expected) {
					std::fprintf(stderr,
					             "TileHardwareReferenceTests: %s (%s): texel (%u, %u) at 0x%llx, "
					             "hardware 0x%llx\n",
					             map.name, map.evidence, x, y,
					             static_cast<unsigned long long>(address),
					             static_cast<unsigned long long>(expected));
					Check(false, "tile.cpp differs from the hardware map");
				}
			}
		}
	}
}

// A five-level 256x256 RGBA8 chain in 64KB_R_X, sampled with the LOD pinned to each level in turn
// at the level's centre texel (side / 2 - 1, side / 2 - 1). The addresses are the ones the console
// fetched (runs pid 131-134, c7-mip-run33/34.log). Levels 2-4 sit in the mip tail, whose start
// for level 2 is 0x8400: 0x8000 is only the lowest address its texels reach.
void TestMipChainFetches() {
	Libs::Graphics::TileSurfaceDescription description {};
	description.format    = Prospero::BufferFormat::k8_8_8_8UNorm;
	description.tile_mode = Prospero::TileMode::kRenderTarget;
	description.width     = 256;
	description.height    = 256;
	description.levels    = 5;

	Libs::Graphics::TileSurfaceLayout layout {};
	Check(Libs::Graphics::TileGetTiledTextureLayout(description, layout),
	      "RGBA8 64KB_R_X chain layout failed");
	Check(layout.first_tail_level == 2, "mip tail does not start at level 2");
	Check(layout.block_slice_size == 0x60000u, "chain is not 0x60000 bytes");

	constexpr uint64_t FETCHED[] = {0x2f0fc, 0x13cfc, 0xb4fc, 0x58fc, 0x8fc};
	constexpr uint64_t STARTS[]  = {0x20000, 0x10000, 0x8400, 0x4800, 0x800};
	const auto&        block     = layout.texture.block;
	for (uint32_t level = 0; level < 5; level++) {
		const auto&    mip    = layout.mips[level];
		const bool     tail   = level >= layout.first_tail_level;
		const uint32_t centre = (256u >> level) / 2u - 1u;

		uint64_t fetched = 0;
		uint64_t start   = 0;
		if (tail) {
			uint32_t local  = 0;
			uint32_t origin = 0;
			Check(Libs::Graphics::TileGetBlockOffset(block, centre + mip.tail_x,
			                                         centre + mip.tail_y, 0, local) &&
			          Libs::Graphics::TileGetBlockOffset(block, mip.tail_x, mip.tail_y, 0, origin),
			      "mip tail lookup failed");
			fetched = mip.offset + local;
			start   = mip.offset + origin;
		} else {
			uint64_t local = 0;
			Check(KytyAddress(block, centre, centre, mip.padded_width / block.block_width, local),
			      "mip level lookup failed");
			fetched = mip.offset + local;
			start   = mip.offset;
		}
		if (fetched != FETCHED[level] || start != STARTS[level]) {
			std::fprintf(stderr,
			             "TileHardwareReferenceTests: level %u fetched 0x%llx (console 0x%llx), "
			             "starts 0x%llx (console 0x%llx)\n",
			             level, static_cast<unsigned long long>(fetched),
			             static_cast<unsigned long long>(FETCHED[level]),
			             static_cast<unsigned long long>(start),
			             static_cast<unsigned long long>(STARTS[level]));
			Check(false, "mip chain differs from the console's fetches");
		}
	}
}

} // namespace

int main() {
	TestBlockMaps();
	TestMipChainFetches();
	std::printf("TileHardwareReferenceTests: all cases passed\n");
	return 0;
}
