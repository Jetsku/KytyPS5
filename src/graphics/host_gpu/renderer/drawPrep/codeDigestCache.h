#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_CODEDIGESTCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_CODEDIGESTCACHE_H_

#include <array>
#include <cstdint>

// KYTY_DRAW_PREP_LEAN digest part (lean.h): a preparing thread's cache of headerless shader code
// digests (shader.cpp HashShaderCodeCertified). Without it every prepared stage hashed its code
// (XXH3 over a few KiB per stage, ~5% of worker time at the Sky Garden) although the code almost
// never changes.
//
// What a cached digest relies on. The digest of [address, address + size) computed at time T0 is
// returned for a preparation at T1 > T0 only if all of these hold, and then the bytes at T1 equal
// the bytes at T0 unless a guest write races the draw:
//  - Publication epoch (syncEpoch.h CurrentPublish), stamped on the slot when the draw is
//    published and equal for the two preparations. Guest CPU writes become visible to GPU work
//    only at ordering points (submission start, waits on memory, cache invalidations, CP packets
//    that write memory), and the publication epoch advances at each of them before any later draw
//    is published (the sequencer advances it itself; the resolver's advances are included). A
//    write ordered before the draw at T1 is therefore ordered before an ordering point that lies
//    between the publication of the T0 draw and that of the T1 draw, and the epochs differ. A
//    write between T0 and T1 that is ordered by no such point races the draw at T1 exactly as it
//    races the serial path's hash (and the log-mode certificate treats it the same way).
//    Speculative slots (KYTY_CP_SEQ_PREFETCH, published before their wait completed) neither read
//    nor fill the cache.
//  - Never GPU-touched (gpuTouchedPages.h, KYTY_CP_SEQ=1): no page of the range was ever touched by
//    a GPU-side transition (GPU-dirty buffers, tracker GPU marks, GPU-modified images, backing
//    publications: the emulator's own writes of backing bytes). The marks are sticky, so a range
//    untouched at T1 was untouched at all times before; without the map nothing is cached.
//  - Backing mapping tags of the range's first and last 64 MiB regions (GuestAddressSpace
//    MappingTag; the global map generation unless KYTY_MAP_REGION_EPOCH=1), equal at T0 (before
//    and after hashing) and at T1: no mapping of the range changed. Ranges over more than two
//    regions are not cached.
//  - Shader map generation equal at T0 and T1 (registration and unmapping of shaders).
// The memory tracker's CPU-mutation epoch was not used: shader code is not in tracked buffer
// regions, so guest writes to it never fault and never advance that epoch.
// The digest still goes into the preparation's read set (RecordDigest), so the certificate at
// commit is the same as without the cache.
namespace Libs::Graphics::DrawPrep {

struct CodeDigestKey {
	uint64_t address        = 0;
	uint64_t size           = 0;
	uint64_t epoch          = 0; // publication epoch of the slot (0: never cached)
	uint64_t map_generation = 0; // shader map generation
	uint64_t tag_begin      = 0; // backing mapping tags of the first and last byte
	uint64_t tag_end        = 0;

	[[nodiscard]] bool operator==(const CodeDigestKey&) const noexcept = default;
};

class CodeDigestCache {
public:
	static constexpr uint32_t Entries = 256; // direct-mapped, per thread

	[[nodiscard]] bool Lookup(const CodeDigestKey& key, uint64_t& digest) const noexcept {
		if (key.epoch == 0 || key.size == 0) {
			return false;
		}
		const auto& entry = m_entries[Index(key)];
		if (entry.valid && entry.key == key) {
			digest = entry.digest;
			return true;
		}
		return false;
	}

	void Store(const CodeDigestKey& key, uint64_t digest) noexcept {
		if (key.epoch == 0 || key.size == 0) {
			return;
		}
		auto& entry  = m_entries[Index(key)];
		entry.key    = key;
		entry.digest = digest;
		entry.valid  = true;
	}

	void Clear() noexcept { m_entries = {}; }

private:
	struct Entry {
		CodeDigestKey key;
		uint64_t      digest = 0;
		bool          valid  = false;
	};

	[[nodiscard]] static uint32_t Index(const CodeDigestKey& key) noexcept {
		const auto mixed = (key.address >> 4u) ^ (key.address >> 13u) ^ (key.size >> 2u);
		return static_cast<uint32_t>(mixed) & (Entries - 1u);
	}

	std::array<Entry, Entries> m_entries {};
};

} // namespace Libs::Graphics::DrawPrep

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_CODEDIGESTCACHE_H_
