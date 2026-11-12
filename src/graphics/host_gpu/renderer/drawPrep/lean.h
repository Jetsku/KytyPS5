#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_LEAN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_LEAN_H_

#include <cstdint>
#include <cstring>

// KYTY_DRAW_PREP_LEAN (live, default off): cheaper draw preparations with unchanged outputs, in
// parts. 1 or "all" turns every part on; otherwise the value lists the parts by letter:
//   d  shader code digest cache (codeDigestCache.h): a headerless shader's code digest is computed
//      once per (address, size, publication epoch, shader map, backing mapping) instead of for
//      every prepared stage;
//   r  read set (readSet.h): reads merge into the certificate ranges as they are recorded when they
//      come in address order (one byte buffer, no order array, no sort); the certificate list
//      merges the digest ranges into them in one linear pass;
//   c  claims (window.h): a worker that finds a backlog claims two slots with one claim-counter
//      update and prepares them in order.
// KYTY_DRAW_PREP_LEAN_VERIFY=1|exit (startup): every digest taken from the cache is checked against
// the bytes (hashed again without recording); a difference is counted (exit: stops), and the fresh
// digest is used. The read-set part is covered by KYTY_DRAW_PREP_VERIFY and the certificate checks
// (its ranges are the ones the plain path builds), and by draw_prep_tests.
namespace Libs::Graphics::DrawPrep::Lean {

enum Part : uint32_t {
	Digest  = 1u << 0u,
	ReadSet = 1u << 1u,
	Claim   = 1u << 2u,
	All     = Digest | ReadSet | Claim,
};

// The parts of a KYTY_DRAW_PREP_LEAN value (nullptr: unset).
[[nodiscard]] inline uint32_t ParseParts(const char* value) noexcept {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	if (std::strcmp(value, "1") == 0 || std::strcmp(value, "all") == 0) {
		return All;
	}
	uint32_t parts = 0;
	for (const char* c = value; *c != '\0'; c++) {
		switch (*c) {
			case 'd': parts |= Digest; break;
			case 'r': parts |= ReadSet; break;
			case 'c': parts |= Claim; break;
			default: break;
		}
	}
	return parts;
}

// drawPrep.cpp (live switch).
[[nodiscard]] uint32_t Parts() noexcept;
[[nodiscard]] inline bool On(Part part) noexcept {
	return (Parts() & part) != 0;
}
// KYTY_DRAW_PREP_LEAN_VERIFY: 0 off, 1 count, 2 exit.
[[nodiscard]] int DigestVerifyMode() noexcept;
// Digest cache totals (any thread, relaxed): cache hits, misses (hashed and stored), lookups that
// could not use the cache, verify checks and mismatches.
void NoteDigest(uint32_t hits, uint32_t misses, uint32_t uncacheable) noexcept;
void NoteDigestVerify(bool mismatch) noexcept;

} // namespace Libs::Graphics::DrawPrep::Lean

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_LEAN_H_
