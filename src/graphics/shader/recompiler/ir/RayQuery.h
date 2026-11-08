#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_RAYQUERY_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_RAYQUERY_H_

#include <cstdint>

// The flags of the RayQuery* IR operations (native ray tracing, see RayTracingPatch.h): each
// patched traversal (site) owns one host ray query per lane, paused at the candidate the lane
// tests next; the wave processes one instance at a time with every lane whose candidate is in it.
namespace Libs::Graphics::ShaderRecompiler::IR {

constexpr uint32_t MaxRayQuerySites = 4;
// Per-lane words a verifying site keeps (RayQueryKeep): the entry values of its result
// registers, the native results, and the flag of a lane that verifies.
constexpr uint32_t RayQueryKeepSlots = 17;
constexpr uint32_t RayQueryKeepFlag  = 16;

// RayQueryCandidate: a field of the lane's candidate.
enum class RayQueryCandidateKind : uint32_t {
	Node = 0, // The guest triangle node pointer of its leaf entry.
	Word = 1, // The info word of its leaf entry.
};

// RayQueryFlag: per-lane state.
enum class RayQueryFlagKind : uint32_t {
	InTraversal = 0, // The lane started the native traversal.
};

// RayQueryCount: the per-ray counters of a verifying site (NativeRayTracing reports them).
enum class RayQueryCounter : uint32_t {
	Rays          = 0, // Rays compared.
	NativeHits    = 1,
	GuestHits     = 2,
	HitMismatch   = 3, // One found a hit, the other not.
	InstanceDiff  = 4, // Both hit, other instances.
	PrimitiveDiff = 5, // Same instance, another triangle (node or info word).
	DistanceDiff  = 6, // Same triangle, |dt| > 1e-4 * t.
	BaryDiff      = 7, // Same triangle, barycentrics differ by more than 1e-3.
	EntryLanes    = 8, // Lanes that entered a patched traversal.
	NativeLanes   = 9, // Of them, lanes that traversed natively (the host had the TLAS).
	Count         = 10,
};

[[nodiscard]] constexpr uint32_t RayQueryFlags(uint32_t site, uint32_t variant) noexcept {
	return site | (variant << 8u);
}
[[nodiscard]] constexpr uint32_t RayQuerySite(uint32_t flags) noexcept {
	return flags & 0xffu;
}
[[nodiscard]] constexpr uint32_t RayQueryVariant(uint32_t flags) noexcept {
	return flags >> 8u;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_RAYQUERY_H_
