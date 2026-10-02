#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPCOMMIT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPCOMMIT_H_

#include <cstdint>

// KYTY_CP_COMMIT: cheaper per-draw commit work on the command processor (Astro Bot's Sky Garden
// start commits ~4,900 draws per flip at ~5.4 us each). Default 0 (off: every part below behaves
// as before); 1 or "all" enables every part; otherwise a comma list of part names. Each part is
// exact: it only removes work that provably changes nothing, or reorders none.
//  - dccguest: a render-target lookup whose DCC metadata the GPU never wrote (its fast-clear key
//    bytes are guest memory) becomes a provable draw-sequence repeat: the repeat re-reads each
//    slice's first key byte and checks that the metadata is still not GPU-written, which is all
//    MaterializeDccClear's decision reads when no key is a clear code (TextureCache::MetadataNoop).
//  - targetalloc: no heap allocation per target acquisition (the bounded-claim block set is a
//    reused member; the HTile surface entry is inserted with try_emplace).
//  - streamread: a small read binding's stream copy reads the guest bytes through the per-thread
//    mapping record for any size (LibKernel::Memory::TryReadBackingDirect), without the backing
//    store's mapping lock and tree lookup; a racing map change redoes it under the lock.
//  - slots: image and buffer slot lookups read a dense per-slot record (common/slotVector.h,
//    Common::g_slot_vector_dense) instead of the slot's tail behind the value in its deque node.
//  - draws: per-draw bookkeeping without dead copies: AcquireVertexBuffers writes into the draw's
//    vertex bindings and builds its vertex range plan only when it has none from the binding
//    plan; the periodic draw-prep console line reads the clock every 256 commits.
//  - texdcc: a sampled texture with DCC metadata is memoized like any other (TextureBindingMemo),
//    with the certificate of its FindImage's DCC decision (TextureCache::MetadataNoop: the
//    dccguest keys, a recorded fill, or the description alone). A hit re-checks it as
//    TryRepeatLookup does and performs MaterializeDccClear's bookkeeping; without it every such
//    binding took the full resolution (about 2,000 per flip at the Sky Garden start).
//  - bindslots: the read-binding epoch memo (BufferCache::BindingMemoSlot) is direct-mapped with
//    32,768 slots instead of 2,048, so fewer recorded ranges are evicted by other ranges before
//    they repeat (about 19,000 of 53,000 read bindings per flip missed their slot). Same entries,
//    same checks.
//  - metaerase: MaterializeDccClear's and its repeats' erase of any other interpretation of the
//    DCC bytes (TextureCache::m_surface_metas) is skipped when that address was erased before and
//    nothing was inserted since (EraseSurfaceMeta).
namespace Libs::Graphics::CpCommit {

enum class Part : uint32_t {
	DccGuest    = 1u << 0u,
	TargetAlloc = 1u << 1u,
	StreamRead  = 1u << 2u,
	Slots       = 1u << 3u,
	Draws       = 1u << 4u,
	TexDcc      = 1u << 5u,
	BindSlots   = 1u << 6u,
	MetaErase   = 1u << 7u,
};
inline constexpr uint32_t AllParts = (1u << 8u) - 1u;

// Parsed once from KYTY_CP_COMMIT (unknown part names stop the emulator).
[[nodiscard]] uint32_t Parts();
[[nodiscard]] inline bool Enabled(Part part) {
	return (Parts() & static_cast<uint32_t>(part)) != 0;
}

} // namespace Libs::Graphics::CpCommit

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPCOMMIT_H_
