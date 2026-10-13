#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_XFRAMEREUSE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_XFRAMEREUSE_H_

#include "common/liveSwitch.h"

#include <atomic>
#include <cstdint>
#include <cstring>

// KYTY_XFRAME_REUSE (live, default 0): cross-frame reuse of the resolved structure of draws that
// start a new draw run (KYTY_DRAW_RUN continues only the previous draw's structure).
//
// Profile (Sky Garden / clock tower, KYTY_CP_COMMIT_STATS sub-phases): a start draw binds about
// 11 textures; their resolution and views take about 1.4 us of its ~7 us, against ~0.05 us for a
// continuation. 70-80% of starts have the structure of a draw of the previous two frames, but in
// PreparedBindings' current set and three-set history only ~30% of their texture sets are found:
// a new set is resolved binding by binding (memo runs with a ~0.5 KB description copy each).
//
// What is reused: a stage's texture set (program, T# words, resolved bindings: image, final
// description, TextureBindingMemo entry tag) that fell out of a stage's history is kept in a
// cross-frame store (pipeline/textureSetCache.h) under its program and words, and becomes the
// stage's current set again when a later draw binds the same program with the same words.
//
// The certificate is the one the current set and the history already use
// (RenderExecutor::RepeatStageTextures, KYTY_DRAW_SEQUENCE_FAST): every binding is still
// described by its memo entry (same tag, so the same key: T# words and resource fields of the
// same program) and passes TryResolve's hit conditions under the texture-cache lock (registered,
// no stencil association or rebind request, resident levels, the first page's structure version,
// alias partners, DCC certificates), so each ResolveTexture would return the same image and
// description and do nothing but its bookkeeping, which is then performed in binding order
// (tick, LRU touch, DCC bookkeeping, BindImage). The views follow in RebindImages
// (TextureBindingMemo::TryRepeatViews: refresh no-op, residency) or the normal acquisition.
// A set that fails is resolved the normal way into the same vectors. Nothing else of the draw
// changes: targets, samplers, pipeline, acquisition and every transition, barrier and descriptor
// take the normal path, so a reused set cannot skip a barrier.
//
// Not reused (measured, see the int17-reuse report): attachments acquisition, transitions, dynamic
// state and descriptors need the command buffer's and rendering instance's state, which a
// cross-frame record cannot certify; the colour/depth target descriptions already hit their
// per-slot memos ~99% of the time (their remaining cost is TryRepeatLookup's certificate).
//
// KYTY_XFRAME_REUSE=1: on. verify: on, and every stage whose set came from the store runs every
// binding's full resolution and the normal view acquisition, which must find the same image,
// memo entry and view; differences are counted and logged (the full results are used). exit: as
// verify, stopping at the first difference. KYTY_XFRAME_REUSE_SLOTS (default 4096): store size.
// One console line every 10 s while on: "XFrameReuse 10s: ...".
//
// Result (Sky Garden, 2026-10-07): ~90% of new stage sets are found in the store, but only 12%
// (default 4096 memo entries) or 41% (KYTY_TEXTURE_MEMO_SLOTS=32768) pass the revalidation: the
// rest lost their memo entry (evicted or recorded again under a new tag). Same-process A/B with
// 32768 entries: -1.0% +- 0.4% fps (+0.32 ms CP/flip). Keep it off; the memo's capacity was the
// real cost of new texture sets.
namespace Libs::Graphics::XFrameReuse {

enum class Mode : uint8_t { Off, On, Verify, Exit };

[[nodiscard]] inline Mode ParseMode(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0 ||
	    std::strcmp(value, "off") == 0) {
		return Mode::Off;
	}
	if (std::strcmp(value, "verify") == 0) {
		return Mode::Verify;
	}
	return std::strcmp(value, "exit") == 0 ? Mode::Exit : Mode::On;
}

namespace Detail {
extern Live::Switch g_mode;
} // namespace Detail

[[nodiscard]] inline Mode GetMode() {
	return static_cast<Mode>(Detail::g_mode.Get());
}
[[nodiscard]] inline bool Enabled() {
	return Detail::g_mode.Get() != 0;
}
[[nodiscard]] inline bool VerifyMode() {
	const auto mode = GetMode();
	return mode == Mode::Verify || mode == Mode::Exit;
}

[[nodiscard]] uint32_t SlotCount();
// The failure breakdown (a second check per failed set): verify mode, KYTY_CP_COMMIT_STATS or
// KYTY_XFRAME_REUSE_STATS=1.
[[nodiscard]] bool StatsEnabled();

struct Totals {
	std::atomic<uint64_t> new_sets {0};    // a stage's words matched neither its set nor its history
	std::atomic<uint64_t> stored {0};      // sets moved into the store
	std::atomic<uint64_t> found {0};       // new sets found in the store
	std::atomic<uint64_t> repeated {0};    // found sets that passed the revalidation
	std::atomic<uint64_t> bindings {0};    // their bindings
	std::atomic<uint64_t> verify_checks {0};
	std::atomic<uint64_t> verify_mismatches {0};
	// Found sets that failed the revalidation, by TextureBindingMemo::RepeatResolveFailure.
	std::atomic<uint64_t> failures[8] {};
	std::atomic<uint32_t> used {0};        // store occupancy (GPU thread)
	std::atomic<uint64_t> evictions {0};
};
[[nodiscard]] Totals& GetTotals();

// Verify mode: one difference (counted, logged; exit mode stops).
void ReportMismatch(const char* what, uint64_t detail = 0);
// GPU thread: the 10-second console line.
void PrintSummary();

} // namespace Libs::Graphics::XFrameReuse

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_XFRAMEREUSE_H_
