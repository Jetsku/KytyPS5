#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_IMGSIDE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_IMGSIDE_H_

// The commit thread's image side (texture/sampler resolution, stage texture sets, views, targets,
// transitions): counters that split its per-draw work by outcome.
//
// KYTY_IMG_STATS=1 (diagnostic, start-up, default off; compiled out with KYTY_CP_DIAGNOSTICS=OFF):
// one "ImgSide 10s:" line every 10 s (with the DrawPrep summary) with per-frame averages of every
// counter below. Nothing it counts feeds back into emulation.

#include "common/onceValue.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>

#ifndef KYTY_CP_DIAGNOSTICS
#define KYTY_CP_DIAGNOSTICS 1
#endif

namespace Libs::Graphics::ImgSide {

enum class C : uint8_t {
	Draws,         // ExecutePreparedDraw calls that reach the bindings
	StagesFull,    // PrepareBindings without kept images
	StagesKept,    // continuation stages (kept images)
	// Texture bindings of PrepareBindings, by the path that resolved them.
	TexSetRepeat,  // bindings of a stage set that passed TryRepeatResolve
	TexRunHit,     // TryResolveRun hits (worker hints)
	TexHit,        // TryResolve hits without revalidation
	TexHitReval,   // TryResolve hits with a revalidation
	TexFull,       // full resolutions (ResolveTextureFull), every caller
	TexFullRebind, // of which: RebindImages' rebind of an image that changed
	// Why TryResolve did not hit (each full resolution of a memoized resource has one).
	MissNoMemo,    // memo off for the resource (depth compare with null compare)
	MissEmpty,     // the slot never held an entry
	MissConflict,  // the slot holds another key
	MissDcc,       // DCC certificate no longer holds
	MissImage,     // image gone / unregistered / stencil association / pending rebind
	MissResident,  // resident levels changed
	MissPage,      // first page's owner list changed and the revalidation failed
	MissPartner,   // alias partner and not the owner
	// What the full resolution left in the memo.
	RecOk,
	RecReject,     // view rebased / other image / DCC without proof / not a first-page answer
	RecNone,       // not offered to the memo (copy bindings, null-image fallbacks)
	// Full resolutions whose key had been resolved in full before (shadow table of key hashes,
	// any frame): the memo lost or never kept an answer it had.
	FullRepeatKey,
	FullRepeatKey2, // ... within the last 2 frames
	// Views (RebindImages).
	ViewRepeat,    // bindings of a stage that passed TryRepeatViews
	ViewRunHit,    // TryAcquireViewRun hits
	ViewHit,       // TryAcquireView hits
	ViewFind,      // FindTexture (memo miss)
	ViewFindNoEnt, // of which: no entry / no recorded view
	ViewMips,      // dynamic-mip views (FindTexture each)
	// Image bookkeeping.
	Touch,         // TextureCache::TouchImage
	TouchSame,     // of which: image already touched this frame and LRU-skippable (no state change but the tick)
	Bind,          // BindImage
	BindTarget,    // BindRenderTarget
	ResetImages,   // ResetBindings: images reset
	// Targets.
	Acquire,       // AcquireRenderTargets
	AcquireSame,   // of which: same colour/depth images, views, slots and layouts as the previous acquisition
	TargetRun,     // DrawRunTargets
	FindTarget,    // FindTargetImage
	FindTargetRep, // of which: TryRepeatLookup answered
	Transit,       // Image::Transit
	TransitNoOp,   // of which: predicted no-op
	Count
};

namespace Detail {
[[nodiscard]] bool ReadEnabled();
void               Add(C counter, uint64_t n);
// The shadow table: whether this key hash was resolved in full before (and within 2 frames).
void NoteFullKey(uint64_t hash);
void PrintStats();
} // namespace Detail

[[nodiscard]] inline bool StatsEnabled() {
#if !KYTY_CP_DIAGNOSTICS
	return false;
#endif
	static std::atomic<int64_t> once {Common::OnceUnset};
	return static_cast<bool>(Common::Once(once, [] { return Detail::ReadEnabled(); }));
}

inline void Count(C counter, uint64_t n = 1) {
	if (StatsEnabled()) [[unlikely]] {
		Detail::Add(counter, n);
	}
}

inline void NoteFullKey(uint64_t hash) {
	if (StatsEnabled()) [[unlikely]] {
		Detail::NoteFullKey(hash);
	}
}

// ---------------------------------------------------------------------------------------------
// The serial reductions (KYTY_IMG_<NAME>, live, 0 off | 1 on | 2 or "verify"; default 0). Verify
// does both, compares, counts mismatches and keeps the old result. Their counters are always kept
// and printed as one "ImgSwitch 10s:" line while any switch is not off (release builds too).
enum class Mode : uint8_t { Off = 0, On = 1, Verify = 2 };

[[nodiscard]] inline Mode ParseMode(const char* value) noexcept {
	if (value == nullptr || value[0] == '\0') {
		return Mode::Off;
	}
	if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) {
		return Mode::On;
	}
	if (std::strcmp(value, "2") == 0 || std::strcmp(value, "verify") == 0) {
		return Mode::Verify;
	}
	return Mode::Off;
}

enum class S : uint8_t {
	Memo2Evict,    // KYTY_IMG_MEMO2: primary entries moved into the second level
	Memo2Hit,      // second-level entries promoted (on) / claimed (verify)
	Memo2Miss,     // primary misses the second level did not hold
	Memo2Checks,   // verify: claims compared with the full resolution
	Memo2Mismatch, // verify: claims that differed
	Count
};

// KYTY_IMG_MEMO2's replacement choice inside a second-level set of `ways` ways: the way holding the
// same key (its older recording), else the first empty way, else round robin (`next` advances).
template <typename SameKey, typename Empty>
[[nodiscard]] uint32_t PickWay(uint32_t ways, uint8_t& next, SameKey&& same_key, Empty&& empty) {
	for (uint32_t w = 0; w < ways; w++) {
		if (same_key(w)) {
			return w;
		}
	}
	for (uint32_t w = 0; w < ways; w++) {
		if (empty(w)) {
			return w;
		}
	}
	return static_cast<uint32_t>(next++ % ways);
}

// Counters of the commit thread (single writer).
inline std::array<std::atomic<uint64_t>, static_cast<size_t>(S::Count)>& SwitchCounters() {
	static std::array<std::atomic<uint64_t>, static_cast<size_t>(S::Count)> counters {};
	return counters;
}
inline void Bump(S counter, uint64_t n = 1) {
	auto& c = SwitchCounters()[static_cast<size_t>(counter)];
	c.store(c.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}
// A verify mismatch: counted, and the first 16 logged ("ImgSwitch verify: <what>").
void ReportMismatch(S counter, const char* what);

// The switches' current values (registered in imgSide.cpp).
[[nodiscard]] Mode Memo2Mode();

// The 10 s lines (stats with KYTY_IMG_STATS; the switches' counters while any is not off).
void PrintSummary();
// Frame boundary (graphics processor reset): the frame count of the averages.
void OnFrame();

} // namespace Libs::Graphics::ImgSide

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_IMGSIDE_H_
