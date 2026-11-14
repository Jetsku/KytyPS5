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
	// Record rejects by reason (RecReject is their sum).
	RejRebased,    // FindImage rebased the view (a mip/slice of a larger image)
	RejOtherImage, // the binding's image is not FindImage's answer (stencil association redirect)
	RejDcc,        // DCC description without a provable certificate
	RejNull,       // null description whose image is not the permanent null image
	RejMode,       // image lookup mode is not FirstPage
	RejImage,      // FindImage's answer unregistered / gone / stencil association
	RejPages,      // the image's range has no page range
	RejBacking,    // FindImageWithSameBacking of the final description is not FindImage's answer
	// MissPartner split: what SyncAliasFromOwner would do for that image now.
	PartnerAtOnce, // returns at once (the image owns the bytes in its sense)
	PartnerNoOwner, // no registered image qualifies as the owner to copy from: a no-op
	PartnerOwner,  // an owner qualifies: it copies (or adopts the owner's serial) and takes over
	// Rejects (rebased / backing) whose FindImage answer came from a pure overlap loop
	// (TextureCache::OverlapLookup::pure), of which over a region of at most 4 pages, of which with
	// one candidate.
	OvlPure,
	OvlPurePages,
	OvlPureOne,
	// Backing rejects split: the lookup's path (first-page answer / overlap loop impure / overlap loop
	// that inserted), and what FindImageWithSameBacking(final description) finds (none / another
	// image), and whether several images on the page have that backing.
	BkFirstPage,
	BkOvlImpure,
	BkOvlInserted,
	BkSbNone,
	BkSbOther,
	BkSbMulti,
	BkDescChanged, // the description FindImage was given differs from the final one (info)
	// AcquireSame split: why the draw-run acquisition reuse (DrawRunAcquireCandidate) did not take
	// an acquisition of the same targets as the previous one (first failing condition).
	AcqWhyCaller,  // not a draw's acquisition (another caller) / reuse not tried
	AcqWhyOff,     // KYTY_DRAW_RUN(_ACQUIRE) off, or a verify mode of the run
	AcqWhyNoPrev,  // no recorded previous draw (m_run_prev_valid false)
	AcqWhyEngine,  // not inside the engine commit
	AcqWhyRecord,  // the record has no acquisition (record ineligible / no depth slot)
	AcqWhyCount,   // colour count differs from the record
	AcqWhyActivity, // activity epoch moved since the record
	AcqWhyCommand, // another command buffer
	AcqWhyTick,    // another scheduler tick
	AcqWhyInstance, // no active rendering instance, or another one
	AcqWhyWritten, // the scissor union differs
	AcqWhyTargets, // colour/depth target infos differ
	AcqWhyFeedback, // a texture is an attachment or lies over one
	AcqWhyImages,  // an image changed (DrawRunImagesUnchanged)
	AcqWhyNone,    // the candidate check passes (reuse taken elsewhere / verify)
	// Of the AcqWhyNoPrev acquisitions: what the previous draw's KYTY_IMG_ACQSEED step did
	// (RenderExecutor::SeedWhy; the first is "the step was not reached").
	SeedWhyNotReached,
	SeedWhyKernel,
	SeedWhyEligible,
	SeedWhyOff,
	SeedWhyNotEngine,
	SeedWhyFeedback,
	SeedWhyRendering,
	SeedWhyWrites,
	SeedWhyClears,
	SeedWhyProgram,
	SeedWhyTextures,
	SeedWhyDepthPromotion,
	SeedWhyOverAttachment,
	SeedWhyRecorded,
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
	Memo2Hit,      // second-level entries promoted (on) / claimed (verify) / found (in place)
	Memo2Miss,     // primary misses the second level did not hold
	Memo2Checks,   // verify: claims compared with the full resolution
	Memo2Mismatch, // verify: claims that differed
	Memo2Copies,   // memo entry copies (~1 KiB each): evictions and promotions
	InplaceHit,    // KYTY_IMG_INPLACE: second-level entries that answered in place
	InplaceChecks, // verify: in-place answers and views compared with the full path
	InplaceMismatch,
	PartnerHit,    // KYTY_IMG_PARTNER: alias-partner entries answered that the old rule refused
	PartnerSync,   // ... of which the hit performed SyncAliasFromOwner (an owner qualified)
	InplaceRunHit, // KYTY_IMG_INPLACE: worker-hinted run hits on second-level entries
	PartnerChecks, // verify: claims compared with the full resolution (and its side effects)
	PartnerMismatch,
	OverlapRecords, // KYTY_IMG_OVERLAP: overlap-loop answers recorded
	OverlapHit,    // ... that answered
	OverlapChecks, // verify: claims compared with the full resolution
	OverlapMismatch,
	AcqSeedRecords, // KYTY_IMG_ACQSEED: acquisitions seeded by run-ineligible draws
	AcqSeedHit,     // ... reused by the next draw
	AcqSeedChecks,  // verify: reusable acquisitions compared with the acquisition
	AcqSeedMismatch,
	// KYTY_IMG_OVERLAP not off: why an overlap-loop answer was not recorded (Record).
	OvlRejTimedTaken, // a time-dependent branch decided to free (or the keep window)
	OvlRejTimedMany,  // more than OverlapLookup::MaxTimed time-dependent candidates
	OvlRejDepth,      // a depth grow / depth overlap branch
	OvlRejExpand,     // an expansion
	OvlRejMerge,      // a mip merge / expansion into the request
	OvlRejContains,   // the Contains free
	OvlRejInserted,   // the lookup inserted the image
	OvlRejPages,      // the region spans more than 4 pages
	OvlRejOther,      // DCC description, result changed, no overlap record
	// KYTY_IMG_ACQSEED not off: why a run-ineligible draw seeded nothing.
	SeedRefAttachment, // a texture is an attachment
	SeedRefProgram,    // address writes or GDS
	SeedRefWrites,     // a buffer write over an attachment's memory
	SeedRefOther,      // feedback, clears, no rendering instance, depth promotion, a texture over an attachment
	// KYTY_IMG_DEPTHALIAS (textureBindingMemo.h): the depth-alias class of FindImage's overlap loop.
	DaRecordable,  // Record: a depth-class answer the rule keeps (count build: counted, not kept)
	DaRejImpure,   // ... not kept: the loop recreated (or more than MaxRebinds alias answers)
	DaRejSpan,     // ... not kept: the pages the answer depends on span more than MaxSpanPages
	DaRejOther,    // ... not kept: rebased view, DCC, result not FindImage's, block subwrites, no page range
	DaRecords,     // entries recorded
	DaHit,         // entries that answered
	DaRebind,      // ... of which rebind effects applied (the depth image was bound or a target)
	DaChecks,      // verify: claims compared with the full resolution
	DaMismatch,
	// KYTY_IMG_ACQFB (render.h): acquisitions of the same targets as the previous one, no run record.
	AcqFbSame,
	AcqFbRepeat,       // ... whose acquisition repeated the previous one (state, feedback, no transition, no keep barrier, marks)
	AcqFbRepeatSerial, // ... and left every attachment's content serial as well
	AcqFbFeedback,     // ... repeats with depth feedback aspects
	AcqFbKeep,         // ... not a repeat: the depth-feedback keep noted a barrier
	AcqFbState,        // ... not a repeat: another RenderState or feedback aspects
	AcqFbTransit,      // ... not a repeat: transitions recorded
	AcqFbMarks,        // ... not a repeat: an attachment mark moved (layout/access/stage/...; serials not compared)
	AcqFbNoInstance,   // repeats without an active rendering instance at the acquisition
	AcqFbHit,
	AcqFbChecks,
	AcqFbMismatch,
	Frames,        // frame boundaries (per-frame figures of the line)
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

// KYTY_IMG_INPLACE's memo slot numbering: primary slots are [0, primary); the second-level entry of
// `set`, `way` is primary + set * ways + way (TextureBinding::memo_slot names either kind).
[[nodiscard]] constexpr uint32_t SecondSlot(uint32_t primary, uint32_t set, uint32_t ways, uint32_t way) {
	return primary + set * ways + way;
}
[[nodiscard]] constexpr bool IsSecondSlot(uint32_t slot, uint32_t primary) {
	return slot >= primary;
}

// KYTY_IMG_PARTNER's decision for a memo entry whose image has an alias partner. old_blocks: the
// old rule refuses it (the image is not the alias owner and has no stencil); sync_noop:
// SyncAliasFromOwner would do nothing now (returns at once, or no image qualifies as the owner).
// HitSync: the hit performs SyncAliasFromOwner itself, at the point FindImage performs it.
enum class PartnerAction : uint8_t { Hit, HitSync, Miss, Claim };
[[nodiscard]] constexpr PartnerAction PartnerDecision(Mode mode, bool old_blocks, bool sync_noop) {
	switch (mode) {
		case Mode::On: return sync_noop ? PartnerAction::Hit : PartnerAction::HitSync;
		case Mode::Verify:
			// The old result (a miss: the full resolution), with a claim to compare where the old rule
			// refused.
			return !old_blocks ? PartnerAction::Hit : PartnerAction::Claim;
		case Mode::Off:
		default: return old_blocks ? PartnerAction::Miss : PartnerAction::Hit;
	}
}

// KYTY_IMG_DEPTHALIAS: the effect of ResolveDepthOverlap's alias branch on the cached image, as a
// memo hit replays it (true: applied; the caller then moves the image-state epoch).
[[nodiscard]] constexpr bool ApplyRebind(bool is_bound, bool is_target, bool& needs_rebind) {
	if (is_bound || is_target) {
		needs_rebind = true;
		return true;
	}
	return false;
}
// Record's decision for a depth-class answer of the overlap loop. pure: no branch with another
// effect or a time-dependent "free"; refused: a rebased view, DCC, block subwrites or no page range;
// span_pages: the pages of the outer region, inner regions and the image's first page.
enum class DaRecord : uint8_t { Keep, Impure, Other, Span };
[[nodiscard]] constexpr DaRecord DepthAliasRecord(bool pure, bool refused, uint64_t span_pages,
                                                  uint64_t max_span) {
	return !pure ? DaRecord::Impure : refused ? DaRecord::Other : span_pages > max_span ? DaRecord::Span : DaRecord::Keep;
}
// The span [lo, hi) of pages a depth-class answer depends on: the outer region, the union of the
// inner regions ([inner_lo, inner_hi), empty when inner_lo >= inner_hi) and the image's first page.
struct PageSpan {
	uint64_t lo = 0;
	uint64_t hi = 0;
};
[[nodiscard]] constexpr PageSpan DepthAliasSpan(uint64_t region_lo, uint64_t region_hi, uint64_t inner_lo,
                                                uint64_t inner_hi, uint64_t image_page) {
	PageSpan span {region_lo < image_page ? region_lo : image_page,
	               region_hi > image_page + 1 ? region_hi : image_page + 1};
	if (inner_lo < inner_hi) {
		span.lo = inner_lo < span.lo ? inner_lo : span.lo;
		span.hi = inner_hi > span.hi ? inner_hi : span.hi;
	}
	return span;
}

// KYTY_IMG_ACQFB: whether an acquisition leaves an attachment's content serial as it found it (the
// unwritten depth target, restored by RestoreContentIfUnwritten with KYTY_ALIAS_SYNC_SKIP) or gives
// it a fresh one (colour targets always: their binding marks a GPU write). The reuse replays this.
[[nodiscard]] constexpr bool AcqFbSerialKept(bool is_depth, bool writes, bool alias_sync_skip) {
	return is_depth && !writes && alias_sync_skip;
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
// KYTY_IMG_INPLACE (needs KYTY_IMG_MEMO2=1): a second-level entry answers where it is stored
// instead of being swapped into its primary slot (textureBindingMemo.h).
[[nodiscard]] Mode InplaceMode();
// KYTY_IMG_PARTNER: an entry with an alias partner answers while SyncAliasFromOwner would do
// nothing (no registered image qualifies as the owner to copy from; textureBindingMemo.h).
[[nodiscard]] Mode PartnerMode();
// KYTY_IMG_OVERLAP: answers of FindImage's overlap loop (rebased views, other-format views at the
// same address) are memoized while the loop was pure and the region's pages keep their versions
// (textureBindingMemo.h).
[[nodiscard]] Mode OverlapMode();
// KYTY_IMG_ACQSEED: an ordinary acquisition of a draw that may not seed a draw run seeds the next
// draw's acquisition reuse (render.h, RenderExecutor::AcquireSeed).
[[nodiscard]] Mode AcqSeedMode();
// KYTY_IMG_DEPTHALIAS: depth-class overlap answers (a kept colour alias of a depth image) are
// memoized with the alias branch's rebind effect replayed (textureBindingMemo.h).
[[nodiscard]] Mode DepthAliasMode();
// KYTY_IMG_ACQFB: same-target acquisition reuse after draws that sample their attachments (render.h).
[[nodiscard]] Mode AcqFbMode();

// The 10 s lines (stats with KYTY_IMG_STATS; the switches' counters while any is not off).
void PrintSummary();
// Frame boundary (graphics processor reset): the frame count of the averages.
void OnFrame();

} // namespace Libs::Graphics::ImgSide

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_IMGSIDE_H_
