#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXTUREBINDINGMEMO_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXTUREBINDINGMEMO_H_

#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/drawPrep/imgSide.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>

namespace Libs::Graphics {

struct TextureBinding;

// Texture binding identity memo (KYTY_TEXTURE_BINDING_MEMO, default on; =0 restores the full
// resolution for every binding).
//
// RenderExecutor::ResolveTexture turns (T# dwords, shader image resource) into a texture-cache
// image and a description, and RebindImages turns that into the sampled view. Both repeat for
// every bound texture of every draw although the answer rarely changes. This memo remembers, per
// exact key (the eight T# dwords plus every ImageResource field the resolution reads), the final
// description, the image FindImage returned and the view FindTexture returned.
//
// When a memoized answer is exact:
//  - FindImage. Only answers of the first-page lookup are recorded: after the slow resolution,
//    FindImageWithSameBacking(final description) must return the same image and FindImage must
//    not have rebased the view (mip/slice-of-a-larger-image answers are never recorded; DCC
//    descriptions are recorded only with KYTY_CP_COMMIT=texdcc, together with the certificate of
//    MaterializeDccClear's decision, which inspects guest metadata on every lookup; a hit checks
//    it). That lookup reads only the owner list of the description's first 1 MiB page, the
//    owners' registered flags and SameBacking fields, which are fixed for an image's lifetime.
//    Owner lists and registered flags change only in TextureCache::RegisterImage/
//    UnregisterImage (every creation, free, expansion, overlap resolution, depth recreate,
//    residency change, idle/garbage/pressure collection and unmap goes through them), which
//    bump the structure version of every page the image covers (TextureCache::PageVersion). An
//    entry is used only while its first page's version is unchanged, so FindImage would return
//    the recorded image again.
//  - FindImage side effects. SyncAliasFromOwner must be a no-op: either the image is its alias
//    owner (checked live) or no other registered image has the same backing range, extent and
//    sample count (checked once when recording, over the first-page owners; such partners start
//    on that page, so they appear only with a new version). The tick/LRU touch is performed
//    exactly as FindImage does. A stencil association (depth_id, checked live; attaching one also
//    bumps the page versions) or a pending rebind (checked live) takes the slow path.
//  - Resident mip levels. FindImage and FindTexture call EnsureResidency with the first level
//    the view can sample (TextureCache::RequestedFirstLevel of the description: base level plus
//    MIN_LOD for sampled views, 0 for storage). The memo records that level and uses an entry
//    only while the image's resident_first is at most it (checked live), so EnsureResidency
//    would do nothing; an extension re-registers the image, which also bumps the page versions.
//  - FindTexture (sampled bindings only; storage bindings always take the slow path because they
//    mark the image GPU-written). RefreshImage must be a no-op: the image is not CPU-dirty,
//    maybe-dirty or buffer-modified, and TrackImage has nothing to do (the resident range is
//    watched as a whole, or, when chunk-tracked, watched with no released chunk); it has no
//    stencil plane to refresh. Views are never destroyed or replaced while their image lives
//    and FindView returns the first view matching the normalized info, so the recorded view of
//    the same image id (slot generation included) is what FindView returns.
// All image state is read under the texture-cache lock. The guest write-fault fast path skips
// that lock only for pages on which no image is registered, so it never changes an image the
// memo can answer for.
// Null descriptors resolve to the texture cache's permanent null images (never registered, never
// freed); their entries do not depend on page versions.
class TextureBindingMemo {
public:
	struct Key {
		std::array<uint32_t, 8>                   words {};
		ShaderRecompiler::IR::ImageResourceClass  resource_class {};
		Prospero::TextureNumericClass             numeric_class {};
		ShaderRecompiler::Decoder::ImageDimension dimension {};
		ShaderRecompiler::IR::ImageMipMode        mip_mode {};
		uint32_t                                  mip_count         = 0;
		Prospero::BufferFormat                    conversion_format {};
		uint32_t                                  shader_swizzle    = 0;
		bool                                      read              = false;
		bool                                      written           = false;
		bool                                      atomic            = false;
		bool                                      atomic64          = false;
		bool                                      depth_compare     = false;
		bool                                      cube              = false;
		bool                                      r128              = false;
		// KYTY_TEXTURE_LAYER_BOUND: the view's layer count follows it (descriptors.cpp).
		uint32_t                                  max_array_index   = UINT32_MAX;

		bool operator==(const Key&) const = default;
	};

	TextureBindingMemo();
	~TextureBindingMemo();
	TextureBindingMemo(const TextureBindingMemo&)            = delete;
	TextureBindingMemo& operator=(const TextureBindingMemo&) = delete;

	[[nodiscard]] static bool Enabled();

	[[nodiscard]] static Key MakeKey(const ShaderRecompiler::IR::ImageResource& resource,
	                                 const uint32_t (&words)[8]);
	// The same hash the texture description cache uses for these words and fields.
	[[nodiscard]] static uint64_t Hash(const Key& key);

	// ResolveTexture fast path. On success `binding` holds exactly what the full resolution
	// would produce (image id and final description; view/layout/mip views are reset by the
	// caller) and the FindImage touch was performed. tag_hint: FindHint's tag for this key (0:
	// none); an entry still holding that tag holds this key, so the key comparison is skipped.
	// verify_hint: compare the keys anyway and report a difference (KYTY_DRAW_PREP_BINDINGS_VERIFY).
	[[nodiscard]] bool TryResolve(TextureCache& cache, const Key& key, uint64_t hash,
	                              TextureBinding& binding, uint64_t tag_hint = 0,
	                              bool verify_hint = false);

	// KYTY_DRAW_PREP_BINDINGS texturememo (P4b-2). FindHint, on a draw-prep thread: the tag of the
	// entry that holds `key` in its slot now, read without the command processor's cooperation
	// (a sequence word guards the published key and tag; Record rewrites them). Tags are unique
	// per recording and an entry's key never changes under a tag, so while the entry keeps the tag
	// it holds the key.
	[[nodiscard]] bool FindHint(const Key& key, uint64_t hash, uint64_t& tag) const;
	// TryResolve for a run of bindings named by hints (tags[i], 0: none; hashes[i] their keys'
	// hashes), under one texture-cache lock: in binding order, every binding whose entry still
	// holds its tag and passes TryResolve's hit conditions without a revalidation gets TryResolve's
	// hit (touches, description, counts), up to the first binding that does not, which is left to
	// TryResolve. Nothing between the hits changes what the next check reads (touches and the
	// callers' BindImage do not), so the order of effects is TryResolve's; other threads see the
	// run as one critical section. apply false: count the hits only (verify mode). Returns them.
	[[nodiscard]] uint32_t TryResolveRun(TextureCache& cache, std::span<const uint64_t> hashes,
	                                     std::span<const uint64_t> tags,
	                                     std::span<TextureBinding> bindings, bool apply = true);
	// TryAcquireView for a run of bindings under one texture-cache lock, in binding order, up to
	// the first binding TryAcquireView would not hit (left to it). apply false: count the hits
	// only. Returns them.
	[[nodiscard]] uint32_t TryAcquireViewRun(TextureCache& cache, std::span<TextureBinding> bindings,
	                                         bool apply = true);
	// After a full resolution: records `binding` when the answer is memoizable. `found` is what
	// FindImage returned, `view_rebased` whether FindImage changed the view's base level/layer.
	// `dcc` (KYTY_CP_COMMIT=texdcc): for a DCC description, the certificate of FindImage's DCC
	// decision (its RepeatLookup's); a DCC description without one is not recorded.
	void Record(TextureCache& cache, const Key& key, uint64_t hash, TextureBinding& binding,
	            ImageId found, bool exact_format, bool view_rebased,
	            const TextureCache::MetadataNoop* dcc = nullptr);
	// Marks `binding` as not described by any entry (its description was built elsewhere).
	static void Forget(TextureBinding& binding);

	// RebindImages fast path for a sampled binding: the FindTexture touch and view, when
	// FindTexture would do nothing else.
	[[nodiscard]] bool TryAcquireView(TextureCache& cache, TextureBinding& binding);
	// Remembers the view FindTexture returned for a binding described by an entry.
	void RecordView(const TextureBinding& binding, vk::ImageView view);

	// KYTY_DRAW_SEQUENCE_FAST (textures). A stage whose program and T# words are the ones its
	// bindings were last resolved from: every binding still described by its entry (the entry holds
	// the binding's tag, so it holds the key of that resource and those words) is what TryResolve
	// would find for it again. True when TryResolve would hit for every binding without
	// revalidation, checked under one texture-cache lock; with `apply` each hit's access
	// bookkeeping is then performed, in binding order, as TryResolve performs it (the bindings
	// keep their image and description). `apply` false only checks (verify mode).
	[[nodiscard]] bool TryRepeatResolve(TextureCache& cache, std::span<TextureBinding> bindings,
	                                    bool apply);
	// Diagnostics (KYTY_XFRAME_REUSE): why TryRepeatResolve fails for `bindings` (checked in the
	// same order): 0 it would not, 1 DCC certificate, 2 the entry no longer holds the binding's tag,
	// 3 the image is gone, unregistered, has a stencil association or awaits a rebind, 4 residency,
	// 5 the first page's structure version moved, 6 alias partner, 7 DCC pages.
	[[nodiscard]] uint32_t RepeatResolveFailure(TextureCache& cache,
	                                            std::span<const TextureBinding> bindings);
	// RebindImages for such a stage: true when, for every binding, RebindImages would neither
	// resolve it again nor do anything but TryAcquireView's hit (sampled bindings only). With
	// `apply` each hit's touch and view, in binding order.
	[[nodiscard]] bool TryRepeatViews(TextureCache& cache, std::span<TextureBinding> bindings,
	                                  bool apply);
	// The view recorded for the entry describing `binding` (verify mode, after TryRepeatViews).
	[[nodiscard]] vk::ImageView EntryView(const TextureBinding& binding) const;

	// KYTY_TEXTURE_MEMO_REVALIDATE (textureBindingMemo.cpp): the last TryResolve hit needed its
	// entry revalidated for a new page version; with the verify mode on, the caller then compares
	// it with the full resolution and reports a difference.
	[[nodiscard]] bool        LastHitRevalidated() const noexcept { return m_last_revalidated; }
	// KYTY_CP_COMMIT=texdcc: the last TryResolve hit was a DCC entry's (its certificate held); the
	// same verify mode then also runs the full resolution, which must make no DCC decision
	// that does anything (TextureCache::DccDecisionEffects).
	[[nodiscard]] bool        LastHitDcc() const noexcept { return m_last_dcc_hit; }
	[[nodiscard]] static bool RevalidateVerify();
	static void               ReportRevalidateMismatch();
	// Always counted (the TextureBindingMemo* frame events need a connected profiler).
	struct Totals {
		uint64_t hits        = 0;
		uint64_t stale       = 0;
		uint64_t revalidated = 0;
	};
	[[nodiscard]] const Totals& GetTotals() const noexcept { return m_totals; }

	// KYTY_IMG_MEMO2 (live 0|1|verify, default 0; drawPrep/imgSide.h): a second level behind the
	// direct-mapped entries. While it is not off, an entry Record() replaces moves into a
	// set-associative second table (KYTY_IMG_MEMO2_SLOTS, start-up, default 32768, 4 ways) instead
	// of being lost. On a primary miss TryResolve looks the key up there:
	//  - on: the entry found is swapped into its primary slot (the primary's occupant moves to the
	//    second level) and TryResolve continues exactly as for a primary hit. Exact because an
	//    entry's validity never depends on where it is stored: every condition of a hit is checked
	//    live against the texture cache at each use (image slot generation, registration, stencil
	//    association, pending rebind, resident levels, the first page's structure version or its
	//    revalidation, alias partner, DCC certificate), entries are never invalidated by any other
	//    path than being overwritten, and tags stay unique per recording, so a binding holding the
	//    tag and slot of the promoted entry describes it again.
	//  - verify: nothing is promoted; the claim (the entry's image and description, if all hit
	//    conditions hold now) is kept and compared with the full resolution that follows
	//    (VerifySecondClaim), which provides the result.
	// FindHint readers see the swap through the same sequence protocol as Record().
	//
	// KYTY_IMG_INPLACE (live 0|1|verify, default 0; only while KYTY_IMG_MEMO2=1): a second-level
	// entry answers where it is stored; nothing is swapped. The binding then holds the entry's tag
	// and its second-level slot (memo slots SlotCount() and above name second-level entries), and
	// every user of (tag, slot) - views, stage set repeats - reads the entry there (SlotEntry). Exact
	// for the same reasons as a promotion: validity is checked live at every use, independent of
	// the storage place; a tag lives in exactly one place at a time (an eviction copies a primary
	// entry whose slot is overwritten at once, a promotion clears the second-level copy), so a
	// binding's (tag, slot) matches only the recording it was given. Verify: the in-place answer is
	// followed by the full resolution (kept on a difference), and a view of a second-level entry is
	// acquired by FindTexture and compared with the entry's (RecordView).
	//
	// KYTY_IMG_PARTNER (live 0|1|verify, default 0): an entry whose image has an alias partner and
	// does not own the bytes is a miss today, although FindImage's SyncAliasFromOwner does nothing
	// unless a registered image of the same range qualifies as the owner to copy from
	// (TextureCache::IsAliasSyncSource: registered, alias owner, GPU-modified, not buffer-owned, same
	// range, grid and samples). Such an image starts at the same address, so it is in the owner list
	// of the entry's first page; while the page version holds that list is the recorded one, and the
	// owners' flags are read live. On: the entry answers when SyncAliasReturnsAtOnce holds or no
	// owner of the page qualifies. Verify: such an answer is claimed, the full resolution runs and
	// must give the same image and description and add no lookup side effect (residency
	// extension, alias synchronization).
	//
	// KYTY_IMG_DEPTHALIAS (live 0|1|verify, default 0; acts only with KYTY_IMG_PARTNER=1, since every
	// such entry has an alias partner): answers of FindImage's overlap loop that went through
	// ResolveDepthOverlap without recreating - Astro Bot samples the D32 memory of its depth target
	// as R16G16_SFLOAT through a kept colour alias (ResolveDepthOverlap's keep_aliases branch) - are
	// memoized. Exact condition (every state the full lookup reads, and what it writes):
	//  reads: the owner lists of the outer region's pages (FindImagesInRegion over the description's
	//    resident prefix) and of each inner region (the alias scan over the merged request's range),
	//    the owners' registered flags, stencil associations and live ranges - all of which move the
	//    structure versions of the pages they cover (TextureCache::NoteStructureChange) - the
	//    candidates' fixed fields (info, backing sample count, generation in the id), the switches
	//    read once (KYTY_IMAGE_ALIAS_AGE, KYTY_IMAGE_CONTAINS, ...), and for time-dependent "keep"
	//    branches OverlapSafeToDelete (re-checked live, as KYTY_IMG_OVERLAP does); then FindImage's
	//    tail as for any entry (requested_first, alias partner, touch).
	//  writes: per alias answer, `cached.binding.needs_rebind = true` and the image-state epoch when
	//    the cached (depth) image is bound or a target at that moment (read live at the hit, applied
	//    in the lookup's order, before SyncAliasFromOwner as in FindImage); SyncAliasFromOwner
	//    (KYTY_IMG_PARTNER's HitSync); the tick and LRU touch.
	//  An entry answers while the structure versions of every page of the span (outer region,
	//  inner regions, the image's first page; at most MaxSpanPages) are the recorded ones: versions
	//  only grow, so the sum of the span's versions is unchanged exactly when every version is. A
	//  lookup that recreated (inserted, copied, freed) or had any other impure branch, a rebased
	//  view, a DCC description, KYTY_BLOCK_SUBRESOURCE_WRITES, or more than MaxRebinds alias
	//  answers is not recorded. Never revalidated. Only TryResolve answers them (the worker-hinted
	//  runs and stage set repeats refuse them: they do not replay effects).
	// Verify: the claim (answer, predicted rebinds with their applied flags, whether the hit would
	// synchronize) is compared with the full resolution: same answer, its lookup depth-class and
	// pure with the same result and the same rebind list and applied flags, and alias
	// synchronizations exactly when predicted (LookupSideEffects).
	enum class ClaimKind : uint8_t { None, Memo2, Partner, Overlap, DepthAlias };
	struct SecondClaim {
		ClaimKind               kind = ClaimKind::None;
		ImageId                 image;
		TextureCache::ImageDesc desc;
		uint64_t                effects = 0; // TextureCache::LookupSideEffects when claimed
		bool                    expect_sync = false; // KYTY_IMG_PARTNER: an owner qualified
		// KYTY_IMG_DEPTHALIAS: the rebinds the hit would perform.
		uint32_t                                                     rebind_count = 0;
		std::array<ImageId, TextureCache::OverlapLookup::MaxRebinds> rebind {};
		std::array<bool, TextureCache::OverlapLookup::MaxRebinds>    rebind_applied {};
	};
	// Verify modes: compares the pending claim of the last TryResolve with the full resolution's
	// binding and clears it (no-op without a claim).
	void VerifyClaim(const TextureCache& cache, const TextureBinding& full);
	// KYTY_IMG_INPLACE=verify: the last TryResolve hit was an in-place second-level answer; the
	// caller runs the full resolution and passes it here. True when they differ (counted; the
	// caller keeps the full resolution then).
	[[nodiscard]] bool LastHitInplaceVerify() const noexcept { return m_inplace_verify; }
	[[nodiscard]] bool InplaceDiffers(const TextureBinding& hit, const TextureBinding& full);

private:
	struct Entry;
	// KYTY_IMG_OVERLAP: the most pages besides the first an overlap-loop entry checks.
	static constexpr uint32_t MaxExtraPages = 4;
	// The versions of every page the entry's answer depends on are the recorded ones.
	[[nodiscard]] static bool PagesHold(const TextureCache& cache, const Entry& entry);
	// Not an overlap-loop entry, or KYTY_IMG_OVERLAP is on (verify: TryResolve claims only).
	[[nodiscard]] static bool OverlapUsable(const Entry& entry);
	// KYTY_IMG_DEPTHALIAS: the most pages a depth-class entry's span covers, the sum of the
	// structure versions of [first, first + count), and whether the last lookup is of the class.
	static constexpr uint32_t MaxSpanPages = 64;
	// The count build (r6) set this: depth-class answers counted (DaRecordable), never recorded.
	static constexpr bool DepthAliasCountOnly = false;
	[[nodiscard]] static uint64_t SpanSum(const TextureCache& cache, uint64_t first, uint32_t count);
	[[nodiscard]] static bool DepthAliasClass(const TextureCache::OverlapLookup& lookup, ImageId found);
	// KYTY_TEXTURE_MEMO_SLOTS (startup, default 4096, 64..4M): the number of direct-mapped entries
	// (~1 KiB each; the size is logged once). A level binding more distinct textures per frame
	// than that evicts entries every frame (Sky Garden: most texture sets KYTY_XFRAME_REUSE kept
	// had lost their entries). 32768 (31 MiB): start draws' texture resolution 1.04 -> 0.85 us
	// (Sky Garden) and 1.22 -> 0.99 us (pirate island) with KYTY_CP_COMMIT_STATS; separate
	// launches without counters were +5.7% fps (Sky Garden, ABBA) but mixed elsewhere (launch
	// noise of 5-10% on the pirate island), so the default stays.
	[[nodiscard]] static uint32_t SlotCount();
	static constexpr uint32_t KeyWords = 8;
	using PackedKey                    = std::array<uint64_t, KeyWords>;

	// Every field of a key, losslessly, in words FindHint compares.
	[[nodiscard]] static PackedKey PackKey(const Key& key);
	// TryResolve's hit conditions for an entry of a registered image (no revalidation): the image
	// to touch, or null (a miss). Caller holds cache.m_lock.
	[[nodiscard]] static Image* HitImage(TextureCache& cache, const Entry& entry);
	// TryResolve's effects of a hit on `binding` (the image touch is the caller's).
	void ApplyHit(Entry& entry, uint32_t slot, TextureBinding& binding);
	// TryAcquireView's conditions without its lock (the entry and image checks). Caller holds
	// cache.m_lock for the image part.
	[[nodiscard]] const Entry* ViewEntry(const TextureBinding& binding) const;
	[[nodiscard]] static bool ViewImageReady(const Entry& entry, const Image& image);

	[[nodiscard]] static bool RefreshIsNoOp(const Image& image);
	// KYTY_CP_COMMIT=texdcc: whether the DCC decision recorded with a DCC entry would be made again
	// (MetadataStateHolds reads outside cache.m_lock what the decision reads there, as
	// TryRepeatLookup does; the page part is checked under the lock with the other conditions),
	// and MaterializeDccClear's bookkeeping for a hit (caller holds cache.m_lock).
	[[nodiscard]] static bool DccStateHolds(TextureCache& cache, const Entry& entry);
	static void               ApplyDcc(TextureCache& cache, const Entry& entry, Image& image);
	// Whether a registered image other than `found` has exactly its backing range, extent and
	// sample count on `page` (so SyncAliasFromOwner may copy into it). Caller holds cache.m_lock.
	[[nodiscard]] static bool HasPartner(const TextureCache& cache, uint64_t page, ImageId found,
	                                     const Image& image);

	// KYTY_IMG_STATS: a rejected answer's overlap-loop class (TextureCache::LastOverlap).
	static void CountOverlap(const TextureCache& cache, ImageId found);
	static void CountBacking(const TextureCache& cache, const TextureCache::ImageDesc& desc,
	                         bool exact_format, ImageId found);
	// KYTY_IMG_MEMO2: the second level (allocated when first used).
	static constexpr uint32_t SecondWays = 4;
	[[nodiscard]] static uint32_t SecondSets();
	[[nodiscard]] static uint32_t SecondSet(uint64_t hash);
	// Field copy of an entry's recorded answer (key, tag and the rest; the atomics are FindHint's
	// view, written by Publish).
	static void CopyAnswer(Entry& to, const Entry& from);
	// Writes `entry`'s key and tag into FindHint's view (odd/even sequence, as Record).
	static void Publish(Entry& entry);
	// The primary entry `slot` holds a recording about to be replaced: keep it in the second level.
	void Evict(const Entry& entry);
	// A primary miss of `key`: on - promote the second-level entry into `slot` (true); verify -
	// set m_claim when its hit conditions hold (false).
	[[nodiscard]] bool TrySecond(TextureCache& cache, const Key& key, uint64_t hash, uint32_t slot,
	                             bool verify);
	// KYTY_IMG_INPLACE: the second-level entry holding `key` and its memo slot, or null.
	[[nodiscard]] Entry* FindSecond(const Key& key, uint64_t hash, uint32_t& slot);
	// The entry a memo slot names (primary below SlotCount(), second level above), or null.
	[[nodiscard]] Entry*       SlotEntry(uint32_t slot);
	[[nodiscard]] const Entry* SlotEntry(uint32_t slot) const;
	[[nodiscard]] static bool  IsSecondSlot(uint32_t slot) { return ImgSide::IsSecondSlot(slot, SlotCount()); }
	// SlotEntry, or an empty entry (tag 0, matching no binding) when the slot names none.
	[[nodiscard]] const Entry& SlotRef(uint32_t slot) const;
	// KYTY_IMG_INPLACE=verify and `slot` is a second-level one: the stage repeats and the view memo
	// leave the binding to the paths that compare (TryResolve's full resolution, FindTexture).
	[[nodiscard]] static bool InplaceVerifySlot(uint32_t slot);
	// TryResolve's alias-partner condition: the entry may not answer (FindImage would synchronize
	// the image from an owner). `partner_switch`: KYTY_IMG_PARTNER's rule (on), else the old one.
	[[nodiscard]] static bool PartnerBlocks(const TextureCache& cache, const Entry& entry,
	                                        const Image& image, bool partner_switch);
	// SyncAliasFromOwner(entry.image) would do nothing: it returns at once, or no image of the
	// entry's first page qualifies as the owner. Caller holds cache.m_lock.
	[[nodiscard]] static bool AliasSyncNoOp(const TextureCache& cache, const Entry& entry,
	                                        const Image& image);
	// What SyncAliasFromOwner(entry.image) would do now: return at once, find no owner (both
	// no-ops), or copy from an owner. Caller holds cache.m_lock.
	enum class AliasSync : uint8_t { AtOnce, NoOwner, Owner };
	[[nodiscard]] static AliasSync ClassifyAliasSync(const TextureCache& cache, const Entry& entry,
	                                                 const Image& image);

	std::unique_ptr<Entry[]> m_second;
	std::unique_ptr<uint8_t[]> m_second_next; // round-robin way per set
	// m_second for FindHint's readers (KYTY_IMG_INPLACE): set once, after it exists (release).
	std::atomic<const Entry*> m_second_published {nullptr};
	SecondClaim              m_claim;
	bool                     m_inplace_verify = false;

	std::unique_ptr<Entry[]> m_entries;
	// m_entries for FindHint's readers: set once, after the entries exist (release).
	std::atomic<const Entry*> m_published {nullptr};
	uint64_t                 m_next_tag         = 1;
	bool                     m_last_revalidated = false;
	bool                     m_last_dcc_hit     = false;
	Totals                   m_totals;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXTUREBINDINGMEMO_H_
