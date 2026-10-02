#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXTUREBINDINGMEMO_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXTUREBINDINGMEMO_H_

#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
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
		bool                                      depth_compare     = false;
		bool                                      cube              = false;
		bool                                      r128              = false;

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

private:
	struct Entry;
	static constexpr uint32_t Slots    = 4096;
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
