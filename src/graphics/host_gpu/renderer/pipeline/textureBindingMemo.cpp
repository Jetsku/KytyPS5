#include "graphics/host_gpu/renderer/pipeline/textureBindingMemo.h"
#include "common/staticOnce.h"

#include "common/assert.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/drawPrep/bindingPlan.h"
#include "graphics/host_gpu/renderer/drawPrep/imgSide.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <xxhash.h>

namespace Libs::Graphics {

struct TextureBindingMemo::Entry {
	Key key;
	// 0: empty. Unique per recording, so a binding that holds this tag holds this description.
	uint64_t tag = 0;
	// The ImagePageTable page holding the description's first byte, and its structure version
	// (TextureCache::PageVersion) when recorded (both unused for null images).
	uint64_t page         = 0;
	uint64_t page_version = 0;
	// KYTY_IMG_OVERLAP: an answer of FindImage's overlap loop. Its other pages (the rest of the
	// lookup region, and the image's own first page when outside it) must keep their versions too,
	// and it is never revalidated.
	bool                                overlap     = false;
	uint8_t                             extra_count = 0;
	std::array<uint64_t, MaxExtraPages> extra_pages {};
	std::array<uint64_t, MaxExtraPages> extra_versions {};
	// The overlap loop's candidates whose "keep" decision depended on time (TextureCache::
	// OverlapLookup::timed): each must still exist and not be OverlapSafeToDelete.
	uint8_t                                                  timed_count = 0;
	std::array<ImageId, TextureCache::OverlapLookup::MaxTimed> timed {};
	// KYTY_IMG_DEPTHALIAS: a depth-class answer. The pages [span_first, span_first + span_count)
	// (outer region, inner regions, the image's first page) keep their structure versions exactly
	// while their sum is the recorded one (versions only grow). `rebind`: the alias branch's
	// rebind effects, replayed by a hit in the lookup's order.
	bool                                                         depth_class  = false;
	uint8_t                                                      rebind_count = 0;
	std::array<ImageId, TextureCache::OverlapLookup::MaxRebinds> rebind {};
	uint64_t                                                     span_first = 0;
	uint32_t                                                     span_count = 0;
	uint64_t                                                     span_sum   = 0;
	// The image's own first page: SyncAliasFromOwner's partners are registered there (= page for
	// first-page answers).
	uint64_t alias_page = 0;
	ImageId  image;
	// TextureCache::RequestedFirstLevel of the description for this image: FindImage and
	// FindTexture extend the image's resident levels (EnsureResidency) unless its resident_first
	// is at most this.
	uint32_t requested_first = 0;
	// Another registered image has the same backing range, extent and sample count, so
	// SyncAliasFromOwner may copy into this one unless it owns the bytes.
	bool                    has_partner = false;
	bool                    null_image  = false;
	// FindImage's exact_format for the description (KYTY_TEXTURE_MEMO_REVALIDATE).
	bool                    exact_format = false;
	vk::ImageView           view        = nullptr;
	TextureCache::ImageDesc desc;
	// KYTY_CP_COMMIT=texdcc: a DCC description, recorded with the certificate of its FindImage's
	// DCC decision (MaterializeDccClear's MetadataNoop); false for every other entry.
	bool                       dcc = false;
	TextureCache::MetadataNoop dcc_noop;
	// FindHint's view of `key` and `tag` (P4b-2): Record makes `seq` odd, stores the packed key
	// and the tag, and makes it even again (release); a reader that sees the same even value
	// before and after its loads read a pair Record wrote together.
	std::atomic<uint32_t>                         seq {0};
	std::atomic<uint64_t>                         published_tag {0};
	std::array<std::atomic<uint64_t>, KeyWords>   packed_key {};
};

namespace {

// KYTY_TEXTURE_MEMO_REVALIDATE (default on; =0 off): an entry whose first page changed its owner
// list (the page version moved: an image registered or unregistered there, often a transient
// render target sharing the 1 MiB page) is checked again instead of being dropped: the lookup
// FindImage makes (FindImageWithSameBacking on the recorded description), the alias-partner scan
// and the version are redone exactly as Record() does them, under the same lock. If the lookup
// still returns the recorded image, the entry (and its view) stay valid for the new version.
bool RevalidateEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_TEXTURE_MEMO_REVALIDATE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_TEXTURE_MEMO_REVALIDATE_VERIFY=1|exit: every revalidated hit is followed by the full
// resolution (RenderExecutor::ResolveTexture), and a different image or description is counted
// (TextureBindingMemoRevalidateMismatches; exit stops on the first).
int RevalidateVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_TEXTURE_MEMO_REVALIDATE_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

} // namespace

bool TextureBindingMemo::HasPartner(const TextureCache& cache, uint64_t page, ImageId found,
                                    const Image& image) {
	// SyncAliasFromOwner's grid rule (SameAliasGrid: equal extents, or with KYTY_ALIAS_BLOCK_VIEWS a
	// block-compressed / uncompressed pair over the same block grid). The extent-only test this
	// replaced missed block-view partners (titles with KYTY_ALIAS_BLOCK_VIEWS=1); without that
	// switch both are the same test.
	return cache.HasAliasPartner(page, found, image);
}

TextureBindingMemo::AliasSync TextureBindingMemo::ClassifyAliasSync(const TextureCache& cache,
                                                                    const Entry&        entry,
                                                                    const Image&        image) {
	if (TextureCache::SyncAliasReturnsAtOnce(image)) {
		return AliasSync::AtOnce;
	}
	// Every image SyncAliasFromOwner could copy from has image's exact range (IsAliasSyncSource),
	// so it starts on the same first page and is in that page's owner list, which is the recorded
	// one while the page version holds (checked before this by every caller).
	bool source = false;
	if (const auto* owners = cache.m_image_page_table.Find(entry.alias_page); owners != nullptr) {
		owners->ForEach([&](ImageId other_id) {
			const auto* other = cache.m_slot_images.try_get(other_id);
			if (other_id != entry.image && other != nullptr &&
			    TextureCache::IsAliasSyncSource(*other, image)) {
				source = true;
			}
		});
	}
	return source ? AliasSync::Owner : AliasSync::NoOwner;
}

uint64_t TextureBindingMemo::SpanSum(const TextureCache& cache, uint64_t first, uint32_t count) {
	uint64_t sum = 0;
	for (uint64_t page = first; page < first + count; page++) {
		sum += cache.PageVersion(page);
	}
	return sum;
}

bool TextureBindingMemo::DepthAliasClass(const TextureCache::OverlapLookup& lookup, ImageId found) {
	// The loop answered through ResolveDepthOverlap without recreating (depth_class), or recreated
	// (impure for that reason): KYTY_IMG_DEPTHALIAS decides (and counts) either.
	return ImgSide::DepthAliasMode() != ImgSide::Mode::Off && lookup.overlap && lookup.result == found &&
	       (lookup.depth_class || lookup.impure_why == TextureCache::OverlapLookup::WhyDepth);
}

bool TextureBindingMemo::PagesHold(const TextureCache& cache, const Entry& entry) {
	if (cache.PageVersion(entry.page) != entry.page_version) {
		return false;
	}
	// KYTY_IMG_DEPTHALIAS: page versions only grow, so an unchanged sum is an unchanged version of
	// every page of the span.
	if (entry.depth_class && SpanSum(cache, entry.span_first, entry.span_count) != entry.span_sum) {
		return false;
	}
	for (uint32_t i = 0; i < entry.extra_count; i++) {
		if (cache.PageVersion(entry.extra_pages[i]) != entry.extra_versions[i]) {
			return false;
		}
	}
	for (uint32_t i = 0; i < entry.timed_count; i++) {
		// The same candidates are there (page versions); each must take its "keep" branch again.
		const auto* candidate = cache.m_slot_images.try_get(entry.timed[i]);
		if (candidate == nullptr || cache.OverlapSafeToDelete(*candidate)) {
			return false;
		}
	}
	return true;
}

bool TextureBindingMemo::OverlapUsable(const Entry& entry) {
	// KYTY_IMG_DEPTHALIAS entries answer only through TryResolve (which replays their rebind
	// effects): the worker-hinted runs and the stage set repeats leave them to it.
	return !entry.overlap || (!entry.depth_class && ImgSide::OverlapMode() == ImgSide::Mode::On);
}

bool TextureBindingMemo::AliasSyncNoOp(const TextureCache& cache, const Entry& entry,
                                       const Image& image) {
	return ClassifyAliasSync(cache, entry, image) != AliasSync::Owner;
}

bool TextureBindingMemo::PartnerBlocks(const TextureCache& cache, const Entry& entry,
                                       const Image& image, bool partner_switch) {
	if (!entry.has_partner) {
		return false;
	}
	if (!partner_switch) {
		return !image.alias_owner && !image.info.HasStencil();
	}
	return !AliasSyncNoOp(cache, entry, image);
}

TextureBindingMemo::Entry* TextureBindingMemo::SlotEntry(uint32_t slot) {
	if (slot < SlotCount()) {
		return m_entries ? &m_entries[slot] : nullptr;
	}
	const auto second = static_cast<size_t>(slot - SlotCount());
	return m_second && second < static_cast<size_t>(SecondSets()) * SecondWays ? &m_second[second]
	                                                                         : nullptr;
}

const TextureBindingMemo::Entry* TextureBindingMemo::SlotEntry(uint32_t slot) const {
	return const_cast<TextureBindingMemo*>(this)->SlotEntry(slot);
}

const TextureBindingMemo::Entry& TextureBindingMemo::SlotRef(uint32_t slot) const {
	static const Entry none {}; // tag 0: matches no binding's tag
	const auto*        entry = SlotEntry(slot);
	return entry != nullptr ? *entry : none;
}

bool TextureBindingMemo::InplaceVerifySlot(uint32_t slot) {
	return IsSecondSlot(slot) && ImgSide::InplaceMode() == ImgSide::Mode::Verify;
}

bool TextureBindingMemo::RevalidateVerify() {
	return RevalidateVerifyMode() != 0;
}

void TextureBindingMemo::ReportRevalidateMismatch() {
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureBindingMemoRevalidateMismatches);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr, "TextureMemoRevalidateVerify: a revalidated entry resolved differently\n");
	}
	if (RevalidateVerifyMode() == 2) {
		EXIT("TextureMemoRevalidateVerify: a revalidated entry resolved differently\n");
	}
}

uint32_t TextureBindingMemo::SlotCount() {
	static constexpr auto slots_init = [] {
		const auto* value = std::getenv("KYTY_TEXTURE_MEMO_SLOTS");
		const auto  count = value != nullptr && value[0] != '\0' ? std::strtoul(value, nullptr, 10) : 0ul;
		return count >= 64 && count <= (1ul << 22u) ? static_cast<uint32_t>(count) : 4096u;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(slots_init())>> slots_once;
	const uint32_t& slots = slots_once.Get(slots_init);
	return slots;
}

TextureBindingMemo::TextureBindingMemo() = default;
TextureBindingMemo::~TextureBindingMemo() = default;

bool TextureBindingMemo::Enabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_TEXTURE_BINDING_MEMO");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}));
	return enabled;
}

TextureBindingMemo::Key TextureBindingMemo::MakeKey(const ShaderRecompiler::IR::ImageResource& resource,
                                                    const uint32_t (&words)[8]) {
	Key key;
	std::memcpy(key.words.data(), words, sizeof(words));
	key.resource_class    = resource.resource_class;
	key.numeric_class     = resource.numeric_class;
	key.dimension         = resource.dimension;
	key.mip_mode          = resource.mip_mode;
	key.mip_count         = resource.mip_count;
	key.conversion_format = resource.conversion_format;
	key.shader_swizzle    = resource.shader_swizzle;
	key.read              = resource.read;
	key.written           = resource.written;
	key.atomic            = resource.atomic;
	key.atomic64          = resource.atomic64;
	key.depth_compare     = resource.depth_compare;
	key.cube              = resource.cube;
	key.r128              = resource.r128;
	key.max_array_index   = resource.max_array_index;
	return key;
}

uint64_t TextureBindingMemo::Hash(const Key& key) {
	return XXH3_64bits_withSeed(key.words.data(), sizeof(key.words),
	                            (static_cast<uint64_t>(key.dimension) << 32u) ^
	                                (static_cast<uint64_t>(key.numeric_class) << 16u) ^
	                                (key.written ? 1u : 0u) ^ (key.depth_compare ? 2u : 0u) ^
	                                (static_cast<uint64_t>(key.mip_mode) << 8u));
}

bool TextureBindingMemo::RefreshIsNoOp(const Image& image) {
	// Mirrors TextureCache::RefreshImage for a registered image without a stencil association:
	// TrackImage returns at once when the resident range (`live`) is watched as a whole, or, for
	// a chunk-tracked image, when it is watched and no chunk was released; the maybe-dirty edge
	// hash runs only for a maybe-dirty image; InitializeImage (upload, partial or whole) runs
	// only for a buffer-modified or CPU-dirty image (chunk writes and residency extensions mark
	// the image CPU-dirty). Any new refresh trigger added there must be added here.
	if (image.IsCpuDirty() || image.IsBufferModified() || !image.IsTracked()) {
		return false;
	}
	// KYTY_LIGHT_GRID_GUEST: the per-submission guest-byte check runs at every binding.
	if (TextureCache::LightGridGuestApplies(image)) {
		return false;
	}
	if (image.ChunkTracked()) {
		return image.chunks.untracked_count == 0;
	}
	return image.track_addr == image.live.address && image.track_addr_end == image.live.End();
}

void TextureBindingMemo::Forget(TextureBinding& binding) {
	binding.memo_tag  = 0;
	binding.memo_slot = 0;
}

bool TextureBindingMemo::DccStateHolds(TextureCache& cache, const Entry& entry) {
	// What MaterializeDccClear's decision read outside the texture-cache lock (the recorded fill,
	// the GPU-dirty state of the metadata, the guest key bytes), as TryRepeatLookup checks it.
	if (cache.MetadataStateHolds(entry.dcc_noop)) {
		return true;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpCommitTexDccRejects);
	return false;
}

void TextureBindingMemo::ApplyDcc(TextureCache& cache, const Entry& entry, Image& image) {
	// MaterializeDccClear's bookkeeping before its decision: the image takes the description's
	// metadata, and no other interpretation of those bytes remains (TryRepeatLookup does the same).
	image.info.metadata = entry.desc.info.metadata;
	cache.EraseSurfaceMeta(entry.desc.info.metadata.range.address);
}

TextureBindingMemo::PackedKey TextureBindingMemo::PackKey(const Key& key) {
	PackedKey packed {};
	for (uint32_t i = 0; i < 4; i++) {
		packed[i] = key.words[2 * i] | (static_cast<uint64_t>(key.words[2 * i + 1]) << 32u);
	}
	packed[4] = key.mip_count | (static_cast<uint64_t>(key.shader_swizzle) << 32u);
	packed[5] = static_cast<uint32_t>(key.resource_class) |
	            (static_cast<uint64_t>(static_cast<uint32_t>(key.numeric_class)) << 32u);
	packed[6] = static_cast<uint32_t>(key.dimension) |
	            (static_cast<uint64_t>(static_cast<uint32_t>(key.mip_mode)) << 32u);
	const uint64_t flags = (key.read ? 1u : 0u) | (key.written ? 2u : 0u) | (key.atomic ? 4u : 0u) |
	                       (key.depth_compare ? 8u : 0u) | (key.cube ? 16u : 0u) |
	                       (key.r128 ? 32u : 0u) | (key.atomic64 ? 64u : 0u);
	packed[7] = static_cast<uint32_t>(key.conversion_format) | (flags << 32u);
	return packed;
}

// FindHint's read of one entry's published key and tag (sequence protocol, see Entry).
static bool ReadHint(const std::atomic<uint32_t>& seq, const std::atomic<uint64_t>& published_tag,
                     const std::array<std::atomic<uint64_t>, 8>& packed_key,
                     const std::array<uint64_t, 8>& packed, uint64_t& tag) {
	const auto before = seq.load(std::memory_order_acquire);
	if ((before & 1u) != 0) {
		return false; // being rewritten
	}
	const auto found = published_tag.load(std::memory_order_relaxed);
	bool       same  = found != 0;
	for (uint32_t i = 0; i < 8; i++) {
		same = packed_key[i].load(std::memory_order_relaxed) == packed[i] && same;
	}
	std::atomic_thread_fence(std::memory_order_acquire);
	if (!same || seq.load(std::memory_order_relaxed) != before) {
		return false;
	}
	tag = found;
	return true;
}

bool TextureBindingMemo::FindHint(const Key& key, uint64_t hash, uint64_t& tag) const {
	const auto* entries = m_published.load(std::memory_order_acquire);
	if (entries == nullptr) {
		return false;
	}
	const auto& entry  = entries[hash % SlotCount()];
	const auto  packed = PackKey(key);
	if (ReadHint(entry.seq, entry.published_tag, entry.packed_key, packed, tag)) {
		return true;
	}
	// KYTY_IMG_INPLACE on: a key that answers from the second level gets its tag there (the
	// commit thread's TryResolveRun looks the tag up in the key's set; Evict publishes the ways).
	if (ImgSide::Memo2Mode() != ImgSide::Mode::On || ImgSide::InplaceMode() != ImgSide::Mode::On) {
		return false;
	}
	const auto* second = m_second_published.load(std::memory_order_acquire);
	if (second == nullptr) {
		return false;
	}
	const auto* ways = &second[static_cast<size_t>(SecondSet(hash)) * SecondWays];
	for (uint32_t w = 0; w < SecondWays; w++) {
		if (ReadHint(ways[w].seq, ways[w].published_tag, ways[w].packed_key, packed, tag)) {
			return true;
		}
	}
	return false;
}

Image* TextureBindingMemo::HitImage(TextureCache& cache, const Entry& entry) {
	auto* image = cache.m_slot_images.try_get(entry.image);
	if (image == nullptr || !image->registered || image->depth_id ||
	    image->binding.needs_rebind || entry.requested_first < image->resident_first ||
	    !OverlapUsable(entry) || !PagesHold(cache, entry) ||
	    PartnerBlocks(cache, entry, *image, ImgSide::PartnerMode() == ImgSide::Mode::On)) {
		return nullptr;
	}
	return image;
}

void TextureBindingMemo::ApplyHit(Entry& entry, uint32_t slot, TextureBinding& binding) {
	binding.image_id = entry.image;
	if (binding.memo_tag == entry.tag && binding.memo_slot == slot) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::TextureBindingDescCopiesAvoided);
	} else {
		binding.desc      = entry.desc;
		binding.memo_tag  = entry.tag;
		binding.memo_slot = slot;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureBindingMemoHits);
	m_totals.hits++;
}

uint32_t TextureBindingMemo::TryResolveRun(TextureCache& cache, std::span<const uint64_t> hashes,
                                           std::span<const uint64_t> tags,
                                           std::span<TextureBinding> bindings, bool apply) {
	m_last_revalidated = false;
	m_last_dcc_hit     = false;
	EXIT_IF(hashes.size() < bindings.size() || tags.size() < bindings.size());
	if (!m_entries || bindings.empty() || tags[0] == 0) {
		return 0;
	}
	uint32_t         hits = 0;
	std::scoped_lock lock {cache.m_lock};
	const auto       tick = cache.m_scheduler.CurrentTick();
	const bool second = m_second && ImgSide::Memo2Mode() == ImgSide::Mode::On &&
	                    ImgSide::InplaceMode() == ImgSide::Mode::On;
	for (; hits < bindings.size(); hits++) {
		auto   slot  = static_cast<uint32_t>(hashes[hits] % SlotCount());
		Entry* found = &m_entries[slot];
		if (tags[hits] == 0) {
			break;
		}
		if (found->tag != tags[hits]) {
			// KYTY_IMG_INPLACE on: the hint may name a second-level entry of the key's set (tags are
			// unique, so a way holding the tag holds the key).
			found = nullptr;
			if (second) {
				const auto set = SecondSet(hashes[hits]);
				for (uint32_t w = 0; w < SecondWays; w++) {
					auto& way = m_second[static_cast<size_t>(set) * SecondWays + w];
					if (way.tag == tags[hits]) {
						found = &way;
						slot  = ImgSide::SecondSlot(SlotCount(), set, SecondWays, w);
						break;
					}
				}
			}
			if (found == nullptr) {
				break;
			}
			ImgSide::Bump(ImgSide::S::InplaceRunHit);
		}
		auto& entry = *found;
		if (entry.dcc) {
			break; // TryResolve checks the DCC certificate outside the lock
		}
		Image* image = nullptr;
		if (!entry.null_image) {
			image = HitImage(cache, entry);
			if (image == nullptr) {
				break; // TryResolve decides: stale, or a revalidation
			}
		}
		if (!apply) {
			continue;
		}
		if (image != nullptr) {
			// FindImage's access bookkeeping for the returned image (as TryResolve).
			image->tick_accessed_last = tick;
			cache.TouchImage(*image);
		}
		ApplyHit(entry, slot, bindings[hits]);
	}
	return hits;
}

uint32_t TextureBindingMemo::SecondSets() {
	static constexpr auto sets_init = [] {
		const auto* value = std::getenv("KYTY_IMG_MEMO2_SLOTS");
		const auto  count = value != nullptr && value[0] != '\0' ? std::strtoul(value, nullptr, 10) : 0ul;
		const auto  slots = count >= 256 && count <= (1ul << 22u) ? static_cast<uint32_t>(count) : 32768u;
		return slots / SecondWays;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(sets_init())>> sets_once;
	const uint32_t& sets = sets_once.Get(sets_init);
	return sets;
}

uint32_t TextureBindingMemo::SecondSet(uint64_t hash) {
	// Other bits than the primary slot's (hash % SlotCount).
	return static_cast<uint32_t>((hash >> 32u) % SecondSets());
}

void TextureBindingMemo::CopyAnswer(Entry& to, const Entry& from) {
	ImgSide::Bump(ImgSide::S::Memo2Copies);
	to.key             = from.key;
	to.tag             = from.tag;
	to.page            = from.page;
	to.page_version    = from.page_version;
	to.overlap         = from.overlap;
	to.extra_count     = from.extra_count;
	to.extra_pages     = from.extra_pages;
	to.extra_versions  = from.extra_versions;
	to.alias_page      = from.alias_page;
	to.timed_count     = from.timed_count;
	to.timed           = from.timed;
	to.depth_class     = from.depth_class;
	to.rebind_count    = from.rebind_count;
	to.rebind          = from.rebind;
	to.span_first      = from.span_first;
	to.span_count      = from.span_count;
	to.span_sum        = from.span_sum;
	to.image           = from.image;
	to.requested_first = from.requested_first;
	to.has_partner     = from.has_partner;
	to.null_image      = from.null_image;
	to.exact_format    = from.exact_format;
	to.view            = from.view;
	to.desc            = from.desc;
	to.dcc             = from.dcc;
	to.dcc_noop        = from.dcc_noop;
}

void TextureBindingMemo::Publish(Entry& entry) {
	const auto seq = entry.seq.load(std::memory_order_relaxed);
	entry.seq.store(seq + 1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);
	const auto packed = PackKey(entry.key);
	for (uint32_t i = 0; i < KeyWords; i++) {
		entry.packed_key[i].store(packed[i], std::memory_order_relaxed);
	}
	entry.published_tag.store(entry.tag, std::memory_order_relaxed);
	entry.seq.store(seq + 2, std::memory_order_release);
}

void TextureBindingMemo::Evict(const Entry& entry) {
	if (entry.tag == 0) {
		return;
	}
	if (!m_second) {
		m_second      = std::make_unique<Entry[]>(static_cast<size_t>(SecondSets()) * SecondWays);
		m_second_next = std::make_unique<uint8_t[]>(SecondSets());
		m_second_published.store(m_second.get(), std::memory_order_release);
		std::printf("TextureBindingMemo: KYTY_IMG_MEMO2 second level %u entries (%zu KiB)\n",
		            SecondSets() * SecondWays,
		            static_cast<size_t>(SecondSets()) * SecondWays * sizeof(Entry) / 1024u);
	}
	const auto set  = SecondSet(Hash(entry.key));
	auto*      ways = &m_second[static_cast<size_t>(set) * SecondWays];
	const auto way = ImgSide::PickWay(
	    SecondWays, m_second_next[set],
	    [&](uint32_t w) { return ways[w].tag != 0 && ways[w].key == entry.key; },
	    [&](uint32_t w) { return ways[w].tag == 0; });
	CopyAnswer(ways[way], entry);
	Publish(ways[way]); // FindHint's readers of the second level (KYTY_IMG_INPLACE)
	ImgSide::Bump(ImgSide::S::Memo2Evict);
}

bool TextureBindingMemo::TrySecond(TextureCache& cache, const Key& key, uint64_t hash, uint32_t slot,
                                   bool verify) {
	uint32_t found_slot = 0;
	Entry*   found      = FindSecond(key, hash, found_slot);
	if (found == nullptr) {
		ImgSide::Bump(ImgSide::S::Memo2Miss);
		return false;
	}
	ImgSide::Bump(ImgSide::S::Memo2Hit);
	if (!verify) {
		// Promote: the found recording takes the primary slot, the primary's occupant goes to
		// its own set of the second level.
		auto& primary = m_entries[slot];
		Entry previous;
		CopyAnswer(previous, primary);
		CopyAnswer(primary, *found);
		found->tag = 0;
		Publish(*found);
		Publish(primary);
		if (previous.tag != 0 && !(previous.key == key)) {
			Evict(previous);
		}
		return true;
	}
	// Verify: the claim, when TryResolve's hit conditions hold for the entry now (read only: a
	// revalidation is evaluated, not stored).
	const auto& entry = *found;
	if (entry.dcc && !cache.MetadataStateHolds(entry.dcc_noop)) {
		return false;
	}
	if (!entry.null_image) {
		std::scoped_lock lock {cache.m_lock};
		const auto*      image = cache.m_slot_images.try_get(entry.image);
		if (image == nullptr || !image->registered || image->depth_id || image->binding.needs_rebind ||
		    entry.requested_first < image->resident_first ||
		    (entry.dcc && !cache.MetadataPagesHold(entry.dcc_noop))) {
			return false;
		}
		if (!OverlapUsable(entry) || (entry.overlap && !PagesHold(cache, entry))) {
			return false;
		}
		bool partner = entry.has_partner;
		if (!entry.overlap && !PagesHold(cache, entry)) {
			if (!RevalidateEnabled() ||
			    cache.FindImageWithSameBacking(entry.desc.info, entry.exact_format) != entry.image) {
				return false;
			}
			partner = HasPartner(cache, entry.page, entry.image, *image);
		}
		if (partner) {
			Entry probe;
			probe.page        = entry.page;
			probe.alias_page  = entry.alias_page;
			probe.image       = entry.image;
			probe.has_partner = true;
			if (PartnerBlocks(cache, probe, *image, ImgSide::PartnerMode() == ImgSide::Mode::On)) {
				return false;
			}
		}
	}
	m_claim.kind    = ClaimKind::Memo2;
	m_claim.image   = entry.image;
	m_claim.desc    = entry.desc;
	m_claim.effects = 0;
	return false;
}

TextureBindingMemo::Entry* TextureBindingMemo::FindSecond(const Key& key, uint64_t hash,
                                                          uint32_t& slot) {
	if (!m_second) {
		return nullptr;
	}
	const auto set  = SecondSet(hash);
	const auto base = static_cast<size_t>(set) * SecondWays;
	for (uint32_t w = 0; w < SecondWays; w++) {
		auto& way = m_second[base + w];
		if (way.tag != 0 && way.key == key) {
			slot = ImgSide::SecondSlot(SlotCount(), set, SecondWays, w);
			return &way;
		}
	}
	return nullptr;
}

static bool SameAnswer(const TextureBinding& a, ImageId image, const TextureCache::ImageDesc& desc) {
	// The fields KYTY_TEXTURE_MEMO_REVALIDATE_VERIFY compares.
	return a.image_id == image && a.desc.type == desc.type && a.desc.view_info == desc.view_info &&
	       a.desc.info.data == desc.info.data && a.desc.info.pixel_format == desc.info.pixel_format;
}

void TextureBindingMemo::VerifyClaim(const TextureCache& cache, const TextureBinding& full) {
	const auto kind = m_claim.kind;
	if (kind == ClaimKind::None) {
		return;
	}
	m_claim.kind = ClaimKind::None;
	if (kind == ClaimKind::Memo2) {
		ImgSide::Bump(ImgSide::S::Memo2Checks);
		if (!SameAnswer(full, m_claim.image, m_claim.desc)) {
			ImgSide::ReportMismatch(ImgSide::S::Memo2Mismatch,
			                        "memo2: a second-level entry resolved differently");
		}
		return;
	}
	if (kind == ClaimKind::DepthAlias) {
		// KYTY_IMG_DEPTHALIAS=verify: same answer; the full lookup was depth-class and pure with the
		// same result, made the same rebinds with the same applied flags (the hit's replay), and
		// synchronized the alias exactly when the hit would have (LookupSideEffects).
		ImgSide::Bump(ImgSide::S::DaChecks);
		const auto& lookup = cache.LastOverlap();
		bool        same_rebinds = lookup.rebind_count == m_claim.rebind_count;
		for (uint32_t i = 0; same_rebinds && i < m_claim.rebind_count; i++) {
			same_rebinds = lookup.rebind[i] == m_claim.rebind[i] &&
			               lookup.rebind_applied[i] == m_claim.rebind_applied[i];
		}
		const auto effects = cache.LookupSideEffects() - m_claim.effects;
		if (!SameAnswer(full, m_claim.image, m_claim.desc)) {
			ImgSide::ReportMismatch(ImgSide::S::DaMismatch, "depthalias: an entry resolved differently");
		} else if (!lookup.overlap || !lookup.pure || !lookup.depth_class || lookup.result != m_claim.image) {
			ImgSide::ReportMismatch(ImgSide::S::DaMismatch, "depthalias: the full lookup took another path");
		} else if (!same_rebinds) {
			ImgSide::ReportMismatch(ImgSide::S::DaMismatch, "depthalias: the full lookup's rebinds differ");
		} else if (m_claim.expect_sync ? effects == 0 : effects != 0) {
			ImgSide::ReportMismatch(ImgSide::S::DaMismatch,
			                        m_claim.expect_sync ? "depthalias: the full resolution did not synchronize the alias"
			                                            : "depthalias: the full resolution had a lookup side effect");
		}
		return;
	}
	if (kind == ClaimKind::Overlap) {
		// KYTY_IMG_OVERLAP=verify: same answer, no lookup side effect, and the full lookup's overlap
		// loop was pure again with the same image (no free, expansion, merge or insertion).
		ImgSide::Bump(ImgSide::S::OverlapChecks);
		const auto& lookup = cache.LastOverlap();
		if (!SameAnswer(full, m_claim.image, m_claim.desc)) {
			ImgSide::ReportMismatch(ImgSide::S::OverlapMismatch,
			                        "overlap: an overlap-loop entry resolved differently");
		} else if (cache.LookupSideEffects() != m_claim.effects || !lookup.overlap || !lookup.pure ||
		           lookup.result != m_claim.image) {
			ImgSide::ReportMismatch(ImgSide::S::OverlapMismatch,
			                        "overlap: the full lookup did more than the entry's answer");
		}
		return;
	}
	// KYTY_IMG_PARTNER=verify: same answer, and FindImage synchronized nothing (nor extended
	// residency) - LookupSideEffects counts both and only the full resolution ran since the claim.
	ImgSide::Bump(ImgSide::S::PartnerChecks);
	if (!SameAnswer(full, m_claim.image, m_claim.desc)) {
		ImgSide::ReportMismatch(ImgSide::S::PartnerMismatch,
		                        "partner: an alias-partner entry resolved differently");
	} else if (const auto effects = cache.LookupSideEffects() - m_claim.effects;
	           m_claim.expect_sync ? effects == 0 : effects != 0) {
		// A no-op claim: FindImage must synchronize nothing (nor extend residency); a sync claim: it
		// must synchronize (the hit would have made that same call).
		ImgSide::ReportMismatch(ImgSide::S::PartnerMismatch,
		                        m_claim.expect_sync ? "partner: the full resolution did not synchronize the alias"
		                                            : "partner: the full resolution synchronized the alias");
	}
}

bool TextureBindingMemo::InplaceDiffers(const TextureBinding& hit, const TextureBinding& full) {
	ImgSide::Bump(ImgSide::S::InplaceChecks);
	if (SameAnswer(full, hit.image_id, hit.desc)) {
		return false;
	}
	ImgSide::ReportMismatch(ImgSide::S::InplaceMismatch,
	                        "inplace: a second-level entry resolved differently");
	return true;
}

bool TextureBindingMemo::TryResolve(TextureCache& cache, const Key& key, uint64_t hash,
                                    TextureBinding& binding, uint64_t tag_hint, bool verify_hint) {
	m_last_revalidated = false;
	m_last_dcc_hit     = false;
	m_claim.kind       = ClaimKind::None;
	m_inplace_verify   = false;
	if (!m_entries) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::TextureBindingMemoMisses);
		ImgSide::Count(ImgSide::C::MissEmpty);
		return false;
	}
	const auto slot       = static_cast<uint32_t>(hash % SlotCount());
	Entry*     entry      = &m_entries[slot];
	uint32_t   entry_slot = slot;
	// A hint naming this entry's tag proves the key (FindHint); otherwise compare it.
	const bool hinted = tag_hint != 0 && entry->tag == tag_hint;
	if (hinted && verify_hint) {
		DrawPrep::CountBindingVerifyCheck();
		if (!(entry->key == key)) {
			DrawPrep::ReportBindingMismatch("texture memo hint");
		}
	}
	bool inplace_hit    = false;
	bool inplace_verify = false;
	if (entry->tag == 0 || !(hinted || entry->key == key)) {
		const auto second  = ImgSide::Memo2Mode();
		const bool empty   = entry->tag == 0;
		const auto inplace = second == ImgSide::Mode::On ? ImgSide::InplaceMode() : ImgSide::Mode::Off;
		bool       found   = false;
		if (inplace != ImgSide::Mode::Off) {
			// KYTY_IMG_INPLACE: the second-level entry answers where it is (continues below exactly
			// as a primary hit, on that entry and its slot).
			uint32_t second_slot = 0;
			if (auto* held = FindSecond(key, hash, second_slot); held != nullptr) {
				ImgSide::Bump(ImgSide::S::Memo2Hit);
				entry          = held;
				entry_slot     = second_slot;
				found          = true;
				inplace_hit    = true;
				inplace_verify = inplace == ImgSide::Mode::Verify;
			} else {
				ImgSide::Bump(ImgSide::S::Memo2Miss);
			}
		} else if (second != ImgSide::Mode::Off) {
			// KYTY_IMG_MEMO2 on: a promoted entry continues as a primary hit below.
			found = TrySecond(cache, key, hash, slot, second == ImgSide::Mode::Verify);
		}
		if (!found) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureBindingMemoMisses);
			ImgSide::Count(empty ? ImgSide::C::MissEmpty : ImgSide::C::MissConflict);
			return false;
		}
	}
	// KYTY_CP_COMMIT=texdcc: MaterializeDccClear would decide the same, and do nothing but its
	// bookkeeping (below), only while the recorded certificate holds.
	if (entry->dcc && !DccStateHolds(cache, *entry)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::TextureBindingMemoStale);
		m_totals.stale++;
		ImgSide::Count(ImgSide::C::MissDcc);
		return false;
	}
	bool revalidated = false;
	if (!entry->null_image) {
		std::scoped_lock lock {cache.m_lock};
		auto*            image = cache.m_slot_images.try_get(entry->image);
		const auto       stale = [&] {
            Profiler::CountFrameEvent(Profiler::FrameEvent::TextureBindingMemoStale);
            m_totals.stale++;
            return false;
		};
		if (entry->dcc && !cache.MetadataPagesHold(entry->dcc_noop)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::CpCommitTexDccRejects);
			ImgSide::Count(ImgSide::C::MissDcc);
			return stale();
		}
		// A stencil association redirects the binding; a pending rebind, a copy from the alias
		// owner and a residency extension (resident levels changed, or fewer than the view
		// samples) are FindImage/RebindImages work: all take the full resolution.
		if (image == nullptr || !image->registered || image->depth_id ||
		    image->binding.needs_rebind || entry->requested_first < image->resident_first) {
			ImgSide::Count(image != nullptr && image->registered && !image->depth_id &&
			                       !image->binding.needs_rebind
			                   ? ImgSide::C::MissResident
			                   : ImgSide::C::MissImage);
			return stale();
		}
		// A structural change on the description's first page may alter what FindImage returns:
		// redo its lookup and the partner scan for the new owner list (RevalidateEnabled).
		if (entry->overlap) {
			// KYTY_IMG_OVERLAP: every page of the lookup region keeps its version (no revalidation).
			if (!PagesHold(cache, *entry)) {
				ImgSide::Count(ImgSide::C::MissPage);
				return stale();
			}
		} else if (const auto version = cache.PageVersion(entry->page); entry->page_version != version) {
			if (!RevalidateEnabled() ||
			    cache.FindImageWithSameBacking(entry->desc.info, entry->exact_format) !=
			        entry->image) {
				ImgSide::Count(ImgSide::C::MissPage);
				return stale();
			}
			entry->has_partner  = HasPartner(cache, entry->page, entry->image, *image);
			entry->page_version = version;
			entry->alias_page   = entry->page;
			revalidated         = true;
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureBindingMemoRevalidated);
			m_totals.revalidated++;
		}
		// KYTY_IMG_DEPTHALIAS: the alias branch's rebind targets are the lookup's candidates, registered
		// on the span's pages: present while the span holds (checked anyway; a miss is the full path).
		for (uint32_t i = 0; i < entry->rebind_count; i++) {
			if (cache.m_slot_images.try_get(entry->rebind[i]) == nullptr) {
				ImgSide::Count(ImgSide::C::MissPage);
				return stale();
			}
		}
		bool hit_sync = false; // KYTY_IMG_PARTNER: the hit synchronizes the alias (after the rebinds)
		if (entry->has_partner) {
			// An alias partner: FindImage would synchronize the image from an owner unless it owns
			// the bytes (the old rule) or, by KYTY_IMG_PARTNER's rule, no image qualifies as an
			// owner (TextureCache::IsAliasSyncSource; SyncAliasFromOwner then does nothing).
			// (On also refuses a stencil image the old rule let answer when, with
			// KYTY_STENCIL_DEPTH_ALIAS, its alias can still be synchronized.)
			const auto partner_mode = ImgSide::PartnerMode();
			const bool old_blocks   = PartnerBlocks(cache, *entry, *image, false);
			const bool classify     = partner_mode == ImgSide::Mode::On ||
			                      (old_blocks && (partner_mode != ImgSide::Mode::Off || ImgSide::StatsEnabled()));
			const auto sync = classify ? ClassifyAliasSync(cache, *entry, *image) : AliasSync::Owner;
			if (old_blocks && ImgSide::StatsEnabled()) {
				ImgSide::Count(sync == AliasSync::AtOnce    ? ImgSide::C::PartnerAtOnce
				               : sync == AliasSync::NoOwner ? ImgSide::C::PartnerNoOwner
				                                            : ImgSide::C::PartnerOwner);
			}
			const auto action = ImgSide::PartnerDecision(partner_mode, old_blocks, sync != AliasSync::Owner);
			if (action == ImgSide::PartnerAction::Miss || action == ImgSide::PartnerAction::Claim) {
				if (action == ImgSide::PartnerAction::Claim) {
					// Claimed; the full resolution provides the result (VerifyClaim).
					m_claim.kind        = ClaimKind::Partner;
					m_claim.image       = entry->image;
					m_claim.desc        = entry->desc;
					m_claim.effects     = cache.LookupSideEffects();
					m_claim.expect_sync = sync == AliasSync::Owner;
				}
				ImgSide::Count(ImgSide::C::MissPartner);
				return stale();
			}
			if (old_blocks) {
				ImgSide::Bump(ImgSide::S::PartnerHit); // an answer the old rule refused
			}
			hit_sync = action == ImgSide::PartnerAction::HitSync;
		}
		const auto class_mode = entry->depth_class ? ImgSide::DepthAliasMode() : ImgSide::OverlapMode();
		if (entry->overlap && class_mode != ImgSide::Mode::On) {
			// KYTY_IMG_OVERLAP=verify / KYTY_IMG_DEPTHALIAS=verify: every hit condition holds; claimed
			// (with the effects the hit would perform), and the full resolution provides the result
			// (VerifyClaim). Off: such entries never answer. Nothing was done yet.
			if (class_mode == ImgSide::Mode::Verify) {
				m_claim.kind         = entry->depth_class ? ClaimKind::DepthAlias : ClaimKind::Overlap;
				m_claim.image        = entry->image;
				m_claim.desc         = entry->desc;
				m_claim.effects      = cache.LookupSideEffects();
				m_claim.expect_sync  = hit_sync;
				m_claim.rebind_count = entry->rebind_count;
				m_claim.rebind       = entry->rebind;
				for (uint32_t i = 0; i < entry->rebind_count; i++) {
					const auto& depth = cache.m_slot_images[entry->rebind[i]];
					m_claim.rebind_applied[i] = depth.binding.is_bound || depth.binding.is_target;
				}
			}
			ImgSide::Count(ImgSide::C::MissPage);
			return false;
		}
		if (entry->rebind_count != 0) {
			// KYTY_IMG_DEPTHALIAS: the alias branch's effect, in the lookup's order, before
			// SyncAliasFromOwner as in FindImage (textureCache.cpp ResolveDepthOverlap).
			for (uint32_t i = 0; i < entry->rebind_count; i++) {
				auto& depth = cache.m_slot_images[entry->rebind[i]];
				if (ImgSide::ApplyRebind(depth.binding.is_bound, depth.binding.is_target,
				                         depth.binding.needs_rebind)) {
					NoteImageStateChange();
					ImgSide::Bump(ImgSide::S::DaRebind);
				}
			}
		}
		if (hit_sync) {
			// FindImage's SyncAliasFromOwner for its (unchanged) answer, at the same point: after
			// the lookup and the (no-op) residency extension, before the touch.
			cache.SyncAliasFromOwner(entry->image);
			ImgSide::Bump(ImgSide::S::PartnerSync);
		}
		// FindImage's access bookkeeping for the returned image.
		image->tick_accessed_last = cache.m_scheduler.CurrentTick();
		cache.TouchImage(*image);
		if (entry->dcc) {
			ApplyDcc(cache, *entry, *image);
		}
		if (entry->overlap) {
			ImgSide::Bump(entry->depth_class ? ImgSide::S::DaHit : ImgSide::S::OverlapHit);
		}
	}
	ApplyHit(*entry, entry_slot, binding);
	ImgSide::Count(revalidated ? ImgSide::C::TexHitReval : ImgSide::C::TexHit);
	if (inplace_hit) {
		ImgSide::Bump(ImgSide::S::InplaceHit);
	}
	m_last_revalidated = revalidated;
	m_last_dcc_hit     = entry->dcc;
	m_inplace_verify   = inplace_verify;
	return true;
}

void TextureBindingMemo::CountOverlap(const TextureCache& cache, ImageId found) {
	// KYTY_IMG_STATS: whether a rejected answer came from a pure overlap loop (TextureCache::LastOverlap).
	const auto& overlap = cache.LastOverlap();
	if (!ImgSide::StatsEnabled() || !overlap.overlap || !overlap.pure || overlap.result != found) {
		return;
	}
	ImgSide::Count(ImgSide::C::OvlPure);
	TextureCache::ImagePageTable::PageRange pages {};
	if (TextureCache::ImagePageTable::TryGetPageRange(overlap.region_address, overlap.region_size, pages) &&
	    pages.last_exclusive - pages.first <= 4) {
		ImgSide::Count(ImgSide::C::OvlPurePages);
		if (overlap.candidates == 1) {
			ImgSide::Count(ImgSide::C::OvlPureOne);
		}
	}
}

void TextureBindingMemo::CountBacking(const TextureCache& cache, const TextureCache::ImageDesc& desc,
                                      bool exact_format, ImageId found) {
	// KYTY_IMG_STATS: why FindImageWithSameBacking(final description) is not FindImage's answer.
	const auto& lookup = cache.LastOverlap();
	if (lookup.result == found) {
		if (lookup.first_page || lookup.inserted || !lookup.pure) {
			ImgSide::Count(lookup.first_page ? ImgSide::C::BkFirstPage
			               : lookup.inserted ? ImgSide::C::BkOvlInserted
			                                 : ImgSide::C::BkOvlImpure);
		}
		if (!(lookup.requested.data == desc.info.data) || lookup.requested.extent != desc.info.extent ||
		    lookup.requested.pixel_format != desc.info.pixel_format ||
		    !(lookup.requested.resources == desc.info.resources) || lookup.requested.type != desc.info.type) {
			ImgSide::Count(ImgSide::C::BkDescChanged);
		}
	}
	const auto other = cache.FindImageWithSameBacking(desc.info, exact_format);
	ImgSide::Count(!other ? ImgSide::C::BkSbNone : ImgSide::C::BkSbOther);
	TextureCache::ImagePageTable::PageRange pages {};
	if (TextureCache::ImagePageTable::TryGetPageRange(desc.info.data.address, desc.info.data.size, pages)) {
		uint32_t matches = 0;
		if (const auto* owners = cache.m_image_page_table.Find(pages.first); owners != nullptr) {
			owners->ForEach([&](ImageId id) {
				const auto* image = cache.m_slot_images.try_get(id);
				if (image != nullptr && image->registered &&
				    TextureCache::SameBacking(image->info, desc.info, exact_format)) {
					matches++;
				}
			});
		}
		if (matches > 1) {
			ImgSide::Count(ImgSide::C::BkSbMulti);
		}
	}
}

void TextureBindingMemo::Record(TextureCache& cache, const Key& key, uint64_t hash,
                                TextureBinding& binding, ImageId found, bool exact_format,
                                bool view_rebased, const TextureCache::MetadataNoop* dcc) {
	Forget(binding);
	const auto& desc = binding.desc;
	// A DCC description only with the certificate of its lookup's DCC decision (texdcc).
	const bool dcc_desc = desc.info.metadata.kind == ImageMetadataKind::Dcc;
	// KYTY_IMG_OVERLAP: answers of FindImage's overlap loop (no first-page answer) are recorded when
	// the loop was pure (OverlapAnswer); a rebased view is such an answer.
	const auto overlap_mode = ImgSide::OverlapMode();
	const bool overlap_ok   = overlap_mode != ImgSide::Mode::Off && !dcc_desc;
	const auto reject       = [&](ImgSide::C reason) {
        Profiler::CountFrameEvent(Profiler::FrameEvent::TextureBindingMemoRejects);
        ImgSide::Count(ImgSide::C::RecReject);
        ImgSide::Count(reason);
	};
	if ((view_rebased && !overlap_ok) || binding.image_id != found ||
	    (dcc_desc && (dcc == nullptr || !dcc->provable))) {
		reject(view_rebased                ? ImgSide::C::RejRebased
		       : binding.image_id != found ? ImgSide::C::RejOtherImage
		                                   : ImgSide::C::RejDcc);
		if (view_rebased) {
			CountOverlap(cache, found);
		}
		return;
	}
	const bool null_image      = desc.info.data.Empty();
	bool       has_partner     = false;
	bool       overlap         = false;
	uint64_t   page            = 0;
	uint64_t   page_version    = 0;
	uint64_t   alias_page      = 0;
	uint32_t   requested_first = 0;
	uint8_t    extra_count     = 0;
	std::array<uint64_t, MaxExtraPages> extra_pages {};
	std::array<uint64_t, MaxExtraPages> extra_versions {};
	uint8_t                                                  timed_count = 0;
	std::array<ImageId, TextureCache::OverlapLookup::MaxTimed> timed {};
	bool                                                       depth_class  = false;
	uint8_t                                                    rebind_count = 0;
	std::array<ImageId, TextureCache::OverlapLookup::MaxRebinds> rebind {};
	uint64_t                                                   span_first = 0;
	uint32_t                                                   span_count = 0;
	uint64_t                                                   span_sum   = 0;
	{
		// One critical section: the checks below and the page versions describe the same state.
		std::scoped_lock lock {cache.m_lock};
		const auto*      image = cache.m_slot_images.try_get(found);
		if (null_image) {
			// FindImage returned the null image of this format; it is never registered or freed.
			if (image == nullptr || image->registered || !image->info.data.Empty()) {
				reject(ImgSide::C::RejNull);
				return;
			}
		} else {
			// Only first-page answers: FindImage would return `found` again, and nothing else,
			// while the owner list of the description's first page is unchanged (SameBacking
			// requires the same start address, so `found` starts on that page too).
			TextureCache::ImagePageTable::PageRange pages {};
			if (cache.m_image_lookup_mode != TextureCache::ImageLookupMode::FirstPage ||
			    image == nullptr || !image->registered || image->depth_id ||
			    !TextureCache::ImagePageTable::TryGetPageRange(image->info.data.address,
			                                                   image->info.data.size, pages)) {
				if (ImgSide::StatsEnabled()) {
					reject(cache.m_image_lookup_mode != TextureCache::ImageLookupMode::FirstPage
					           ? ImgSide::C::RejMode
					       : image == nullptr || !image->registered || image->depth_id ? ImgSide::C::RejImage
					                                                                  : ImgSide::C::RejPages);
				} else {
					reject(ImgSide::C::RejImage);
				}
				return;
			}
			const bool first_page = !view_rebased &&
			                        cache.FindImageWithSameBacking(desc.info, exact_format) == found;
			if (!first_page && DepthAliasClass(cache.LastOverlap(), found)) {
				// KYTY_IMG_DEPTHALIAS: a depth-class answer of the overlap loop (see textureBindingMemo.h).
				const auto& lookup = cache.LastOverlap();
				TextureCache::ImagePageTable::PageRange region {};
				const bool refused = view_rebased || dcc_desc || TextureCache::BlockSubwritesOn() ||
				                     !TextureCache::ImagePageTable::TryGetPageRange(lookup.region_address,
				                                                                    lookup.region_size, region);
				// Every page whose owner list the answer read: the outer scan's region, the inner scans'
				// regions, the image's own first page (SyncAliasFromOwner's partners).
				const auto span = ImgSide::DepthAliasSpan(region.first, region.last_exclusive, lookup.span_first,
				                                          lookup.span_last, pages.first);
				const auto decision = ImgSide::DepthAliasRecord(lookup.pure, refused, span.hi - span.lo, MaxSpanPages);
				const uint64_t span_lo = span.lo;
				const uint64_t span_hi = span.hi;
				ImgSide::Bump(decision == ImgSide::DaRecord::Keep     ? ImgSide::S::DaRecordable
				              : decision == ImgSide::DaRecord::Impure ? ImgSide::S::DaRejImpure
				              : decision == ImgSide::DaRecord::Span   ? ImgSide::S::DaRejSpan
				                                                      : ImgSide::S::DaRejOther);
				if (decision != ImgSide::DaRecord::Keep || DepthAliasCountOnly) {
					reject(ImgSide::C::RejBacking);
					return;
				}
				overlap     = true;
				depth_class = true;
				timed_count = static_cast<uint8_t>(lookup.timed_count);
				timed       = lookup.timed;
				rebind_count = static_cast<uint8_t>(lookup.rebind_count);
				rebind       = lookup.rebind;
				page         = region.first;
				span_first   = span_lo;
				span_count   = static_cast<uint32_t>(span_hi - span_lo);
				span_sum     = SpanSum(cache, span_first, span_count);
				ImgSide::Bump(ImgSide::S::DaRecords);
			} else if (!first_page) {
				// KYTY_IMG_OVERLAP: the overlap loop's answer, when pure (see textureBindingMemo.h).
				const auto& lookup = cache.LastOverlap();
				TextureCache::ImagePageTable::PageRange region {};
				// (A depth-class answer is KYTY_IMG_DEPTHALIAS's: its rebind effect needs the replay.)
				if (!overlap_ok || !lookup.overlap || !lookup.pure || lookup.depth_class ||
				    lookup.result != found ||
				    !TextureCache::ImagePageTable::TryGetPageRange(lookup.region_address,
				                                                   lookup.region_size, region) ||
				    region.last_exclusive - region.first > MaxExtraPages) {
					if (overlap_ok) {
						using L = TextureCache::OverlapLookup;
						const bool same = lookup.overlap && lookup.result == found;
						ImgSide::Bump(!same                                    ? ImgSide::S::OvlRejOther
						              : lookup.inserted                        ? ImgSide::S::OvlRejInserted
						              : lookup.impure_why == L::WhyTimedTaken  ? ImgSide::S::OvlRejTimedTaken
						              : lookup.impure_why == L::WhyTimedMany   ? ImgSide::S::OvlRejTimedMany
						              : lookup.impure_why == L::WhyDepth || lookup.depth_class ? ImgSide::S::OvlRejDepth
						              : lookup.impure_why == L::WhyExpand      ? ImgSide::S::OvlRejExpand
						              : lookup.impure_why == L::WhyMerge       ? ImgSide::S::OvlRejMerge
						              : lookup.impure_why == L::WhyContains    ? ImgSide::S::OvlRejContains
						              : !lookup.pure                           ? ImgSide::S::OvlRejOther
						                                                       : ImgSide::S::OvlRejPages);
					}
					reject(view_rebased ? ImgSide::C::RejRebased : ImgSide::C::RejBacking);
					CountOverlap(cache, found);
					if (!view_rebased && ImgSide::StatsEnabled()) {
						CountBacking(cache, desc, exact_format, found);
					}
					return;
				}
				overlap     = true;
				timed_count = static_cast<uint8_t>(lookup.timed_count);
				timed       = lookup.timed;
				// The description's first page (FindImageWithSameBacking's) is the region's first;
				// the other pages of the region (FindImagesInRegion's owner lists), and the image's own
				// first page (SyncAliasFromOwner's partners) when it lies outside the region.
				page = region.first;
				for (auto p = region.first + 1; p < region.last_exclusive; p++) {
					extra_pages[extra_count++] = p;
				}
				if (pages.first < region.first || pages.first >= region.last_exclusive) {
					extra_pages[extra_count++] = pages.first;
				}
				ImgSide::Bump(ImgSide::S::OverlapRecords);
			} else {
				page = pages.first;
			}
			// What FindImage/FindTexture pass to EnsureResidency for this description; they
			// already made these levels resident, and only EnsureResidency (re-registration)
			// changes resident_first.
			requested_first = cache.RequestedFirstLevel(desc, image->info.resources.levels);
			// SyncAliasFromOwner copies only from images with this exact backing range, grid
			// and sample count; they start on the image's own first page. Their flags (alias
			// owner, GPU-modified, stencil) are read live at each use (PartnerBlocks).
			alias_page   = pages.first;
			has_partner  = HasPartner(cache, alias_page, found, *image);
			page_version = cache.PageVersion(page);
			for (uint32_t i = 0; i < extra_count; i++) {
				extra_versions[i] = cache.PageVersion(extra_pages[i]);
			}
		}
	}
	if (!m_entries) {
		m_entries = std::make_unique<Entry[]>(SlotCount());
		m_published.store(m_entries.get(), std::memory_order_release);
		std::printf("TextureBindingMemo: %u entries (%zu KiB)\n", SlotCount(),
		            static_cast<size_t>(SlotCount()) * sizeof(Entry) / 1024u);
	}
	const auto slot  = static_cast<uint32_t>(hash % SlotCount());
	auto&      entry = m_entries[slot];
	// KYTY_IMG_MEMO2: the recording this one replaces stays in the second level.
	if (ImgSide::Memo2Mode() != ImgSide::Mode::Off && entry.tag != 0 && !(entry.key == key)) {
		Evict(entry);
	}
	// FindHint's readers: odd while the key and tag change (the fence orders the odd value before
	// the new words), even again once both are stored (release).
	const auto seq = entry.seq.load(std::memory_order_relaxed);
	entry.seq.store(seq + 1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);
	entry.key             = key;
	entry.tag             = m_next_tag++;
	const auto packed     = PackKey(key);
	for (uint32_t i = 0; i < KeyWords; i++) {
		entry.packed_key[i].store(packed[i], std::memory_order_relaxed);
	}
	entry.published_tag.store(entry.tag, std::memory_order_relaxed);
	entry.seq.store(seq + 2, std::memory_order_release);
	entry.page            = page;
	entry.page_version    = page_version;
	entry.alias_page      = alias_page;
	entry.overlap         = overlap;
	entry.extra_count     = extra_count;
	entry.extra_pages     = extra_pages;
	entry.extra_versions  = extra_versions;
	entry.timed_count     = timed_count;
	entry.timed           = timed;
	entry.depth_class     = depth_class;
	entry.rebind_count    = rebind_count;
	entry.rebind          = rebind;
	entry.span_first      = span_first;
	entry.span_count      = span_count;
	entry.span_sum        = span_sum;
	entry.image           = found;
	entry.requested_first = requested_first;
	entry.has_partner     = has_partner;
	entry.null_image      = null_image;
	entry.exact_format    = exact_format;
	entry.view            = nullptr;
	entry.desc            = desc;
	entry.dcc             = dcc_desc;
	entry.dcc_noop        = dcc_desc ? *dcc : TextureCache::MetadataNoop {};
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureBindingMemoFills);
	ImgSide::Count(ImgSide::C::RecOk);
	if (dcc_desc) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpCommitTexDccRecords);
	}
	if (overlap && (depth_class ? ImgSide::DepthAliasMode() : overlap_mode) != ImgSide::Mode::On) {
		// KYTY_IMG_OVERLAP=verify / KYTY_IMG_DEPTHALIAS=verify: the entry exists for TryResolve's
		// claims only; the binding stays undescribed, as before (no view memo, no stage repeat
		// through it).
		return;
	}
	binding.memo_tag  = entry.tag;
	binding.memo_slot = slot;
}

const TextureBindingMemo::Entry* TextureBindingMemo::ViewEntry(const TextureBinding& binding) const {
	if (!m_entries || binding.memo_tag == 0 ||
	    binding.desc.type != TextureCache::BindingType::Texture) {
		return nullptr;
	}
	const auto* entry = SlotEntry(binding.memo_slot);
	if (entry == nullptr || entry->tag != binding.memo_tag || entry->image != binding.image_id ||
	    entry->view == nullptr) {
		return nullptr;
	}
	// KYTY_IMG_INPLACE=verify: a second-level entry's view is acquired by FindTexture and compared
	// with the entry's (RecordView).
	if (IsSecondSlot(binding.memo_slot) && ImgSide::InplaceMode() == ImgSide::Mode::Verify) {
		return nullptr;
	}
	return entry;
}

bool TextureBindingMemo::ViewImageReady(const Entry& entry, const Image& image) {
	// FindTexture's rediscovery checks, a no-op EnsureResidency (the view's levels are resident),
	// a no-op RefreshImage, and no stencil plane refresh.
	return image.info.data.Empty() ||
	       (image.registered && !image.depth_id && !image.binding.needs_rebind &&
	        entry.requested_first >= image.resident_first && !image.info.HasStencil() &&
	        RefreshIsNoOp(image));
}

bool TextureBindingMemo::TryAcquireView(TextureCache& cache, TextureBinding& binding) {
	if (!m_entries || binding.memo_tag == 0 ||
	    binding.desc.type != TextureCache::BindingType::Texture) {
		return false;
	}
	const auto* entry = ViewEntry(binding);
	if (entry == nullptr) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::TextureViewMemoMisses);
		ImgSide::Count(ImgSide::C::ViewFindNoEnt);
		return false;
	}
	std::scoped_lock lock {cache.m_lock};
	auto*            image = cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || !ViewImageReady(*entry, *image)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::TextureViewMemoMisses);
		return false;
	}
	cache.TouchImage(*image);
	binding.image_view = entry->view;
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureViewMemoHits);
	ImgSide::Count(ImgSide::C::ViewHit);
	return true;
}

uint32_t TextureBindingMemo::TryAcquireViewRun(TextureCache& cache,
                                               std::span<TextureBinding> bindings, bool apply) {
	if (!m_entries || bindings.empty() || ViewEntry(bindings[0]) == nullptr) {
		return 0;
	}
	uint32_t         hits = 0;
	std::scoped_lock lock {cache.m_lock};
	for (; hits < bindings.size(); hits++) {
		auto&       binding = bindings[hits];
		const auto* entry   = ViewEntry(binding);
		auto*       image   = entry != nullptr ? cache.m_slot_images.try_get(binding.image_id) : nullptr;
		if (image == nullptr || !ViewImageReady(*entry, *image)) {
			break; // TryAcquireView decides
		}
		if (apply) {
			cache.TouchImage(*image);
			binding.image_view = entry->view;
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureViewMemoHits);
		}
	}
	return hits;
}

bool TextureBindingMemo::TryRepeatResolve(TextureCache& cache, std::span<TextureBinding> bindings,
                                          bool apply) {
	m_last_revalidated = false;
	m_last_dcc_hit     = false;
	if (!m_entries) {
		return false;
	}
	// KYTY_CP_COMMIT=texdcc: DCC certificates first, outside the lock (as TryResolve checks them).
	for (const auto& binding: bindings) {
		const auto& entry = SlotRef(binding.memo_slot);
		if (binding.memo_tag != 0 && entry.tag == binding.memo_tag && entry.dcc &&
		    !DccStateHolds(cache, entry)) {
			return false;
		}
	}
	std::scoped_lock lock {cache.m_lock};
	for (const auto& binding: bindings) {
		const auto& entry = SlotRef(binding.memo_slot);
		if (binding.memo_tag == 0 || entry.tag != binding.memo_tag || entry.image != binding.image_id ||
		    InplaceVerifySlot(binding.memo_slot)) {
			return false;
		}
		if (entry.null_image) {
			continue;
		}
		// TryResolve's conditions for a hit that needs no revalidation.
		const auto* image = cache.m_slot_images.try_get(entry.image);
		if (image == nullptr || !image->registered || image->depth_id ||
		    image->binding.needs_rebind || entry.requested_first < image->resident_first ||
		    !OverlapUsable(entry) || !PagesHold(cache, entry) ||
		    PartnerBlocks(cache, entry, *image, ImgSide::PartnerMode() == ImgSide::Mode::On) ||
		    (entry.dcc && !cache.MetadataPagesHold(entry.dcc_noop))) {
			return false;
		}
	}
	if (!apply) {
		return true;
	}
	const auto tick = cache.m_scheduler.CurrentTick();
	for (const auto& binding: bindings) {
		const auto& entry = SlotRef(binding.memo_slot);
		if (!entry.null_image) {
			auto& image              = cache.m_slot_images[entry.image];
			image.tick_accessed_last = tick;
			cache.TouchImage(image);
			if (entry.dcc) {
				ApplyDcc(cache, entry, image);
			}
		}
	}
	// The hits TryResolve would count (each finds its binding's own entry: no description copy).
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureBindingDescCopiesAvoided, bindings.size());
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureBindingMemoHits, bindings.size());
	m_totals.hits += bindings.size();
	return true;
}

uint32_t TextureBindingMemo::RepeatResolveFailure(TextureCache&                    cache,
                                                 std::span<const TextureBinding> bindings) {
	if (!m_entries) {
		return 2;
	}
	for (const auto& binding: bindings) {
		const auto& entry = SlotRef(binding.memo_slot);
		if (binding.memo_tag != 0 && entry.tag == binding.memo_tag && entry.dcc &&
		    !cache.MetadataStateHolds(entry.dcc_noop)) {
			return 1;
		}
	}
	std::scoped_lock lock {cache.m_lock};
	for (const auto& binding: bindings) {
		const auto& entry = SlotRef(binding.memo_slot);
		if (binding.memo_tag == 0 || entry.tag != binding.memo_tag || entry.image != binding.image_id) {
			return 2;
		}
		if (entry.null_image) {
			continue;
		}
		const auto* image = cache.m_slot_images.try_get(entry.image);
		if (image == nullptr || !image->registered || image->depth_id || image->binding.needs_rebind) {
			return 3;
		}
		if (entry.requested_first < image->resident_first) {
			return 4;
		}
		if (!OverlapUsable(entry) || !PagesHold(cache, entry)) {
			return 5;
		}
		if (PartnerBlocks(cache, entry, *image, ImgSide::PartnerMode() == ImgSide::Mode::On)) {
			return 6;
		}
		if (entry.dcc && !cache.MetadataPagesHold(entry.dcc_noop)) {
			return 7;
		}
	}
	return 0;
}

bool TextureBindingMemo::TryRepeatViews(TextureCache& cache, std::span<TextureBinding> bindings,
                                        bool apply) {
	if (!m_entries) {
		return false;
	}
	std::scoped_lock lock {cache.m_lock};
	for (const auto& binding: bindings) {
		const auto& entry = SlotRef(binding.memo_slot);
		if (binding.memo_tag == 0 || binding.desc.type != TextureCache::BindingType::Texture ||
		    entry.tag != binding.memo_tag || entry.image != binding.image_id ||
		    entry.view == nullptr || InplaceVerifySlot(binding.memo_slot)) {
			return false;
		}
		const auto* image = cache.m_slot_images.try_get(binding.image_id);
		// RebindImages resolves a binding again when its image is gone, unregistered (null images
		// never are) or awaits a rebind; then TryAcquireView's conditions.
		if (image == nullptr || image->binding.needs_rebind) {
			return false;
		}
		if (!image->info.data.Empty() &&
		    (!image->registered || image->depth_id ||
		     entry.requested_first < image->resident_first || image->info.HasStencil() ||
		     !RefreshIsNoOp(*image))) {
			return false;
		}
	}
	if (!apply) {
		return true;
	}
	for (auto& binding: bindings) {
		const auto& entry = SlotRef(binding.memo_slot);
		cache.TouchImage(cache.m_slot_images[binding.image_id]);
		binding.image_view = entry.view;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureViewMemoHits, bindings.size());
	return true;
}

vk::ImageView TextureBindingMemo::EntryView(const TextureBinding& binding) const {
	if (!m_entries || binding.memo_tag == 0) {
		return nullptr;
	}
	const auto& entry = SlotRef(binding.memo_slot);
	return entry.tag == binding.memo_tag ? entry.view : nullptr;
}

void TextureBindingMemo::RecordView(const TextureBinding& binding, vk::ImageView view) {
	if (!m_entries || binding.memo_tag == 0 || view == nullptr ||
	    binding.desc.type != TextureCache::BindingType::Texture) {
		return;
	}
	auto* entry = SlotEntry(binding.memo_slot);
	if (entry != nullptr && entry->tag == binding.memo_tag && entry->image == binding.image_id) {
		// KYTY_IMG_INPLACE=verify: FindTexture's view of a second-level entry (ViewEntry refused it)
		// must be the view the entry holds.
		if (entry->view != nullptr && InplaceVerifySlot(binding.memo_slot)) {
			ImgSide::Bump(ImgSide::S::InplaceChecks);
			if (entry->view != view) {
				ImgSide::ReportMismatch(ImgSide::S::InplaceMismatch,
				                        "inplace: FindTexture's view differs from the second-level entry's");
			}
		}
		entry->view = view;
	}
}

} // namespace Libs::Graphics
