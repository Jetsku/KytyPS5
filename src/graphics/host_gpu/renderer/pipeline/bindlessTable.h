#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class PageManager;

// The engine method for the guest's texture heaps (Senaxx, engine-method.md): heap entries are
// noticed when the guest writes them, through write-watched pages, instead of by a rolling scan.
// On by default; KYTY_BINDLESS_ENGINE=0 restores the rolling scan and the per-draw image walk.
[[nodiscard]] bool BindlessEngineEnabled();

// The descriptor-set layout of set 1 (BindlessBindings.h), created on first use by whichever
// thread needs it first (a pipeline compile, or the table itself); null when the device has no
// bindless support (GraphicContext::bindless_enabled). Thread-safe.
[[nodiscard]] vk::DescriptorSetLayout BindlessSetLayout(GraphicContext& graphics);
// Slots of each image array of set 1 (the same count for every array).
[[nodiscard]] uint32_t BindlessImagesPerArray(const GraphicContext& graphics);

// KYTY_BINDLESS_VIEW_KEY (Erlou93/KytyPS5-Yotei v0.1.4 TableView): what of a shader's image resource
// changes the view ResolveBindlessKey makes of a T# (numeric class, dimension, mip mode and count,
// conversion format, swizzle, depth compare, cube, R128), hashed; never 0.
[[nodiscard]] uint64_t BindlessViewKey(const ShaderRecompiler::IR::ImageResource& resource);
// 0 off, 1 log heaps whose users view them differently, 2 such users get heaps (and shared slots)
// of their own.
[[nodiscard]] uint32_t BindlessViewKeyMode();

// A guest descriptor heap as one view type samples it: a region of the translation (and
// feedback) buffer, one entry per key, and the textures resolved so far.
struct BindlessHeap {
	uint64_t base         = 0;
	uint32_t table_offset = 0;
	// Record pointer heaps (BindlessHeapUse::record_pointer): key k's T# is at the 64-bit
	// pointer stored at base + k * stride + record_pointer, plus table_offset.
	uint32_t stride         = 32;
	uint32_t record_pointer = UINT32_MAX;
	// Nested heaps (BindlessHeapUse::nested_vsharp): key = record * nested_keys + inner key.
	uint32_t nested_vsharp  = UINT32_MAX;
	uint32_t nested_heap    = 0;
	uint32_t nested_keys    = 0;
	uint32_t binding      = 0;
	uint32_t region       = 0;
	// Position in BindlessTable::Heaps() (append-only), for KYTY_BINDLESS_SYNC_INDEX.
	uint32_t serial       = 0;
	// Keys the region holds. The guest appends textures to a heap as it streams them in and
	// its descriptor grows; each draw patches its own descriptor's entry count.
	uint32_t entries = 0;
	// The most keys a draw's heap descriptor has covered (entries adds headroom to it).
	uint32_t guest_entries = 0;
	// Engine method: the entries WatchHeap watched, and the watch generation then. A draw calls
	// WatchHeap only when either differs (the heap grew, or an unmap dropped its watch).
	uint32_t                            watched_entries  = 0;
	uint64_t                            watch_generation = 0;
	ShaderRecompiler::IR::ImageResource resource;
	// KYTY_BINDLESS_VIEW_KEY=2: the view fields of `resource` (BindlessViewKey) are part of the heap's
	// key; 0 without the switch.
	uint64_t                            view_key = 0;
	// Per key, 0 = not resolved; else the translation entry: the slot in this heap's binding, or
	// (BindlessSlotArrayFlag, KYTY_BINDLESS_3D_AS_ARRAY) in the 2D-array binding.
	// BindlessTable::SlotBinding/SlotIndex decode it.
	std::vector<uint32_t>               slots;
	std::vector<uint8_t>                settled;   // per key; resolved, or known placeholder
	std::vector<ImageId>                resolved;  // images to keep readable for draws
	std::vector<ImageId>                unchecked; // settled since the last bindless draw (E5)
	// Per settled key, the T# it was settled from and the image it resolved to (none for a
	// placeholder). The guest rewrites entries as it streams textures out and others in; a key
	// whose T# changed is settled again.
	std::vector<std::array<uint32_t, 8>> descriptors;
	std::vector<ImageId>                 images;
	uint32_t                             revalidate_cursor = 0; // next key to compare
	uint64_t                             processed_frame    = UINT64_MAX; // record heaps: last ProcessRecordHeap
	// KYTY_BINDLESS_SYNC_RETIRE: the last frame a draw prepared this heap, and whether a sync
	// skipped a write into its keys since (the next draw that uses it compares every key first).
	uint64_t                             sync_used_frame    = 0;
	bool                                 sync_stale         = false;
	uint64_t                             processed_snapshot = 0; // record heaps: snapshot serial taken
	// What RenderExecutor::CommitBindings last checked: g_bindless_state_generation then, and
	// how many resolved images. Unchanged since, only the images resolved after need a look.
	uint64_t checked_generation = 0;
	size_t   checked_count      = 0;
	// KYTY_RUNTIME_IMAGES: a runtime heap (BindlessTable::RuntimeHeap). Its keys are T#s shaders
	// loaded at run time and asked for, kept here, not entries of a guest table; nothing of it is
	// read from guest memory.
	bool                                 runtime          = false;
	// KYTY_BINDLESS_PREFILL: the heap's keys were settled at its first use (RenderExecutor::
	// PrefillBindlessHeap).
	bool                                 prefilled        = false;
	// KYTY_BINDLESS_REFILL: the scheduler tick of the last table check and the table bytes' digest then.
	uint64_t                             refill_tick      = UINT64_MAX;
	uint64_t                             refill_digest    = 0;
	// KYTY_BINDLESS_TOUCH: the last frame its resolved images were touched (TextureCache::TouchImage).
	uint64_t                             touched_frame    = UINT64_MAX;
	uint32_t                             runtime_next_key = 0;
	std::vector<std::array<uint32_t, 8>> runtime_dwords;
	// KYTY_BINDLESS_REFRESH: the last frame every resolved key was looked up again, and the keys
	// whose image the GPU writes (looked up again by every draw that uses the heap).
	uint64_t                             refresh_frame = UINT64_MAX;
	std::vector<uint32_t>                refresh_gpu_keys;
};

// Descriptor set 1 of every pipeline that samples bindless images: typed arrays of sampled
// images a shader indexes with a slot it looks up per pixel, the key -> slot translation the
// host fills per guest descriptor heap, and a bitset where shaders flag keys whose texture is
// not resident yet. Nothing is created until the first draw or dispatch that samples bindless
// images (EnsureCreated); programs without them never touch the table.
class BindlessTable {
public:
	enum Binding : uint32_t {
		Images2D      = 0,
		Images2DArray = 1,
		ImagesCube    = 2,
		Images3D      = 3,
		Translation   = 4,
		Feedback      = 5,
	};
	static_assert(Images2D == ShaderRecompiler::IR::BindlessImages2D &&
	              Images2DArray == ShaderRecompiler::IR::BindlessImages2DArray &&
	              ImagesCube == ShaderRecompiler::IR::BindlessImagesCube &&
	              Images3D == ShaderRecompiler::IR::BindlessImages3D &&
	              Translation == ShaderRecompiler::IR::BindlessTranslation &&
	              Feedback == ShaderRecompiler::IR::BindlessFeedback);
	// The runtime image bindings (BindlessBindings.h); the enum values above index nothing else.
	static constexpr uint32_t RuntimeTableBinding    = ShaderRecompiler::IR::BindlessRuntimeTable;
	static constexpr uint32_t RuntimeRequestsBinding = ShaderRecompiler::IR::BindlessRuntimeRequests;
	// KYTY_BINDLESS_SAMPLERS: the sampler array (BindlessBindings.h), at most MaxSamplers slots.
	static constexpr uint32_t SamplersBinding = ShaderRecompiler::IR::BindlessSamplers;
	static constexpr uint32_t MaxSamplers     = 4096;
	// Keys of each binding's runtime heap (the T#s runtime images requested).
	static constexpr uint32_t RuntimeKeys = 8192;
	static constexpr uint32_t ImageArrays        = 4;
	static constexpr uint32_t PlaceholderColors  = 3;
	static_assert(ShaderRecompiler::IR::BindlessPendingSlot < PlaceholderColors);
	static constexpr uint32_t Placeholders       = ImageArrays * PlaceholderColors;
	static constexpr uint32_t MaxImagesPerArray  = 65536;
	static constexpr uint32_t TranslationEntries = 1u << 20u;
	// KYTY_BINDLESS_TRANSLATION_MI=<n> (default 4, 1..16): translation entries in Mi (the
	// translation and feedback buffers hold one dword each). Heaps take regions of it and never
	// give them back; a call table read through record pointers takes 256 Ki keys. With 1 Mi,
	// Miles Morales' ray-tracing material tables (two call tables, several images each) filled it
	// and every later heap sampled the placeholder ("translation buffer is full").
	[[nodiscard]] static uint32_t TranslationCapacity();

	using Heap = BindlessHeap;

	BindlessTable(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager);
	~BindlessTable();
	KYTY_CLASS_NO_COPY(BindlessTable);

	// The device supports bindless images (the table can be created).
	[[nodiscard]] bool Supported() const noexcept;
	// Creates the pool, the set, the buffers and the placeholders on first use (GPU thread, with
	// a command buffer being recorded: the placeholders are cleared in it). False when the device
	// has no support or creation failed; the caller then treats every key as the placeholder.
	[[nodiscard]] bool EnsureCreated();
	[[nodiscard]] bool Enabled() const noexcept { return m_set != nullptr; }
	[[nodiscard]] vk::DescriptorSet Set() const noexcept { return m_set; }
	[[nodiscard]] uint32_t          ImagesPerArray() const noexcept { return m_images_per_array; }

	// The heap for (base, table offset, view binding), created with every entry pending and
	// room to grow, or moved to a larger region when a descriptor covers more keys; null when
	// the translation buffer is full.
	[[nodiscard]] Heap* FindOrCreateHeap(uint64_t base, uint32_t table_offset, uint32_t binding,
	                                     uint32_t entries,
	                                     const ShaderRecompiler::IR::ImageResource& resource,
	                                     uint32_t stride = 32, uint32_t record_pointer = UINT32_MAX,
	                                     uint32_t nested_vsharp = UINT32_MAX, uint32_t nested_heap = 0,
	                                     uint32_t nested_keys = 0);
	[[nodiscard]] std::deque<Heap>& Heaps() noexcept { return m_heaps; }
	// KYTY_BINDLESS_SYNC_INDEX: the serials of heaps whose region was (re)allocated since the last
	// call (their key span may have grown), moved out.
	void TakeResizedHeaps(std::vector<uint32_t>& out) {
		out.clear();
		out.swap(m_resized_heaps);
	}
	// 0 when the array is full. A released slot is reused once the GPU has finished the work
	// recorded before its release.
	[[nodiscard]] uint32_t AllocateSlot(uint32_t binding);
	// The binding and array slot a Heap::slots entry names.
	[[nodiscard]] static uint32_t SlotBinding(const Heap& heap, uint32_t entry) noexcept {
		return (entry & ShaderRecompiler::IR::BindlessSlotFlagMask) ==
		               ShaderRecompiler::IR::BindlessSlotArrayFlag
		           ? static_cast<uint32_t>(Images2DArray)
		           : heap.binding;
	}
	[[nodiscard]] static uint32_t SlotIndex(uint32_t entry) noexcept {
		return (entry & ShaderRecompiler::IR::BindlessSlotFlagMask) ==
		               ShaderRecompiler::IR::BindlessSlotArrayFlag
		           ? entry & ~ShaderRecompiler::IR::BindlessSlotFlagMask
		           : entry;
	}
	// The image view slot `slot` of `binding` holds (null: a placeholder or nothing).
	[[nodiscard]] vk::ImageView SlotView(uint32_t binding, uint32_t slot) const noexcept {
		return binding < ImageArrays && slot < m_slot_views[binding].size() ? m_slot_views[binding][slot]
		                                                                     : vk::ImageView {};
	}
	// KYTY_BINDLESS_REFRESH: slot `slot` of `binding` now shows `view` of `new_id`; every key that
	// samples the slot through `old_id` moves its image reference to `new_id` (the shared slot too).
	// True when `old_id` has no reference left.
	bool RemapSlotImage(uint32_t binding, uint32_t slot, ImageId old_id, ImageId new_id, vk::ImageView view,
	                    vk::ImageLayout layout);
	// Queue a slot for reuse once the GPU is past the current tick (at most once; IDXTRI).
	void FreeSlot(uint32_t binding, uint32_t slot);
	void WriteSlot(uint32_t binding, uint32_t slot, vk::ImageView view, vk::ImageLayout layout);
	// Resolve a key to a slot (0 = placeholder), or back to pending.
	void SetTranslation(const Heap& heap, uint32_t key, uint32_t slot);
	// KYTY_BINDLESS_WATCH_KEY: the watched key's translation and slot view in every heap.
	void DebugWatchReport(uint64_t frame);
	[[nodiscard]] static uint32_t WatchKey();
	// The host reads the feedback flags from a copy in host-cached memory. The feedback buffer
	// itself is device memory, and reading every key of three heaps from it uncached took a
	// fifth of the GPU thread. Records that copy at the current point of the command stream.
	void RecordFeedbackSnapshot();
	[[nodiscard]] bool SnapshotRecorded() const noexcept { return m_snapshot_tick != 0; }
	// Whether the GPU has executed the recorded copy; never waits.
	[[nodiscard]] bool SnapshotReady();
	// The recorded snapshot is the one read from now on; its chunk summary picks the chunks the next
	// snapshot copies.
	void               ConsumeSnapshot();
	// Snapshot traffic since the last call (diagnostics): bytes copied, flagged chunks.
	[[nodiscard]] std::pair<uint64_t, uint64_t> TakeSnapshotTraffic() noexcept {
		return {std::exchange(m_copied_bytes, 0), std::exchange(m_flagged_chunks, 0)};
	}
	[[nodiscard]] std::pair<uint64_t, uint64_t> TakeScanCounts() noexcept {
		return {std::exchange(m_scanned_words, 0), std::exchange(m_nonzero_words, 0)};
	}
	// Pending keys flagged in the snapshot; their flags are cleared.
	void TakeRequests(const Heap& heap, std::vector<uint32_t>& keys);
	// Diagnostics: feedback word 0 (the last out-of-range key, top bit set) in the snapshot,
	// cleared.
	[[nodiscard]] uint32_t TakeWordZero();
	[[nodiscard]] uint32_t UsedEntries() const noexcept { return m_next_region; }
	// KYTY_BINDLESS_EVICT usage probe (adapted from Senaxx 8585096e6). Every draw that samples a
	// guest heap key stores the key's feedback word (0 resident, 1 pending), so marking every
	// resident key's word with ProbeMark (ArmUsageProbe, through the mapped buffer, as the request
	// flags are cleared) and reading a snapshot recorded some frames later shows which keys no
	// draw sampled in between. CollectUnusedImages lists the images all of whose keys still carry
	// the mark there. Runtime heaps are not marked: their images always count as sampled.
	static constexpr uint32_t ProbeMark = 2;
	void                      ArmUsageProbe();
	void                      CollectUnusedImages(std::vector<ImageId>& unused);
	// The host-mapped translation buffer (diagnostics).
	[[nodiscard]] const uint32_t* TranslationWords() const noexcept {
		return m_translation != nullptr ? reinterpret_cast<const uint32_t*>(m_translation->Mapped().data()) : nullptr;
	}
	void                   AddImageReference(ImageId id, Heap& heap, uint32_t key);
	// The key no longer samples what it was settled to: it is pending again, and its image loses
	// the key's reference. Its slot keeps the old view, which frames in flight may still sample;
	// a new resolution takes a new slot. True when the image has no reference left, so it need
	// not stay pinned in the texture cache. free_slot = false: the caller keeps the slot (to fall
	// back to it, or free it itself; IDXTRI).
	[[nodiscard]] bool ReleaseKey(Heap& heap, uint32_t key, bool free_slot = true);

	// KYTY_BINDLESS_PREFILL: one slot per (heap binding, T# words) for prefilled keys, shared by
	// every key of any heap that holds the same T#. A shared slot counts its extra keys; FreeSlot
	// frees it with the last one (and forgets its words).
	struct SharedSlot {
		uint32_t entry = 0; // a Heap::slots entry (slot, or BindlessSlotArrayFlag | slot)
		ImageId  image;
		uint32_t slot_binding = 0;
		uint32_t slot_index   = 0;
	};
	// Null when unknown, or when the slot no longer shows an image (a placeholder was written to
	// it, e.g. ReleaseViews when the image's views were dropped): the words are forgotten then.
	[[nodiscard]] const SharedSlot* FindSharedSlot(uint32_t binding, const std::array<uint32_t, 8>& words,
	                                               uint64_t view_key = 0);
	void NoteSharedSlot(const Heap& heap, const std::array<uint32_t, 8>& words, uint32_t entry, ImageId image);
	void AddSharedSlotRef(uint32_t binding, uint32_t slot);
	// Shared slots found holding a placeholder and forgotten (diagnostics).
	[[nodiscard]] uint64_t SharedSlotsForgotten() const noexcept { return m_shared_forgotten; }

	// Engine method, E1. The guest pages holding a heap's entries are write-watched; a CPU write
	// to one (a page fault on a game thread, or an explicit invalidation such as a file read into
	// the heap) stops watching that page and records it as written. TakeWrittenRanges, on the GPU
	// thread at the start of a guest submission, watches the written pages again before it
	// returns them, so a write after that is caught by the next submission; the caller then
	// reads the entries and compares them with the mirror. Heaps of different view bindings over
	// the same guest table share one watch.
	void WatchHeap(Heap& heap);
	[[nodiscard]] bool NeedsWatch(const Heap& heap) const noexcept {
		return heap.watched_entries != heap.guest_entries ||
		       heap.watch_generation != m_watch_generation.load(std::memory_order_acquire);
	}
	void OnCpuWrite(uint64_t vaddr, uint64_t size) noexcept; // any thread
	void UnwatchRange(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasWrittenRanges() const noexcept {
		return m_heap_writes.load(std::memory_order_acquire);
	}
	void TakeWrittenRanges(std::vector<std::pair<uint64_t, uint64_t>>& ranges);

	// The texture cache dropped a resolved image: point its slots back at the placeholder and
	// make its keys pending again, so a draw that still needs it asks for it anew.
	// replaced (TextureCache::ExpandImage): another image holds the contents now; with
	// KYTY_BINDLESS_REPLACED_RESOLVE (default on) the keys are kept for TakeReplacedKeys, which
	// resolves them before the next draw or dispatch that uses bindless images, instead of their
	// sampling the placeholder until the GPU feedback asks for them a few frames later.
	void OnImageUnregistered(ImageId id, bool replaced = false);
	// The keys OnImageUnregistered made pending for a replaced image (taken once).
	void TakeReplacedKeys(std::vector<std::pair<Heap*, uint32_t>>& keys) {
		keys.swap(m_replaced_keys);
		m_replaced_keys.clear();
	}
	[[nodiscard]] bool HasReplacedKeys() const noexcept { return !m_replaced_keys.empty(); }
	// KYTY_BINDLESS_UNMAP_RERESOLVE: the keys OnImageUnregistered sent back to pending, of heap,
	// to settle again without waiting for a shader to flag them; taken (cleared) by the call.
	void TakeReresolve(const Heap* heap, std::vector<uint32_t>& keys);
	// KYTY_BINDLESS_UNMAP_RERESOLVE (live, default off; Miles preset): the guest unmaps part of a streamed texture,
	// the texture cache frees the image, and its keys went back to pending until a shader flagged
	// them and the flag came round (several frames of placeholder, on alternate call tables).
	[[nodiscard]] static bool UnmapReresolve();
	// Keys made pending because the texture cache dropped their image (statistics; reset on read).
	[[nodiscard]] uint64_t TakeUnregisteredKeys() noexcept {
		return std::exchange(m_unregistered_keys, 0);
	}
	// The texture cache is dropping an image with these views (IDXTRI's stale view guard, Senaxx's
	// ReleaseViews): a slot that still holds one of them is repointed to the placeholder and the
	// keys that use it are pending again. The image itself is destroyed only after the GPU has
	// finished the current tick, and the repointed descriptors are what everything recorded from
	// now on (and the current recording, update-after-bind) samples. True when a slot held one.
	bool ReleaseViews(std::span<const CachedImageView> views);

	// KYTY_RUNTIME_IMAGES (adapted from usa-ratt/KytyPS5-GT7 2fa6c6514): whether set 1 has the
	// runtime image table and request buffer (the switch with bindless images).
	[[nodiscard]] static bool RuntimeImagesEnabled();
	// The runtime heap of a set-1 image binding, created on first use with `resource` (the first
	// runtime image that reads it); null when the switch is off or the translation buffer is full.
	[[nodiscard]] Heap* RuntimeHeap(uint32_t binding, const ShaderRecompiler::IR::ImageResource* resource);
	// The T#s (and bindings) shaders requested before the last snapshot; the request buffer is
	// cleared for new requests.
	void TakeRuntimeRequests(std::vector<std::pair<std::array<uint32_t, 8>, uint32_t>>& requests);
	// The key of a T# in its binding's runtime heap, added (pending) when new, and published in
	// the runtime image table; UINT32_MAX when the heap is full.
	[[nodiscard]] uint32_t RuntimeKey(Heap& heap, const std::array<uint32_t, 8>& dwords);

	// KYTY_BINDLESS_SAMPLERS (a minimal port of Senaxx's mirrored sampler heaps, 59cb8a83e):
	// whether set 1 has the sampler array (the switch), and its slots on this device.
	[[nodiscard]] static bool     SamplersEnabled();
	[[nodiscard]] static uint32_t SamplerSlots(const GraphicContext& graphics);
	// A guest sampler heap as one kind of use samples it: a region of the sampler array that
	// mirrors its S# records, key for key. Uses differ in whether the depth-compare function is
	// kept, point filtering forced, border colours integer or cube edges clamped (the flags), so
	// each kind gets its own region.
	struct SamplerHeap {
		uint64_t                             base          = 0;
		uint32_t                             table_offset  = 0;
		uint32_t                             stride        = 16;
		uint32_t                             flags         = 0;
		uint32_t                             region        = 0; // first slot; 0 = none yet
		uint32_t                             capacity      = 0;
		std::vector<std::array<uint32_t, 4>> records;           // mirrored S# records
		uint64_t                             checked_frame = UINT64_MAX;
	};
	[[nodiscard]] SamplerHeap& FindOrCreateSamplerHeap(uint64_t base, uint32_t table_offset,
	                                                   uint32_t stride, uint32_t flags);
	// Mirrors the records into the heap's region. New records are appended in place (their slots
	// were never used); a changed record, or more records than the region holds, moves the heap to
	// a new region, because slots the GPU may be reading are never rewritten. False when the
	// sampler array is full; the heap then keeps its previous records.
	bool MirrorSamplerHeap(SamplerHeap& heap, std::span<const std::array<uint32_t, 4>> records,
	                       const std::function<vk::Sampler(const std::array<uint32_t, 4>&)>& make);
	// Slot 0 of the sampler array, used for keys outside their heap; written once.
	void WriteDefaultSampler(vk::Sampler sampler);

private:
	void PublishRuntimeEntry(const Heap& heap, uint32_t key, const std::array<uint32_t, 8>& dwords);
	[[nodiscard]] bool AllocateRegion(Heap& heap, uint32_t entries);
	// Slots 0-2 of every image array: a 1x1 texture of that view type, sampled for keys outside a
	// heap and while a texture is pending. Created and cleared once, in the current recording.
	[[nodiscard]] bool CreatePlaceholders();
	// The placeholder view of `binding`'s view type and colour index (slot).
	[[nodiscard]] vk::ImageView PlaceholderView(uint32_t binding, uint32_t color) const noexcept {
		return m_placeholder_views[color * ImageArrays + binding];
	}
	[[nodiscard]] bool IsPlaceholder(vk::ImageView view) const noexcept;
	// The keys of `binding` that use `slot` go back to pending and lose their image references;
	// returns how many there were.
	uint32_t ReleaseSlotKeys(uint32_t binding, uint32_t slot);
	void     Destroy();

	[[nodiscard]] static uint64_t ImageKey(ImageId id) {
		return static_cast<uint64_t>(id.index) | (static_cast<uint64_t>(id.generation) << 32u);
	}

	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;
	PageManager&      m_page_manager;
	bool              m_creation_failed = false;
	// Write-watched guest ranges of the heaps, page by page (TRACKER_PAGE_SIZE).
	struct Watch {
		uint64_t             begin = 0;
		uint64_t             end   = 0;
		std::vector<uint8_t> watched;
		std::vector<uint8_t> written;
	};
	std::mutex         m_watch_mutex;
	std::vector<Watch> m_watches;
	std::atomic_bool   m_heap_writes {false};
	// Bounds of every watch, so a write far from the heaps returns without the lock.
	std::atomic<uint64_t> m_watch_low {UINT64_MAX};
	std::atomic<uint64_t> m_watch_high {0};
	std::atomic<uint64_t> m_watch_generation {1};
	uint64_t              m_unregistered_keys = 0;
	std::vector<std::pair<Heap*, uint32_t>> m_replaced_keys;
	std::mutex                                 m_reresolve_mutex;
	std::vector<std::pair<const Heap*, uint32_t>> m_reresolve;
	std::deque<Heap>      m_heaps;
	std::vector<uint32_t> m_resized_heaps; // TakeResizedHeaps()
	// FindOrCreateHeap's heaps by guest base (KYTY_BINDLESS_SMALL_HEAPS: Ghost of Yotei creates
	// tens of thousands; a linear search per draw was the old lookup).
	std::unordered_map<uint64_t, std::vector<Heap*>> m_heaps_by_base;
	// Released slots per array, with the tick after which no command buffer reads them, and per
	// slot whether it waits there (FreeSlot keeps each slot on the list once).
	std::array<std::deque<std::pair<uint32_t, uint64_t>>, ImageArrays> m_free_slots;
	std::array<std::vector<uint8_t>, ImageArrays>                      m_slot_free;
	std::unordered_map<uint64_t, std::vector<std::pair<Heap*, uint32_t>>> m_image_refs;
	// KYTY_BINDLESS_PREFILL: shared slots by (binding, T# words), the words of each shared slot
	// ((binding << 32) | slot), and each shared slot's keys beyond the first.
	struct SharedSlotKey {
		uint32_t                binding = 0;
		std::array<uint32_t, 8> words {};
		uint64_t                view = 0; // KYTY_BINDLESS_VIEW_KEY=2: Heap::view_key
		bool operator==(const SharedSlotKey&) const = default;
	};
	struct SharedSlotKeyHash {
		size_t operator()(const SharedSlotKey& key) const noexcept {
			uint64_t hash = (0xcbf29ce484222325ull ^ key.binding ^ key.view) * 0x100000001b3ull;
			for (const auto word: key.words) hash = (hash ^ word) * 0x100000001b3ull;
			return static_cast<size_t>(hash);
		}
	};
	std::unordered_map<SharedSlotKey, SharedSlot, SharedSlotKeyHash> m_shared_slots;
	std::unordered_map<uint64_t, SharedSlotKey>                      m_shared_slot_words;
	std::unordered_map<uint64_t, uint32_t>                           m_shared_slot_refs;
	uint64_t                                                         m_shared_forgotten = 0;
	// The slot's words are forgotten (no new key shares it); drop_refs: its keys were released
	// together, so the next FreeSlot frees it.
	void ForgetSharedSlot(uint32_t binding, uint32_t slot, bool drop_refs);
	uint32_t m_next_region = 1; // translation[0] is the out-of-range entry
	std::array<uint32_t, ImageArrays> m_next_slot {PlaceholderColors, PlaceholderColors,
	                                               PlaceholderColors, PlaceholderColors};
	std::array<VulkanImage, Placeholders>   m_placeholders {};
	std::array<vk::ImageView, Placeholders> m_placeholder_views {};
	vk::DescriptorPool                      m_pool             = nullptr;
	vk::DescriptorSet                       m_set              = nullptr;
	uint32_t                                m_images_per_array = 0;
	std::unique_ptr<Buffer>                 m_translation;
	std::unique_ptr<Buffer>                 m_feedback;
	std::array<std::unique_ptr<Buffer>, 2>  m_feedback_snapshots;
	uint32_t                                m_snapshot_record = 0; // index the next snapshot is copied to
	uint32_t                                m_snapshot_read   = 1; // index the host reads
	std::vector<uint32_t>                   m_copy_chunks;      // flagged chunks for the next copy
	std::vector<uint32_t>                   m_inflight_chunks;  // chunks in the recorded snapshot
	struct PendingChunk {
		uint64_t                                                          serial = 0;
		std::array<uint32_t, 1u << ShaderRecompiler::IR::BindlessFeedbackChunkShift> words {};
	};
	std::unordered_map<uint32_t, PendingChunk> m_pending_chunks; // copied chunks whose flags wait for their heap
	uint64_t                                m_snapshot_serial = 0;
	static constexpr uint32_t MaxCopyChunks = 2048;
	static constexpr uint32_t SnapshotWords =
	    ShaderRecompiler::IR::BindlessFeedbackFirstRegion + (MaxCopyChunks << ShaderRecompiler::IR::BindlessFeedbackChunkShift);
	std::vector<vk::BufferCopy>             m_copy_regions;
	uint64_t                                m_copied_bytes   = 0;
	uint64_t                                m_flagged_chunks = 0;
	uint64_t                                m_scanned_words  = 0;
	uint64_t                                m_nonzero_words  = 0;
	std::unique_ptr<Buffer>                 m_runtime_table;
	std::unique_ptr<Buffer>                 m_runtime_requests;
	std::unique_ptr<Buffer>                 m_runtime_requests_snapshot;
	std::array<Heap*, ImageArrays>          m_runtime_heaps {};
	uint32_t                                m_sampler_slots           = 0;
	uint32_t                                m_next_sampler_slot       = 1; // slot 0: the stand-in
	bool                                    m_default_sampler_written = false;
	std::deque<SamplerHeap>                 m_sampler_heaps;
	std::map<std::pair<uint32_t, std::array<uint32_t, 8>>, uint32_t> m_runtime_keys;
	uint64_t                                m_snapshot_tick = 0; // 0 = none recorded
	// The view each image slot holds, and the slots each view is in (placeholders excluded).
	std::array<std::vector<vk::ImageView>, ImageArrays>                          m_slot_views;
	std::unordered_map<VkImageView, std::vector<std::pair<uint32_t, uint32_t>>> m_view_slots;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
