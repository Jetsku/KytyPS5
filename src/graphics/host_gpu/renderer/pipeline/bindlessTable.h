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

// A guest descriptor heap as one view type samples it: a region of the translation (and
// feedback) buffer, one entry per key, and the textures resolved so far.
struct BindlessHeap {
	uint64_t base         = 0;
	uint32_t table_offset = 0;
	uint32_t binding      = 0;
	uint32_t region       = 0;
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
	std::vector<uint32_t>               slots;     // per key; 0 = not resolved
	std::vector<uint8_t>                settled;   // per key; resolved, or known placeholder
	std::vector<ImageId>                resolved;  // images to keep readable for draws
	std::vector<ImageId>                unchecked; // settled since the last bindless draw (E5)
	// Per settled key, the T# it was settled from and the image it resolved to (none for a
	// placeholder). The guest rewrites entries as it streams textures out and others in; a key
	// whose T# changed is settled again.
	std::vector<std::array<uint32_t, 8>> descriptors;
	std::vector<ImageId>                 images;
	uint32_t                             revalidate_cursor = 0; // next key to compare
	// What RenderExecutor::CommitBindings last checked: g_bindless_state_generation then, and
	// how many resolved images. Unchanged since, only the images resolved after need a look.
	uint64_t checked_generation = 0;
	size_t   checked_count      = 0;
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
	static constexpr uint32_t ImageArrays        = 4;
	static constexpr uint32_t PlaceholderColors  = 3;
	static_assert(ShaderRecompiler::IR::BindlessPendingSlot < PlaceholderColors);
	static constexpr uint32_t Placeholders       = ImageArrays * PlaceholderColors;
	static constexpr uint32_t MaxImagesPerArray  = 65536;
	static constexpr uint32_t TranslationEntries = 1u << 20u;

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
	                                     const ShaderRecompiler::IR::ImageResource& resource);
	[[nodiscard]] std::deque<Heap>& Heaps() noexcept { return m_heaps; }
	// 0 when the array is full. A released slot is reused once the GPU has finished the work
	// recorded before its release.
	[[nodiscard]] uint32_t AllocateSlot(uint32_t binding);
	// Queue a slot for reuse once the GPU is past the current tick (at most once; IDXTRI).
	void FreeSlot(uint32_t binding, uint32_t slot);
	void WriteSlot(uint32_t binding, uint32_t slot, vk::ImageView view, vk::ImageLayout layout);
	// Resolve a key to a slot (0 = placeholder), or back to pending.
	void SetTranslation(const Heap& heap, uint32_t key, uint32_t slot);
	// The host reads the feedback flags from a copy in host-cached memory. The feedback buffer
	// itself is device memory, and reading every key of three heaps from it uncached took a
	// fifth of the GPU thread. Records that copy at the current point of the command stream.
	void RecordFeedbackSnapshot();
	[[nodiscard]] bool SnapshotRecorded() const noexcept { return m_snapshot_tick != 0; }
	// Whether the GPU has executed the recorded copy; never waits.
	[[nodiscard]] bool SnapshotReady();
	void               ConsumeSnapshot() noexcept { m_snapshot_tick = 0; }
	// Pending keys flagged in the snapshot; their flags are cleared.
	void TakeRequests(const Heap& heap, std::vector<uint32_t>& keys);
	// Diagnostics: feedback word 0 (the last out-of-range key, top bit set) in the snapshot,
	// cleared.
	[[nodiscard]] uint32_t TakeWordZero();
	void                   AddImageReference(ImageId id, Heap& heap, uint32_t key);
	// The key no longer samples what it was settled to: it is pending again, and its image loses
	// the key's reference. Its slot keeps the old view, which frames in flight may still sample;
	// a new resolution takes a new slot. True when the image has no reference left, so it need
	// not stay pinned in the texture cache. free_slot = false: the caller keeps the slot (to fall
	// back to it, or free it itself; IDXTRI).
	[[nodiscard]] bool ReleaseKey(Heap& heap, uint32_t key, bool free_slot = true);

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
	void OnImageUnregistered(ImageId id);
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

private:
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
	std::deque<Heap>      m_heaps;
	// Released slots per array, with the tick after which no command buffer reads them, and per
	// slot whether it waits there (FreeSlot keeps each slot on the list once).
	std::array<std::deque<std::pair<uint32_t, uint64_t>>, ImageArrays> m_free_slots;
	std::array<std::vector<uint8_t>, ImageArrays>                      m_slot_free;
	std::unordered_map<uint64_t, std::vector<std::pair<Heap*, uint32_t>>> m_image_refs;
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
	std::unique_ptr<Buffer>                 m_feedback_snapshot;
	uint64_t                                m_snapshot_tick = 0; // 0 = none recorded
	// The view each image slot holds, and the slots each view is in (placeholders excluded).
	std::array<std::vector<vk::ImageView>, ImageArrays>                          m_slot_views;
	std::unordered_map<VkImageView, std::vector<std::pair<uint32_t, uint32_t>>> m_view_slots;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
