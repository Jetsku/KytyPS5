#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_

#include "common/alignment.h"
#include "common/assert.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <algorithm>
#include <compare>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
struct ImageTestAccess;

using ImageId = Common::SlotId;

struct CachedImageView {
	ImageViewInfo info;
	vk::ImageView view = nullptr;
};

struct ImageUsage {
	bool texture       = false;
	bool storage       = false;
	bool render_target = false;
	bool depth_target  = false;
	bool video_out     = false;
};

struct ImageBinding {
	vk::ImageLayout  attachment_layout = vk::ImageLayout::eUndefined;
	vk::AccessFlags2 attachment_access;
	bool             is_bound      = false;
	bool             is_target     = false;
	bool             needs_rebind  = false;
	bool             force_general = false;
	bool             shader_write  = false;
};

class Image final {
public:
	// sparse_first_level > 0 (KYTY_TEXTURE_SPARSE_RESIDENCY, TextureCache::InsertImage): try a sparse
	// residency image with memory behind levels >= it only (GraphicContext::CreateSparseImage);
	// a format or usage that cannot be sparse gets an ordinary image.
	Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& info,
	      uint32_t sparse_first_level = 0);
	~Image();
	KYTY_CLASS_NO_COPY(Image);

	[[nodiscard]] vk::ImageView FindView(const ImageViewInfo& view_info);
	using Barriers = std::vector<vk::ImageMemoryBarrier2>;
	// KYTY_GUEST_STORAGE_REPEAT=1 (default off; ported from chenxiao07/KytyPS5 397112a04): a shader
	// write that guest draws and dispatches repeat in the same layout and access gets no barrier
	// between them, as nothing but the guest's own barriers (EmitGlobalBarrier) orders them on the
	// console. A repeated write after any other access (an upload, a copy, a helper pass) keeps its
	// barrier. The scope marks the transitions of a guest draw's or dispatch's bindings.
	struct GuestTransitScope {
		GuestTransitScope();
		~GuestTransitScope();
		KYTY_CLASS_NO_COPY(GuestTransitScope);
		bool m_previous;
	};
	[[nodiscard]] Barriers GetBarriers(vk::ImageLayout                      destination_layout,
	                                   vk::AccessFlags2                     destination_access,
	                                   vk::PipelineStageFlags2              destination_stage,
	                                   std::optional<ImageSubresourceRange> range);
	// deferrable: see CommandBuffer::BatchImageBarriers (render.h); only for callers that record
	// no memory-accessing command through command_buffer before the next barrier flush point.
	// KYTY_IMAGE_TRANSIT_SKIP (default on; =0 off): returns at once when TransitIsNoOp.
	// KYTY_IMAGE_TRANSIT_SKIP_VERIFY=1|exit runs GetBarriers after such a decision and records
	// what it returns: a barrier there is a mismatch (counted, logged; exit stops). FrameEvents
	// ImageTransitSkips, ImageTransitVerify{Checks,Mismatches}.
	void Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
	             std::optional<ImageSubresourceRange> range, vk::CommandBuffer command_buffer,
	             bool deferrable = false);
	// Transit calls that produced barriers (all images; KYTY_DRAW_RUN=verify reads the difference
	// around a draw's attachment acquisition and binding commit).
	[[nodiscard]] static uint64_t RecordedTransitions() noexcept;
	// GetBarriers would return no barrier and change no state: the image has one state (no
	// per-subresource states), `range` covers every level and layer (a volume's range counts as
	// one layer, as in GetBarriers), and that state already has the layout and the access, which
	// includes no write (a repeated write needs a barrier). GetBarriers' early return.
	[[nodiscard]] bool TransitIsNoOp(vk::ImageLayout                             destination_layout,
	                                 vk::AccessFlags2                            destination_access,
	                                 const std::optional<ImageSubresourceRange>& range) const noexcept;
	void Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	            uint64_t size);
	void Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	              uint64_t size);
	void CopyImage(Image& source);
	// vkCmdCopyImage between a depth aspect and a compatible color format (VK_KHR_maintenance8).
	void CopyDepthColorImage(Image& source);
	void Resolve(Image& source, const ImageSubresourceRange& source_range,
	             const ImageSubresourceRange& destination_range);
	void CopyImageWithBuffer(Image& source, Buffer& buffer);
	void CopyMip(Image& source, uint32_t mip, uint32_t layer);

	// Native contents identity. Every recorded write to this image gives it a fresh serial
	// (Image copy/upload/resolve methods here, TextureCache::MarkImageGpuModified for draws,
	// dispatches, clears and helper passes). A bit-exact copy from another image may adopt
	// the source's serial afterwards: equal nonzero serials then prove equal native bits.
	// NoteContentWrite: a write that is recorded (also counted in DefiniteWrites).
	// NotePossibleWrite: a binding that may write (render/depth target, storage image); the
	// caller can restore the previous serial once it knows the binding wrote nothing.
	[[nodiscard]] uint64_t ContentSerial() const noexcept { return m_content_serial; }
	[[nodiscard]] uint64_t DefiniteWrites() const noexcept { return m_definite_writes; }
	void                   NoteContentWrite() noexcept;
	void                   NotePossibleWrite() noexcept;
	void AdoptContentSerial(uint64_t serial) noexcept { m_content_serial = serial; }
	// A fresh serial from the same sequence, for ordering other writers against image writes
	// (BufferCache::InvalidateContentRevisions stamps unbounded buffer writers with it).
	[[nodiscard]] static uint64_t NextContentSerial() noexcept;

	void InvalidateCpuWrite(uint64_t vaddr, uint64_t size) {
		if (ImageRangeOverlaps(live.address, live.size, vaddr, size)) {
			m_cpu_dirty        = true;
			m_maybe_cpu_dirty  = false;
			m_maybe_hash_valid = false;
			// Whole-image invalidation carries no chunk information.
			m_partial_valid = false;
			NoteDirtySpan(vaddr, size);
		} else if (ImagePageRangesOverlap(live.address, live.size, vaddr, size)) {
			m_maybe_cpu_dirty = true;
			NoteDirtySpan(vaddr, size);
		}
	}

	// Resident mip levels (TextureCache, KYTY_TEXTURE_RESIDENT_MIPS). PS5 mip chains are stored
	// smallest level first, so levels [resident_first, levels) occupy the prefix `live` of the
	// guest range. The cache registers, watches, overlap-tests and uploads only `live`; levels
	// below resident_first hold undefined native contents that no view can sample (every
	// sampled view clamps its minimum LOD at or above resident_first, see TextureCache::
	// RequestedFirstLevel). Any other use makes the image fully resident first.
	[[nodiscard]] bool FullyResident() const noexcept { return resident_first == 0; }
	// Every resident level must be refreshed from guest memory (newly resident levels).
	void MarkResidencyDirty() noexcept {
		m_cpu_dirty        = true;
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		m_partial_valid    = false;
	}

	// Chunk-granular CPU write tracking (TextureCache, KYTY_TEXTURE_PARTIAL_UPLOAD). Only for
	// page-aligned images, so every write to one of their pages overlaps their bytes. While
	// tracked, the image watches its whole page range except the chunks marked untracked; a
	// chunk is untracked only by a CPU write, which also marks it dirty. Dirty chunks are the
	// guest bytes that may differ from the native contents; they are cleared by a refresh.
	struct ChunkState {
		uint64_t              base        = 0; // AlignDown(data.address, chunk size)
		uint32_t              count       = 0; // 0: whole-image tracking
		uint32_t              shift       = 0;
		uint32_t              dirty_count = 0;
		uint32_t              untracked_count = 0;
		// Consecutive refreshes that found (nearly) every chunk dirty. Such an image is
		// rewritten whole by the CPU: its next write releases every chunk at once, so it takes
		// one fault per refresh instead of one per chunk.
		uint32_t              full_streak = 0;
		// Refresh cycles since the last chunk-by-chunk cycle; every 8th cycle probes again.
		uint32_t              whole_cycles   = 0;
		bool                  whole_released = false;
		std::vector<uint64_t> dirty;
		std::vector<uint64_t> untracked;
		// KYTY_TEXTURE_PARTIAL_VERIFY=1 only: guest-byte hash of each chunk at the last upload.
		std::vector<uint64_t> hashes;
	};
	[[nodiscard]] bool ChunkTracked() const noexcept { return chunks.count != 0; }
	void               EnableChunkTracking(uint32_t shift) {
		chunks.shift = shift;
		chunks.base  = live.address & ~((uint64_t {1} << shift) - 1);
		const auto end   = live.End();
		const auto count = ((end - chunks.base) + (uint64_t {1} << shift) - 1) >> shift;
		chunks.count     = static_cast<uint32_t>(count);
		chunks.dirty.assign((count + 63) / 64, 0);
		chunks.untracked.assign((count + 63) / 64, 0);
		chunks.dirty_count     = 0;
		chunks.untracked_count = 0;
		chunks.full_streak     = 0;
		chunks.whole_cycles    = 0;
		chunks.whole_released  = false;
		chunks.hashes.clear();
	}
	[[nodiscard]] static bool ChunkBit(const std::vector<uint64_t>& bits, uint32_t index) noexcept {
		return (bits[index >> 6] >> (index & 63u)) & 1u;
	}
	static void SetChunkBit(std::vector<uint64_t>& bits, uint32_t index) noexcept {
		bits[index >> 6] |= uint64_t {1} << (index & 63u);
	}
	static void ClearChunkBit(std::vector<uint64_t>& bits, uint32_t index) noexcept {
		bits[index >> 6] &= ~(uint64_t {1} << (index & 63u));
	}
	// Records a CPU write to chunk `index` (the caller removes the watch). Returns true when
	// the chunk was clean before.
	bool MarkChunkDirty(uint32_t index) noexcept {
		if (ChunkBit(chunks.dirty, index)) {
			return false;
		}
		SetChunkBit(chunks.dirty, index);
		chunks.dirty_count++;
		return true;
	}
	// A CPU write to guest bytes of this chunk-tracked image: the image needs a refresh.
	void NoteChunkWrite(uint64_t vaddr, uint64_t size) noexcept {
		m_cpu_dirty        = true;
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		NoteDirtySpan(vaddr, size);
	}
	void ClearChunkDirty() noexcept {
		if (chunks.dirty_count != 0) {
			std::fill(chunks.dirty.begin(), chunks.dirty.end(), 0);
			chunks.dirty_count = 0;
		}
	}
	// Whether any chunk overlapping guest range [vaddr, vaddr + size) is dirty.
	[[nodiscard]] bool ChunkRangeDirty(uint64_t vaddr, uint64_t size) const noexcept {
		if (size == 0 || vaddr < chunks.base) {
			return true;
		}
		const auto first = (vaddr - chunks.base) >> chunks.shift;
		const auto last  = (vaddr + size - 1 - chunks.base) >> chunks.shift;
		if (last >= chunks.count) {
			return true;
		}
		for (auto index = first; index <= last; index++) {
			if (ChunkBit(chunks.dirty, static_cast<uint32_t>(index))) {
				return true;
			}
		}
		return false;
	}
	// True while every native byte outside the dirty chunks equals the detiled guest bytes:
	// set only after an upload from guest memory, cleared by any other write to the image.
	[[nodiscard]] bool PartialValid() const noexcept { return m_partial_valid; }
	void               SetPartialValid(bool valid) noexcept { m_partial_valid = valid; }

	// Transfer attribution (diagnostics only): union of the guest ranges that dirtied this image
	// since its last refresh, clipped to the image, and why it was last refreshed.
	void NoteDirtySpan(uint64_t vaddr, uint64_t size) noexcept {
		const auto begin = std::max(vaddr, live.address);
		const auto end   = std::min(vaddr + size, live.End());
		if (begin >= end) {
			return;
		}
		if (m_dirty_begin == m_dirty_end) {
			m_dirty_begin = begin;
			m_dirty_end   = end;
		} else {
			m_dirty_begin = std::min(m_dirty_begin, begin);
			m_dirty_end   = std::max(m_dirty_end, end);
		}
	}
	[[nodiscard]] uint64_t DirtySpanBytes() const noexcept { return m_dirty_end - m_dirty_begin; }
	[[nodiscard]] uint64_t DirtySpanBegin() const noexcept { return m_dirty_begin; }
	void                   ClearDirtySpan() noexcept { m_dirty_begin = m_dirty_end = 0; }
	// False until the first refresh (upload, or overwrite of the initial guest contents).
	[[nodiscard]] bool     WasEverUploaded() const noexcept { return m_uploads != 0 || m_refreshed; }
	void                   NoteUpload() noexcept { m_uploads++; }
	[[nodiscard]] bool     DirtyFromEdgeHash() const noexcept { return m_dirty_from_hash; }

	[[nodiscard]] bool IsCpuDirty() const { return m_cpu_dirty || m_maybe_cpu_dirty; }
	[[nodiscard]] bool IsDefinitelyCpuDirty() const { return m_cpu_dirty; }
	[[nodiscard]] bool IsMaybeCpuDirty() const { return m_maybe_cpu_dirty; }
	void               MarkMaybeCpuDirty() {
		if (!m_cpu_dirty) {
			m_maybe_cpu_dirty = true;
		}
	}
	[[nodiscard]] bool NeedsMaybeCpuHash() const {
		return m_maybe_cpu_dirty && !m_maybe_hash_valid;
	}
	void SetMaybeCpuHash(uint64_t hash) {
		if (!NeedsMaybeCpuHash()) {
			EXIT("image cannot initialize maybe-dirty hash\n");
		}
		m_maybe_cpu_hash   = hash;
		m_maybe_hash_valid = true;
	}
	[[nodiscard]] bool ResolveMaybeCpuHash(uint64_t hash) {
		if (!m_maybe_cpu_dirty || !m_maybe_hash_valid || m_cpu_dirty) {
			EXIT("image cannot resolve maybe-dirty hash\n");
		}
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		m_dirty_from_hash  = hash != m_maybe_cpu_hash;
		m_cpu_dirty |= m_dirty_from_hash;
		if (!m_cpu_dirty) {
			ClearDirtySpan();
		}
		return m_cpu_dirty;
	}

	void RefreshComplete() {
		if (!IsCpuDirty()) {
			EXIT("clean image cannot complete a refresh\n");
		}
		m_cpu_dirty        = false;
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		m_dirty_from_hash  = false;
		m_refreshed        = true;
		// Also reached without an upload (the image is being overwritten on the GPU); a
		// guest-sourced refresh sets partial validity again afterwards.
		m_partial_valid = false;
		ClearChunkDirty();
		ClearDirtySpan();
	}

	[[nodiscard]] bool IsGpuModified() const noexcept { return m_gpu_modified; }
	void               MarkGpuModified() noexcept {
		m_gpu_modified  = true;
		m_partial_valid = false;
	}
	void ClearGpuModified() noexcept {
		m_gpu_modified = false;
		m_owned.reset();
		claimed_rect_valid = false;
	}

	// Byte ownership of a GPU-modified image (TextureCache, KYTY_ALIAS_BYTES): the guest bytes of
	// info.data whose current contents are this image's native contents. All of them unless an
	// owned set is present: a write of another alias or a bounded buffer write took the others, or
	// the image claimed only the blocks a draw's scissor covers. An empty set is transient: a
	// binding marked the image before the write that claims bytes.
	[[nodiscard]] bool OwnsAllBytes() const noexcept { return m_gpu_modified && !m_owned; }
	[[nodiscard]] bool OwnsNoBytes() const noexcept {
		return !m_gpu_modified || (m_owned && m_owned->Empty());
	}
	// The owned bytes when not all of them; nullptr when all (or none: not GPU-modified).
	[[nodiscard]] const RangeSet* OwnedSet() const noexcept {
		return m_gpu_modified ? m_owned.get() : nullptr;
	}
	[[nodiscard]] bool OwnsBytesIn(uint64_t address, uint64_t size) const noexcept {
		if (!m_gpu_modified || size == 0 || !ImageRangeOverlaps(info.data.address, info.data.size,
		                                                        address, size)) {
			return false;
		}
		return !m_owned || m_owned->Intersects(address, size);
	}
	[[nodiscard]] bool OwnsBytes(uint64_t address, uint64_t size) const noexcept {
		if (!m_gpu_modified || size == 0) {
			return false;
		}
		const auto begin = std::max(address, info.data.address);
		const auto end   = std::min(address + size, info.data.End());
		if (begin >= end || begin != address || end != address + size) {
			return false;
		}
		return !m_owned || m_owned->Contains(address, size);
	}
	// func(begin, end) for each owned byte range inside [address, address + size).
	template <typename Func>
	void ForEachOwnedRange(uint64_t address, uint64_t size, Func&& func) const {
		if (!m_gpu_modified || size == 0) {
			return;
		}
		const auto begin = std::max(address, info.data.address);
		const auto end   = std::min(address + size, info.data.End());
		if (begin >= end) {
			return;
		}
		if (!m_owned) {
			func(begin, end);
			return;
		}
		m_owned->ForEachInRange(begin, end - begin, func);
	}
	// A binding that has not written yet: GPU-modified, owning no byte until its write claims some.
	void OwnNoBytes() {
		m_owned            = std::make_unique<RangeSet>();
		claimed_rect_valid = false;
	}
	void OwnAllBytes() noexcept { m_owned.reset(); }
	// Adds [address, address + size) (clipped to info.data) to the owned bytes.
	void OwnBytes(uint64_t address, uint64_t size) {
		const auto begin = std::max(address, info.data.address);
		const auto end   = std::min(address + size, info.data.End());
		if (!m_owned || begin >= end) {
			return;
		}
		m_owned->Add(begin, end - begin);
		if (m_owned->Contains(info.data.address, info.data.size)) {
			m_owned.reset();
		}
	}
	// Removes [address, address + size) from the owned bytes, and GPU ownership once none is left.
	// Returns whether the image owned any of those bytes.
	bool DisownBytes(uint64_t address, uint64_t size) {
		if (!OwnsBytesIn(address, size)) {
			return false;
		}
		const auto begin = std::max(address, info.data.address);
		const auto end   = std::min(address + size, info.data.End());
		if (!m_owned) {
			if (begin == info.data.address && end == info.data.End()) {
				ClearGpuModified();
				return true;
			}
			m_owned = std::make_unique<RangeSet>();
			m_owned->Add(info.data.address, info.data.size);
		}
		m_owned->Subtract(begin, end - begin);
		claimed_rect_valid = false;
		if (m_owned->Empty()) {
			ClearGpuModified();
		}
		return true;
	}

	[[nodiscard]] bool IsBufferModified() const noexcept { return m_buffer_modified; }
	void               MarkBufferModified() noexcept {
		m_buffer_modified = true;
		m_partial_valid   = false;
	}
	void               ClearBufferModified() noexcept { m_buffer_modified = false; }

	// Against the registered (resident) guest range; see `live`.
	[[nodiscard]] bool Overlaps(uint64_t address, uint64_t size,
	                            bool pages = false) const noexcept {
		return pages ? ImagePageRangesOverlap(live.address, live.size, address, size)
		             : ImageRangeOverlaps(live.address, live.size, address, size);
	}
	[[nodiscard]] bool SafeToDownload() const noexcept {
		return OwnsAllBytes() && !IsBufferModified() && !IsCpuDirty();
	}
	[[nodiscard]] bool IsTracked() const noexcept { return track_addr != 0 && track_addr_end != 0; }
	[[nodiscard]] uint64_t AccountedSize() const noexcept {
		return backing.image == nullptr ? 0 : Common::AlignUp(info.data.size, 1024);
	}
	[[nodiscard]] uint64_t HashGuestEdges() const;

	ImageInfo        info;
	// The guest bytes of the resident levels: info.data, or its prefix when resident_first > 0.
	// Changed only while the image is unregistered.
	GuestRange       live;
	uint32_t         resident_first = 0;
	// KYTY_TEXTURE_RESIDENT_MIPS=poison: non-resident levels were filled with a marker.
	bool             residency_poisoned = false;
	// The next refresh uploads levels made resident by a residency change (attribution only).
	bool             residency_refresh = false;
	VulkanImage      backing;
	std::vector<CachedImageView> views;
	ImageUsage       usage;
	ImageBinding     binding;
	bool             registered     = false;
	mutable uint32_t query_epoch    = 0;
	uint64_t         track_addr     = 0;
	uint64_t         track_addr_end = 0;
	ImageId          depth_id {};
	uint64_t         tick_accessed_last  = 0;
	uint64_t         frame_accessed_last = 0; // presented guest frames, see TextureCache::AdvanceFrame
	size_t           lru_id              = 0;
	// While registered: the tick of TextureCache's LRU item lru_id (KYTY_IMAGE_LRU_SKIP). Set from
	// the item after every Insert (RegisterImage) and Touch (TouchImage), its only writers.
	uint64_t         lru_tick            = 0;
	// Last GPU writer among overlapping aliases; cleared when another alias takes the bytes.
	bool             alias_owner         = false;
	// KYTY_IMAGE_EXACT_RANGE_INVALIDATE (TextureCache::InvalidateMemoryFromGPU): a GPU buffer write
	// over exactly another image's range partly overlapped this image and left it unrebuilt (in
	// count mode: would have). Cleared by the image's next GPU write, and (on) by a refresh upload.
	bool             exact_range_stale   = false;
	// KYTY_ALIAS_BYTES: a level-0 texel rectangle whose 64 KiB blocks this image owns (bounded
	// render-target claims skip the claim when a draw's scissor lies inside it). Dropped whenever
	// the image loses owned bytes.
	vk::Rect2D       claimed_rect {};
	bool             claimed_rect_valid  = false;
	ChunkState       chunks;
	// Last download into a cache buffer for texel-buffer reads (SynchronizeBufferFromImage):
	// the buffer revision it produced and this image's content serial at the time.
	struct TexelSyncMark {
		Common::SlotId buffer {};
		uint64_t       revision = 0;
		uint64_t       epoch    = 0;
		uint64_t       serial   = 0;
		uint64_t       size     = 0;
		bool           valid    = false;
	};
	TexelSyncMark    texel_sync;
	// Depth attachments (KYTY_DEPTH_FEEDBACK_KEEP, RenderExecutor::NoteDepthFeedback): the
	// rendering instance this image is attached to with no attachment write since it began, the
	// content serial at that point (0: none), and the last instance it was attached to at all.
	uint64_t         feedback_instance = 0;
	uint64_t         feedback_serial   = 0;
	uint64_t         feedback_attached = 0;
	// Replaces the tracked state without recording a barrier. Only for callers that proved no
	// access pair of the old and new scopes can conflict (see KYTY_DEPTH_FEEDBACK_KEEP).
	void AdoptState(vk::PipelineStageFlags2 stage, vk::AccessFlags2 access,
	                vk::ImageLayout layout) noexcept {
		backing.state = {stage, access, layout};
	}

private:
	friend struct ImageTestAccess;

	[[nodiscard]] static vk::ImageAspectFlags FullAspectMask(vk::Format format) noexcept;
	void                                      CopyImageRegions(Image& source);
	[[nodiscard]] static uint32_t             CopyRows(uint64_t row_size, uint32_t rows,
	                                                   uint64_t capacity) noexcept;
	[[nodiscard]] static std::pair<uint32_t, uint32_t>
	SanitizeCopyLayers(const Image& source, const Image& destination, uint32_t depth);

	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;
	uint64_t          m_maybe_cpu_hash   = 0;
	bool              m_cpu_dirty        = false;
	bool              m_maybe_cpu_dirty  = false;
	bool              m_maybe_hash_valid = false;
	bool              m_gpu_modified     = false;
	bool              m_buffer_modified  = false;
	bool              m_dirty_from_hash  = false;
	bool              m_refreshed        = false;
	bool              m_partial_valid    = false;
	// See OwnedSet(): the owned bytes of a GPU-modified image, nullptr for all of them.
	std::unique_ptr<RangeSet> m_owned;
	uint64_t          m_content_serial   = 0;
	uint64_t          m_definite_writes  = 0;
	uint64_t          m_dirty_begin      = 0;
	uint64_t          m_dirty_end        = 0;
	uint32_t          m_uploads          = 0;
	// What the device refused for this image and the image was created without
	// (DeviceCompat::OptionalImageCreateFallbacks): FindView stops at a view that needs it.
	vk::ImageUsageFlags  m_dropped_usage {};
	vk::ImageCreateFlags m_dropped_flags {};
};

namespace ImageOps {

void                                 Validate(const ImageInfo& info);
[[nodiscard]] Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element);
// KYTY_TILER_IMAGE_DIRECT_BC (default on; needs KYTY_TILER_IMAGE_DIRECT): block-compressed images
// get storage usage (through block-texel-compatible uncompressed views, where the device supports
// it) so that TileManager::DetileToImage can write their blocks directly.
[[nodiscard]] bool BlockStorageUploadsEnabled();

} // namespace ImageOps

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
