#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/hangTrace.h"
#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/profiler.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/coherenceLog.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/imageCacheGcPolicy.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/dccClear.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <array>
#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class DccClearHelper;
class RenderExecutor;
class StagingCopier;
class TextureBindingMemo;
struct TextureCacheTestAccess;

class TextureCache {
public:
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	// A render-target binding with CMASK fast clears enabled (CB_COLORn_INFO.FAST_CLEAR):
	// the target's CMASK bytes and CB_COLORn_CLEAR_WORD0/1 (KYTY_CMASK_FAST_CLEAR).
	struct ColorFastClear {
		GuestRange range;
		uint32_t   clear_word0 = 0;
		uint32_t   clear_word1 = 0;
		bool       valid       = false;
	};

	struct ImageDesc {
		ImageInfo      info;
		ImageViewInfo  view_info;
		BindingType    type = BindingType::Texture;
		ColorFastClear cmask;
	};

	// KYTY_DRAW_SEQUENCE_FAST: why a metadata decision of FindImage (MaterializeDccClear,
	// MaterializeCmaskClear) did nothing, in terms a later decision for the same description can
	// be checked against. `provable`: the decision read nothing but the description and fixed
	// image fields, or only the state recorded here: the texture-cache structure over the metadata
	// bytes (page versions: which images alias them), the recorded fills (BufferCache
	// KnownFillGeneration; 0 when none was read) and the GPU-dirty state (set) of `gpu_ranges`.
	// With all of it unchanged the same decision is made again, and again does nothing.
	struct MetadataNoop {
		static constexpr uint32_t MaxPages     = 2;
		static constexpr uint32_t MaxGpuRanges = 4;
		static constexpr uint32_t MaxGuestKeys = 4;

		bool                                 provable        = false;
		uint64_t                             first_page      = 0;
		uint32_t                             page_count      = 0;
		std::array<uint64_t, MaxPages>       page_versions   {};
		uint64_t                             fill_generation = 0;
		uint32_t                             gpu_range_count = 0;
		std::array<GuestRange, MaxGpuRanges> gpu_ranges      {};
		// KYTY_CP_COMMIT=dccguest: a DCC decision on metadata the GPU had not written, none of whose
		// slices' first key decoded to a clear (so nothing else of the slice was read). It stays the
		// same decision while `guest_range` is still not GPU-dirty and every recorded first key
		// still reads as recorded (guest_key_count keys; 0: no such certificate).
		GuestRange                             guest_range     {};
		uint32_t                               guest_key_count = 0;
		std::array<uint64_t, MaxGuestKeys>     guest_key_addresses {};
		std::array<uint8_t, MaxGuestKeys>      guest_key_codes {};
	};
	// KYTY_DRAW_SEQUENCE_FAST: what a FindImage of a description returned, and why the same
	// lookup would return it again with nothing to do but its access bookkeeping (see
	// TryRepeatLookup). Filled by FindImage(desc, exact_format, &record); `valid` only when every
	// part is proven.
	//  - Lookup: the answer came from the first-page lookup (FindImageWithSameBacking), which
	//    reads only the owner list of the description's first page and the owners' registered
	//    flags and SameBacking fields; every change of those bumps the page's version.
	//  - EnsureResidency: the image's resident_first is at most `requested_first` (checked live).
	//  - SyncAliasFromOwner: returns at once (checked live) or no registered image has the same
	//    backing range, extent and sample count (`has_partner` false: such images start on the
	//    same page, so a new one bumps its version).
	//  - MaterializeDccClear / MaterializeCmaskClear: `dcc` and `cmask`.
	struct RepeatLookup {
		ImageId      image;
		uint64_t     page            = 0;
		uint64_t     page_version    = 0;
		uint32_t     requested_first = 0;
		bool         exact_format    = false;
		bool         has_partner     = false;
		bool         valid           = false;
		MetadataNoop dcc;
		MetadataNoop cmask;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	// `record` (optional): filled as described at RepeatLookup.
	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false,
	                                      RepeatLookup* record = nullptr);
	// The lookup `record` describes, repeated for the same `desc` and `exact_format` (the caller
	// guarantees the description is equal to the recorded lookup's): true when every recorded proof
	// still holds, so FindImage would return record.image and do nothing else but its access
	// bookkeeping; with `apply` that bookkeeping (and the metadata bookkeeping of a DCC
	// description) is performed as FindImage performs it. `apply` false only checks.
	[[nodiscard]] bool          TryRepeatLookup(const ImageDesc& desc, bool exact_format,
	                                            const RepeatLookup& record, bool apply);
	// Diagnostics (KYTY_DRAW_SEQUENCE_VERIFY): residency extensions and alias synchronizations
	// FindImage has performed; a repeat must add none.
	[[nodiscard]] uint64_t      LookupSideEffects() const noexcept { return m_lookup_side_effects; }
	// Diagnostics (KYTY_CP_COMMIT=texdcc verify): DCC decisions of MaterializeDccClear that did
	// something (a slice clear, or a native GPU inspection); a memo hit's full resolution adds none.
	[[nodiscard]] uint64_t      DccDecisionEffects() const noexcept { return m_dcc_decision_effects; }
	// KYTY_DRAW_SEQUENCE_VERIFY: a GPU-owned metadata range `record` relied on is no longer
	// GPU-modified. FindImage never ends GPU ownership of such a range before a decision that
	// differs from the recorded one, so after a repeat check this means another thread (a guest
	// write fault's flush, a side readback) changed it: the full lookup raced the check.
	[[nodiscard]] bool          RepeatGpuRangesLost(const RepeatLookup& record);
	void                        UpdateImage(ImageId id);
	// buffer_sync: the caller copies the image into the buffer at its own range (texel reads),
	// where GPU-dirty bytes it supersedes do not block it (SafeToSyncIntoBuffer).
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true,
	                                               bool buffer_sync  = false);
	// FindImageFromRange's selection among the images FindImagesInRegion finds for
	// [address, address + query_size) (KYTY_TEXEL_IMAGE_LOOKUP). Caller holds m_lock.
	[[nodiscard]] ImageId       FindImageFromRangeLocked(uint64_t address, uint64_t size,
	                                                     uint64_t query_size, bool ensure_valid,
	                                                     bool buffer_sync);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	// written: the draw's scissor (framebuffer pixels, unclamped); the target then owns only the
	// 64 KiB blocks under it (KYTY_ALIAS_BYTES). nullptr: the whole image may be written.
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc,
	                                             const vk::Rect2D* written = nullptr);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);

	// Content identity around an attachment binding (see Image::ContentSerial). Take a mark
	// before FindDepthTarget; once the draw is known not to write the attachment, restoring it
	// keeps the image's serial when nothing else (upload, copy, clear) wrote the image meanwhile.
	struct ContentMark {
		uint64_t serial          = 0;
		uint64_t definite_writes = 0;
	};
	[[nodiscard]] ContentMark MarkContent(ImageId id);
	void                      RestoreContentIfUnwritten(ImageId id, const ContentMark& mark);

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size);
	// KYTY_IMAGE_EXACT_RANGE_INVALIDATE (default off; =1 on, =count dry run): when a GPU buffer
	// write covers exactly one image's range, InvalidateMemoryFromGPU rebuilds only the images
	// inside that range; images it partly overlaps lose ownership of the written bytes but are not
	// rebuilt (they are other surfaces sharing the memory at other times, e.g. the DRS-sized
	// render targets of one heap). Count mode rebuilds them as before and only counts. Both modes
	// count, per 10 s on the console, the sampled/storage uses of such images before their next
	// GPU write (RebindImages calls this): a stale read the rule would expose.
	void NoteExactRangeStaleUse(Image& image, bool storage) {
		if (m_exact_range_mode != 0 && image.exact_range_stale) [[unlikely]] {
			NoteExactRangeStaleUseSlow(image, storage);
		}
	}
	// Diagnostics only (changes nothing): images that InvalidateMemoryFromGPU(address, size) would
	// invalidate but that no range in `written` overlaps.
	[[nodiscard]] uint32_t CountImagesOutsideGpuWrite(uint64_t address, uint64_t size,
	                                                  std::span<const GuestRange> written);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t address, uint64_t size);
	// Whether any cached image overlaps the range (RenderContext::UnmapMemory: CPU-only unmaps skip
	// the GPU drain).
	[[nodiscard]] bool IsRegionRegistered(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	// KYTY_META_CLEAR_MEMO=1 (default off; BryanKAdams/KytyPS5 c36bbff): the last answer is kept for
	// its address while m_surface_meta_generation holds, without m_lock. Called by the GPU (CP)
	// thread only (the depth-target resolution of draws).
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	void ProcessDownloadImages();
	// Called once per completed guest flip. Alias lifetimes are measured in presented frames:
	// a frame spans hundreds of scheduler ticks, so tick age alone treats every alias used
	// earlier in the same frame as stale.
	void AdvanceFrame() noexcept { m_frame.fetch_add(1, std::memory_order_relaxed); }
	void RunGarbageCollector();
	// KYTY_VRAM_STATS report lines (vramStats.h): images by binding kind, frame age, residency,
	// render-target sizes and aliases, the largest images, the GC thresholds (GPU thread).
	void ReportVram();

	// KYTY_ALIAS_BYTES (default on; =0 restores whole-image ownership). GPU ownership follows the
	// bytes a write can reach: a write takes only those bytes from overlapping images (a draw only
	// the 64 KiB blocks under its scissor), and every other image keeps the rest of its bytes.
	// Bytes an image owns reach the buffer cache (the owner's contents are tiled into it) before
	// a read goes through buffer bytes: a sampled or copied image's rebuild, a texel-buffer read,
	// a buffer copy source; and before a GPU-modified image is freed (except by an unmap).
	// Needs aliases aged by frames (KYTY_IMAGE_ALIAS_AGE).
	[[nodiscard]] static bool AliasBytesEnabled();

private:
	enum class TransferDirection { Upload, Download };
	// Why an image is refreshed: a read of its contents (sampling, copies, presentation) or a
	// binding that writes it (render, depth and storage targets). With KYTY_ALIAS_BYTES a read
	// rebuild first moves bytes other images own into the buffer; a write binding's does not
	// (the write usually covers them; it costs a tile pass per aliased target and frame).
	enum class RefreshIntent : uint8_t { Read, Write };
	// The bytes a GPU write may write (KYTY_ALIAS_BYTES): ranges == nullptr means the whole
	// image; rect is the level-0 texel rectangle behind bounded render-target claims.
	struct WriteClaim {
		const RangeSet*           ranges = nullptr;
		std::optional<vk::Rect2D> rect;
	};
	enum class ImageLookupMode { FirstPage, Legacy, Verify };
	struct TextureTransfer;
	struct ImageDownload;

	struct MetaDataInfo {
		enum class Type : uint8_t { CMask, FMask, HTile };

		Type     type;
		uint32_t clear_mask = UINT32_MAX;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
	};

	struct GpuDccInspection {
		GuestRange metadata;
		BufferContentRevision revision;
		// Clear codes (DccClearHelper::ClearCodes bits) the inspected interpretation accepted.
		uint32_t decodable_mask = 0;
	};

	using ImageIds       = InlinePageOwnerList<ImageId, 16>;
	using ImagePageTable = MultiLevelPageTable<ImageIds, 20, 40, 10>;

	// Callers have validated the nonempty 40-bit range with TryGetPageRange.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		using FuncReturn = typename std::invoke_result<Func, uint64_t>::type;
		static constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			if constexpr (RETURNS_BOOL) {
				if (func(page)) {
					break;
				}
			} else {
				func(page);
			}
		}
	}

	// resident_first > 0: register only the prefix holding levels [resident_first, levels)
	// (resident_prefix bytes, computed when 0); stays fully resident when not applicable.
	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info, uint32_t resident_first = 0,
	                                      uint64_t resident_prefix = 0);
	// Resident mip levels (Image::live, KYTY_TEXTURE_RESIDENT_MIPS). Caller holds m_lock.
	// The finest level a view of `desc` can read, for an image with `levels` levels.
	[[nodiscard]] uint32_t    RequestedFirstLevel(const ImageDesc& desc, uint32_t levels) const;
	[[nodiscard]] uint64_t    ResidentPrefixSize(const ImageInfo& info, uint32_t first_level) const;
	// Makes levels >= first_level resident (re-registration; the next refresh uploads them).
	// `sampling`: requested by a sampled view (counted as an extension, else a fallback).
	void                      EnsureResidency(ImageId id, uint32_t first_level, bool sampling);
	// Whole chain resident and refreshed now, before a non-sampling use records anything.
	void                      RequireFullResidency(ImageId id);
	void                      PoisonNonResidentLevels(Image& image);
	void                      RetireIdlePartialImages();
	[[nodiscard]] ImageId     GetNullImage(const ImageDesc& desc);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	void                      FreeImage(ImageId id, HangTrace::ImageFreeReason reason =
	                                                    HangTrace::ImageFreeReason::Other);
	void                      TouchImage(Image& image);
	void                      SyncAliasFromOwner(ImageId id);
	void                      TrackImage(ImageId id);
	// Chunk-granular tracking (Image::ChunkState). Caller holds m_lock.
	[[nodiscard]] bool        ChunkTrackingEligible(const Image& image) const;
	void                      TrackChunkImage(Image& image);
	template <bool track>
	uint32_t                  UpdateChunkWatchers(Image& image, uint32_t first, uint32_t last);
	void                      InvalidateChunks(Image& image, uint64_t address, uint64_t size);
	struct PartialUploadResult {
		bool     done        = false;
		uint64_t bytes       = 0; // tiled guest bytes detiled and uploaded
		uint64_t dirty_bytes = 0; // dirty chunk bytes that caused the refresh
	};
	[[nodiscard]] PartialUploadResult TryPartialUpload(Image& image);
	// Whole-image refresh of a tiled sampled texture whose guest bytes are copied to staging
	// by m_staging_copier. False when the ordinary refresh path must be used.
	[[nodiscard]] bool        TryAsyncFullUpload(Image& image);
	void                      RecordChunkHashes(Image& image);
	[[nodiscard]] bool        VerifyCleanChunks(Image& image);
	[[nodiscard]] bool        KeepOverlappedImage(const Image& cached, uint64_t current_frame) const;
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	// All callers hold m_lock; negative ownership proofs contain no guest values.
	void                      MarkImageGpuModified(Image& image);
	// Logs [address, address + size) to the coherence log (all memory by default).
	void InvalidateCleanImageProofs(uint64_t address = 0, uint64_t size = UINT64_MAX,
	                                Coherence::Source source = Coherence::Source::Universe);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);
	[[nodiscard]] bool               SafeToDownload(const Image& image);
	// The image's native contents are newer than every GPU-dirty buffer byte in its range.
	[[nodiscard]] bool SupersedesGpuDirtyBytes(const Image& image);
	// SafeToDownload for a copy of the image into the buffer at its own range (texel reads).
	[[nodiscard]] bool SafeToSyncIntoBuffer(const Image& image);

	// Caller holds m_lock; it also serializes the per-image query epoch.
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size,
	                                               bool page_overlap) const;
	// Caller holds m_lock. Equal backing ranges must begin in the same indexed page.
	[[nodiscard]] ImageId       FindImageWithSameBacking(const ImageInfo& requested,
	                                                     bool exact_format) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id,
	                                         RefreshIntent intent = RefreshIntent::Read);
	// `noop` (optional): how the decision can be repeated (MetadataNoop), reset first.
	void                        MaterializeDccClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer,
	                                                MetadataNoop* noop = nullptr);
	// A render-target binding whose CMASK marks every tile of a bound slice fast-cleared: clears
	// the slice to the CLEAR_WORD colour and leaves the CMASK expanded, as the colour block's
	// reads and the fast-clear eliminate would (KYTY_CMASK_FAST_CLEAR).
	void                        MaterializeCmaskClear(ImageId id, const ImageDesc& desc,
	                                                  uint32_t metadata_base_layer,
	                                                  MetadataNoop* noop = nullptr);
	// MetadataNoop helpers. Caller holds m_lock: the page versions over `range` (false when it
	// covers more than MetadataNoop::MaxPages pages).
	[[nodiscard]] bool CaptureMetadataPages(GuestRange range, MetadataNoop& noop) const;
	[[nodiscard]] bool MetadataPagesHold(const MetadataNoop& noop) const;
	// Without m_lock: the recorded fills and GPU-dirty states are still as recorded.
	[[nodiscard]] bool MetadataStateHolds(const MetadataNoop& noop);
	// KYTY_CP_COMMIT=dccguest: the guest-key certificate of `noop` still holds.
	[[nodiscard]] bool GuestKeysHold(const MetadataNoop& noop);
	// m_surface_metas.erase(address) (caller holds m_lock); KYTY_CP_COMMIT=metaerase skips it
	// when it would erase nothing (m_meta_erases).
	void EraseSurfaceMeta(uint64_t address);
	// SyncAliasFromOwner returns before looking for an owner (caller holds m_lock).
	[[nodiscard]] static bool SyncAliasReturnsAtOnce(const Image& image);
	// Whether a registered image other than `found` has exactly its backing range, extent and
	// sample count on `page` (SyncAliasFromOwner's only possible owners). Caller holds m_lock.
	[[nodiscard]] bool HasAliasPartner(uint64_t page, ImageId found, const Image& image) const;
	// Returns DccGpuRecords or DccGpuReuses when the native path handled every slice, otherwise
	// the first failing reason (Profiler::FrameEvent::DccFallback*) for the CPU fallback.
	[[nodiscard]] Profiler::FrameEvent TryMaterializeGpuDccClear(ImageId id, const ImageDesc& desc,
	                                                             uint32_t metadata_base_layer);
	// The native inspection behind TryMaterializeGpuDccClear for any metadata whose uniform
	// bytes select a clear value (values/decodable in DccClearHelper::ClearCodes order): clears
	// the bound slices whose bytes all equal an accepted code and leaves those bytes 0xFF. cmask:
	// CMASK bytes (only code 0x00), not counted as DCC work.
	[[nodiscard]] Profiler::FrameEvent TryMaterializeGpuMetadataClear(
	    DccClearHelper& helper, ImageId id, const ImageDesc& desc, GuestRange range,
	    uint32_t metadata_base_layer, const DccClearHelper::ClearValues& values, uint32_t decodable,
	    bool cmask);
	void                        InitializeImage(ImageId id,
	                                            RefreshIntent intent = RefreshIntent::Read);
	// KYTY_ALIAS_BYTES. Moves the bytes of `range` that GPU-modified images own into the buffer
	// cache: each owner is tiled into a scratch copy of its owned ranges' buffer bytes, which are
	// copied back; the buffer then owns them (GPU-dirty) and the image no longer does. only:
	// just that image's bytes, without the ones another image also owns (a copy's source and
	// destination); except: every owner but that one. Returns the bytes moved. Caller holds
	// m_lock; records GPU work.
	// target: the cache buffer to move the bytes into, when it covers them (a texel read's).
	uint64_t MaterializeOwnedBytes(GuestRange range, ImageId only, ImageId except,
	                               const char* reason, Buffer* target = nullptr);
	// Whether any GPU-modified image other than `except` owns a byte of `range`.
	[[nodiscard]] bool OtherImagesOwnBytes(GuestRange range, ImageId except);
	// False when no image is registered on the 1 MiB pages of the range (lock-free, any thread).
	[[nodiscard]] bool MayOverlapImages(uint64_t address, uint64_t size) const;
	// The 64 KiB blocks of a single-level, single-layer render target (64 KiB render-target
	// tiling) under `rect` (level-0 texels, clamped to the extent). False: not such a target, or
	// the rectangle covers the whole image (claim it all).
	[[nodiscard]] bool BlocksUnderRect(const Image& image, vk::Rect2D& rect, RangeSet& blocks) const;
	// The destination of a copy owns what the source owned there (and the source no longer).
	void TakeOverOwnedBytes(Image& destination, Image& source);
	[[nodiscard]] TextureTransfer
	BuildTextureTransfer(const Image& image, BindingType binding, TransferDirection direction) const;
	[[nodiscard]] static TextureTransfer BuildTextureTransfer(const ImageInfo& info,
	                                                          uint32_t backing_samples,
	                                                          BindingType binding,
	                                                          TransferDirection direction);
	// Drops the regions (and tiles) of non-resident levels and packs the detiled scratch.
	void RestrictToResidentLevels(const Image& image, TextureTransfer& transfer) const;
	[[nodiscard]] ImageDownload BuildDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                       uint64_t destination_size, ImageDownload transfer);
	// The same into any device buffer with `destination_capacity` bytes from destination_offset.
	void DownloadImageTo(Image& image, vk::Buffer destination, uint64_t destination_offset,
	                     uint64_t destination_capacity, uint64_t destination_size,
	                     ImageDownload transfer);
	void DownloadDepth(Image& image, vk::Buffer destination, uint64_t destination_offset,
	                   uint64_t destination_capacity);
	void CommitGpuWrite(Image& image);
	void CommitGpuWrite(Image& image, const WriteClaim& claim);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	// Returns false, after reporting once per signature, when `id` cannot take the clear (a depth
	// image for a colour range, a stencil plane record, a level or layer range beyond the image):
	// nothing was changed, so the caller must not consume the guest's clear key or fill.
	[[nodiscard]] bool ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                              const vk::ImageSubresourceRange& range,
	                              const vk::ClearValue& clear, const char* site);
	void PrepareImageCopy(Image& image);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	// Depth <-> color reinterpretation without a staging buffer (VK_KHR_maintenance8 copy or a
	// one-pass shader). Returns the path name, or nullptr when nothing was recorded.
	[[nodiscard]] const char* TryDirectReinterpret(Image& destination, Image& source);
	// Returns true when the destination's copied subresources now hold exactly the source's
	// native bits for every subresource of both images (same shape, lossless path).
	bool CopyImage(ImageId destination, ImageId source, const char* context = "other");
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	[[nodiscard]] bool DownloadImageMemory(ImageId id);
	void RunPressureGarbageCollector(uint64_t tick);
	// KYTY_VRAM_IDLE_FRAMES=N (default 0: off): once every 32 frames, registered images unused for
	// more than N presented frames that the submission-age collector frees without a download (not
	// GPU-modified, not bound, a target, a stencil association or a presentation surface) are freed,
	// whatever the memory usage, so earlier scenes' images do not stay until the budget runs out.
	// A later use recreates and refreshes the image.
	// KYTY_VRAM_PRESSURE_FRAMES=K (default 0: off): above the collection trigger, instead of the
	// submission-age collection (images unused for 16-160 completed submissions, a fraction of a
	// frame, are freed and recreated by the next frame; it also stops at the first LRU entries it
	// may not free), images unused for K frames are freed the same way, once per frame, oldest
	// first, until usage is under the trigger (K/4, at least 2, above the critical mark).
	// m_frame_ticks: the GC tick each recent frame began at (LRU ticks are GC ticks).
	[[nodiscard]] bool IdleRetirable(const Image& image, uint64_t frame, uint64_t min_age) const;
	void               NoteFrameTick(uint64_t frame, uint64_t tick);
	// Returns the AccountedSize bytes freed (stops past `enough`).
	uint64_t           RetireUnusedImages(uint64_t frame, uint64_t min_age, size_t max_count,
	                                      uint64_t enough);
	uint64_t           m_idle_frames          = 0;
	uint64_t           m_idle_next_frame      = 0;
	uint64_t           m_idle_freed           = 0;
	uint64_t           m_pressure_frames      = 0;
	uint64_t           m_pressure_frame       = 0;
	uint64_t           m_pressure_freed_bytes = 0;
	// KYTY_VRAM_GC_BUDGET (graphics/host_gpu/vramBudget.h): frame-aged collection against the
	// budget once per frame. Returns true when it handled this collection.
	bool               RunBudgetGarbageCollector(uint64_t frame);
	uint64_t           m_budget_frame       = UINT64_MAX;
	uint64_t           m_budget_freed_bytes = 0;
	uint64_t           m_budget_freed       = 0;
	uint64_t           m_budget_last        = 0; // planning budget at the last collection
	std::deque<std::pair<uint64_t, uint64_t>> m_frame_ticks;
	// KYTY_VRAM_STATS: images created and freed (per HangTrace::ImageFreeReason) since the last
	// report, {count, native bytes}.
	void NoteVramFree(const Image& image, HangTrace::ImageFreeReason reason);
	std::array<std::pair<uint64_t, uint64_t>, static_cast<size_t>(HangTrace::ImageFreeReason::Count)>
	                              m_vram_frees {};
	std::pair<uint64_t, uint64_t> m_vram_creates {};

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	std::unique_ptr<DccClearHelper>                    m_dcc_clear;
	// Native CMASK inspection when KYTY_DCC_GPU does not create m_dcc_clear (created on first use).
	std::unique_ptr<DccClearHelper>                    m_cmask_clear;
	std::unordered_map<uint64_t, GpuDccInspection>      m_gpu_dcc_inspections;
	uint64_t m_gpu_dcc_attempts = 0;
	uint64_t m_gpu_dcc_records = 0;
	uint64_t m_gpu_dcc_reuses = 0;
	uint64_t m_gpu_dcc_fallbacks = 0;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	std::unordered_map<vk::Format, ImageId>           m_null_images;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	std::atomic<uint64_t>                             m_frame {0};
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	// KYTY_CP_COMMIT=metaerase (EraseSurfaceMeta): insertions into m_surface_metas so far, and per
	// recently erased address the insertion count at its erase. While the count is unchanged the
	// address is still absent (every other change of the map is a removal).
	struct MetaErase {
		uint64_t address = UINT64_MAX;
		uint64_t inserts = 0;
	};
	uint64_t                                          m_surface_meta_inserts = 0;
	std::array<MetaErase, 64>                         m_meta_erases {};
	// KYTY_CP_COMMIT=targetalloc: FindRenderTarget's bounded-claim block list (m_lock held).
	RangeSet                                          m_claim_blocks;
	// Bumped under m_lock after every change of m_surface_metas (an entry added or removed, a
	// clear_mask changed): validates m_meta_clear_memo (IsMetaCleared).
	std::atomic<uint64_t> m_surface_meta_generation {0};
	void                  NoteSurfaceMetaChange() noexcept {
		m_surface_meta_generation.fetch_add(1, std::memory_order_release);
	}
	struct MetaClearMemo {
		uint64_t address    = 0;
		uint64_t generation = 0;
		uint32_t clear_mask = 0;
		bool     found      = false;
		bool     valid      = false;
	} m_meta_clear_memo;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_registered_image_memory = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	uint64_t         m_gc_tick                = 0;
	ImageCacheGcPolicy m_pressure_gc_policy;
	bool             m_pressure_gc_enabled   = false;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t>::Cursor m_pressure_gc_cursor;
	uint64_t m_pressure_retirement_bytes = 0;
	mutable uint32_t m_image_query_epoch      = 0;
	bool             m_direct_dirty_image_query = false;
	struct CleanImagePageProof {
		uint64_t page = 0;
		uint64_t epoch = 0;
	};
	std::array<CleanImagePageProof, 256> m_clean_image_pages {};
	uint64_t m_clean_image_epoch = 1;
	bool m_clean_image_proofs = false;
	ImageLookupMode  m_image_lookup_mode = ImageLookupMode::FirstPage;
	uint64_t         m_image_lookup_checks = 0;
	uint64_t         m_image_lookup_mismatches = 0;
	// LookupSideEffects (GPU thread).
	uint64_t         m_lookup_side_effects = 0;
	// DccDecisionEffects (GPU thread).
	uint64_t         m_dcc_decision_effects = 0;
	bool             m_readback_linear_images = false;
	// Texture streaming (see TextureCache constructor for the environment switches).
	bool             m_partial_upload      = true;
	bool             m_partial_bands       = true;
	bool             m_partial_verify      = false;
	uint32_t         m_chunk_shift         = 16;
	uint64_t         m_overlap_keep_frames = 0;
	// Set by FindImage for the ResolveOverlap calls of one lookup: many live overlapping
	// images already share the requested range, so overlapped images are not kept.
	bool             m_overlap_crowded     = false;
	uint64_t         m_partial_verify_mismatches = 0;
	// KYTY_TEXTURE_ASYNC_STAGING=0: null, staging copies stay on the GPU thread.
	std::unique_ptr<StagingCopier> m_staging_copier;
	// Device-local host-visible (resizable BAR) staging ring for StagingCopier jobs, so detile
	// reads VRAM instead of system memory over PCIe (KYTY_TEXTURE_STAGING_REBAR=0: none).
	std::unique_ptr<StreamBuffer>  m_texture_staging;
	// Registered images per ImagePageTable page, readable without m_lock
	// (KYTY_TEXTURE_FAULT_FAST_PATH=0 always takes the lock). RegisterImage counts a page before
	// the image enters its owner list, UnregisterImage uncounts it after the image left it (all
	// sequentially consistent), so an image FindImagesInRegion can find is always counted.
	std::unique_ptr<std::atomic<uint32_t>[]> m_image_page_counts;
	bool                                     m_fault_fast_path = true;
	// No image is registered on the ImagePageTable pages [address, address + size) spans, from
	// m_image_page_counts without m_lock: FindImagesInRegion would find nothing there. Each page's
	// zero count means no image was findable on it at that load (see above); an image registered
	// after a load (on that page) belongs after this query in any serialization of the two.
	[[nodiscard]] bool NoImagesOnPages(uint64_t address, uint64_t size) const noexcept;
	// KYTY_GPU_WRITE_IMAGE_SKIP (default on; =0 off): a GPU buffer write's image checks
	// (InvalidateMemoryFromGPU, BufferCache::PreserveImagesForGpuWrite) skip m_lock and the page
	// walk when NoImagesOnPages: the walk would find no image to act on. True: the caller skips.
	// KYTY_GPU_WRITE_IMAGE_SKIP_VERIFY=1|exit takes the lock after such a decision and runs the
	// walk (then returns false, so the caller does too): an image found while its pages are
	// still uncounted is a mismatch (exit stops), one whose pages are counted by now was
	// registered after the decision (a race). FrameEvents GpuWriteImageSkip*.
	[[nodiscard]] bool SkipGpuWriteImageWalk(uint64_t address, uint64_t size);
	// KYTY_IMAGE_EXACT_RANGE_INVALIDATE: 0 off, 1 on, 2 count (see InvalidateMemoryFromGPU), and
	// its totals (GPU thread) for the 10 s console summary.
	void NoteExactRangeStaleUseSlow(Image& image, bool storage);
	void NoteExactRangeRewrite(Image& image);
	void PrintExactRangeSummary();
	int  m_exact_range_mode = 0;
	struct ExactRangeTotals {
		uint64_t exact_writes = 0; // buffer writes whose range equals an image's range
		uint64_t kept         = 0; // partly overlapped images left unrebuilt (count: would be)
		uint64_t stale_uses   = 0; // sampled/storage uses of such an image before its next write
		uint64_t rewrites     = 0; // such images written again by the GPU (or refreshed, on)
	};
	ExactRangeTotals m_exact_range_totals;
	ExactRangeTotals m_exact_range_printed;
	uint64_t         m_exact_range_print_ns = 0;
	uint32_t         m_exact_range_logged   = 0;
	bool m_gpu_write_skip        = true;
	int  m_gpu_write_skip_verify = 0;
	// Why TryMaterializeGpuMetadataClear last refused an image (DccImageState* FrameEvent).
	Profiler::FrameEvent m_image_state_reason = Profiler::FrameEvent::DccImageStateUnregistered;
	// Why the last MaterializeDccClear / MaterializeCmaskClear decision was not a provable no-op
	// (TargetRecordDcc* / TargetRecordCmask* FrameEvents; FindImage counts them for draw-sequence
	// target records that end invalid). Instrumentation only.
	Profiler::FrameEvent m_dcc_noop_refusal   = Profiler::FrameEvent::TargetRecordDccGuest;
	Profiler::FrameEvent m_cmask_noop_refusal = Profiler::FrameEvent::TargetRecordCmaskOther;
	// KYTY_DCC_GPU_REFRESH (default off; =1 on, needs KYTY_DCC_GPU=1): the native DCC inspection
	// (TryMaterializeGpuMetadataClear) also accepts a registered, matching, fully resident image
	// that waits for a refresh (buffer-modified, CPU-dirty) or is not GPU-owned, which the CPU
	// fallback handles today after draining the GPU for the metadata. The image is refreshed first
	// (InitializeImage), exactly as that fallback's ClearImage refreshes it before a layer clear,
	// then inspected and committed as a GPU write like any inspected image: the decision stays on
	// the GPU behind the metadata's writer. FrameEvent DccGpuRefreshes (slices).
	// KYTY_DCC_GPU_REFRESH_VERIFY=1|exit: for such an image, also take the CPU fallback's decision
	// (after a drain), copy the inspected slices out behind the inspection and compare at
	// completion: a slice the fallback clears must have its key consumed (all 0xFF), any other
	// must be untouched. DccGpuRefreshVerify{Checks,Mismatches}; exit stops on a mismatch.
	bool m_dcc_gpu_refresh        = false;
	int  m_dcc_gpu_refresh_verify = 0;
	// Atomic: the verify's deferred comparison runs on whichever thread pops pending operations.
	struct DccRefreshTotals {
		std::atomic<uint64_t> refreshes {0}; // slices inspected through the refresh
		std::atomic<uint64_t> verify_checks {0};
		std::atomic<uint64_t> verify_mismatches {0};
	};
	DccRefreshTotals m_dcc_refresh_totals;
	// Caller holds m_lock, right after RecordSlice of the refreshed image.
	void RecordDccRefreshVerify(Buffer& metadata, uint64_t offset, uint64_t slice_size,
	                            std::vector<uint8_t> before, std::vector<uint8_t> clears,
	                            GuestRange range);
	struct GpuWriteSkipTotals {
		std::atomic<uint64_t> skips {0};
		std::atomic<uint64_t> verify_checks {0};
		std::atomic<uint64_t> verify_races {0};
		std::atomic<uint64_t> verify_mismatches {0};
	};
	GpuWriteSkipTotals m_gpu_write_skip_totals;
	// KYTY_IMAGE_LRU_SKIP (default on; =0 off): TouchImage skips the LRU call when the image's
	// mirror of its item's tick (Image::lru_tick) is not older than m_gc_tick: the image was
	// touched in this GC tick already, and LeastRecentlyUsedCache::Touch would return at once.
	// The LRU order and every GC decision stay the same; only the item's cache line is not read.
	// KYTY_IMAGE_LRU_SKIP_VERIFY=1|exit checks the mirror against the item on every decision of
	// TouchImage for a registered image (linked, equal ticks); a mismatch takes today's path (the
	// touch) and is counted and logged (exit stops). FrameEvents ImageLruTouchSkips,
	// ImageLruVerify{Checks,Mismatches}.
	[[nodiscard]] bool LruMirrorHolds(Image& image, bool skip);
	bool m_lru_touch_skip        = true;
	int  m_lru_touch_skip_verify = 0;
	// Written by the thread that touches images (the GPU thread; GetImage touches without m_lock)
	// with a relaxed load and store, read by tests.
	struct LruTouchTotals {
		std::atomic<uint64_t> skips {0};
		std::atomic<uint64_t> touches {0};
		std::atomic<uint64_t> verify_checks {0};
		std::atomic<uint64_t> verify_mismatches {0};
	};
	LruTouchTotals m_lru_touch_totals;
	enum class ResidencyMode : uint8_t { Off, On, Poison };
	ResidencyMode                            m_residency            = ResidencyMode::On;
	// KYTY_TEXTURE_SPARSE_RESIDENCY: partially resident textures are sparse residency images.
	bool                                     m_sparse_textures      = false;
	uint64_t                                 m_residency_violations = 0;
	uint64_t                                 m_residency_overlap_logs = 0;
	// Partially resident images (stale ids are dropped by the once-per-frame scan).
	std::vector<ImageId>                     m_partial_images;
	uint64_t                                 m_partial_scan_frame   = 0;
	uint64_t                                 m_resident_idle_frames = 30;
	// KYTY_TEXEL_SYNC_SKIP=0 downloads image contents for every texel-buffer read.
	bool                                     m_texel_sync_skip = true;
	[[nodiscard]] StreamBuffer& StagingRing();
	// Structural generations for TextureBindingMemo (pipeline/textureBindingMemo.h), one per
	// ImagePageTable page: NoteStructureChange(image) bumps every page the image's registered
	// range covers. Called by RegisterImage and UnregisterImage (every change of a page's owner
	// list or of an image's registered flag, i.e. whenever FindImage's first-page lookup on that
	// page may answer differently) and when a stencil association is attached. Changing a
	// registered image's SameBacking fields (address, size, extent, resources, samples, block
	// size, tile mode, format, type) in place must call it too. Caller holds m_lock.
	std::unique_ptr<uint64_t[]> m_page_versions;
	void NoteStructureChange(const Image& image) {
		ImagePageTable::PageRange pages {};
		if (!ImagePageTable::TryGetPageRange(image.live.address, image.live.size, pages)) {
			return;
		}
		if (!m_page_versions) {
			m_page_versions = std::make_unique<uint64_t[]>(
			    size_t {1} << (ImagePageTable::kAddressSpaceBits - ImagePageTable::kPageBits));
		}
		for (auto page = pages.first; page < pages.last_exclusive; ++page) {
			++m_page_versions[page];
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::TextureCacheStructureChanges);
	}
	[[nodiscard]] uint64_t PageVersion(uint64_t page) const noexcept {
		return m_page_versions ? m_page_versions[page] : 0;
	}

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
	friend class TextureBindingMemo;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
