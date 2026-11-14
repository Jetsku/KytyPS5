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
#include <string>
#include <deque>
#include <functional>
#include <bitset>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class NanScrubber;
class BufferCache;

// KYTY_META_CLEAR_COVER / KYTY_META_TRACE (textureCache.cpp).
[[nodiscard]] bool     MetaClearCoverEnabled();
[[nodiscard]] bool     StencilPlaneFillClearEnabled();
[[nodiscard]] uint64_t MetaTraceAddress();
void                   SetMetaTraceAddressAuto(uint64_t address);
void                   SetMetaTraceAddressStencil(uint64_t address);
[[nodiscard]] bool     MetaTraceLine();
// Null images (adapted from Erlou93/KytyPS5-Yotei NullTextureDesc / GetNullImage / NullCompareBinding;
// textureCache.cpp). KYTY_NULL_TEXTURE_ZERO=1: a sampled null float texture is an RGBA32F image and
// every null colour image is cleared to (0, 0, 0, 0), what the hardware returns for a null T#.
// KYTY_NULL_COMPARE_DEPTH=1: a null image on a depth-compare binding is a 1x1 D16 cleared to the
// depth that fails the S#'s comparison (SetNullCompareFuncs: DEPTH_COMPARE_FUNC bits of the binding).
[[nodiscard]] bool     NullTextureZeroEnabled();
// KYTY_STENCIL_RESTORE_GUARD=1 (Erlou93/KytyPS5-Yotei): after a stencil plane swap restored the
// plane, its record is uploaded only for a GPU buffer write, not a CPU-dirty mark (FindDepthTarget).
[[nodiscard]] bool     StencilRestoreGuardEnabled();
[[nodiscard]] bool     ImageContainsEnabled();
[[nodiscard]] bool     NullCompareDepthEnabled();
void                   SetNullCompareFuncs(uint32_t funcs);
class CommandBuffer;
class CommandScheduler;
class DccClearHelper;
class RenderExecutor;
class StagingCopier;
class TextureBindingMemo;
struct TextureCacheTestAccess;

class TextureCache {
public:
	// Bindless table hooks (Senaxx; RenderContext sets them). DeleteImage, on the GPU thread, calls
	// on_bindless_unregister for a bindless-pinned image and then on_image_views_dropped with the
	// image's views, before it unregisters it; its destruction waits for the current tick.
	// replaced: another image already holds the dropped image's contents (ExpandImage), so the
	// keys that sampled it can be resolved again at once.
	std::function<void(ImageId, bool replaced)>           on_bindless_unregister;
	std::function<bool(std::span<const CachedImageView>)> on_image_views_dropped;

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
	// Diagnostic (KYTY_DUMP_CS_IMAGES, renderCompute.cpp): records a copy of mip 0 / layer 0 of the
	// colour image whose guest base is `address` into a download buffer and, once the tick
	// completed, writes its texels to <prefix>_<address>_<w>x<h>_vk<format>_b<bytes>.raw. Returns
	// false when no such image is cached (or it is a depth/MSAA/compressed image). GPU thread, at
	// an emission safe point.
	// width/height 0: any extent; otherwise an image of that extent is preferred.
	bool DebugDumpImage(uint64_t address, uint32_t width, uint32_t height, const std::string& prefix);
	// KYTY_NAN_SCAN (renderCompute.cpp, diagnostics): downloads a band of rows of the cached image at
	// `address` (layer 0, mip 0) after the current recording and counts NaN/Inf texels of float
	// formats when the submission completed; the line names `label` (op order, shader, image).
	// False when the image is not cached or its format cannot hold NaN.
	bool DebugNanScan(uint64_t address, uint32_t width, uint32_t height, const std::string& label);
	// One image of DebugDumpImage (m_lock held).
	bool DebugDumpOne(ImageId id, uint64_t address, const std::string& prefix);
	// KYTY_PASS_DUMP (renderCompute.cpp, diagnostic): samples every `step`-th row and column of the
	// cached image at `address` (layer 0, mip 0; `aspect` 0 colour, 1 depth, 2 stencil of a depth
	// image; an image of want_width x want_height preferred) after the current recording. When the
	// submission completed, `emit` gets one text record: extent, format, per-channel mean/min/max,
	// NaN/Inf and all-zero counts, a hash of the sampled bytes and a 16x9 grid of region means; with
	// `thumb_path` the sampled texels are also written there as float RGBA. False when no such
	// image is cached or the format is block-compressed / multisampled.
	bool DebugPassSample(uint64_t address, uint32_t want_width, uint32_t want_height, int aspect, uint32_t step,
	                     const std::string& thumb_path, std::function<void(std::string)> emit);
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
	// KYTY_IMG_STATS / KYTY_IMG_OVERLAP: how the last FindImage reached its answer when the first-page
	// lookup found nothing. `pure`: every ResolveOverlap branch it took returned without an effect
	// or a time-dependent decision, and nothing was inserted, so the same lookup over the same
	// owner lists of the region's pages returns `result` again with nothing else done.
	struct OverlapLookup {
		bool     overlap        = false;
		bool     pure           = false;
		bool     inserted       = false;
		bool     first_page     = false; // the first-page lookup answered (no overlap loop)
		// Candidates whose branch depended on time (safe_to_delete) and was decided "keep, do
		// nothing" (safe_to_delete false): the same branch is taken again exactly while
		// OverlapSafeToDelete stays false for each (checked live by the memo). More than
		// MaxTimed such candidates make the lookup impure.
		static constexpr uint32_t MaxTimed = 4;
		uint32_t                  timed_count = 0;
		std::array<ImageId, MaxTimed> timed {};
		// The first reason the loop was impure (KYTY_IMG_OVERLAP counters).
		enum Why : uint8_t { WhyNone, WhyTimedTaken, WhyTimedMany, WhyDepth, WhyExpand, WhyMerge, WhyContains };
		uint8_t impure_why = WhyNone;
		// KYTY_IMG_DEPTHALIAS: ResolveDepthOverlap answered at least once without recreating (the
		// cached image itself, or a kept alias found by its inner region scan). Such a lookup is pure
		// apart from the alias branch's one effect, listed in `rebind`: the cached (depth) image gets
		// needs_rebind (and the image-state epoch moves) when it is bound or a target at that moment
		// (`rebind_applied`: what the lookup did). The inner scans' pages are [span_first, span_last).
		bool                          depth_class = false;
		static constexpr uint32_t     MaxRebinds  = 2;
		uint32_t                      rebind_count = 0;
		std::array<ImageId, MaxRebinds> rebind {};
		std::array<bool, MaxRebinds>    rebind_applied {};
		uint64_t                      span_first = UINT64_MAX;
		uint64_t                      span_last  = 0;
		ImageInfo requested;             // the description's info as FindImage got it
		uint32_t candidates     = 0;
		uint64_t region_address = 0;
		uint64_t region_size    = 0;
		ImageId  result;
	};
	[[nodiscard]] const OverlapLookup& LastOverlap() const noexcept { return m_last_overlap; }
	// ResolveOverlap's safe_to_delete for a candidate image, as it would be decided now (GPU
	// thread; caller holds m_lock).
	[[nodiscard]] bool OverlapSafeToDelete(const Image& cached) const;
	// Marks the current lookup's overlap loop impure, keeping the first reason.
	void NoteOverlapImpure(uint8_t why);
	// KYTY_BLOCK_SUBRESOURCE_WRITES (FindImage's ApplyBlockSubwrites runs for every lookup).
	[[nodiscard]] static bool BlockSubwritesOn();
	// KYTY_ALIAS_SYNC_SKIP (RestoreContentIfUnwritten restores a serial only with it).
	[[nodiscard]] static bool AliasSyncSkipOn();
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
	// KYTY_SPARSE_HOLES=1 (default off; adapted from usa-ratt/KytyPS5-GT7 89be9768d): a depth image
	// over memory only partly mapped (sparse, in 64 KB blocks), sampled with a depth comparison,
	// has the texels of its unmapped blocks reset to `empty`, the depth that passes the comparison
	// (1.0 for LESS/LEQUAL, 0.0 for GREATER/GEQUAL), once each time its contents change. The
	// console drops the writes that land there and reports reads there as not resident, which the
	// shaders sampling such shadow maps take as lit. Without it the host's texels there, drawn
	// into, shadow the ground in noisy patches with straight edges (GT7).
	void               ResetUnmappedDepth(ImageId id, float empty);
	// KYTY_NAN_SCRUB=1 (textureCache.cpp), GPU thread, at a guest flip's preparation: one changed
	// GPU-written B10G11R11 or half-float array of six layers or more has its texels that are not
	// finite rewritten (a NaN to zero, an infinity to the largest finite value).
	void               RunNanScrub();
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
	// KYTY_SRT_STALE_IMAGE_READS (kernel/memory.cpp): whether a GPU-modified image is still the
	// source of some guest byte of the range: it owns bytes there and no CPU write superseded it
	// (a definitely CPU-dirty image without kept ranges is refreshed whole from guest memory and is
	// no source of guest bytes, as MaterializeOwnedBytes treats it). IsRegionGpuModified asks only
	// whether a GPU-modified image overlaps the range.
	[[nodiscard]] bool IsRegionGpuOwnedForRead(uint64_t address, uint64_t size);
	// KYTY_SRT_STALE_IMAGE_READS: moves the bytes of the range that GPU-modified images own into the
	// buffer cache (MaterializeOwnedBytes, KYTY_ALIAS_BYTES), so a GPU-thread read can download them
	// like any GPU-written buffer bytes. GPU thread; records GPU work. Returns the bytes moved.
	uint64_t MaterializeForRead(uint64_t address, uint64_t size);
	// Diagnostics (KYTY_INDIRECT_ARGS_DIAG / KYTY_READINESS_DIAG): the GPU-modified images over the
	// range, "image 0x<addr>+0x<size> <format> <w>x<h> ..." joined by "; " (empty when none).
	[[nodiscard]] std::string DescribeGpuModifiedImages(uint64_t address, uint64_t size);
	// KYTY_BDA_SETTLE_IMAGE_STARTS (BufferCache::SettleBdaWrittenRange): InvalidateMemoryFromGPU
	// restricted to the images whose first byte lies in [address, address + size) (Erlou93/
	// KytyPS5-Yotei's rule, which acts on images starting at the written address only). GPU thread.
	void InvalidateMemoryFromGPUImageStarts(uint64_t address, uint64_t size);
	// Whether a registered image has `info`'s backing (FindImageWithSameBacking under m_lock;
	// descriptors.cpp HoldLayerRange / HoldFromBaseLevel).
	[[nodiscard]] bool HoldsImage(const ImageInfo& info);
	// Whether an image over the range was written by the GPU (KYTY_TEXTURE_REDUCE leaves those whole).
	[[nodiscard]] bool HasGpuWrittenImageOverlap(uint64_t address, uint64_t size);
	// Whether any cached image overlaps the range (RenderContext::UnmapMemory: CPU-only unmaps skip
	// the GPU drain).
	[[nodiscard]] bool IsRegionRegistered(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	// KYTY_CMASK_MSAA: a colour target's CMASK starts at this address (MaterializeCmaskClear).
	[[nodiscard]] bool IsColorMeta(uint64_t address);
	// KYTY_META_CLEAR_MEMO=1 (default off; BryanKAdams/KytyPS5 c36bbff): the last answer is kept for
	// its address while m_surface_meta_generation holds, without m_lock. Called by the GPU (CP)
	// thread only (the depth-target resolution of draws).
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	// KYTY_HTILE_SLICE_CLEAR=1 (adapted from Erlou93/KytyPS5-Yotei FIX_HTILE_SLICE_CLEAR, M122;
	// PPSA26344 preset, default off; =0 is ClearMeta(address)): a clear of `size` bytes at a
	// layered HTILE clears the layers it covers only. At the HTILE's base address the first
	// size / slice-size layers (the other layers keep their state; ClearMeta takes every layer as
	// cleared); inside it (a single layer's or a few layers' HTILE bytes) those layers, which
	// ClearMeta did not see at all. True only for a clear at the base (the caller may consume it).
	[[nodiscard]] bool ClearMetaRange(uint64_t address, uint64_t size);
	// KYTY_HTILE_SIZE_RULE=1 (Erlou93/KytyPS5-Yotei ClearMetaWrite): a compute write at an HTILE base
	// larger than that HTILE is an ordinary write, not a clear: the entry is erased and true returned.
	[[nodiscard]] bool OversizedMetaWrite(uint64_t address, uint64_t size);
	[[nodiscard]] static bool HtileSizeRuleEnabled();
	[[nodiscard]] static bool HtileSliceClearEnabled();
	// KYTY_META_CLEAR_COVER: marks cleared every surface metadata entry that starts inside
	// (address, address + size) (the exact start is ClearMeta's); returns how many.
	[[nodiscard]] uint32_t ClearMetaCovered(uint64_t address, uint64_t size);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	void ProcessDownloadImages();
	// With storage readback (Config::ReadbackStorageImagesEnabled, upstream 9678c59b7), a GL2
	// writeback makes GPU-written images visible to the CPU; returns whether a download was
	// recorded.
	[[nodiscard]] bool WriteBackImages();
	// The tick of the latest storage-readback write-back pending over the range's pages, or 0.
	[[nodiscard]] uint64_t WriteBackTick(uint64_t address, uint64_t size);
	// Called once per completed guest flip. Alias lifetimes are measured in presented frames:
	// a frame spans hundreds of scheduler ticks, so tick age alone treats every alias used
	// earlier in the same frame as stale.
	void AdvanceFrame() noexcept { m_frame.fetch_add(1, std::memory_order_relaxed); }
	[[nodiscard]] uint64_t Frame() const noexcept { return m_frame.load(std::memory_order_relaxed); }
	// The image of a slot id, or nullptr when the slot is free (GPU thread).
	[[nodiscard]] Image* TryGetImage(ImageId id) noexcept { return m_slot_images.try_get(id); }
	// KYTY_MS_DEPTH_SAMPLE0=1 (from usa-ratt/KytyPS5-GT7 rounds 122-124): a texture binding that
	// reads a multisample depth image through a single-sample dimension (the shader then has a
	// single-sample image: ResourceMaterialization's DescriptorDimension) reads a single-sample
	// copy of its first sample, as the console's texture unit does. SingleSampleDepth: a D32
	// image of the depth (each layer of an array), SingleSampleStencil (argument: the depth
	// image): an R8_UINT image of the stencil. The copy has no guest memory (never registered),
	// is refreshed when its source's content serial moved, and goes with its source image. Any
	// other image (or the switch off): the argument. GPU thread.
	[[nodiscard]] static bool MsDepthSample0Enabled();
	[[nodiscard]] ImageId SingleSampleDepth(ImageId id);
	[[nodiscard]] ImageId SingleSampleStencil(ImageId depth_id);
	// KYTY_PRESERVE_STENCIL=1 (adapted from Erlou93/KytyPS5-Yotei WORKAROUND_PRESERVE_STENCIL,
	// STENCIL_PLANE_SYNC and FIX_STENCIL_PLANE_WRITES, M119; PPSA26344 preset, default off). The
	// PS5 keeps each stencil plane in its own memory; the host keeps one stencil aspect per depth
	// image. When a depth target is bound with another stencil plane than the one its stencil
	// aspect holds, the held plane is saved (a GPU buffer per depth address, plane and extent) and
	// the new plane's saved copy, if any, is restored (SwapStencilPlane). A shader read of a plane
	// the depth image does not hold now reads that plane's saved copy (StencilPlaneReader) instead
	// of the other plane, and a refresh of such a plane from guest memory waits until the plane is
	// swapped back in, so a guest write to it no longer lands on the plane the depth image holds
	// (Erlou: the game's fill of one plane overwrote the water bits of another: no water lighting).
	// Single-sample, one-layer depth images only. GPU thread.
	[[nodiscard]] static bool PreserveStencilEnabled();
	// The image a sampled read of the stencil plane `record_id` (a stencil association record)
	// binds: its depth image, or (copy = true) a copy of the plane's saved contents.
	[[nodiscard]] ImageId StencilPlaneReader(ImageId record_id, bool& copy);
	// KYTY_STENCIL_READ_DEPTH=1 (default off): sampled stencil-plane reads bind the depth image (see
	// the .cpp). StencilPlaneDepth: the single-sample depth image that swaps to the plane at this
	// address (SwapStencilPlane saw it), or none.
	[[nodiscard]] static bool StencilReadDepthEnabled();
	// KYTY_STENCIL_READ_HELD (default on, see the .cpp): with KYTY_STENCIL_READ_DEPTH a plane record
	// the depth image does not hold is still read from its saved copy.
	[[nodiscard]] static bool StencilReadHeldEnabled();
	// KYTY_STENCIL_PLANE_FILL_SAVED (default on): a fill of a plane the depth image does not hold
	// fills the plane's saved copy (ClearImageFromBuffer).
	[[nodiscard]] static bool StencilPlaneFillSavedEnabled();
	// KYTY_STENCIL_PLANE_SYNC=2 (default; =0 off; based on Erlou93/KytyPS5-Yotei
	// WORKAROUND_STENCIL_PLANE_SYNC level 2): at swap-in, a 1-byte colour image at the plane's address
	// and the depth image's extent that a shader or clear wrote after the plane's saved copy is the
	// plane's contents (loaded instead of the stale copy).
	[[nodiscard]] static int StencilPlaneSyncLevel();
	bool FillStencilPlaneSaved(CommandBuffer& command, ImageId depth_id, ImageId record_id, uint8_t value,
	                           const char* what);
	[[nodiscard]] ImageId     StencilPlaneDepth(uint64_t plane_address);
	// KYTY_STENCIL_READ_PARTNER=1 (default off; Erlou93/KytyPS5-Yotei parity, see the .cpp): a sampled
	// read redirected by KYTY_STENCIL_READ_DEPTH binds the plane the depth image swapped away from
	// (StencilPlanePartnerRecord: its stencil association record, or none).
	[[nodiscard]] static bool StencilReadPartnerEnabled();
	[[nodiscard]] ImageId     StencilPlanePartnerRecord(uint64_t plane_address);
	// KYTY_STENCIL_ALIAS_COPY=1 (adapted from Erlou93/KytyPS5-Yotei WORKAROUND_STENCIL_ALIAS_COPY,
	// a logged approximation for dynamic resolution; PPSA26344 preset, default off): a 1-byte
	// image at a stencil plane's address with another extent than its depth image (the plane read
	// at another resolution) is a separate image nothing writes; before it is sampled the
	// overlapping top-left region of the plane (the depth image's stencil aspect, or the plane's
	// saved copy with KYTY_PRESERVE_STENCIL) is copied into it. True when it copied.
	[[nodiscard]] static bool StencilAliasCopyEnabled();
	// KYTY_STENCIL_STORAGE_SHADOW=1 (adapted from Erlou93/KytyPS5-Yotei
	// WORKAROUND_STENCIL_STORAGE_SHADOW; PPSA26344 preset, default off): a storage view of a depth
	// image's stencil aspect is invalid Vulkan (no storage usage). The binding gets an R8_UINT
	// shadow holding the stencil aspect's contents instead; after the dispatch the shadow is copied
	// back into the stencil aspect (FlushStencilStorageShadows). Single-sample one-layer depth
	// images whose stencil holds the bound plane only; anything else returns the depth image.
	[[nodiscard]] static bool StencilStorageShadowEnabled();
	// KYTY_PRESERVE_STENCIL_RECREATE=1 (Erlou93/KytyPS5-Yotei WORKAROUND_PRESERVE_STENCIL's
	// SaveStencil/RestoreStencil; default off): a depth/stencil image replaced by a colour image
	// at its memory (a compute shader writes the depth as R32 storage) is freed, and the depth
	// image made again when the game binds the depth target copies only the depth aspect back: the
	// stencil aspect of the new image was undefined (RenderDoc: "UNDEFINED IMG" at the torii's
	// Water Lighting, whose classifier then lit tiles in the sky). The replaced image's stencil is
	// saved and written into the next depth image made at the same address and extent.
	// Single-sample, one-layer depth images only (as KYTY_PRESERVE_STENCIL).
	[[nodiscard]] static bool PreserveStencilRecreateEnabled();
	[[nodiscard]] ImageId StencilStorageShadow(ImageId record_id, bool& shadow);
	[[nodiscard]] bool HasStencilStorageShadows() const noexcept { return !m_pending_stencil_shadows.empty(); }
	void FlushStencilStorageShadows();
	bool CopyStencilPlaneToAlias(ImageId target_id);
	// KYTY_SUB_EXTENT_COPY=1 (from usa-ratt/KytyPS5-GT7 429692dd3, Franseven's case): a texture
	// narrower or shorter than the single-sample colour image at its address, with the same guest
	// layout (same memory size and tiling: UI text drawn with a narrower viewport into a wider
	// target, then sampled at the narrower width). A view cannot shrink the extent, so normalized
	// coordinates would reach the image's columns past the texture's: the texture reads a copy of
	// the image's corner of its extent, made again whenever the image changed (content serial).
	// The image keeps all its pixels. Any other request (or the switch off): `id`. GPU thread.
	[[nodiscard]] static bool SubExtentCopyEnabled();
	[[nodiscard]] ImageId SubExtentCopy(ImageId id, const ImageInfo& requested);
	// KYTY_LIGHT_GRID_CAP=1 (default off; the GT7 preset; usa-ratt/KytyPS5-GT7 ecef51e59): a pixel
	// shader's light grid (R32G32_UINT linear array of two or more layers, read-only;
	// descriptors.cpp) has its shape remembered here, and uploads of linear R32G32_UINT arrays of
	// a shape seen are checked for cells whose packed count asks for more than 1024 lights, which
	// game data never does: the shader then loops over up to 65535 lights a pixel, tens of
	// milliseconds a draw (GT7's reflection freezes). Such cells are uploaded empty when the upload
	// comes from guest bytes (not from a GPU-written cache buffer). Any thread.
	[[nodiscard]] static bool LightGridCapEnabled();
	static void NoteLightGridShape(const ImageInfo& info);
	// KYTY_LIGHT_GRID_PROBE=1 (diagnostics, default off): every light grid a pixel shader binds
	// (descriptors.cpp) has its guest bytes scanned at bind time and logged (garbage cells, the
	// largest and summed light counts). Any thread.
	[[nodiscard]] static bool LightGridProbeEnabled();
	// KYTY_LIGHT_GRID_GUEST=1 (default off; needs KYTY_LIGHT_GRID_CAP): a light grid's image is
	// compared with its guest bytes once per guest submission and uploaded from them (repaired)
	// when they changed, even when GPU writes of other data covered the grid's memory, which
	// otherwise makes the refresh take the stale GPU-written bytes of the buffer cache.
	[[nodiscard]] static bool LightGridGuestEnabled();
	[[nodiscard]] static bool LightGridGuestApplies(const Image& image);
	static void ProbeLightGrid(const ImageInfo& info, uint64_t shader_hash);
	void RunGarbageCollector();
	// KYTY_BINDLESS_GUEST_DROP_FRAMES: no heap key refers to this bindless image any more (the
	// guest replaced or cleared its entries): free it once no draw has used it for that many
	// frames, whatever the memory pressure (RunGarbageCollector). Any thread but m_lock's holder.
	void ReleaseDroppedByGuest(ImageId id);
	// KYTY_BINDLESS_EVICT: bindless images no draw sampled during the last usage probe
	// (RenderExecutor::UpdateBindlessUsageProbe); replaces the previous list. They stay pinned,
	// but while the cache is under pressure RunGarbageCollector frees the clean, unbound ones.
	// Returns their accounted bytes. GPU thread; takes m_lock.
	uint64_t SetBindlessEvictable(std::span<const ImageId> ids);
	// Whether the cache's images (or, with a reported budget, the device's usage) are at its
	// pressure threshold.
	[[nodiscard]] bool BindlessUnderPressure() const noexcept {
		return m_total_used_memory >= m_pressure_gc_memory;
	}
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

	// KYTY_BINDLESS_HOLD_EXTENT (adapted from upstream 95bbc9c65): whether a bindless heap lookup
	// of `info` should wait instead of creating an image. Heaps hold a descriptor per
	// dynamic-resolution extent of a render target, and only one extent at an address holds data:
	// true when no registered image has the lookup's backing but a registered bindless-pinned
	// image of the same type at the same address has another extent. Takes m_lock.
	[[nodiscard]] bool HeapExtentHeld(const ImageInfo& info, bool exact_format);

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

		Type type;
		// PS5 DB_DEPTH_VIEW encodes 13-bit base and last array-slice indices.
		std::bitset<8192> clear_mask;
		// KYTY_HTILE_SLICE_CLEAR: one layer's HTILE bytes and the layers, from a depth target bound
		// with all of its image's layers (0 = not known).
		uint64_t slice_size = 0;
		uint32_t layers     = 0;
		// KYTY_HTILE_SIZE_RULE: the largest HTILE size a depth target bound with it (0 = not known).
		uint64_t size       = 0;
		// The depth image that last acquired this HTile range (upstream 2d84e36b4).
		ImageId owner;
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
	// KYTY_TEXTURE_SPARSE_RESIDENCY growth: the bind ordered after every submitted tick (Bug A).
	void                      BindSparseLevelsAfterSubmitted(Image& image, uint32_t first_level);
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
	// KYTY_BLOCK_SUBRESOURCE_WRITES (caller holds m_lock).
	void                      NoteBlockSubwrite(ImageId id);
	void                      ApplyBlockSubwrites(ImageId id);
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
	// KYTY_IMAGE_GPU_KEEP (default on; =0 the whole-image path): a CPU write to, or a partial unmap
	// of, a GPU-written image replaces only those guest bytes; the image keeps the GPU's other
	// bytes and refreshes only the subresources over the replaced ones (Image::NoteKeptCpuWrite).
	// KeepEligible: the state the partial refresh needs. KeepGpuBytes marks [address, address +
	// size) (CPU write: widened to KeptWriteGranule, those pages released; unmap: exact, pages kept
	// watched) and returns false where the caller must take the whole-image path.
	[[nodiscard]] bool        KeepEligible(const Image& image) const;
	[[nodiscard]] bool        KeepGpuBytes(Image& image, uint64_t address, uint64_t size, bool unmap);
	// Uploads the subresources over image.kept_dirty from guest memory; false: nothing recorded,
	// the caller uploads the whole image.
	[[nodiscard]] PartialUploadResult UploadKeptRanges(Image& image);
	// KYTY_IMAGE_PARTIAL_GPU_WRITE (see UploadGpuWrittenRanges).
	[[nodiscard]] bool GpuWriteTrackEligible(const Image& image) const;
	[[nodiscard]] bool GpuWritePartialCandidate(const Image& image);
	void               MaterializeForRebuild(ImageId id, RefreshIntent intent);
	[[nodiscard]] PartialUploadResult UploadGpuWrittenRanges(ImageId id, RefreshIntent intent);
	[[nodiscard]] std::optional<ImageInfo> GpuWriteDepthInfo(const Image& image) const;
	void VerifyGpuWrittenRanges(ImageId id, RefreshIntent intent);
	void NoteRefreshSync(ImageId id, bool gpu_partial);
	void PrintPartialGpuWriteSummary();
	[[nodiscard]] bool        UploadKeptDepthLayers(Image& image, PartialUploadResult& result);
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
	// KYTY_DCC_GPU_ALIASED: whether images over a target's metadata keep its inspection on the CPU
	// (m_lock held).
	// *aliased: images lie over the metadata and were accepted (KYTY_DCC_GPU_WIDE / _ALIASED=2).
	[[nodiscard]] bool          MetadataAliasBlocks(ImageId id, GuestRange range, bool* aliased = nullptr);
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size,
	                                               bool page_overlap) const;
	// Caller holds m_lock. Equal backing ranges must begin in the same indexed page.
	[[nodiscard]] ImageId       FindImageWithSameBacking(const ImageInfo& requested,
	                                                     bool exact_format) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	// How the last ResolveDepthOverlap answered (KYTY_IMG_DEPTHALIAS): None (no image), Same (the
	// cached image, no effect), Alias (a kept alias of the inner region m_depth_inner_*; the
	// cached image's rebind effect applied or not: m_depth_rebind_applied), Recreate (anything else).
	enum class DepthPath : uint8_t { None, Same, Alias, Recreate };
	DepthPath m_depth_path            = DepthPath::None;
	uint64_t  m_depth_inner_address   = 0;
	uint64_t  m_depth_inner_size      = 0;
	bool      m_depth_rebind_applied  = false;
	// ResolveOverlap's record of a depth answer into m_last_overlap.
	void NoteDepthOverlap(ImageId cached_id);
	// KYTY_DEPTH_GROW_RECREATE: a depth target bound at the address of a cached one with the same
	// layout but a larger width and/or height (a dynamic-resolution target that grows inside a fixed
	// pitch). The pitch/height comparison of ResolveOverlap never recreates it, so draws covered the
	// larger extent of a smaller attachment (GT7 License Center, upstream PR #2082).
	[[nodiscard]] static bool   IsDepthGrowth(const ImageInfo& requested, const ImageInfo& cached);
	// Caller holds m_lock. A new image of the requested size: guest bytes everywhere, then the
	// cached image's native depth and stencil over its own extent; the cached image is freed.
	[[nodiscard]] ImageId       GrowDepthTarget(const ImageInfo& requested, ImageId cached);
	// KYTY_VIEW_PLACEMENT: where a binding the overlap loop found inside `result` lies in it
	// (loop_own: the loop's last overlap placed the binding itself there, at view_mip/view_layer);
	// clears `result` when it has no place there its view fits in (it gets an image of its own).
	[[nodiscard]] static bool   ViewPlacementEnabled();
	void                        PlaceBindingView(const ImageDesc& desc, ImageId& result, bool loop_own,
	                                             int32_t& view_mip, int32_t& view_layer) const;
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	// KYTY_GUEST_HASH_IMAGES_KB: small linear images the CPU writes, compared with their guest bytes
	// once per guest submission instead of trusting the page tracking alone.
	[[nodiscard]] bool          ChecksGuestContent(const Image& image);
	void                        RefreshByContent(ImageId id, RefreshIntent intent);
	// KYTY_PENDING_FILLS (BufferCache::FillOfNewImage), caller holds m_lock: a fresh target made
	// inside a known GPU fill is cleared to it; false when nothing was done (the caller uploads).
	bool                        ClearFromKnownFill(ImageId id);
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
	// Whether SyncAliasFromOwner, for `image`, would take `other` (another image of its region) as
	// the owner to copy from (its loop condition; caller holds m_lock).
	[[nodiscard]] static bool IsAliasSyncSource(const Image& other, const Image& image);
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
	// CMASK bytes (only code 0x00), not counted as DCC work. consume=false (video-out bindings,
	// KYTY_DCC_GPU_VIDEOOUT): the keys stay as they are, and no inspection is retained for reuse.
	[[nodiscard]] Profiler::FrameEvent TryMaterializeGpuMetadataClear(
	    DccClearHelper& helper, ImageId id, const ImageDesc& desc, GuestRange range,
	    uint32_t metadata_base_layer, const DccClearHelper::ClearValues& values, uint32_t decodable,
	    bool cmask, bool keep_keys = false);
	// KYTY_LIGHT_GRID_CAP: a light grid's staged upload bytes, repaired before the copy is recorded.
	void RepairLightGridUpload(const Image& image, Buffer& source, uint64_t source_offset);
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
	// KYTY_TEXEL_SYNC_MEMO: the image registration epoch (InvalidateCleanImageProofs), or 0 when it
	// is not maintained (KYTY_TEXTURE_CLEAN_PROOFS off) or saturated.
	[[nodiscard]] uint64_t RegistrationEpoch() const {
		const auto epoch = m_clean_image_epoch.load(std::memory_order_acquire);
		return m_clean_image_proofs && epoch != UINT64_MAX ? epoch : 0;
	}
	// Whether a registered image starts at `address` (whatever its state).
	[[nodiscard]] bool AnyImageStartsAt(uint64_t address);
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
	// The depth image's stencil plane record covering exactly range, or none (upstream 84419aef7).
	[[nodiscard]] ImageId FindStencilPlane(const Image& depth, GuestRange range) const;
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	// `done` (optional) is set once the downloaded bytes are in guest memory (the deferred write-back ran).
	[[nodiscard]] bool DownloadImageMemory(ImageId id, std::shared_ptr<std::atomic_bool> done = nullptr);
	// KYTY_IMAGE_WRITEBACK_GUARD: a recorded image download not yet written into guest memory. It
	// watches the pages it has not given up (`watched`); a CPU write to them before the write-back
	// releases them and marks them CPU-written, and the write-back leaves those to the CPU.
	// A storage-readback image download whose guest pages stay inaccessible until it lands
	// (upstream 9678c59b7).
	struct StorageWriteBack {
		GuestRange range;
		uint64_t   tick;
	};
	[[nodiscard]] bool DownloadQueuedImages();

	struct PendingWriteback {
		KeptRanges::List watched;
		KeptRanges::List cpu_written;
	};
	// GPU thread, m_lock held.
	[[nodiscard]] std::shared_ptr<PendingWriteback> BeginPendingWriteback(GuestRange range);
	// CPU write faults (InvalidateMemory), m_lock held.
	void NotePendingWritebackCpuWrite(uint64_t address, uint64_t size);
	// Priority worker: the download into guest memory (every byte when `pending` is null).
	void WriteBackImageBytes(GuestRange range, const uint8_t* data,
	                         const std::shared_ptr<PendingWriteback>& pending);
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
	                                      uint64_t enough, bool idle_pass = false);
	// KYTY_VRAM_IDLE_KEEP_POOLS_MB=<n> (default 0: off): the idle pass (KYTY_VRAM_IDLE_FRAMES)
	// passes over sampled-only one-level texture arrays of n MiB and more while the device budget
	// is 12 GiB or more; pressure and budget collection still take them.
	[[nodiscard]] bool KeptIdlePool(const Image& image) const;
	uint64_t           m_idle_keep_pool_bytes = 0;
	uint64_t           m_idle_frames          = 0;
	uint64_t           m_idle_next_frame      = 0;
	uint64_t           m_idle_freed           = 0;
	// KYTY_BINDLESS_GUEST_DROP_FRAMES: images the guest dropped from its texture heaps, with the
	// frame of the drop (ReleaseDroppedByGuest).
	std::vector<std::pair<ImageId, uint64_t>> m_guest_dropped;
	uint64_t                                  m_guest_dropped_freed       = 0;
	uint64_t                                  m_guest_dropped_freed_bytes = 0;
	void                                      FreeGuestDroppedImages(uint64_t frame);
	// KYTY_BINDLESS_EVICT (SetBindlessEvictable).
	std::vector<ImageId> m_bindless_evictable;
	uint64_t             m_bindless_evicted       = 0;
	uint64_t             m_bindless_evicted_bytes = 0;
	bool                 m_bindless_evicting      = false; // FreeImage frees a pinned image
	void                 FreeBindlessEvictable(uint64_t frame);
	uint64_t           m_pressure_frames      = 0;
	uint64_t           m_pressure_frame       = 0;
	uint64_t           m_pressure_freed_bytes = 0;
	// KYTY_VRAM_GC_BUDGET (graphics/host_gpu/vramBudget.h): frame-aged collection against the
	// budget once per frame. Returns true when it handled this collection.
	bool               RunBudgetGarbageCollector(uint64_t frame);
	uint64_t           m_budget_frame       = UINT64_MAX;
	uint64_t           m_budget_freed_bytes = 0;
	// KYTY_VRAM_GC_FIX (textureCache.cpp, CollectGpuWrittenImages; adapted from Erlou93/KytyPS5-Yotei):
	// GPU-written colour images the budget collector downloads into guest memory and frees once the
	// bytes are there, if the image was neither written nor used since.
	struct GcFixEviction {
		ImageId                           id;
		uint64_t                          serial = 0; // content serial when the download was recorded
		uint64_t                          frame  = 0; // frame of the recording
		uint64_t                          bytes  = 0;
		std::shared_ptr<std::atomic_bool> done;
	};
	struct GcFixStats {
		uint64_t downloads = 0, download_bytes = 0, freed = 0, freed_bytes = 0, cancelled = 0;
		uint64_t failed = 0, skip_budget = 0, verify_exact = 0, verify_mismatch = 0, verify_skipped = 0;
	};
	uint64_t           CollectGpuWrittenImages(uint64_t frame, uint64_t min_age, uint64_t enough, bool critical);
	void               FinishGcFixEvictions(uint64_t frame);
	[[nodiscard]] bool GcFixEligible(const Image& image, uint64_t frame, uint64_t min_age);
	// KYTY_VRAM_GC_FIX_VERIFY: the native texels of the image and of an image rebuilt from the
	// downloaded guest bytes are compared before the free (1: equal, 0: differ, -1: not checked).
	int                VerifyGcFixRoundTrip(ImageId id);
	[[nodiscard]] uint64_t LimitTickForAge(uint64_t frame, uint64_t min_age) const;
	std::vector<GcFixEviction>   m_gc_fix_pending;
	std::unordered_set<uint32_t> m_gc_fix_pending_ids; // ImageId indices in m_gc_fix_pending
	GcFixStats                   m_gc_fix_stats;
	uint64_t                     m_gc_fix_log_frame = 0;
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
	std::unordered_map<uint64_t, ImageId>             m_null_images;
	// SingleSampleDepth's and SingleSampleStencil's copies, by the multisample depth image they
	// copy: the copy and the source content serial it holds.
	struct SingleSampleCopy {
		ImageId  copy {};
		uint64_t serial = UINT64_MAX;
	};
	std::unordered_map<ImageId, SingleSampleCopy>      m_single_sample_depths;
	// KYTY_CMASK_MSAA: the addresses colour targets' CMASK starts at (IsColorMeta), at most 4096.
	std::unordered_set<uint64_t>                       m_color_metas;
	std::unordered_map<ImageId, SingleSampleCopy>      m_single_sample_stencils;
	// KYTY_PRESERVE_STENCIL: saved stencil planes by StencilPlaneKey (depth address, plane address,
	// extent): the buffer, its extent, the save count it holds and the reader copy made of it.
	struct SavedStencilPlane {
		std::shared_ptr<Buffer> buffer;
		uint32_t                width  = 0;
		uint32_t                height = 0;
		uint64_t                serial = 0;
		// KYTY_STENCIL_PLANE_SYNC: Image::NextContentSerial() when the copy was last written.
		uint64_t                content_serial = 0;
		ImageId                 reader {};
		uint64_t                reader_serial = 0;
	};
	std::unordered_map<uint64_t, SavedStencilPlane> m_stencil_planes;
	uint64_t                                        m_stencil_plane_serial = 0;
	// KYTY_STENCIL_READ_DEPTH: plane address -> the depth image that swapped to or from it.
	std::unordered_map<uint64_t, ImageId>           m_stencil_plane_depth;
	// KYTY_STENCIL_READ_PARTNER: plane address -> the plane the depth image held before it.
	std::unordered_map<uint64_t, uint64_t>          m_stencil_plane_partner;
	std::shared_ptr<Buffer>                         m_stencil_alias_staging;
	// KYTY_PRESERVE_STENCIL_RECREATE: the stencil of the last depth image replaced by a colour image.
	void SaveRecreatedStencil(Image& depth);
	bool RestoreRecreatedStencil(Image& depth);
	std::shared_ptr<Buffer>                         m_recreate_stencil;
	uint64_t                                        m_recreate_stencil_address = 0;
	uint64_t                                        m_recreate_stencil_plane   = 0;
	uint32_t                                        m_recreate_stencil_width   = 0;
	uint32_t                                        m_recreate_stencil_height  = 0;
	bool                                            m_recreate_stencil_valid   = false;
	// KYTY_STENCIL_STORAGE_SHADOW: shadow image per depth image, and the pairs to copy back.
	std::unordered_map<ImageId, ImageId>            m_stencil_shadows;
	std::vector<std::pair<ImageId, ImageId>>        m_pending_stencil_shadows; // shadow, depth
	// Whether the depth image of a stencil record holds the record's plane (KYTY_PRESERVE_STENCIL).
	[[nodiscard]] bool StencilPlaneHeld(const Image& record) const;
	// True when the incoming plane was restored from its saved copy (or loaded, STENCIL_PLANE_SYNC=2).
	bool SwapStencilPlane(ImageId depth_id, uint64_t next_plane);
	// SubExtentCopy's copies, by the image they copy (at most 4 extents each).
	struct SubExtentCopyEntry {
		ImageId  copy {};
		uint32_t width  = 0;
		uint32_t height = 0;
		uint64_t serial = UINT64_MAX;
		uint64_t used   = 0;
	};
	std::unordered_map<ImageId, std::vector<SubExtentCopyEntry>> m_sub_extent_copies;
	[[nodiscard]] ImageId SingleSampleCopyOf(ImageId id, bool stencil);
	// Releases an image that was never registered (a SingleSample* copy) once the GPU is done.
	void ReleaseUnregisteredImage(ImageId id);
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	std::mutex                                        m_storage_write_back_mutex;
	std::vector<StorageWriteBack>                     m_storage_write_backs;
	std::atomic<uint64_t>                             m_frame {0};
	// KYTY_DISCARDED_TARGET_SKIP (textureCache.cpp): ranges of GPU-written render/depth targets freed
	// without their pixels reaching guest memory, with the frame of the free.
	struct DiscardedTarget {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t frame = 0;
	};
	static constexpr size_t DiscardedTargetCount = 32;
	std::array<DiscardedTarget, DiscardedTargetCount> m_discarded_targets {};
	size_t                                            m_discarded_next = 0;
	void               NoteDiscardedTarget(ImageId id, const Image& image);
	[[nodiscard]] bool WasDiscardedTarget(const Image& image) const;
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
	// KYTY_IMAGE_GC_FILTER (RunGarbageCollector's submission-age walk).
	bool             m_gc_filter_candidates  = true;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t>::Cursor m_pressure_gc_cursor;
	uint64_t m_pressure_retirement_bytes = 0;
	// ExpandImage is freeing the image it copied into a new one (on_bindless_unregister's replaced).
	bool m_freeing_replaced = false;
	mutable uint32_t m_image_query_epoch      = 0;
	bool             m_direct_dirty_image_query = false;
	struct CleanImagePageProof {
		uint64_t page = 0;
		uint64_t epoch = 0;
	};
	std::array<CleanImagePageProof, 256> m_clean_image_pages {};
	// Bumped on every image register/unregister and GPU-modified transition (with clean proofs
	// on); atomic so that BufferCache's texel-sync memo can read it on the GPU thread.
	std::atomic<uint64_t> m_clean_image_epoch {1};
	bool m_clean_image_proofs = false;
	ImageLookupMode  m_image_lookup_mode = ImageLookupMode::FirstPage;
	uint64_t         m_image_lookup_checks = 0;
	uint64_t         m_image_lookup_mismatches = 0;
	// LookupSideEffects (GPU thread).
	uint64_t         m_lookup_side_effects = 0;
	// DccDecisionEffects (GPU thread).
	uint64_t         m_dcc_decision_effects = 0;
	bool             m_readback_linear_images = false;
	bool             m_readback_tiled_images  = false;
	bool             m_readback_storage_images = false;
	// Texture streaming (see TextureCache constructor for the environment switches).
	bool             m_partial_upload      = true;
	bool             m_partial_bands       = true;
	bool             m_partial_verify      = false;
	uint32_t         m_chunk_shift         = 16;
	// KYTY_IMAGE_GPU_KEEP (see KeepEligible).
	bool             m_gpu_keep            = true;
	// KYTY_IMAGE_WRITEBACK_GUARD (see WriteBackImageBytes). Lock order: m_lock, then
	// m_writeback_mutex; the priority worker takes only the latter.
	bool             m_writeback_guard     = true;
	std::mutex       m_writeback_mutex;
	std::vector<std::shared_ptr<PendingWriteback>> m_pending_writebacks;
	std::atomic<size_t> m_pending_writeback_count {0};
	uint64_t         m_overlap_keep_frames = 0;
	// Set by FindImage for the ResolveOverlap calls of one lookup: many live overlapping
	// images already share the requested range, so overlapped images are not kept.
	bool             m_overlap_crowded     = false;
	// Set by ResolveOverlap when a branch has an effect (free, expansion, merge, depth copy) or a
	// time-dependent decision (safe_to_delete, the keep window); reset by FindImage per lookup.
	bool             m_overlap_impure      = false;
	// The last FindImage's overlap path (GPU thread): see LastOverlap.
	OverlapLookup    m_last_overlap;
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
	bool m_invalidate_image_starts_only = false; // InvalidateMemoryFromGPUImageStarts
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
	// KYTY_IMAGE_PARTIAL_GPU_WRITE=1|verify, KYTY_IMAGE_PARTIAL_GPU_WRITE_STENCIL=keep.
	int  m_partial_gpu_write              = 0;
	bool m_partial_gpu_write_verify       = false;
	bool m_partial_gpu_write_stencil_keep = false;
	bool m_stencil_draw_suppressed        = false;
	// KYTY_DEPTH_GROW_RECREATE=1 (GrowDepthTarget).
	bool m_depth_grow_recreate = false;
	bool m_bind_mark                      = false; // a binding's mark right before its refresh
	struct PartialGpuWriteTotals {
		uint64_t refreshes       = 0; // refreshes of buffer-modified images
		uint64_t untracked       = 0; // written ranges not tracked
		uint64_t stale           = 0; // tracked, but other changes since:
		uint64_t stale_foreign   = 0; //   another image wrote its bytes / not in sync
		uint64_t stale_serial    = 0; //   its native contents were written
		uint64_t stale_cpu       = 0; //   CPU writes
		uint64_t partial         = 0; // partial rebuilds
		uint64_t partial_bytes   = 0; // tiled bytes they rebuilt
		uint64_t whole_bytes     = 0; // bytes the whole rebuilds would have read
		uint64_t fallback_layout = 0;
		uint64_t fallback_most   = 0;
		uint64_t fallback_others = 0;
		uint64_t overflows       = 0;
	};
	PartialGpuWriteTotals m_pgw_totals;
	PartialGpuWriteTotals m_pgw_printed;
	uint64_t              m_pgw_print_ns = 0;
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
	// KYTY_DCC_GPU_WIDE=<mask> (default 0 = off, needs KYTY_DCC_GPU=1): GPU-written colour metadata
	// that the native inspection used to refuse, so that the CPU fallback drained the GPU for it
	// (a command-processor wait for the current recording, ~2 ms each, 6-8 ms per GT7 race frame),
	// is inspected on the GPU too. The CPU fallback reads the metadata from the buffer cache
	// (BufferCache::ReadMemory downloads buffer bytes only, never an image's), and the inspection
	// reads the same buffer bytes in queue order, so both take the same decision; the image layer
	// cleared and the keys consumed are the ones the fallback's ClearImage + key write would touch.
	//  1: images registered over the metadata bytes (any, also GPU-modified ones; not the target
	//     itself) no longer refuse (KYTY_DCC_GPU_ALIASED=2 does the same for DCC only);
	//  2: the image found for the binding differs from the description (data range or extent, e.g.
	//     a texture view of a larger target): the image's whole layer is inspected and cleared, as
	//     the fallback's ClearImage(id) clears it;
	//  4: video-out bindings (flips): inspected without consuming the keys (the fallback leaves a
	//     flip's keys as they are), never reused from an earlier inspection.
	//  8: with 4, also video-out surfaces with DCC compression (GT7's 3840x2160 flips, whose
	//     fallback was a side readback of their metadata at every flip, on the command processor).
	// KYTY_DCC_GPU_WIDE_VERIFY=1|exit: for an inspection accepted only through these, also take the
	// CPU fallback's decision (after a drain) and compare it with the inspection's decision copied
	// out behind it: "DCC wide verify" mismatches are counted and logged (exit stops).
	uint32_t m_dcc_gpu_wide        = 0;
	int      m_dcc_gpu_wide_verify = 0;
	[[nodiscard]] uint32_t DccGpuWide() const;

public:
	[[nodiscard]] static int64_t ParseDccGpuWide(const char* value);

private:
	struct DccWideTotals {
		std::atomic<uint64_t> aliased {0};   // inspections accepted with images over the metadata
		std::atomic<uint64_t> mismatch {0};  // inspections of an image that differs from the binding
		std::atomic<uint64_t> video_out {0}; // flip inspections (keys kept)
		std::atomic<uint64_t> verify_checks {0};
		std::atomic<uint64_t> verify_mismatches {0};
	};
	DccWideTotals m_dcc_wide_totals;
	// Caller holds m_lock, after the slices' RecordSlice copied their decisions to `decisions`
	// (16 bytes per slice, download ring): compares them with the fallback's (`expected`) at
	// completion.
	void RecordDccWideVerify(const uint8_t* decisions, uint64_t download_offset, std::vector<uint8_t> expected,
	                         GuestRange range, uint32_t type);
	void ReportDccWide();
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

	// KYTY_NAN_SCRUB: the arrays the GPU wrote (by address), when each was last scrubbed (content
	// serial, frame), the round-robin position, the last frame a scrub ran, its pipeline and scratch.
	struct NanScrubCandidate {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t serial  = UINT64_MAX;
		uint64_t frame   = 0;
	};
	void                           NoteNanScrubCandidate(const Image& image);
	std::vector<NanScrubCandidate> m_nan_candidates;
	size_t                         m_nan_next  = 0;
	uint64_t                       m_nan_frame = UINT64_MAX;
	std::unique_ptr<NanScrubber>   m_nan_scrubber;
	std::shared_ptr<Buffer>        m_nan_scratch;

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
	friend class TextureBindingMemo;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
