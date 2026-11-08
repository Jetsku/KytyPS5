#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include "graphics/host_gpu/renderer/drawPrep/ownedCounters.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/hangTrace.h"
#include "common/liveSwitch.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/rendererBatch.h"
#include "common/skipInvalid.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/cache/bufferLookupStats.h"
#include "graphics/host_gpu/renderer/cpCommit.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawPrep/bindingPlan.h"
#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/host_gpu/renderer/drawPrep/xframeReuse.h"
#include "graphics/host_gpu/renderer/drawPrep/drawRun.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/lodStats.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/guest_gpu/command_processor/passCensus.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/syncEpoch.h"
#include "graphics/host_gpu/watchRange.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/WriteRangeAnalysis.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

// Texture edge cases adapted from Erlou93/KytyPS5-Yotei (descriptors.cpp; the PPSA26344 preset turns
// them on, default off). Each one binds a T# the strict checks below bound as null:
//   KYTY_TEXTURE_MIN_LOD_CLAMP=1: a MIN_LOD past the last mip (streaming textures publish 4095 before
//     their mips are resident) is clamped to the last level, as the sampler does.
//   KYTY_TEXTURE_BASE_LAYER_CLAMP=1: a 1D/2D view of a one-layer image whose BASE_ARRAY names a layer
//     the image does not have reads layer 0 (RDNA2 ISA table 45 lists BASE_ARRAY for array, 3D, cube
//     and MSAA resources only).
//   KYTY_TEXTURE_MISALIGNED_BASE=1: a base less aligned than its swizzle block but 256-byte aligned
//     (the T# base is only required to be 256-byte aligned) is bound at its address.
//   KYTY_DEPTH_ENCODING_FIELDS=1: a sampled depth T# that differs from the known encoding only in
//     fields the texture unit does not use for a non-array 1D/2D resource is bound (=2 also binds
//     any other encoding, Erlou's DEPTH_ENCODING_LENIENT approximation; logged).
bool EnvFlag(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] == '1';
}

// KYTY_NULL_COMPARE_DEPTH (Erlou93/KytyPS5-Yotei ImageCompareFuncs): DEPTH_COMPARE_FUNC values, as
// 1 << func, of the S#s that compare-sample image `index` (a GPU-selected candidate is sampled with
// the S#s of its table root); 0 when the image has no compare sample or the switch is off.
uint32_t ImageCompareFuncs(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                           const ShaderRecompiler::IR::ResourceSnapshot& snapshot, uint32_t index) {
	if (!NullCompareDepthEnabled()) {
		return 0u;
	}
	const auto& images = program.info.images;
	if (index >= images.size() || !images[index].depth_compare) {
		return 0u;
	}
	const auto root = images[index].indirect_root != ShaderRecompiler::IR::ImageResource::NoIndirectImage
	                      ? images[index].indirect_root
	                      : index;
	const auto& samplers = program.info.samplers;
	uint32_t    funcs    = 0;
	for (const auto& pair: program.info.sampled_pairs) {
		if ((pair.image != index && pair.image != root) || pair.sampler >= samplers.size()) {
			continue;
		}
		const auto slot = samplers[pair.sampler].snapshot_index;
		if (slot < snapshot.samplers.size()) {
			funcs |= 1u << ((snapshot.samplers[slot].dwords[0] >> 12u) & 7u);
		}
	}
	return funcs;
}

// KYTY_TEXTURE_CLAMP_LAST_LEVEL=1 (Erlou93/KytyPS5-Yotei, default on there): a LAST_LEVEL past the
// allocated MAX_MIP whose mip view cannot be remapped is cut to the allocated mips (else null).
bool TextureClampLastLevel() {
	static const bool on = EnvFlag("KYTY_TEXTURE_CLAMP_LAST_LEVEL");
	return on;
}

// KYTY_DCC_NO_META_RAW=1 (Erlou93/KytyPS5-Yotei DIAG_TEXTURE_DCC_WITHOUT_METADATA): a DCC-compressed T#
// with META_ADDR 0 is sampled from its raw data, without metadata (else null, "invalid DCC metadata").
bool DccNoMetaRaw() {
	static const bool on = EnvFlag("KYTY_DCC_NO_META_RAW");
	return on;
}

// KYTY_STENCIL_STORAGE_NULL=1: a storage write to a stencil plane without the stencil storage
// shadow binds a null storage image (Erlou93/KytyPS5-Yotei), not the depth/stencil image.
bool StencilStorageNull() {
	static const bool on = EnvFlag("KYTY_STENCIL_STORAGE_NULL");
	return on;
}

bool TextureMinLodClamp() {
	static const bool on = EnvFlag("KYTY_TEXTURE_MIN_LOD_CLAMP");
	return on;
}

bool TextureBaseLayerClamp() {
	static const bool on = EnvFlag("KYTY_TEXTURE_BASE_LAYER_CLAMP");
	return on;
}

bool TextureMisalignedBase() {
	static const bool on = EnvFlag("KYTY_TEXTURE_MISALIGNED_BASE");
	return on;
}

// KYTY_ARRAY_VIEW_SUBRANGE=1 (adapted from Erlou93/KytyPS5-Yotei ARRAY_VIEW_SUBRANGE /
// ImageInfo::ViewLayersAlone): a sampled view of a single-level array that starts past layer 0 is
// described as an image of its layers alone (Ghost of Yotei binds one-layer views of 2 GiB arrays),
// not of the whole array. Not for storage or DCC textures (their metadata and validation describe
// the whole array).
bool ArrayViewSubrange() {
	static const bool on = EnvFlag("KYTY_ARRAY_VIEW_SUBRANGE");
	return on;
}

uint32_t DepthEncodingFieldsMode() {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_DEPTH_ENCODING_FIELDS");
		return value == nullptr ? 0u : value[0] == '1' ? 1u : value[0] == '2' ? 2u : 0u;
	}();
	return mode;
}

// A view's BASE_ARRAY past the image's layers: true when KYTY_TEXTURE_BASE_LAYER_CLAMP reads layer 0
// instead (a 1D/2D view of a one-layer image).
bool ClampsBaseLayer(ShaderRecompiler::Decoder::ImageDimension dimension, uint32_t image_layers) {
	using Dim = ShaderRecompiler::Decoder::ImageDimension;
	return TextureBaseLayerClamp() && image_layers == 1u &&
	       (dimension == Dim::Dim1D || dimension == Dim::Dim2D || dimension == Dim::Dim2DMsaa);
}

} // namespace

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::MipStats:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::UniformBuffers: return vk::DescriptorType::eUniformBuffer;
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
			// KYTY_SRT_UNIFORM: the emitter declares both tables as uniform blocks.
			return ShaderRecompiler::Spirv::SrtUniformEnabled() ? vk::DescriptorType::eUniformBuffer
			                                                    : vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

uint32_t NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding) {
	return binding.resources.empty() ? 1u : static_cast<uint32_t>(binding.resources.size());
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(!texture.image_id || view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static const char* ShaderStageResourceName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "Vertex";
		case ShaderType::Mesh: return "Mesh";
		case ShaderType::Local: return "Local";
		case ShaderType::TessellationControl: return "Hull";
		case ShaderType::TessellationEvaluation: return "Domain";
		case ShaderType::Pixel: return "Pixel";
		case ShaderType::Compute: return "Compute";
		default: return "Unknown";
	}
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1DArray: return Prospero::ImageType::kColor1D;
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

// KYTY_PRECISE_WRITE_RANGES=0 treats every writable storage binding as written in full (the
// behaviour before write-range proofs).
static bool PreciseWriteRangesEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_PRECISE_WRITE_RANGES");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}));
	return enabled;
}

// KYTY_WRITE_RANGE_LOG=N logs the first N writable storage bindings: shader, guest range, the
// proven written spans (or why the binding stays whole) and the evaluated address intervals.
static uint32_t WriteRangeLogLimit() {
	static std::atomic<int64_t> limit_once {Common::OnceUnset};
	const auto limit = static_cast<uint32_t>(Common::Once(limit_once, [] {
		const auto* value = std::getenv("KYTY_WRITE_RANGE_LOG");
		return value == nullptr ? 0u : static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
	}));
	return limit;
}

// KYTY_WRITE_RANGE_STATS=0 skips the image lookup behind FrameEvent.WriteRangeImagesSpared.
static bool WriteRangeImageStatsEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_WRITE_RANGE_STATS");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}));
	return enabled;
}

// KYTY_STORAGE_RANGE_CAP (default auto: on for NVIDIA, 1 = always, 0 = never) limits a storage
// buffer descriptor's range to just under 2 GiB. NVIDIA reports maxStorageBufferRange 4 GiB - 1,
// but OpArrayLength of a binding with a larger range returns 0 there (seen with a 3 GiB range), so
// every bounds-checked load reads 0. Ratchet & Clank binds its whole vertex pool with one 3 GiB V#:
// all world vertices collapsed to one point and the G-buffer stayed empty. Data past the cap reads
// 0 under the cap, as all of it did before.
static uint64_t StorageRangeCap(const GraphicContext& graphics) {
	static std::atomic<int64_t> mode_once {Common::OnceUnset};
	const auto mode = static_cast<int>(Common::Once(mode_once, [] {
		const auto* value = std::getenv("KYTY_STORAGE_RANGE_CAP");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "auto") == 0) {
			// KYTY_STORAGE_LENGTH_ZERO=1 makes shaders survive the zero length: bind the whole range.
			const auto* length_zero = std::getenv("KYTY_STORAGE_LENGTH_ZERO");
			return length_zero != nullptr && std::strcmp(length_zero, "0") != 0 ? int64_t {0}
			                                                                    : int64_t {2};
		}
		return int64_t {std::strcmp(value, "0") != 0 ? 1 : 0};
	}));
	constexpr uint64_t NvidiaVendor = 0x10de;
	const bool on =
	    mode == 1 || (mode == 2 && graphics.GetPhysicalDeviceProperties().vendorID == NvidiaVendor);
	return on ? uint64_t {0x7ffffffc} : std::numeric_limits<uint64_t>::max();
}

// KYTY_READ_BUFFER_TAIL (default on, =0 off; usa-ratt/KytyPS5-GT7 d875ff10f, round 84): a read-only
// storage binding's range, which the shaders' bounds checks floor to whole dwords, is rounded up to
// whole dwords instead of down. Byte and halfword reads index up to four and two times that
// length, so a range ending inside a dword hid its last bytes: GT7's car glass reads a
// 17,370-byte table (one byte per control point of its 965 patches), its last patch read the last
// two bytes as zero and one corner rose half a metre above the roof (the roof spike). A written
// binding keeps the rounding down, so a dword store cannot reach past its range.
// Default on since claude/u59-yoter7-cloth (=0: off): Ghost of Yotei's menu cloth solver reads its
// per-particle inverse masses as 8_UNORM bytes from a 70-byte V# (cs 0x79b0066f0cd1f3cd); bytes
// 68-69 read as zero, the 68-69 distance constraint divided 0 by 0, and the NaN guard reset the
// upper banner to its skinned pose every frame (stiff, sideways katana banner).
static bool ReadBufferTail() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_READ_BUFFER_TAIL");
		return value == nullptr || value[0] == '\0' || value[0] != '0';
	}();
	return enabled;
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, ShaderType stage,
                    uint32_t slot, uint32_t& buffer_offset,
                    const std::vector<GuestRange>* written_ranges, bool uniform_view) {
	buffer_offset = 0;

	const auto& [address, size, id] = source;
	using LookupOutcome = BufferLookupStats::Outcome;
	if (address == 0 || size == 0) {
		BufferLookupStats::AddBinding(LookupOutcome::Null, 0);
		// KYTY_NULL_BUFFER_SHARE: the slot may run a live V#'s specialization; the marker offset
		// puts every access past the null buffer's 16 bytes (and tells formatted reads it is null).
		if (ShaderRecompiler::GetCodegenOptions().null_buffer_share) {
			buffer_offset = ShaderRecompiler::NullBufferMarkerOffset;
		}
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	const auto& graphics  = context.GetGraphics();
	// A buffer bound as a uniform buffer (KYTY_UNIFORM_BUFFERS) meets both offset alignments, so
	// the one base adjustment in shader data serves either view.
	const auto  alignment =
	    uniform_view ? std::max<vk::DeviceSize>(
	                       graphics.StorageMinAlignment(),
	                       graphics.GetPhysicalDeviceProperties().limits.minUniformBufferOffsetAlignment)
	                 : graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	// A proven write range narrows what becomes GPU-owned and which images go stale; everything
	// else about the binding (synchronization, descriptor range) is unchanged.
	const bool narrowed = resource.written && written_ranges != nullptr;
	// Raw reads synchronize GPU image aliases like formatted ones only with the opt-in setting
	// (upstream 743fdc699 / 470669c67).
	const bool sync_images =
	    resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled());
	const auto lookup_start = BufferLookupStats::Now();
	auto [buffer, offset] =
	    narrowed ? context.GetBufferCache().ObtainWrittenBuffer(address, size, *written_ranges, id)
	             : context.GetBufferCache().ObtainBuffer(address, size, resource.written,
	                                                     sync_images, id);
	if (!resource.written && !sync_images && BufferCache::RawReadImageSyncEnabled()) {
		// KYTY_RAW_READ_IMAGE_SYNC (BufferCache::SyncRawReadFromWrittenImage).
		(void)context.GetBufferCache().SyncRawReadFromWrittenImage(*buffer, address, size);
	}
	if (lookup_start != 0) {
		// KYTY_BUFFER_LOOKUP_STATS.
		const auto outcome = BufferLookupStats::TakeOutcome();
		BufferLookupStats::AddBinding(narrowed            ? LookupOutcome::WrittenNarrow
		                              : resource.written   ? LookupOutcome::Written
		                              : resource.formatted ? LookupOutcome::Texel
		                                                   : outcome,
		                              BufferLookupStats::Now() - lookup_start);
	}
	BufferLookupStats::PartScope descriptor_scope(BufferLookupStats::Part::Descriptor);
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	// Shaders bounds-check storage buffers in whole dwords (OpArrayLength floors the range), and
	// may leave plain dword loads to robustBufferAccess2 (HostBufferRobustness). NVIDIA then
	// returns data for a dword that is only partly inside the range, so bind whole dwords: a
	// no-op for every shader-side check, and it makes the device check match them.
	auto range =
	    std::min(size + adjustment >= 4u ? Common::AlignDown(size + adjustment, uint64_t {4})
	                                     : size + adjustment,
	             StorageRangeCap(graphics));
	// KYTY_READ_BUFFER_TAIL: a read-only binding is rounded up to whole dwords instead, so byte
	// and halfword reads of the last partial dword are not hidden by the dword bounds checks.
	if (ReadBufferTail() && !resource.written && (size + adjustment) % 4u != 0) {
		const auto whole = Common::AlignUp(size + adjustment, uint64_t {4});
		// The rounded range must still fit the host buffer and the device's descriptor limit.
		if (aligned_offset + whole <= buffer->Size() && whole <= max_range) {
			range = std::min(whole, StorageRangeCap(graphics));
		}
	}
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, range};
	if (size > 0x7ffffffcu) {
		// Bindings of 2 GiB or more (KYTY_STORAGE_RANGE_CAP / KYTY_STORAGE_LENGTH_ZERO): each guest
		// range once, at most 64 lines.
		static std::mutex                             logged_mutex;
		static std::set<std::pair<uint64_t, uint64_t>> logged;
		std::scoped_lock                              lock {logged_mutex};
		if (logged.size() < 64 && logged.emplace(address, size).second) {
			std::printf("Storage binding over 2 GiB: %s slot=%u guest=0x%016" PRIx64 " size=0x%" PRIx64
			            " host range=0x%" PRIx64 " read=%d written=%d formatted=%d\n",
			            ShaderStageResourceName(stage), slot, address, size,
			            static_cast<uint64_t>(range), resource.read ? 1 : 0, resource.written ? 1 : 0,
			            resource.formatted ? 1 : 0);
		}
	}
	descriptor_scope.Stop();
	const auto invalidate_start = resource.written ? BufferLookupStats::Now() : 0;
	if (narrowed) {
		for (const auto& range: *written_ranges) {
			context.GetTextureCache().InvalidateMemoryFromGPU(range.address, range.size);
		}
	} else if (resource.written) {
		context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
	}
	if (invalidate_start != 0) {
		BufferLookupStats::AddPart(BufferLookupStats::Part::Invalidate,
		                           BufferLookupStats::Now() - invalidate_start);
	}
	const char* access = "Read";
	if (resource.written && resource.read) {
		access = "ReadWrite";
	} else if (resource.written) {
		access = "Write";
	}
	SetVulkanObjectNameF(
	    graphics.device, result.buffer,
	    "Kyty.{}.StorageBuffer[slot={} guest=0x{:016x} size=0x{:x} access={} formatted={}]",
	    ShaderStageResourceName(stage), slot, address, size, access, resource.formatted);
	return result;
}

// Reserved and unsupported T# fields must be zero; perf_mod only scales sampler state.
static bool IsCanonicalTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	if ((descriptor.fields[1] & field1_reserved_mask) != 0 ||
	    (descriptor.fields[2] & field2_reserved_mask) != 0 || descriptor.fields[3] != field3_expected) {
		return false;
	}
	if (r128) {
		return descriptor.fields[4] == 0 && descriptor.fields[5] == 0 && descriptor.fields[6] == 0 &&
		       descriptor.fields[7] == 0;
	}
	return descriptor.fields[4] == field4_expected && descriptor.fields[5] == field5_expected;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	if (!IsCanonicalTextureEncoding(descriptor, r128)) {
		return false;
	}
	if (r128) {
		return true;
	}
	if (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type())) {
		return false;
	}
	const auto metadata_control = descriptor.fields[6] & 0x00ffffffu;
	if (metadata_control == 0) {
		return true;
	}
	constexpr uint32_t htile_control = 0x00280000u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && metadata_addr != 0 &&
	       metadata_addr < TRACKER_ADDRESS_SIZE && (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

bool IsSupportedSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                    const ShaderTextureResource& descriptor,
                                    vk::Format image_format, vk::Format view_format) {
	return IsSupportedSampledDepthResource(resource) &&
	       IsSupportedDepthTextureEncoding(descriptor, resource.r128) &&
	       IsSupportedSampledDepthView(image_format, view_format, descriptor.DstSelXYZW());
}

uint64_t SampledDepthBindingSignature(const ShaderRecompiler::IR::ImageResource& resource,
                                      const ShaderTextureResource& descriptor,
                                      vk::Format image_format, vk::Format view_format) {
	const uint32_t extra[] = {static_cast<uint32_t>(resource.resource_class),
	                          static_cast<uint32_t>(resource.numeric_class),
	                          static_cast<uint32_t>(resource.dimension),
	                          static_cast<uint32_t>(resource.mip_mode),
	                          static_cast<uint32_t>(image_format),
	                          static_cast<uint32_t>(view_format),
	                          (resource.read ? 1u : 0u) | (resource.written ? 2u : 0u) |
	                              (resource.atomic ? 4u : 0u) | (resource.depth_compare ? 8u : 0u) |
	                              (resource.r128 ? 16u : 0u)};
	XXH64_state_t* state = XXH64_createState();
	XXH64_reset(state, 0);
	XXH64_update(state, descriptor.fields, sizeof(descriptor.fields));
	XXH64_update(state, extra, sizeof(extra));
	const auto result = XXH64_digest(state);
	XXH64_freeState(state);
	return result;
}

// KYTY_DEPTH_ENCODING_FIELDS (adapted from Erlou93/KytyPS5-Yotei DepthEncodingFieldsVerdict):
// IsSupportedDepthTextureEncoding compares words 4-7 whole. For a non-array 1D/2D T# the texture
// unit does not use word 4 apart from BASE_ARRAY (checked as 0 here), word 5 apart from MAX_MIP and
// PERF_MOD, nor words 6-7 when COMPRESSION_EN is 0 (no metadata is read) (RDNA2 ISA table 45). The
// strict check on a copy with those fields cleared: 1 = only unused fields differ, 2 = they differ
// and CORNER_SAMPLES is set (not emulated), 0 = something else.
uint32_t DepthEncodingFieldsVerdict(const ShaderTextureResource& descriptor, bool r128) {
	if (r128 ||
	    (descriptor.Type() != Prospero::ImageType::kColor1D &&
	     descriptor.Type() != Prospero::ImageType::kColor2D) ||
	    descriptor.BaseArray5() != 0) {
		return 0;
	}
	auto       used       = descriptor;
	const bool compressed = descriptor.MetaCompress();
	used.fields[4]        = 0;
	used.fields[5]        = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                 (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	if (!compressed) {
		used.fields[6] = 0;
		used.fields[7] = 0;
	}
	if (!IsSupportedDepthTextureEncoding(used, false)) {
		return 0;
	}
	return descriptor.CornerSample() ? 2u : 1u;
}

// A sampled binding the cache resolved to a depth image that the host cannot sample as the shader
// asked. Never fatal: it is reported once per descriptor signature and the caller binds a null
// texture instead.
static bool ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	const bool encoding_ok = IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return true;
	}
	if (const auto mode = DepthEncodingFieldsMode(); mode != 0 && resource_ok && view_ok) {
		const auto verdict = DepthEncodingFieldsVerdict(descriptor, resource.r128);
		if (verdict == 1u || mode == 2u) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
				std::printf("Depth T# bound with %s (KYTY_DEPTH_ENCODING_FIELDS=%u): addr=0x%016" PRIx64
				            " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
				            verdict == 1u   ? "fields the texture unit does not use"
				            : verdict == 2u ? "CORNER_SAMPLES set (approximation)"
				                            : "an unknown encoding (approximation)",
				            mode, descriptor.Base40(), descriptor.fields[0], descriptor.fields[1],
				            descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
				            descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
			}
			return true;
		}
	}
	static std::mutex                   reported_mutex;
	static std::unordered_set<uint64_t> reported;
	{
		std::scoped_lock lock {reported_mutex};
		if (!reported
		         .insert(SampledDepthBindingSignature(resource, descriptor, image.info.pixel_format,
		                                              view_format))
		         .second) {
			return false;
		}
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	LOGF_COLOR(Log::Color::BrightYellow,
	           "unsupported sampled depth image (binding a null texture): resource=%d encoding=%d "
	           "view=%d "
	           "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	           "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	           "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	           "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	           " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	           resource_ok, encoding_ok, view_ok,
	           static_cast<uint32_t>(resource.resource_class),
	           static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	           static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	           resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	           descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	           static_cast<int>(view_format), image.info.resources.layers,
	           static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	           descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	           descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	           descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
	return false;
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d) &&
	       supported_tile && descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() == 0 && supported_swizzle && descriptor.BCSwizzle() == 0 &&
	       !descriptor.MsaaDepth();
}

bool ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size, bool fatal) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsCanonicalTextureEncoding(descriptor, resource.r128);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	// Signed 32-bit atomics through a uint view (upstream aec8c8875).
	// KYTY_RAW_SMALL_SINT_STORAGE: write-only 16/8-bit SINT storage through a uint view.
	const bool raw_small_sint_storage =
	    ShaderRecompiler::GetCodegenOptions().raw_small_sint_storage &&
	    (format == Prospero::BufferFormat::k16SInt || format == Prospero::BufferFormat::k8SInt) &&
	    uint_resource && resource.written && !resource.read && !resource.atomic;
	const bool raw_sint_storage = raw_small_sint_storage ||
	                              (format == Prospero::BufferFormat::k32SInt && uint_resource &&
	                               resource.written &&
	                               (resource.atomic ? !resource.atomic64 : !resource.read));
	const auto numeric_class = Prospero::SampledTextureNumericClass(format);
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return true;
	}
	PassCensus::Null("unsupported storage texture bound as null");
	static std::atomic_bool logged {false};
	if (!fatal && !Common::LogOnce(logged)) {
		return false;
	}
	char report[1024] {};
	std::snprintf(report, sizeof(report),
	    "unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
	if (fatal) {
		EXIT("%s", report);
	}
	std::printf("Warning: %s(bound as null storage; KYTY_SKIP_INVALID, logged once)\n", report);
	return false;
}

TextureCache::ImageDesc NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               TextureCache::BindingType                  binding) {
	TextureCache::ImageDesc desc {};
	switch (resource.numeric_class) {
		case Prospero::TextureNumericClass::Float:
			// KYTY_NULL_TEXTURE_ZERO (Erlou93/KytyPS5-Yotei): a sampled null float texture is the
			// RGBA32F null image, cleared to (0, 0, 0, 0) (textureCache.cpp GetNullImage). Storage: R32F.
			desc.info.guest_format = NullTextureZeroEnabled() &&
			                                 binding == TextureCache::BindingType::Texture &&
			                                 !resource.depth_compare
			                             ? Prospero::BufferFormat::k32_32_32_32Float
			                             : Prospero::BufferFormat::k32Float;
			break;
		case Prospero::TextureNumericClass::Uint:
			desc.info.guest_format = resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
			                                         : Prospero::BufferFormat::k32UInt;
			break;
		case Prospero::TextureNumericClass::Sint:
			desc.info.guest_format = Prospero::BufferFormat::k32SInt;
			break;
		default: EXIT("null image has unsupported numeric class\n");
	}
	// KYTY_NULL_COMPARE_DEPTH (Erlou NullCompareBinding): a 1x1 D16 holding the depth that fails the
	// S#'s comparison; before, a D32 that was never written (undefined compare result).
	const bool compare_d16    = resource.depth_compare && NullCompareDepthEnabled() &&
	                         binding == TextureCache::BindingType::Texture;
	if (compare_d16) {
		desc.info.guest_format = Prospero::BufferFormat::k16UNorm;
	}
	desc.info.pixel_format    = compare_d16              ? vk::Format::eD16Unorm
	                            : resource.depth_compare ? vk::Format::eD32Sfloat
	                                                     : VulkanFormat(desc.info.guest_format);
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = Prospero::NumBytesPerElement(desc.info.guest_format);
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = resource.atomic64 ? vk::Format::eR64Uint : desc.info.pixel_format;
	desc.view_info.type       = vk::ImageViewType::e2D;
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			desc.info.type      = Prospero::ImageType::kColor1D;
			desc.view_info.type = vk::ImageViewType::e1D;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			desc.info.type      = Prospero::ImageType::kColor1D;
			desc.view_info.type = vk::ImageViewType::e1DArray;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			desc.info.type      = Prospero::ImageType::kColor3D;
			desc.view_info.type = vk::ImageViewType::e3D;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
			desc.view_info.type = vk::ImageViewType::e2DArray;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			desc.info.samples = 4;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			desc.info.samples   = 4;
			desc.view_info.type = vk::ImageViewType::e2DArray;
			break;
		default: break;
	}
	// Comparison sampling requires a depth-format view, even for a null descriptor.
	desc.view_info.aspect     = resource.depth_compare ? vk::ImageAspectFlagBits::eDepth
	                                                   : vk::ImageAspectFlagBits::eColor;
	desc.view_info.usage      = binding == TextureCache::BindingType::Storage
	                                ? vk::ImageUsageFlagBits::eStorage
	                                : vk::ImageUsageFlagBits::eSampled;
	desc.type                 = binding;
	return desc;
}

static void PopulateTextureMipLayout(ImageInfo& info) {
	if (info.IsVolume() && info.tile_mode != Prospero::TileMode::kLinear) {
		TileSurfaceLayout            surface {};
		const TileSurfaceDescription description {
		    info.guest_format,  info.tile_mode,    TileSurfaceDimension::Dim3D, info.extent.width,
		    info.extent.height, info.extent.depth, info.resources.levels,       1};
		if (!TileGetTiledTextureLayout(description, surface)) {
			EXIT("unsupported normalized volume texture layout\n");
		}
		for (uint32_t level = 0; level < info.resources.levels; level++) {
			const auto& mip        = surface.mips[level];
			info.mip_layout[level] = {
			    mip.offset,
			    mip.size,
			    mip.padded_width,
			    mip.padded_height,
			};
		}
		return;
	}

	TileSizeOffset levels[16] {};
	TilePaddedSize padded[16] {};
	TileGetTextureSize(info.guest_format, info.extent.width, info.extent.height,
	                   info.resources.levels, info.tile_mode, nullptr, levels, padded);
	const auto texel_shift = info.IsBlock() ? 2u : 0u;
	for (uint32_t level = 0; level < info.resources.levels; level++) {
		const auto offset =
		    levels[level].src_size != 0 ? levels[level].src_offset : levels[level].offset;
		auto size = static_cast<uint64_t>(levels[level].src_size != 0 ? levels[level].src_size
		                                                              : levels[level].size);
		if (info.IsVolume()) {
			size *= std::max(info.extent.depth >> level, 1u);
		} else {
			size *= info.resources.layers;
		}
		info.mip_layout[level] = {
		    offset,
		    size,
		    padded[level].width >> texel_shift,
		    padded[level].height >> texel_shift,
		};
	}
}

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u && !TextureMinLodClamp()) {
		EXIT("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	// KYTY_TEXTURE_MIN_LOD_CLAMP: the sampler clamps a minimum LOD past the last level to that level.
	const auto min_lod  = std::min<uint32_t>(descriptor.MinLod(), descriptor.LastLevel() * 256u);
	const auto base_lod = view.base_level * 256u;
	if (min_lod > base_lod) {
		view.min_lod = min_lod - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || resource.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			view.type       = vk::ImageViewType::e1D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers && ClampsBaseLayer(resource.dimension, image_layers)) {
				view.base_layer = 0;
			}
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			view.type       = vk::ImageViewType::e1DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers && ClampsBaseLayer(resource.dimension, image_layers)) {
				view.base_layer = 0;
			}
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipView(const TileSurfaceDescription& description, bool metadata,
                                   uint32_t view_levels, uint32_t& levels, uint32_t& base_level) {
	TileSurfaceLayout physical {};
	TileSurfaceLayout view {};
	auto              view_description = description;
	view_description.levels            = levels;
	if (!TileGetTiledTextureLayout(description, physical) ||
	    !TileGetTiledTextureLayout(view_description, view)) {
		return false;
	}
	if (physical.first_tail_level == view.first_tail_level &&
	    physical.block_slice_size == view.block_slice_size &&
	    physical.total_size == view.total_size &&
	    std::equal(std::begin(physical.mips), std::begin(physical.mips) + description.levels,
	               std::begin(view.mips))) {
		return true;
	}
	if (metadata || ((description.layers > 1 || description.depth > 1) &&
	                 physical.block_slice_size != view.block_slice_size)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= description.levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (physical.mips[target] != view.mips[source] ||
			    (target >= physical.first_tail_level) != (source >= view.first_tail_level) ||
			    std::max(description.width >> target, 1u) != std::max(description.width >> source, 1u) ||
			    std::max(description.height >> target, 1u) != std::max(description.height >> source, 1u) ||
			    std::max(description.depth >> target, 1u) != std::max(description.depth >> source, 1u)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			levels     = description.levels;
			base_level = base;
			return true;
		}
	}
	return false;
}

// KYTY_SKIP_INVALID (chenxiao07/KytyPS5 b310b97c9, 5cb588f96, 4a4faf72a, 0a1f06d92): a descriptor the
// emulator cannot represent (a stale table slot while a level loads, which the draw usually never
// reads) is bound as a null image instead of ending the session; a storage view of it loses its
// writes for that draw. Each reason is logged once (`logged` is the caller's flag).
static TextureCache::ImageDesc SkipTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                     const ShaderTextureResource& descriptor,
                                                     const char* why, std::atomic_bool& logged);

static TextureCache::ImageDesc BuildTextureDescription(
    const ShaderRecompiler::IR::ImageResource& resource, const ShaderTextureResource& descriptor) {
	const bool storage = resource.written;
	const bool skip    = Common::SkipInvalidEnabled();
	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	auto       view_levels = multisampled || single_storage_mip
	                             ? 1u
	                             : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		if (skip) {
			static std::atomic_bool logged {false};
			return SkipTextureDescriptor(resource, descriptor, "unsupported texture mip view", logged);
		}
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	// KYTY_DEPTH_BITS_READ (upstream aafe014d0): a raw load of a kDepth k16UInt T# binds the D16
	// surface as UNorm (ShaderRecompiler::IR::ReadsDepthBits; the shader converts back).
	const auto format         = ShaderRecompiler::IR::ReadsDepthBits(resource)
	                                ? Prospero::BufferFormat::k16UNorm
	                                : descriptor.Format();
	if (skip) {
		// (TextureGetSurfaceFormatInfo exits over it; a format without a texel layout leaves
		// TileGetTextureSize's size 0, the invalid-footprint path below.)
		if (!TextureSurfaceFormatKnown(format)) {
			static std::atomic_bool logged {false};
			return SkipTextureDescriptor(resource, descriptor, "texture format without a host format",
			                             logged);
		}
	}
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool sampled_numeric_class =
	    storage || resource.numeric_class == Prospero::SampledTextureNumericClass(format);
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	const auto    image_layers = layered ? depth : 1u;
	auto          view_base    = static_cast<uint32_t>(base_level);
	if (levels > physical_levels) {
		const TileSurfaceDescription physical {
		    format, tile, volume ? TileSurfaceDimension::Dim3D : TileSurfaceDimension::Dim2D,
		    width, height, volume ? depth : 1u, physical_levels, image_layers};
		const bool resolved = ResolveTextureMipView(
		    physical, !resource.r128 && descriptor.MetaCompress(), view_levels, levels, view_base);
		if (!resolved && TextureClampLastLevel()) {
			// KYTY_TEXTURE_CLAMP_LAST_LEVEL (Erlou93/KytyPS5-Yotei FIX_TEXTURE_CLAMP_LAST_LEVEL, logged
			// APPROXIMATION there): a view whose LAST_LEVEL exceeds the allocated MAX_MIP (partially
			// resident streaming textures) keeps only the allocated mips; a BASE_LEVEL beyond MAX_MIP
			// views the last allocated mip. Before: bound as null.
			static std::atomic_uint32_t clamp_logged {0};
			if (clamp_logged.fetch_add(1, std::memory_order_relaxed) < 8u) {
				std::printf("KYTY_TEXTURE_CLAMP_LAST_LEVEL: base=%u last=%u max=%u extent=%ux%u\n",
				            static_cast<uint32_t>(base_level), static_cast<uint32_t>(last_level),
				            static_cast<uint32_t>(max_mip), width, height);
			}
			const auto first = std::min<uint32_t>(base_level, max_mip);
			view_levels      = static_cast<uint32_t>(max_mip) - first + 1u;
			levels           = physical_levels;
			view_base        = first;
		} else if (!resolved) {
			if (skip) {
				static std::atomic_bool logged {false};
				return SkipTextureDescriptor(resource, descriptor,
				                             "unsupported texture mip view changes the physical layout",
				                             logged);
			}
			EXIT("unsupported texture mip view changes physical layout: base=%u last=%u max=%u "
			     "extent=%ux%ux%u tile=%u\n",
			     base_level, last_level, max_mip, width, height, depth,
			     static_cast<uint32_t>(tile));
		}
	}
	uint32_t      pitch = 0;
	TileSizeAlign size {};
	if (multisampled) {
		const auto bytes = Prospero::NumBytesPerElement(format);
		pitch            = depth_tile ? TileGetDepthPitch(width, bytes, last_level)
		                              : TileGetRenderTargetPitch(width, bytes, last_level);
		if (pitch == 0 || !TileGetRenderTargetSize(width, height, pitch, bytes, size, last_level) ||
		    size.size > UINT32_MAX / image_layers) {
			if (skip) {
				static std::atomic_bool logged {false};
				return SkipTextureDescriptor(resource, descriptor, "unsupported multisample texture layout",
				                             logged);
			}
			EXIT("unsupported multisample texture layout\n");
		}
		size.size *= image_layers;
	} else {
		pitch = TileGetTexturePitch(format, width, tile);
		TileGetTextureTotalSize(format, width, height, volume ? depth : image_layers,
		                        physical_levels, tile, volume, size);
	}
	// KYTY_TEXTURE_MISALIGNED_BASE: a 256-byte aligned base below its swizzle block's alignment is
	// bound at its address (the swizzle offsets are base-relative). Erlou logged only SW_4KB_S
	// textures at mip-tail positions (Ghost of Yotei's small RGBA16F mips written and read per level).
	const bool misaligned_base = TextureMisalignedBase() && size.size != 0 && size.align != 0 &&
	                             (address & (static_cast<uint64_t>(size.align) - 1u)) != 0 &&
	                             (address & 0xffu) == 0;
	if (misaligned_base) {
		static std::atomic<uint32_t> logged {0};
		const auto n = logged.fetch_add(1, std::memory_order_relaxed) + 1u;
		if (n <= 8u || std::has_single_bit(n)) {
			std::printf("Texture misaligned base #%u (KYTY_TEXTURE_MISALIGNED_BASE): addr=0x%016" PRIx64
			            " align=0x%x %ux%u format=%u tile=%u size=0x%x %s\n",
			            n, address, size.align, width, height, static_cast<uint32_t>(format),
			            static_cast<uint32_t>(tile), size.size, storage ? "storage" : "sampled");
		}
	}
	if (!misaligned_base && (size.size == 0 || size.align == 0 ||
	                         (address & (static_cast<uint64_t>(size.align) - 1u)) != 0)) {
		if (storage && skip) {
			static std::atomic_bool logged {false};
			return SkipTextureDescriptor(resource, descriptor,
			                             "storage texture with an invalid footprint (size or alignment)", logged);
		}
		if (storage) {
			EXIT("Not implemented (storage texture with size=0x%x align=0x%x addr=0x%016" PRIx64
			     ")\n",
			     size.size, size.align, address);
		}
		// A sampled texture whose descriptor has no valid footprint (zero size, or a base address
		// the tile layout cannot align to): bind a null texture like a null descriptor.
		PassCensus::Null("sampled texture with invalid footprint");
		static std::atomic_uint32_t reported {0};
		if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::printf("Warning: texture with invalid footprint bound as null: addr=0x%016" PRIx64
			            " size=0x%x align=0x%x %ux%u format=%u type=%u tile=%u "
			            "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
			            address, size.size, size.align, width, height,
			            static_cast<uint32_t>(format), static_cast<uint32_t>(type),
			            static_cast<uint32_t>(tile), descriptor.fields[0], descriptor.fields[1],
			            descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
			            descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
		}
		return NullTextureDesc(resource, TextureCache::BindingType::Texture);
	}
	if (storage && !ValidateStorageTexture(resource, descriptor, size.size, !skip)) {
		return NullTextureDesc(resource, TextureCache::BindingType::Storage);
	}
	// A T# of no guest memory (the texture cache's registration would exit over it).
	if (skip && (address >= TRACKER_ADDRESS_SIZE || size.size > TRACKER_ADDRESS_SIZE - address)) {
		static std::atomic_bool logged {false};
		return SkipTextureDescriptor(resource, descriptor, "texture data outside guest memory", logged);
	}

	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	// KYTY_RAW_SMALL_SINT_STORAGE: the uint view of a 16/8-bit SINT storage image (Erlou93/KytyPS5-Yotei).
	const bool small_sint_view = storage && ShaderRecompiler::GetCodegenOptions().raw_small_sint_storage &&
	                             resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const auto storage_view_format = resource.atomic64 ? vk::Format::eR64Uint
	                                 : storage && (resource.atomic ||
	                                               format == Prospero::BufferFormat::k32SInt)
	                                     ? vk::Format::eR32Uint
	                                 : small_sint_view && format == Prospero::BufferFormat::k16SInt
	                                     ? vk::Format::eR16Uint
	                                 : small_sint_view && format == Prospero::BufferFormat::k8SInt
	                                     ? vk::Format::eR8Uint
	                                     : SrgbStorageViewFormat(pixel_format);
	auto view_format = storage && storage_view_format != vk::Format::eUndefined ? storage_view_format
	                                                                            : pixel_format;
	// KYTY_SRGB8_VIEW: a sampled 8 sRGB texture reads its R8_UNORM image (MUTABLE_FORMAT) through an
	// R8_SRGB view: the sampler decodes before filtering, as the PS5 does.
	if (!storage && !resource.depth_compare && format == Prospero::BufferFormat::k8Srgb &&
	    pixel_format == vk::Format::eR8Unorm && Prospero::Srgb8ViewActive()) {
		view_format = vk::Format::eR8Srgb;
	}
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	desc.info.data         = {address, size.size};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.pitch        = pitch;
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	const bool dcc_without_metadata =
	    !resource.r128 && descriptor.MetaCompress() && descriptor.MetaAddr() == 0 && DccNoMetaRaw();
	if (dcc_without_metadata) {
		static std::atomic_uint32_t warned {0};
		if (warned.fetch_add(1, std::memory_order_relaxed) < 8u) {
			std::printf("KYTY_DCC_NO_META_RAW: DCC texture without metadata sampled raw: "
			            "addr=0x%016" PRIx64 " format=%u extent=%ux%u\n",
			            address, static_cast<uint32_t>(format), width, height);
		}
	}
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth &&
	    !desc.info.IsDepth() && !dcc_without_metadata) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.kind          = ImageMetadataKind::Dcc;
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
		const auto& meta = desc.info.metadata.range;
		// (ImageOps::Validate exits over "invalid DCC metadata".)
		if (skip && (meta.address == 0 || meta.address >= TRACKER_ADDRESS_SIZE ||
		             (meta.size != 0 && meta.size > TRACKER_ADDRESS_SIZE - meta.address))) {
			static std::atomic_bool logged {false};
			return SkipTextureDescriptor(resource, descriptor, "invalid DCC metadata", logged);
		}
	}
	if (samples > 1) {
		desc.info.mip_layout[0] = {0, size.size, pitch, height};
	} else {
		PopulateTextureMipLayout(desc.info);
	}
	if (skip) {
		// usa-ratt/KytyPS5-GT7 767c1e25e, 5ef1f4c3c: a minimum LOD past the last mip, a base
		// layer past the image's layers or a component selector no texture has (each an EXIT in
		// TextureViewInfo): a bindless heap entry read while the guest rewrote it (GT7 races), or
		// a T# read after its reuse. Bound as null (a bindless key stays on its placeholder).
		if (descriptor.MinLod() > descriptor.LastLevel() * 256u && !TextureMinLodClamp()) {
			static std::atomic_bool logged {false};
			return SkipTextureDescriptor(resource, descriptor, "texture minimum LOD past its last mip",
			                             logged);
		}
		if (resource.dimension != ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
		    descriptor.BaseArray5() >= desc.info.resources.layers &&
		    !ClampsBaseLayer(resource.dimension, desc.info.resources.layers)) {
			static std::atomic_bool logged {false};
			return SkipTextureDescriptor(resource, descriptor, "texture base layer past its layers",
			                             logged);
		}
		if (!storage && resource.conversion_format == Prospero::BufferFormat::kInvalid &&
		    !IsValidImageSwizzle(descriptor.DstSelXYZW())) {
			static std::atomic_bool logged {false};
			return SkipTextureDescriptor(resource, descriptor,
			                             "texture component selector no texture has", logged);
		}
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.view_info.base_level = view_base;
	ClampImageLevelsToExtent(desc.info, desc.view_info);
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	const auto first_layer = desc.view_info.base_layer;
	if (ArrayViewSubrange() && layered && !storage && first_layer != 0 &&
	    first_layer < desc.info.resources.layers && desc.info.resources.levels == 1 && samples == 1 &&
	    desc.info.metadata.kind == ImageMetadataKind::None &&
	    desc.info.data.size % desc.info.resources.layers == 0) {
		const auto slice  = desc.info.data.size / desc.info.resources.layers;
		const auto layers = desc.info.resources.layers - first_layer;
		// The layers' bytes are one slice each (one level), so the sub-array is a contiguous range
		// with the whole array's slice layout.
		if (desc.info.mip_layout[0].offset == 0 && desc.info.mip_layout[0].size == desc.info.data.size &&
		    (slice & (static_cast<uint64_t>(size.align) - 1u)) == 0) {
			desc.info.data             = {address + slice * first_layer, slice * layers};
			desc.info.resources.layers = layers;
			desc.info.mip_layout[0].size = slice * layers;
			desc.view_info.base_layer  = 0;
			desc.view_info.layer_count = std::min(desc.view_info.layer_count, layers);
			static std::atomic<uint32_t> logged {0};
			const auto n = logged.fetch_add(1, std::memory_order_relaxed) + 1u;
			if (n <= 4u || std::has_single_bit(n)) {
				std::printf("Array view subrange #%u (KYTY_ARRAY_VIEW_SUBRANGE): addr=0x%016" PRIx64
				            " %ux%u layers %u..%u slice=0x%" PRIx64 "\n",
				            n, address, width, height, first_layer, first_layer + layers - 1u, slice);
			}
		}
	}

	return desc;
}

static TextureCache::ImageDesc SkipTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                     const ShaderTextureResource& descriptor,
                                                     const char* why, std::atomic_bool& logged) {
	PassCensus::Null(why);
	if (Common::LogOnce(logged)) {
		std::printf("Warning: %s: %s view bound as null (KYTY_SKIP_INVALID; logged once): "
		            "addr=0x%016" PRIx64 " format=%u type=%u tile=%u "
		            "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		            why, resource.written ? "storage" : "sampled", descriptor.Base40(),
		            static_cast<uint32_t>(descriptor.Format()), static_cast<uint32_t>(descriptor.Type()),
		            static_cast<uint32_t>(descriptor.TileMode()), descriptor.fields[0], descriptor.fields[1],
		            descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		            descriptor.fields[6], descriptor.fields[7]);
	}
	return NullTextureDesc(resource, resource.written ? TextureCache::BindingType::Storage
	                                                  : TextureCache::BindingType::Texture);
}

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value) {
	TextureBinding binding;
	ResolveTexture(resource, value, binding);
	return binding;
}

void RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                    const ShaderRecompiler::IR::DescriptorValue& value,
                                    TextureBinding& binding, const uint64_t* hash_hint,
                                    uint64_t tag_hint) {
	// The compare functions a caller set apply to this resolution only (KYTY_NULL_COMPARE_DEPTH).
	struct ResetCompareFuncs {
		~ResetCompareFuncs() { SetNullCompareFuncs(0u); }
	} reset_compare_funcs;
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	// The same state a freshly returned binding had: no view yet, no mip views (their capacity
	// is kept), undefined layout.
	binding.image_view = nullptr;
	binding.layout     = vk::ImageLayout::eUndefined;
	binding.mip_views.clear();

	auto&      texture_cache     = m_context.GetTextureCache();
	// KYTY_NULL_COMPARE_DEPTH: a null depth-compare binding depends on the S#s, which the memo key
	// does not hold, so depth-compare bindings are not memoized.
	const bool memo              = TextureBindingMemo::Enabled() &&
	                  !(resource.depth_compare && NullCompareDepthEnabled());
	const bool description_cache =
	    Common::RendererBatchEnabled() && resource.indirect_resources.size() <= 256u;
	TextureBindingMemo::Key memo_key;
	uint64_t                hash = 0;
	if (memo || description_cache) {
		memo_key = TextureBindingMemo::MakeKey(resource, descriptor.fields);
		// KYTY_DRAW_PREP_BINDINGS textures: the preparing worker hashed the same key.
		if (hash_hint != nullptr && DrawPrep::BindingsVerifyMode() == 0) {
			hash = *hash_hint;
		} else {
			hash = TextureBindingMemo::Hash(memo_key);
			if (hash_hint != nullptr) {
				DrawPrep::CountBindingVerifyCheck();
				if (*hash_hint != hash) {
					DrawPrep::ReportBindingMismatch("texture memo hash");
				}
			}
		}
	}
	// Exactly the answer of the full resolution below (see textureBindingMemo.h), including the
	// FindImage access bookkeeping; validation of the key's resource already passed.
	if (memo && m_texture_memo.TryResolve(texture_cache, memo_key, hash, binding, tag_hint,
	                                      tag_hint != 0 && DrawPrep::BindingsVerifyMode() != 0)) {
		if (!descriptor.IsNull() && HangTrace::Enabled()) {
			HangTrace::RecordTexture(descriptor.fields);
		}
		const bool dcc_hit = m_texture_memo.LastHitDcc();
		if ((m_texture_memo.LastHitRevalidated() || dcc_hit) && TextureBindingMemo::RevalidateVerify()) {
			// KYTY_TEXTURE_MEMO_REVALIDATE_VERIFY: the full resolution (without the memo) must
			// give the same image and description; after a DCC entry's hit (KYTY_CP_COMMIT=texdcc)
			// its DCC decision must also do nothing (it would clear what the hit did not).
			const auto     effects = texture_cache.DccDecisionEffects();
			TextureBinding full;
			ResolveTextureFull(resource, descriptor, memo_key, hash, false, full);
			const bool dcc_differs = dcc_hit && texture_cache.DccDecisionEffects() != effects;
			if (full.image_id != binding.image_id || full.desc.type != binding.desc.type ||
			    !(full.desc.view_info == binding.desc.view_info) ||
			    full.desc.info.data != binding.desc.info.data ||
			    full.desc.info.pixel_format != binding.desc.info.pixel_format || dcc_differs) {
				if (dcc_differs) {
					std::fprintf(stderr, "TextureMemoRevalidateVerify: a DCC entry's hit skipped a DCC "
					                     "decision that does something (KYTY_CP_COMMIT=texdcc)\n");
				}
				TextureBindingMemo::ReportRevalidateMismatch();
				binding = std::move(full);
			}
		}
		return;
	}
	ResolveTextureFull(resource, descriptor, memo_key, hash, memo, binding);
}

void RenderExecutor::ResolveTextureFull(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource&               descriptor,
                                        const TextureBindingMemo::Key& memo_key, uint64_t hash,
                                        bool memo, TextureBinding& binding) {
	auto&      texture_cache     = m_context.GetTextureCache();
	const bool storage           = resource.written;
	const bool description_cache =
	    Common::RendererBatchEnabled() && resource.indirect_resources.size() <= 256u;
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	if (storage) {
		ValidateStorageImageResource(resource);
	}
	TextureBindingMemo::Forget(binding);
	auto& desc = binding.desc;

	// The recompiler's specialization treats a sampled texture whose format the host cannot
	// represent as a null image (ResourceMaterialization.cpp); bind a null texture for it here.
	const bool unsupported_sampled_format =
	    !storage && !resource.depth_compare &&
	    !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
	    Prospero::SampledTextureNumericClass(descriptor.Format()) ==
	        Prospero::TextureNumericClass::Unsupported;
	if (descriptor.IsNull() || unsupported_sampled_format) {
		desc = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                         : TextureCache::BindingType::Texture);
		binding.image_id = texture_cache.FindImage(desc);
		if (memo) {
			m_texture_memo.Record(texture_cache, memo_key, hash, binding, binding.image_id, false,
			                      false);
		}
		return;
	}

	if (HangTrace::Enabled()) {
		HangTrace::RecordTexture(descriptor.fields);
	}

	if (description_cache) {
		const TextureDescriptionKey key {resource.resource_class, resource.numeric_class,
		                                 resource.dimension,      resource.mip_mode,
		                                 resource.mip_count,      resource.conversion_format,
		                                 resource.shader_swizzle, resource.read,
		                                 resource.written,        resource.atomic,
		                                 resource.depth_compare,  resource.cube,
		                                 resource.r128,           resource.atomic64};
		// TextureBindingMemo::Hash is this cache's hash of the dwords and key.
		auto& entry = m_texture_descriptions[hash % m_texture_descriptions.size()];
		if (entry.valid && entry.key == key &&
		    std::ranges::equal(entry.words, descriptor.fields)) {
			desc = entry.desc;
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureDescriptionHits);
		} else {
			desc = BuildTextureDescription(resource, descriptor);
			entry.key = key;
			std::ranges::copy(descriptor.fields, entry.words.begin());
			entry.desc = desc;
			entry.valid = true;
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureDescriptionMisses);
		}
	} else {
		desc = BuildTextureDescription(resource, descriptor);
	}
	if (desc.info.data.Empty()) {
		// Bound as null (an invalid footprint; KYTY_SKIP_INVALID): as a null descriptor above.
		binding.image_id = texture_cache.FindImage(desc);
		if (memo) {
			m_texture_memo.Record(texture_cache, memo_key, hash, binding, binding.image_id, false,
			                      false);
		}
		return;
	}
	// Only descriptor-derived geometry is retained. FindImage may alter the local
	// description and must still resolve ownership, aliases, uploads and metadata.
	const auto pixel_format = desc.info.pixel_format;
	const auto view_format = desc.view_info.format;
	const auto byte_size = desc.info.data.size;
	const auto base_level = desc.view_info.base_level;
	const auto base_layer = desc.view_info.base_layer;
	const bool shader_conversion = TextureGetSurfaceFormatInfo(descriptor.Format()).conversion_format !=
	                               Prospero::BufferFormat::kInvalid;
	// KYTY_CP_COMMIT=texdcc: a DCC description's lookup also records the certificate of its DCC
	// decision, with which the memo may answer later lookups of this binding
	// (TextureBindingMemo::Record). Recording changes nothing FindImage does.
	const bool dcc_memo = memo && desc.info.metadata.kind == ImageMetadataKind::Dcc &&
	                      CpCommit::Enabled(CpCommit::Part::TexDcc);
	TextureCache::RepeatLookup dcc_record;
	auto id = texture_cache.FindImage(desc, shader_conversion, dcc_memo ? &dcc_record : nullptr);
	// KYTY_WATCH_RANGE: a texture binding over the watched range, with the T#'s format and the
	// cached image it resolved to (several images can share one address).
	if (WatchRange::Enabled()) {
		const auto& found_image = texture_cache.GetImage(id);
		WatchRange::Note(fmt::format("bind {} T# fmt {} {}x{} view {} -> image {}:{} {} {}x{} depth_id {} meta {} 0x{:x}+0x{:x}",
		                             storage ? "storage" : "texture",
		                             static_cast<uint32_t>(descriptor.Format()), desc.info.extent.width,
		                             desc.info.extent.height, vk::to_string(desc.view_info.format), id.index,
		                             id.generation, vk::to_string(found_image.backing.format),
		                             found_image.backing.extent.width, found_image.backing.extent.height,
		                             found_image.depth_id ? 1 : 0, static_cast<uint32_t>(desc.info.metadata.kind),
		                             desc.info.metadata.range.address, desc.info.metadata.range.size)
		                     .c_str(),
		                 desc.info.data.address, desc.info.data.size);
	}
	const auto found               = id;
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	// KYTY_MS_DEPTH_SAMPLE0=1 (usa-ratt/KytyPS5-GT7 rounds 122-124): the shader reads a
	// multisample depth T# through a single-sample dimension (DescriptorDimension made its image a
	// single-sample one): the binding gets a copy of the first sample (GT7 replays: the light
	// sprites' occlusion test, the depth reduction the CPU reads back).
	const bool single_sample_read =
	    resource.dimension != ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa &&
	    resource.dimension != ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray;
	const bool ms_depth_sample0 = TextureCache::MsDepthSample0Enabled() && single_sample_read;
	bool       copy_binding     = false;
	// KYTY_STENCIL_READ_DEPTH: a 1-byte colour image at the address of a plane a depth image swaps
	// to (no stencil association of its own) is read as that depth image's stencil, at its extent.
	ImageId plane_depth {};
	if (!stencil_association && !storage && TextureCache::StencilReadDepthEnabled() &&
	    image->info.bytes_per_block == 1u && !image->info.IsDepth()) {
		plane_depth = texture_cache.StencilPlaneDepth(image->info.data.address);
		if (plane_depth) {
			const auto& depth = texture_cache.GetImage(plane_depth);
			if (depth.backing.extent.width != image->backing.extent.width ||
			    depth.backing.extent.height != image->backing.extent.height) {
				plane_depth = {};
			}
		}
	}
	if (plane_depth) {
		id           = plane_depth;
		image        = &texture_cache.GetImage(id);
		copy_binding = true; // not memoized: the plane's depth image is looked up each time
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
			LOGF("KYTY_STENCIL_READ_DEPTH: plane 0x%012" PRIx64 " read as the stencil of depth 0x%012" PRIx64 "\n",
			     texture_cache.GetImage(found).info.data.address, image->info.data.address);
		}
		// KYTY_STENCIL_READ_PARTNER: the plane the depth image swapped away from (Erlou parity).
		if (TextureCache::StencilReadPartnerEnabled()) {
			const auto plane_address = texture_cache.GetImage(found).info.data.address;
			const auto partner       = texture_cache.StencilPlanePartnerRecord(plane_address);
			if (partner) {
				bool       plane_copy = false;
				const auto reader     = texture_cache.StencilPlaneReader(partner, plane_copy);
				if (plane_copy) {
					id                    = reader;
					image                 = &texture_cache.GetImage(id);
					desc.view_info.format = vk::Format::eR8Uint;
					desc.view_info.aspect = vk::ImageAspectFlagBits::eColor;
				}
				static std::atomic<uint32_t> partner_logged {0};
				if (partner_logged.fetch_add(1, std::memory_order_relaxed) < 4) {
					LOGF("KYTY_STENCIL_READ_PARTNER: plane 0x%012" PRIx64 " read as plane 0x%012" PRIx64 " (%s)\n",
					     plane_address, texture_cache.GetImage(partner).info.data.address,
					     plane_copy ? "saved copy" : "held");
				}
			}
		}
	} else if (stencil_association) {
		const auto record_id = id;
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
		if (storage && TextureCache::StencilStorageShadowEnabled()) {
			bool       shadowed = false;
			const auto shadow   = texture_cache.StencilStorageShadow(record_id, shadowed);
			if (shadowed) {
				id                    = shadow;
				image                 = &texture_cache.GetImage(id);
				copy_binding          = true;
				desc.view_info.format = vk::Format::eR8Uint;
				desc.view_info.aspect = vk::ImageAspectFlagBits::eColor;
			}
		}
		if (storage && !copy_binding) {
			// A storage view of a depth/stencil image's stencil aspect (Erlou: STENCIL_STORAGE_SHADOW).
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
				std::printf("Stencil plane bound as a storage image: plane 0x%012" PRIx64
				            " depth 0x%012" PRIx64 " %ux%u format %d\n",
				            texture_cache.GetImage(record_id).info.data.address,
				            image->info.data.address, image->backing.extent.width,
				            image->backing.extent.height, static_cast<int>(image->backing.format));
			}
			// KYTY_STENCIL_STORAGE_NULL=1 (Erlou93/KytyPS5-Yotei DIAG_DEPTH_STORAGE_NULL): a storage
			// view of the depth/stencil image is invalid Vulkan (D32S8 has no storage usage), so a
			// null storage image takes the write (not memoized: the next resolution logs again).
			if (StencilStorageNull()) {
				desc             = NullTextureDesc(resource, TextureCache::BindingType::Storage);
				binding.image_id = texture_cache.FindImage(desc);
				return;
			}
		} else if (TextureCache::PreserveStencilEnabled() &&
		           (!TextureCache::StencilReadDepthEnabled() || TextureCache::StencilReadHeldEnabled())) {
			// KYTY_PRESERVE_STENCIL: a plane the depth image does not hold is read from its saved copy
			// (KYTY_STENCIL_READ_DEPTH with KYTY_STENCIL_READ_HELD=0: the depth image's stencil instead).
			bool       plane_copy = false;
			const auto reader     = texture_cache.StencilPlaneReader(record_id, plane_copy);
			if (plane_copy) {
				id           = reader;
				image        = &texture_cache.GetImage(id);
				copy_binding = true;
				// As the stencil aspect view reads: the plane's bytes as integers.
				desc.view_info.format = vk::Format::eR8Uint;
				desc.view_info.aspect = vk::ImageAspectFlagBits::eColor;
			}
		}
		if (ms_depth_sample0 && !storage && image->backing.samples > 1) {
			id           = texture_cache.SingleSampleStencil(id);
			image        = &texture_cache.GetImage(id);
			copy_binding = image->backing.samples == 1;
		}
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		if (!ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format, byte_size)) {
			desc = NullTextureDesc(resource, TextureCache::BindingType::Texture);
			binding.image_id = texture_cache.FindImage(desc);
			return;
		}
		if (ms_depth_sample0 && image->backing.samples > 1) {
			id           = texture_cache.SingleSampleDepth(id);
			image        = &texture_cache.GetImage(id);
			copy_binding = image->backing.samples == 1;
		}
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		// KYTY_STENCIL_ALIAS_COPY: a stencil plane read at another extent than its depth image.
		if (TextureCache::StencilAliasCopyEnabled() && image->info.bytes_per_block == 1u &&
		    texture_cache.CopyStencilPlaneToAlias(id)) {
			image        = &texture_cache.GetImage(id);
			copy_binding = true; // not memoized: each resolution copies the plane again
		}
		if (!IsSupportedSampledColorView(image->info.pixel_format, pixel_format,
		                                 descriptor.DstSelXYZW()) &&
		    Common::SkipInvalidEnabled()) {
			// usa-ratt/KytyPS5-GT7 5ef1f4c3c: an image the T# cannot view (a stale bindless heap
			// entry, a reused T#): bound as null instead of ending the emulator.
			PassCensus::Null("sampled texture over an image it cannot view");
			static std::atomic_bool logged {false};
			if (Common::LogOnce(logged)) {
				std::printf("Warning: sampled texture over an image it cannot view (image format %d, "
				            "view format %d, swizzle 0x%x) bound as null (KYTY_SKIP_INVALID; logged "
				            "once): addr=0x%016" PRIx64 "\n",
				            static_cast<int>(image->info.pixel_format), static_cast<int>(pixel_format),
				            descriptor.DstSelXYZW(), descriptor.Base40());
			}
			desc             = NullTextureDesc(resource, TextureCache::BindingType::Texture);
			binding.image_id = texture_cache.FindImage(desc);
			return;
		}
		(void)SelectSampledColorView(image->info.pixel_format, pixel_format,
		                             descriptor.DstSelXYZW());
		// KYTY_SUB_EXTENT_COPY=1 (usa-ratt/KytyPS5-GT7 429692dd3, Franseven): a texture smaller
		// than the colour image at its address, with the same memory and tiling (UI text drawn
		// into part of a wider target), reads a copy of the image's corner: a view of the image
		// would stretch its coordinates over the whole image.
		if (TextureCache::SubExtentCopyEnabled()) {
			if (const auto copy_id = texture_cache.SubExtentCopy(id, desc.info); copy_id != id) {
				id           = copy_id;
				image        = &texture_cache.GetImage(id);
				copy_binding = true; // not memoized either: each resolution refreshes the copy
			}
		}
	}
	if (ms_depth_sample0 && image->backing.samples > 1) {
		// A multisample image left for a single-sample shader image (a colour image under a
		// depth-tiled multisample T#, or a depth image no copy was made of): a mismatch Vulkan
		// leaves undefined (wrong reads or a lost device), so a null texture instead.
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("Multisample depth read as single-sample: 0x%012" PRIx64 " %ux%u fmt %d, %u "
			     "samples, %u layers, %u levels, %s%s: no single-sample copy, a null texture\n",
			     image->info.data.address, image->info.extent.width, image->info.extent.height,
			     static_cast<int>(image->info.pixel_format), image->backing.samples,
			     image->info.resources.layers, image->info.resources.levels,
			     image->info.IsDepth() ? "a depth image" : "a colour image",
			     stencil_association ? " (its stencil)" : "");
		}
		desc = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                         : TextureCache::BindingType::Texture);
		binding.image_id = texture_cache.FindImage(desc);
		return;
	}
	binding.image_id = id;
	if (copy_binding) {
		// Not memoized: each resolution refreshes the copy from its source.
		return;
	}
	if (memo) {
		const bool view_rebased =
		    desc.view_info.base_level != base_level || desc.view_info.base_layer != base_layer;
		const bool dcc_proven = dcc_memo && dcc_record.valid && dcc_record.image == found;
		m_texture_memo.Record(texture_cache, memo_key, hash, binding, found, shader_conversion,
		                      view_rebased, dcc_proven ? &dcc_record.dcc : nullptr);
	}
}

static bool SamplerMemoEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_SAMPLER_MEMO");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_CUBE_EDGE_CLAMP=1 (from usa-ratt/KytyPS5-GT7 round 142). A cube map is sampled as a 2D
// array of its faces (face coordinates and face index from V_CUBE*, CubeAxis/CubeLayer), so the
// sampler's address mode applies inside each face. The console's cube addressing ignores it and
// filters across the face edges; with WRAP (what a sky's sampler often has) bilinear filtering at
// an edge blends in the face's opposite edge: a thin line along the face edges in the sky and a
// shade step where a reflection cube's coarse mips meet (GT7). CLAMP_LAST_TEXEL keeps each face's
// filtering inside it.
static bool CubeEdgeClamp() {
	static std::atomic<int64_t> slot {Common::OnceUnset};
	return Common::Once(slot, [] {
		       const char* value   = std::getenv("KYTY_CUBE_EDGE_CLAMP");
		       const bool  enabled = value != nullptr && value[0] == '1';
		       if (enabled) {
			       LOGF("Cube maps: samplers used only with cube maps clamp at the face edges "
			            "(KYTY_CUBE_EDGE_CLAMP=1)\n");
		       }
		       return enabled ? 1 : 0;
	       }) != 0;
}

// Whether every image this sampler is paired with is a cube map.
static bool SamplesCubesOnly(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                             uint32_t                                        sampler) {
	bool any = false;
	for (const auto& pair: program.info.sampled_pairs) {
		if (pair.sampler != sampler) {
			continue;
		}
		if (pair.image >= program.info.images.size() || !program.info.images[pair.image].cube) {
			return false;
		}
		any = true;
	}
	return any;
}

ShaderSamplerResource NativeSamplerDescriptor(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                              uint32_t                                        index,
                                              const ShaderRecompiler::IR::DescriptorValue&    value) {
	auto descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	if (CubeEdgeClamp() && SamplesCubesOnly(program, index)) {
		// CLAMP_X and CLAMP_Y (bits 0-2 and 3-5) to CLAMP_LAST_TEXEL (2).
		descriptor.fields[0] = (descriptor.fields[0] & ~0x3fu) | 2u | (2u << 3u);
	}
	if (!program.info.samplers[index].depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (program.info.samplers[index].force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	return descriptor;
}

void AppendUserShaderData(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                          const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
                          std::vector<uint32_t>&                          shader_data) {
	for (const auto reg: program.bindings.user_data_registers) {
		shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
}

bool WriteMipStatsFields(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                         const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
                         std::vector<uint32_t>&                          shader_data) {
	// GET_LOD_STATS field per image, 32 bits (LodStatsReport::ImageField): MipStatsCntId, the
	// BASE_LEVEL added to recorded levels, a no-counter flag and the LOD threshold below which a
	// sample is counted (the T# MIN_LOD with KYTY_LOD_STATS_COUNT=clamp, every sample otherwise).
	// KYTY_MIP_STATS_BASE_LEVEL=0 reports levels relative to BASE_LEVEL (the U27 behaviour).
	static const bool absolute_levels = [] {
		const char* value = std::getenv("KYTY_MIP_STATS_BASE_LEVEL");
		return !(value != nullptr && value[0] == '0');
	}();
	const auto& layout        = program.bindings;
	const bool  count_clamped = LodStatsCounter::CountClamped();
	bool        active        = false;
	for (uint32_t i = 0; i < layout.mip_stats_count; i++) {
		const auto field = LodStatsReport::ImageField(snapshot.images.at(i).dwords.data(),
		                                              absolute_levels, count_clamped);
		shader_data[layout.MipStatsOffsetDword() + i] = field;
		active |= (field & LodStatsReport::NoCounterFlag) == 0u;
	}
	return active;
}

vk::Sampler RenderExecutor::NativeSampler(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                          uint32_t                                        index,
                                          const ShaderRecompiler::IR::DescriptorValue&    value,
                                          bool normalize_unnormalized) {
	auto descriptor = NativeSamplerDescriptor(program, index, value);
	if (normalize_unnormalized) {
		// KYTY_UNNORMALIZED_FIX: the hull shader's copy divides its coordinates by the mip's size
		// (SpirvNormalizedExplicitLod), so FORCE_UNNORMALIZED (word 0 bit 15) goes, and the S#'s
		// LOD range and clamps apply as they are.
		descriptor.fields[0] &= ~(1u << 15u);
	}
	const bool integer_border = program.info.samplers[index].integer_border;
	if (!SamplerMemoEnabled()) {
		return m_context.GetSamplerCache().GetSampler(descriptor, integer_border);
	}
	// Keyed on exactly the SamplerCache key (the four final dwords and the border class).
	// SamplerMemoWays entries per set, most recently used first: the desert stamps bind eight
	// samplers per draw, two of which shared a direct-mapped slot and missed on every draw
	// (1,815 misses per flip).
	static_assert(sizeof(descriptor.fields) == sizeof(SamplerMemoEntry::fields));
	const auto slot = (descriptor.fields[0] * 0x9e3779b1u ^ descriptor.fields[1] * 0x85ebca6bu ^
	                   descriptor.fields[2] * 0xc2b2ae35u ^ descriptor.fields[3]) >>
	                  26u;
	constexpr uint32_t sets = static_cast<uint32_t>(std::tuple_size_v<decltype(m_sampler_memo)>) /
	                          SamplerMemoWays;
	auto* const set = m_sampler_memo.data() + (slot % sets) * SamplerMemoWays;
	for (uint32_t way = 0; way < SamplerMemoWays; way++) {
		auto& entry = set[way];
		if (entry.sampler != nullptr && entry.integer_border == integer_border &&
		    std::memcmp(entry.fields.data(), descriptor.fields, sizeof(descriptor.fields)) == 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::SamplerMemoHits);
			const auto sampler = entry.sampler;
			if (way != 0) {
				std::rotate(set, set + way, set + way + 1);
			}
			return sampler;
		}
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::SamplerMemoMisses);
	const auto sampler = m_context.GetSamplerCache().GetSampler(descriptor, integer_border);
	// The least recently used way makes room at the front.
	std::rotate(set, set + SamplerMemoWays - 1, set + SamplerMemoWays);
	std::memcpy(set[0].fields.data(), descriptor.fields, sizeof(descriptor.fields));
	set[0].integer_border = integer_border;
	set[0].sampler        = sampler;
	return sampler;
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	auto& command_buffer = context.GetCommandScheduler().Current();
	EXIT_IF(command_buffer.IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

// KYTY_SRT_GPU_FILL_DIAG=1 (startup, default off): the SRT GPU fill's 10 s console counts (also the
// recompiler's per-plan lines, ResourceMaterialization.cpp).
static bool SrtGpuFillDiag() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SRT_GPU_FILL_DIAG");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_SRT_GPU_FILL (port of senaxx 0845beadf): the flat buffer of a refresh that left slots to
// the GPU (ResourceSnapshot::gpu_fills). Uploaded on its own (never shared through the upload
// dedup: its placeholders are not its values); each slot's dword is then copied on the GPU from
// the buffer that holds the GPU-written bytes, behind the writes that produced them and before
// the command that reads the table. A dword that stopped being GPU-written since the walk (or a
// rebind off the command processor) is read on the host instead, as the walk would have.
static vk::DescriptorBufferInfo UploadFlatWithGpuFills(RenderContext&                    context,
                                                       const ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                                                       std::span<const uint32_t> flat_words) {
	struct Copy {
		vk::Buffer source;
		uint64_t   source_offset;
		uint32_t   flat_offset;
	};
	thread_local std::vector<uint32_t> words;
	thread_local std::vector<Copy>     copies;
	words.assign(flat_words.begin(), flat_words.end());
	copies.clear();
	auto&      cache      = context.GetBufferCache();
	const bool gpu_thread = GuestGpu::IsGpuThread();
	uint64_t   host_reads = 0;
	for (const auto& fill: snapshot.gpu_fills) {
		if (fill.flat_offset >= words.size()) {
			continue;
		}
		vk::Buffer source        = nullptr;
		uint64_t   source_offset = 0;
		if (gpu_thread && cache.GpuFillSource(fill.address, &source, &source_offset)) {
			copies.push_back({source, source_offset, fill.flat_offset});
			continue;
		}
		uint32_t word = 0;
		std::memcpy(&word, reinterpret_cast<const void*>(fill.address), sizeof(word));
		words[fill.flat_offset] = word;
		host_reads++;
	}
	const auto flat = NativeUpload(context, words);
	if (!copies.empty()) {
		auto& command = context.GetCommandScheduler().Current();
		command.EndRendering();
		const auto      native = command.HelperSink();
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite |
		                       vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
		                       nullptr);
		for (const auto& copy: copies) {
			const vk::BufferCopy region {copy.source_offset,
			                             flat.offset + uint64_t {copy.flat_offset} * sizeof(uint32_t),
			                             sizeof(uint32_t)};
			native.copyBuffer(copy.source, flat.buffer, 1, &region);
		}
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eUniformRead;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr, 0,
		                       nullptr);
	}
	// KYTY_SRT_GPU_FILL_DIAG=1: one console line every 10 s while fills happen.
	if (!SrtGpuFillDiag()) {
		return flat;
	}
	static std::atomic<uint64_t> tables {0};
	static std::atomic<uint64_t> filled {0};
	static std::atomic<uint64_t> read_host {0};
	static std::atomic<int64_t>  last_ms {0};
	tables.fetch_add(1, std::memory_order_relaxed);
	filled.fetch_add(copies.size(), std::memory_order_relaxed);
	read_host.fetch_add(host_reads, std::memory_order_relaxed);
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	auto last = last_ms.load(std::memory_order_relaxed);
	if (last == 0) {
		last_ms.store(now, std::memory_order_relaxed);
	} else if (now - last >= 10000 && last_ms.compare_exchange_strong(last, now)) {
		std::printf("SrtGpuFill %.0fs: %" PRIu64 " flat tables, %" PRIu64
		            " dwords copied on the GPU, %" PRIu64 " read on the host\n",
		            static_cast<double>(now - last) / 1000.0, tables.exchange(0), filled.exchange(0),
		            read_host.exchange(0));
	}
	return flat;
}

// KYTY_UPLOAD_DEDUP=0 restores the previous per-draw upload memo (64 hashed slots, only with
// KYTY_RENDERER_BATCH=1).
static bool UploadDedupEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_UPLOAD_DEDUP");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}));
	return enabled;
}

// KYTY_UPLOAD_DEDUP_TABLE=1: every shader-data upload is also looked up in a hashed content table
// after its site's last upload (the U54 behaviour). By default only the site's last upload is
// checked: in the U54 captures the table matched about 14 of 9,200 lookups per flip at the Sky
// Garden start and 0.6 of 900 in the desert, so hashing every upload cost more than it saved.
static bool UploadDedupTableEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_UPLOAD_DEDUP_TABLE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}));
	return enabled;
}

vk::DescriptorBufferInfo RenderExecutor::UploadShaderData(std::span<const uint32_t> data,
                                                          uint32_t site, bool* fresh) {
	// These shader tables are read-only and ring allocations live until their GPU
	// tick retires. A wrap that reuses them must submit/wait and advance that tick.
	if (fresh != nullptr) {
		*fresh = true;
	}
	const auto reused = [fresh](const vk::DescriptorBufferInfo& allocation) {
		if (fresh != nullptr) {
			*fresh = false;
		}
		return allocation;
	};
	if (UploadDedupEnabled()) {
		// Identical bytes uploaded earlier in the same tick (the same command buffer) reuse that
		// allocation: the ring never overwrites a range during the tick that allocated it (a wrap
		// that reaches it first submits and completes that tick, which changes CurrentTick()),
		// and the tables are read-only. Equal contents then give equal descriptors, which the
		// push-descriptor shadow can skip.
		if (data.size_bytes() > 16u * 1024u) {
			return NativeUpload(m_context, data);
		}
		EXIT_IF(data.empty());
		const auto tick = m_context.GetCommandScheduler().CurrentTick();
		if (!UploadDedupTableEnabled()) {
			auto& last = m_upload_site_last[site % m_upload_site_last.size()];
			if (last.allocation.buffer != nullptr && last.tick == tick &&
			    std::ranges::equal(data, last.words)) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadLastHits);
				Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadReuseHits);
				Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadBytesAvoided,
				                          data.size_bytes());
				return last.allocation;
			}
			const auto allocation = NativeUpload(m_context, data);
			last.words.assign(data.begin(), data.end());
			last.allocation = allocation;
			// The upload may have submitted (ring wrap): the allocation belongs to the tick after it.
			last.tick = m_context.GetCommandScheduler().CurrentTick();
			Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadReuseMisses);
			return allocation;
		}
		auto&      last_slot = m_upload_last_slot[site % m_upload_last_slot.size()];
		if (const auto& last = m_upload_dedup[last_slot];
		    last.allocation.buffer != nullptr && last.tick == tick &&
		    std::ranges::equal(data, last.words)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadLastHits);
			Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadReuseHits);
			Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadBytesAvoided,
			                          data.size_bytes());
			return reused(last.allocation);
		}
		const auto hash = XXH3_64bits(data.data(), data.size_bytes());
		const auto slot = static_cast<uint32_t>(hash % m_upload_dedup.size());
		auto&      entry = m_upload_dedup[slot];
		last_slot        = slot;
		if (entry.allocation.buffer != nullptr && entry.tick == tick && entry.hash == hash &&
		    std::ranges::equal(data, entry.words)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadReuseHits);
			Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadBytesAvoided,
			                          data.size_bytes());
			return reused(entry.allocation);
		}
		const auto allocation = NativeUpload(m_context, data);
		entry.words.assign(data.begin(), data.end());
		entry.allocation = allocation;
		entry.hash       = hash;
		// The upload may have submitted (ring wrap): the allocation belongs to the tick after it.
		entry.tick = m_context.GetCommandScheduler().CurrentTick();
		Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadReuseMisses);
		return allocation;
	}
	if (!Common::RendererBatchEnabled() || data.size_bytes() > 16u * 1024u) {
		return NativeUpload(m_context, data);
	}
	EXIT_IF(data.empty());
	auto& scheduler = m_context.GetCommandScheduler();
	const auto tick = scheduler.CurrentTick();
	const auto hash = XXH3_64bits(data.data(), data.size_bytes());
	auto& cached = m_shader_uploads[hash % m_shader_uploads.size()];
	if (cached.allocation.buffer != nullptr && cached.tick == tick && cached.hash == hash &&
	    std::ranges::equal(data, cached.words)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadReuseHits);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadBytesAvoided, data.size_bytes());
		return reused(cached.allocation);
	}
	const auto allocation = NativeUpload(m_context, data);
	cached.words.assign(data.begin(), data.end());
	cached.allocation = allocation;
	cached.hash = hash;
	cached.tick = scheduler.CurrentTick();
	Profiler::CountFrameEvent(Profiler::FrameEvent::ShaderUploadReuseMisses);
	return allocation;
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.info.data.Empty()) {
		return;
	}
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	if (!Common::RendererBatchEnabled() || (!image.binding.is_bound && !image.binding.is_target)) {
		m_bound_images.push_back(id);
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	if (!Common::RendererBatchEnabled() || (!image.binding.is_bound && !image.binding.is_target)) {
		m_bound_images.push_back(id);
	}
	image.binding.is_target = true;
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) {
			if (image->binding.is_target) {
				// A bindless slot samples it again once it is readable (E5).
				NoteBindlessStateChange(*image);
			}
			image->binding = {};
		}
	}
	m_bound_images.clear();
}

// KYTY_GDS_TRACE=1 logs the user SGPRs of the first draws or dispatches of every program that
// binds GDS (up to 32 programs, 4 uses each). A DS_APPEND/DS_CONSUME takes its GDS range from
// M0, which such programs load from a user SGPR (the GdsCounterRange direct resource: base in
// bits 31:16, size in bits 15:0), so the log shows which counters a pass uses.
static void NoteGdsProgram(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                           const ShaderRecompiler::IR::ResourceSnapshot&   snapshot) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_GDS_TRACE");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	if (!enabled) {
		return;
	}
	constexpr uint32_t          MaxPrograms = 32;
	constexpr uint32_t          MaxUses     = 4;
	static std::atomic<uint64_t> hashes[MaxPrograms] {};
	static std::atomic<uint32_t> uses[MaxPrograms] {};
	uint32_t                     slot        = MaxPrograms;
	for (uint32_t i = 0; i < MaxPrograms; i++) {
		auto expected = uint64_t {0};
		if (hashes[i].load(std::memory_order_relaxed) == program.shader_hash ||
		    hashes[i].compare_exchange_strong(expected, program.shader_hash)) {
			slot = i;
			break;
		}
	}
	if (slot == MaxPrograms || uses[slot].fetch_add(1, std::memory_order_relaxed) >= MaxUses) {
		return;
	}
	std::string values;
	const auto  count = std::min<size_t>(snapshot.user_data.size(), 24);
	for (size_t i = 0; i < count; i++) {
		values += fmt::format(" s{}=0x{:08x}", program.user_data_base + i, snapshot.user_data[i]);
	}
	LOGF("GDS trace: program stage=%u hash=0x%016" PRIx64 " user data:%s\n",
	     static_cast<uint32_t>(program.stage), program.shader_hash, values.c_str());
}

bool RenderExecutor::RepeatStageTextures(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                         const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
                                         PreparedBindings&                               prepared) {
	if (!DrawSequenceEnabled(DrawSequencePart::Textures) || !TextureBindingMemo::Enabled()) {
		return false;
	}
	auto&      totals  = m_draw_sequence_totals;
	bool       xframe  = false; // the set came from the cross-frame store
	const auto matches = [&](const ShaderRecompiler::IR::CompiledShaderInfo*  set_program,
	                         const std::vector<ShaderRecompiler::IR::DescriptorValue>& words) {
		return set_program == &program && std::ranges::equal(words, snapshot.images);
	};
	if (matches(prepared.texture_program, prepared.texture_words)) {
		CommitStats::AddItems(CommitStats::Item::TexSetCurrent, 1);
	} else {
		auto&      history = prepared.texture_history;
		const auto found   = std::ranges::find_if(
            history, [&](const auto& set) { return matches(set.program, set.words); });
		CommitStats::AddItems(found == history.end() ? CommitStats::Item::TexSetMiss
		                                             : CommitStats::Item::TexSetHistory,
		                      1);
		if (found == history.end()) {
			// New words: the current set becomes the most recent earlier one, and the caller
			// resolves into the oldest one's vectors (their capacity is reused).
			if (prepared.texture_program != nullptr) {
				std::rotate(history.begin(), history.end() - 1, history.end());
				std::swap(history.front().program, prepared.texture_program);
				std::swap(history.front().words, prepared.texture_words);
				std::swap(history.front().images, prepared.images);
			}
			// KYTY_XFRAME_REUSE (drawPrep/xframeReuse.h): the set that fell out of the history
			// (now in prepared) moves into the cross-frame store, and a stored set of these words
			// becomes the current one, validated below like the current set's. Otherwise the caller
			// resolves into the vectors the store gave back.
			if (!XFrameReuse::Enabled() || !TakeStoredTextureSet(program, snapshot, prepared)) {
				prepared.texture_program = nullptr;
				return false;
			}
			xframe = true;
		} else {
			// An earlier set of these words becomes the current one (the current set takes its
			// place); its bindings are validated below like the current set's.
			std::swap(found->program, prepared.texture_program);
			std::swap(found->words, prepared.texture_words);
			std::swap(found->images, prepared.images);
			totals.texture_history_hits++;
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawSequenceTextureHistoryHits);
		}
	}
	if (prepared.images.empty()) {
		return false;
	}
	auto&      texture_cache = m_context.GetTextureCache();
	const bool xframe_verify = xframe && XFrameReuse::VerifyMode();
	const bool verify        = DrawSequenceVerifyMode() != 0 || xframe_verify;
	if (!m_texture_memo.TryRepeatResolve(texture_cache, prepared.images, !verify)) {
		totals.texture_misses++;
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawSequenceTextureMisses);
		if (xframe && XFrameReuse::StatsEnabled()) {
			XFrameReuse::GetTotals()
			    .failures[m_texture_memo.RepeatResolveFailure(texture_cache, prepared.images) & 7u]
			    .fetch_add(1, std::memory_order_relaxed);
		}
		return false;
	}
	if (xframe) {
		auto& xframe_totals = XFrameReuse::GetTotals();
		xframe_totals.repeated.fetch_add(1, std::memory_order_relaxed);
		xframe_totals.bindings.fetch_add(prepared.images.size(), std::memory_order_relaxed);
		// Verify mode: RebindImages compares the views it acquires with the entries' views.
		prepared.xframe_verify_views = xframe_verify;
	}
	totals.texture_repeats++;
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawSequenceTextureRepeats);
	CommitStats::AddItems(CommitStats::Item::TexSetRepeated, 1);
	const auto storage = [](const TextureBinding& binding) {
		return binding.desc.type == TextureCache::BindingType::Storage;
	};
	if (verify) {
		// KYTY_DRAW_SEQUENCE_VERIFY: every binding's full resolution must find its own entry again
		// (same image and description, no revalidation: same tag); it provides the result. Only
		// the GPU thread changes what the check read (short of a guest unmap racing the draw).
		totals.verify_checks++;
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawSequenceVerifyChecks);
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto&      binding       = prepared.images[i];
			const auto claimed_image = binding.image_id;
			const auto claimed_tag   = binding.memo_tag;
			SetNullCompareFuncs(ImageCompareFuncs(program, snapshot, i));
			ResolveTexture(program.info.images[i], snapshot.images[i], binding);
			if (xframe_verify) {
				XFrameReuse::GetTotals().verify_checks.fetch_add(1, std::memory_order_relaxed);
				if (binding.image_id != claimed_image || binding.memo_tag != claimed_tag) {
					XFrameReuse::ReportMismatch(binding.image_id != claimed_image ? "image" : "memo entry",
					                            i);
				}
			} else if (binding.image_id != claimed_image || binding.memo_tag != claimed_tag) {
				totals.verify_mismatches++;
				ReportDrawSequenceMismatch("a repeated stage texture binding");
			}
			BindImage(binding.image_id, storage(binding));
		}
		return true;
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		auto& binding = prepared.images[i];
		// ResolveTexture's work besides the lookup TryRepeatResolve repeated.
		binding.image_view = nullptr;
		binding.layout     = vk::ImageLayout::eUndefined;
		binding.mip_views.clear();
		if (HangTrace::Enabled()) {
			const auto descriptor =
			    DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[i]);
			if (!descriptor.IsNull()) {
				HangTrace::RecordTexture(descriptor.fields);
			}
		}
		BindImage(binding.image_id, storage(binding));
	}
	return true;
}

bool RenderExecutor::TakeStoredTextureSet(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                          const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
                                          PreparedBindings&                               prepared) {
	auto& totals = XFrameReuse::GetTotals();
	totals.new_sets.fetch_add(1, std::memory_order_relaxed);
	if (!m_xframe_sets) {
		m_xframe_sets = std::make_unique<XFrameTextureSets>(XFrameReuse::SlotCount());
	}
	auto& store = *m_xframe_sets;
	if (prepared.texture_program != nullptr) {
		store.Put(prepared.texture_program, prepared.texture_words, prepared.images);
		totals.stored.fetch_add(1, std::memory_order_relaxed);
	}
	const std::span<const ShaderRecompiler::IR::DescriptorValue> words(snapshot.images);
	const bool found = store.Take(&program, words, XFrameTextureSets::Hash(&program, words),
	                              prepared.texture_program, prepared.texture_words, prepared.images);
	totals.used.store(store.Used(), std::memory_order_relaxed);
	totals.evictions.store(store.Evictions(), std::memory_order_relaxed);
	if (found) {
		totals.found.fetch_add(1, std::memory_order_relaxed);
	}
	return found;
}

// ---------------------------------------------------------------------------------------------
// Bindless images (port of Senaxx's engine method, senaxx/wolverine c5a6961b6 and the commits
// before it; hardening from IDXTRI wolverine-v2). A shader whose image handle reads a T# from a
// guest descriptor heap at a runtime key samples set 1 (BindlessTable) at the slot the
// translation buffer gives for the key. Keys the shaders flag as pending are resolved here through
// the texture cache, at most a budget per frame; with the engine method the heap pages are
// write-watched and rewritten entries are settled again at the start of the next submission.
// ---------------------------------------------------------------------------------------------

namespace {

// Engine-method counters (SyncBindlessHeaps), logged and reset with the heap statistics.
struct BindlessEngineStats {
	std::atomic<uint64_t> syncs {0};
	std::atomic<uint64_t> written_bytes {0};
	std::atomic<uint64_t> view_changes {0};
	std::atomic<uint64_t> resource_changes {0};
	std::atomic<uint64_t> resettled {0};
	std::atomic<uint64_t> requested {0}; // keys draws asked for through feedback
	std::atomic<uint64_t> kept_old {0};  // changed T# not resolvable yet: the old view stays
	std::atomic<uint64_t> record_retries {0}; // record T# not written yet (KYTY_BINDLESS_RECORD_RETRY)
	std::atomic<uint64_t> record_unfilled {0}; // record keys read with a null pointer or an all-zero T#
	std::atomic<uint64_t> record_filled {0};   // unfilled records found filled on a later read
	std::atomic<uint64_t> host_ns {0};         // CPU time in ResolveBindlessRequests + ProcessRecordHeap
};
BindlessEngineStats g_bindless_engine_stats;

// Adds the scope's CPU time to BindlessEngineStats::host_ns (logged with KYTY_BINDLESS_HEAP_STATS).
struct BindlessHostTimer {
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	~BindlessHostTimer() {
		g_bindless_engine_stats.host_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                              std::chrono::steady_clock::now() - start)
		                                              .count()),
		                                          std::memory_order_relaxed);
	}
};

// KYTY_BINDLESS_ENGINE_VERIFY=1 (with the engine method): the rolling scan keeps running and logs
// every rewritten entry the write watch did not report.
bool BindlessEngineVerify() {
	static const bool verify = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE_VERIFY");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return verify;
}

// KYTY_BINDLESS_VERIFY=1: CommitBindings checks every resolved image of a stage's heaps, and stops
// if one it would have skipped needed a transition.
bool BindlessCommitVerify() {
	static const bool verify = std::getenv("KYTY_BINDLESS_VERIFY") != nullptr;
	return verify;
}

// T# bits that select a view of the same texture (RDNA 2 ISA 8.2.6): min LOD; dst_sel, base and
// last level, BC swizzle; base array; min LOD warning, perf mod, mip-stats enable; mip-stats slot
// and cache policy. Everything else describes the texture itself.
constexpr std::array<uint32_t, 8> BindlessViewMask {0u,          0x000fff00u, 0u,          0x0e0fffffu,
                                                    0x1fff0000u, 0x027fff00u, 0x000003ffu, 0u};

// The T#s of keys [first, first + count) of a plain heap whose keys' T#s are `stride` bytes apart
// from `table` on: one read when they are contiguous, one read of the covering span otherwise (a
// T# inside a larger per-key record, KYTY_BINDLESS_STRIDE; adapted from usa-ratt/KytyPS5-GT7
// a6cd035f0).
bool ReadHeapDescriptors(uint64_t table, uint32_t stride, uint64_t first, uint64_t count,
                         std::array<uint32_t, 8>* out) {
	if (count == 0) {
		return true;
	}
	if (stride == 32u) {
		return LibKernel::Memory::TryReadBacking(table + first * 32u, out, count * 32u);
	}
	thread_local std::vector<uint8_t> span;
	const uint64_t bytes = (count - 1u) * stride + 32u;
	span.resize(bytes);
	if (!LibKernel::Memory::TryReadBacking(table + first * stride, span.data(), bytes)) {
		return false;
	}
	for (uint64_t i = 0; i < count; i++) {
		std::memcpy(out[i].data(), span.data() + i * stride, 32u);
	}
	return true;
}

// Keys whose T# lies within `span` bytes of the table start (span / 32 for a plain T# array).
uint64_t HeapEntriesInSpan(uint64_t span, uint32_t stride) {
	return span < 32u ? 0 : (span - 32u) / stride + 1u;
}

// Diagnostics: the first time a draw uses a bindless heap, read its texture descriptors and log
// how many are valid, of which types, and roughly how much memory they would take as host
// textures. Nothing is created.
void SurveyBindlessHeap(const ShaderRecompiler::IR::BindlessHeapUse& use) {
	constexpr uint64_t MaxEntries = 1u << 20u;
	const uint64_t     span       = use.size > use.table_offset ? use.size - use.table_offset : 0;
	const uint64_t     entries = std::min<uint64_t>(HeapEntriesInSpan(span, use.stride), MaxEntries);
	uint64_t           valid = 0, bytes = 0, unreadable = 0;
	std::array<uint64_t, 16> types {};
	std::vector<std::array<uint32_t, 8>> chunk(1024u);
	for (uint64_t first = 0; first < entries; first += chunk.size()) {
		const auto count = std::min<uint64_t>(chunk.size(), entries - first);
		if (!ReadHeapDescriptors(use.base + use.table_offset, use.stride, first, count,
		                         chunk.data())) {
			unreadable += count;
			continue;
		}
		for (uint64_t i = 0; i < count; i++) {
			ShaderRecompiler::IR::DescriptorValue value {};
			value.dword_count = 8u;
			std::copy_n(chunk[i].begin(), 8, value.dwords.begin());
			const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
			if (descriptor.IsNull()) {
				continue;
			}
			valid++;
			types[static_cast<uint32_t>(descriptor.Type()) & 15u]++;
			const uint64_t width  = static_cast<uint64_t>(descriptor.Width5()) + 1u;
			const uint64_t height = static_cast<uint64_t>(descriptor.Height5()) + 1u;
			const uint64_t bpe =
			    std::max<uint32_t>(Prospero::NumBytesPerElement(descriptor.Format()), 1u);
			bytes += width * height * bpe * 4u / 3u;
		}
	}
	std::string histogram;
	for (uint32_t t = 0; t < types.size(); t++) {
		if (types[t] != 0) {
			histogram += fmt::format(" t{}={}", t, types[t]);
		}
	}
	LOGF("Bindless heap: base=0x%016" PRIx64 " size=0x%" PRIx64 " offset=%u entries=%" PRIu64
	     " valid=%" PRIu64 " unreadable=%" PRIu64 " est=%" PRIu64 "MB types:%s\n",
	     use.base, use.size, use.table_offset, entries, valid, unreadable, bytes >> 20u,
	     histogram.c_str());
}

// The set-1 array a bindless image samples (the same choice as the SPIR-V emitter's).
uint32_t BindlessBindingFor(const ShaderRecompiler::IR::ImageResource& resource) {
	using ShaderRecompiler::Decoder::ImageDimension;
	if (resource.cube) {
		return ShaderRecompiler::IR::BindlessImagesCube;
	}
	if (resource.dimension == ImageDimension::Dim3D) {
		return ShaderRecompiler::IR::BindlessImages3D;
	}
	if (resource.dimension == ImageDimension::Dim2DArray ||
	    resource.dimension == ImageDimension::Dim2DMsaaArray) {
		return ShaderRecompiler::IR::BindlessImages2DArray;
	}
	return ShaderRecompiler::IR::BindlessImages2D;
}

// A heap entry the bindless array of this binding can hold: a float-sampled, single-sample
// texture of a matching shape. Anything else stays the placeholder rather than reaching
// ResolveTexture, which exits on descriptors a draw would never bind that way.
// allow_integer (KYTY_RUNTIME_INT_IMAGES, runtime heaps): uint and sint textures too, which the
// shader's loads read through integer aliases of the binding.
bool BindlessCompatible(const ShaderTextureResource& descriptor, uint32_t binding,
                        bool allow_integer = false) {
	if (descriptor.IsNull()) {
		return false;
	}
	const auto numeric_class = Prospero::SampledTextureNumericClass(descriptor.Format());
	if (numeric_class != Prospero::TextureNumericClass::Float &&
	    !(allow_integer && (numeric_class == Prospero::TextureNumericClass::Uint ||
	                        numeric_class == Prospero::TextureNumericClass::Sint))) {
		return false;
	}
	switch (descriptor.Type()) {
		case Prospero::ImageType::kColor2D:
		case Prospero::ImageType::kColor2DArray:
			// KYTY_BINDLESS_3D_AS_ARRAY: a 3D image instruction reads a 2D or 2D-array T# as a
			// 2D array (BindlessSlotArrayFlag).
			return binding == ShaderRecompiler::IR::BindlessImages2D ||
			       binding == ShaderRecompiler::IR::BindlessImages2DArray ||
			       (binding == ShaderRecompiler::IR::BindlessImages3D &&
			        ShaderRecompiler::GetCodegenOptions().bindless_3d_as_array);
		// A cube T# sampled by a 2D-array instruction reads as the array of its faces (the GPU
		// addresses by the instruction's dimension; the shader computes face and slice itself, as
		// Miles Morales' sky material does with its cube array of 384 faces). Bound images do
		// the same (TextureType).
		case Prospero::ImageType::kCube:
			return binding == ShaderRecompiler::IR::BindlessImagesCube ||
			       binding == ShaderRecompiler::IR::BindlessImages2DArray;
		case Prospero::ImageType::kColor3D: return binding == ShaderRecompiler::IR::BindlessImages3D;
		default: return false;
	}
}

// Bytes of a T#'s mip chain from first_level down, ignoring tiling padding: an estimate.
uint64_t BindlessMipChainBytes(const ShaderTextureResource& descriptor, uint32_t first_level) {
	const auto     format = descriptor.Format();
	const auto     block  = Prospero::BlockCompressedBytesPerBlock(format);
	const auto     texel  = block != 0 ? 0u : Prospero::NumBytesPerElement(format);
	const uint64_t width  = descriptor.Width5() + 1u;
	const uint64_t height = descriptor.Height5() + 1u;
	uint64_t       layers = descriptor.Type() == Prospero::ImageType::kColor2D
	                            ? 1u
	                            : static_cast<uint64_t>(descriptor.Depth()) + 1u;
	if (descriptor.Type() == Prospero::ImageType::kCube) {
		layers = 6u;
	}
	uint64_t bytes = 0;
	for (uint32_t level = first_level; level <= descriptor.MaxMip(); level++) {
		const auto w = std::max<uint64_t>(width >> level, 1u);
		const auto h = std::max<uint64_t>(height >> level, 1u);
		bytes += layers * (block != 0 ? ((w + 3u) / 4u) * ((h + 3u) / 4u) * block : w * h * texel);
	}
	return bytes;
}

// Diagnostics for the engine method: KYTY_BINDLESS_HEAP_STATS=1 logs every 300 frames, per heap,
// the entries that hold a texture this heap's view type can sample, the distinct textures among
// them, their full and resident (at or above each T#'s min LOD) mip chains and those of the
// textures draws asked for, the rewrites since the last line (view fields only, or the
// resource), and the entries enabling mip statistics; then the write watch's totals.
void LogBindlessHeapStats(BindlessTable& table, uint64_t frame) {
	static const bool enabled = std::getenv("KYTY_BINDLESS_HEAP_STATS") != nullptr;
	static uint64_t   next    = 0;
	if (!enabled || frame < next) {
		return;
	}
	next = frame + 300;
	static std::unordered_map<const BindlessHeap*, std::vector<std::array<uint32_t, 8>>> previous;
	std::vector<std::array<uint32_t, 8>> words;
	for (const auto& heap: table.Heaps()) {
		const auto entries = std::min(heap.guest_entries, static_cast<uint32_t>(heap.settled.size()));
		if (entries == 0 || heap.record_pointer != UINT32_MAX || heap.runtime) {
			continue;
		}
		words.resize(entries);
		if (!ReadHeapDescriptors(heap.base + heap.table_offset, heap.stride, 0, entries,
		                         words.data())) {
			continue;
		}
		uint32_t valid = 0, mip_stats = 0, settled = 0, view_changes = 0, resource_changes = 0;
		struct Texture {
			uint64_t all      = 0;
			uint64_t resident = 0;
			bool     sampled  = false;
		};
		std::unordered_map<uint64_t, Texture> textures;
		auto&                                 last = previous[&heap];
		for (uint32_t key = 0; key < entries; key++) {
			settled += heap.settled[key] != 0 ? 1u : 0u;
			if (key < last.size() && last[key] != words[key]) {
				bool resource = false;
				for (uint32_t i = 0; i < 8; i++) {
					resource |= ((last[key][i] ^ words[key][i]) & ~BindlessViewMask[i]) != 0;
				}
				(resource ? resource_changes : view_changes)++;
			}
			ShaderTextureResource descriptor {};
			std::copy_n(words[key].data(), 8, descriptor.fields);
			if (!BindlessCompatible(descriptor, heap.binding)) {
				continue;
			}
			valid++;
			mip_stats += descriptor.MipStatsCntEn() ? 1u : 0u;
			const auto first = std::max<uint32_t>(descriptor.BaseLevel(), descriptor.MinLod() >> 8u);
			auto&      texture = textures[descriptor.Base40()];
			texture.all        = std::max(texture.all, BindlessMipChainBytes(descriptor, 0));
			texture.resident   = std::max(texture.resident, BindlessMipChainBytes(descriptor, first));
			texture.sampled |= heap.settled[key] != 0;
		}
		uint64_t all = 0, resident = 0, sampled = 0, sampled_textures = 0;
		for (const auto& [address, texture]: textures) {
			all += texture.all;
			resident += texture.resident;
			if (texture.sampled) {
				sampled += texture.resident;
				sampled_textures++;
			}
		}
		last = words;
		LOGF("Bindless heap stats: frame=%" PRIu64 " base=0x%" PRIx64 "+0x%x binding=%u entries=%u "
		     "valid=%u textures=%zu sampled_textures=%" PRIu64 " all_mb=%" PRIu64
		     " resident_mb=%" PRIu64 " sampled_mb=%" PRIu64 " settled=%u rewrites_view=%u "
		     "rewrites_resource=%u mip_stats=%u\n",
		     frame, heap.base, heap.table_offset, heap.binding, entries, valid, textures.size(),
		     sampled_textures, all >> 20u, resident >> 20u, sampled >> 20u, settled, view_changes,
		     resource_changes, mip_stats);
	}
	auto&      stats   = g_bindless_engine_stats;
	const auto traffic = table.TakeSnapshotTraffic();
	const auto scans   = table.TakeScanCounts();
	LOGF("Bindless engine stats: frame=%" PRIu64 " syncs=%" PRIu64 " written=%" PRIu64
	     " KiB view=%" PRIu64 " resource=%" PRIu64 " resettled=%" PRIu64 " requested=%" PRIu64
	     " unregistered_keys=%" PRIu64 " kept_old=%" PRIu64 " record_retries=%" PRIu64 " record_unfilled=%" PRIu64 " record_filled=%" PRIu64 " host_ms=%.1f snapshot_words=%u snapshot_kib=%" PRIu64 " flagged_chunks=%" PRIu64 " scanned=%" PRIu64 " nonzero=%" PRIu64 "\n",
	     frame, stats.syncs.exchange(0), stats.written_bytes.exchange(0) >> 10u,
	     stats.view_changes.exchange(0), stats.resource_changes.exchange(0),
	     stats.resettled.exchange(0), stats.requested.exchange(0), table.TakeUnregisteredKeys(),
	     stats.kept_old.exchange(0), stats.record_retries.exchange(0), stats.record_unfilled.exchange(0), stats.record_filled.exchange(0), static_cast<double>(stats.host_ns.exchange(0)) / 1e6, table.UsedEntries(), traffic.first >> 10u, traffic.second, scans.first, scans.second);
}

} // namespace

bool BindlessKeyCompatible(const ShaderTextureResource& descriptor, uint32_t binding) {
	return BindlessCompatible(descriptor, binding);
}

uint64_t RenderExecutor::BindlessFrame() const {
	return static_cast<uint64_t>(m_context.GetGpu().GetFrameNum());
}

// KYTY_R128_TABLE_IMAGES: a heap of 128-bit T#s (R128=1) holds four words per key; the four read
// past them belong to the next entry and are zeroed, so they neither change the view nor make an
// unchanged key look rewritten.
static void MaskR128HeapWords(const BindlessHeap& heap, uint32_t* words) {
	if (heap.resource.r128 && ShaderRecompiler::GetCodegenOptions().r128_table_images) {
		std::fill_n(words + 4, 4, 0u);
	}
}

// Key `key`'s T#: in the heap itself, or behind the key's record pointer.
static bool ReadBindlessEntry(const BindlessHeap& heap, uint32_t key, uint32_t* words) {
	if (heap.runtime) {
		// KYTY_RUNTIME_IMAGES: the T# a shader requested (BindlessTable::RuntimeKey), not a guest
		// heap entry.
		if (key >= heap.runtime_dwords.size()) {
			return false;
		}
		std::copy_n(heap.runtime_dwords[key].begin(), 8, words);
		return true;
	}
	if (heap.record_pointer == UINT32_MAX) {
		if (!LibKernel::Memory::TryReadBacking(
		        heap.base + heap.table_offset + static_cast<uint64_t>(key) * heap.stride, words, 32u)) {
			return false;
		}
		MaskR128HeapWords(heap, words);
		return true;
	}
	const bool nested = heap.nested_vsharp != UINT32_MAX && heap.nested_keys != 0;
	const auto record = nested ? key / heap.nested_keys : key;
	std::array<uint32_t, 2> pointer {};
	if (!LibKernel::Memory::TryReadBacking(heap.base + static_cast<uint64_t>(record) * heap.stride +
	                                           heap.record_pointer,
	                                       pointer.data(), 8u)) {
		return false;
	}
	auto address =
	    static_cast<uint64_t>(pointer[0]) | (static_cast<uint64_t>(pointer[1] & 0xffffu) << 32u);
	if (address == 0) {
		return false;
	}
	if (nested) {
		// The V# at nested_vsharp past the pointer, the heap V# at nested_heap of its buffer, and
		// the inner key's T# in that heap.
		std::array<uint32_t, 4> vsharp {};
		if (!LibKernel::Memory::TryReadBacking(address + heap.nested_vsharp, vsharp.data(), 16u)) {
			return false;
		}
		// nested_heap UINT32_MAX (KYTY_HEAP_KEY_MASK, PPSA03671's material callbacks): that V# is
		// the heap itself.
		const auto buffer =
		    static_cast<uint64_t>(vsharp[0]) | (static_cast<uint64_t>(vsharp[1] & 0xffffu) << 32u);
		if (heap.nested_heap != UINT32_MAX &&
		    (buffer == 0 || static_cast<uint64_t>(heap.nested_heap) + 16u > vsharp[2] ||
		     !LibKernel::Memory::TryReadBacking(buffer + heap.nested_heap, vsharp.data(), 16u))) {
			return false;
		}
		const auto inner = key % heap.nested_keys;
		const auto heap_base =
		    static_cast<uint64_t>(vsharp[0]) | (static_cast<uint64_t>(vsharp[1] & 0xffffu) << 32u);
		const auto end = static_cast<uint64_t>(heap.table_offset) + uint64_t {inner} * 32u + 32u;
		return heap_base != 0 && end <= vsharp[2] &&
		       LibKernel::Memory::TryReadBacking(heap_base + heap.table_offset +
		                                             uint64_t {inner} * 32u,
		                                         words, 32u);
	}
	return LibKernel::Memory::TryReadBacking(address + heap.table_offset, words, 32u);
}

// KYTY_BINDLESS_RECORD_RETRY (live; ResolveBindlessKey).
static Live::Switch g_bindless_record_retry("KYTY_BINDLESS_RECORD_RETRY", [](const char* value) -> int64_t {
	return value == nullptr || value[0] != '0' ? 1 : 0;
});

// KYTY_BINDLESS_SHARE_SLOTS=1 (startup, default off): every key resolved through ResolveBindlessKey (feedback,
// resettle, prefill) shares the slot of an earlier key with the same T# words and view binding
// (BindlessTable::FindSharedSlot / NoteSharedSlot, the KYTY_BINDLESS_PREFILL bookkeeping, refcounted in FreeSlot)
// instead of taking a new slot. Ghost of Yotei builds new per-frame heaps of the same textures: each resolve took
// a new slot, ~400 slots/s on the menu (KYTY_BINDLESS_SLOT_LOG); with KYTY_BINDLESS_SLOT_REUSE=0 binding 0 was full
// (65536) after ~160 s and the menu broke right after (leaf band, grey floor, red hilt); with reuse the slots
// recycle continuously.
static bool BindlessTouchImages() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_TOUCH");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

static bool BindlessShareSlots() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_SHARE_SLOTS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool RenderExecutor::ResolveBindlessKey(BindlessHeap& heap, uint32_t key) {
	auto&                                 table = m_context.GetBindlessTable();
	ShaderRecompiler::IR::DescriptorValue value {};
	value.dword_count             = 8u;
	m_bindless_transient_failure  = false;
	heap.settled[key]             = 1;
	heap.descriptors[key]         = {};
	// KYTY_BINDLESS_WATCH_KEY: each resolve of that key, with its T# address (behind the record
	// pointer), whether the GPU wrote it, the words read and the outcome.
	const bool watched = key == BindlessTable::WatchKey();
	const auto watch   = [&](const char* outcome) {
		if (!watched) return;
		uint64_t address = 0;
		if (heap.record_pointer != UINT32_MAX && heap.nested_vsharp == UINT32_MAX) {
			std::array<uint32_t, 2> pointer {};
			if (LibKernel::Memory::TryReadBacking(heap.base + uint64_t {key} * heap.stride + heap.record_pointer,
			                                      pointer.data(), 8u)) {
				address = (static_cast<uint64_t>(pointer[0]) | (static_cast<uint64_t>(pointer[1] & 0xffffu) << 32u)) +
				          heap.table_offset;
			}
		}
		const bool dirty = address != 0 && m_context.GetBufferCache().HasGpuDirtyBytes(address, 32u);
		LOGF("Bindless watch resolve: frame=%" PRIu64 " heap 0x%" PRIx64 "+%u b%u key %u T# at 0x%" PRIx64
		     "%s words %08x %08x %08x %08x %08x %08x %08x %08x: %s\n",
		     BindlessFrame(), heap.base, heap.table_offset, heap.binding, key, address, dirty ? " (gpu-dirty)" : "",
		     value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3], value.dwords[4], value.dwords[5],
		     value.dwords[6], value.dwords[7], outcome);
	};
	struct WatchExit {
		const decltype(watch)& log;
		const char*            outcome = "resolved";
		~WatchExit() { log(outcome); }
	} watch_exit {watch};
	// KYTY_BINDLESS_RECORD_RETRY=0 (default 1, live): a record pointer heap's key whose pointer is null or
	// unreadable, or whose T# reads as all zeros, may be a record the guest has not filled yet rather
	// than a final answer. The key samples the placeholder (or keeps its old view, ResettleBindlessKey)
	// and is read again 4, 8, 16, 32 and 64 frames later (m_bindless_record_retry); its descriptor is
	// left as zeros, so revalidation also catches a T# written later. Miles Morales rebuilds its two
	// ping-pong call tables with new material pointers; the T# behind a moved pointer read as zeros,
	// the key stayed on the placeholder (revalidation compared against the stored zeros only when its
	// window came round), and the RT reflection of the sky went dark in every other frame.
	const bool record_retry = g_bindless_record_retry.Get() != 0;
	m_bindless_record_unfilled = false;
	const auto not_written_yet = [&](const char* outcome) {
		watch_exit.outcome           = outcome;
		heap.descriptors[key]        = {};
		m_bindless_transient_failure = true;
		m_bindless_record_unfilled   = true;
		table.SetTranslation(heap, key, 0u);
		QueueBindlessRecordRetry(heap, key, m_bindless_record_attempt);
		return false;
	};
	const bool record_heap = heap.record_pointer != UINT32_MAX && !heap.runtime;
	if (!ReadBindlessEntry(heap, key, value.dwords.data())) {
		if (record_heap) g_bindless_engine_stats.record_unfilled.fetch_add(1, std::memory_order_relaxed);
		if (record_retry && record_heap) {
			return not_written_yet("unreadable (retry later)");
		}
		watch_exit.outcome = "unreadable";
		static const bool log_unreadable = std::getenv("KYTY_BINDLESS_KEY_LOG") != nullptr;
		static std::atomic<uint32_t> logged = 0;
		if (log_unreadable && logged.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("Bindless key: heap 0x%016" PRIx64 " key %u: entry unreadable (placeholder)\n", heap.base, key);
		}
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	if (record_heap &&
	    std::ranges::all_of(value.dwords.begin(), value.dwords.begin() + 8, [](uint32_t w) { return w == 0u; })) {
		g_bindless_engine_stats.record_unfilled.fetch_add(1, std::memory_order_relaxed);
		if (record_retry) return not_written_yet("zero T# (retry later)");
	}
	std::copy_n(value.dwords.data(), 8, heap.descriptors[key].begin());
	// KYTY_BINDLESS_SHARE_SLOTS: a T# another key already resolved shares that key's slot.
	if (BindlessShareSlots() && !heap.runtime) {
		if (const auto* found = table.FindSharedSlot(heap.binding, heap.descriptors[key], heap.view_key); found != nullptr) {
			const auto id    = found->image;
			const auto entry = found->entry;
			auto*      image = m_context.GetTextureCache().m_slot_images.try_get(id);
			if (image != nullptr && image->registered && image->bindless_pinned) {
				heap.slots[key]  = entry;
				heap.images[key] = id;
				if (std::find(heap.resolved.begin(), heap.resolved.end(), id) == heap.resolved.end()) {
					heap.resolved.push_back(id);
				}
				heap.unchecked.push_back(id);
				table.AddImageReference(id, heap, key);
				table.AddSharedSlotRef(BindlessTable::SlotBinding(heap, entry), BindlessTable::SlotIndex(entry));
				table.SetTranslation(heap, key, entry);
				watch_exit.outcome = "shared slot";
				return true;
			}
		}
	}
	const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	// A 1D image instruction's heap (it shares the 2D binding) has no view of a 2D texture: such a
	// key stays on the placeholder instead of reaching an invalid 1D view of a 2D image (a call
	// table's record T#s, Miles Morales, once their heaps got translation regions).
	const auto dimension    = heap.resource.dimension;
	const bool one_d_heap   = dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D ||
	                          dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray;
	const bool integer_views = heap.runtime && ShaderRecompiler::GetCodegenOptions().runtime_int_images;
	if (one_d_heap || !BindlessCompatible(descriptor, heap.binding, integer_views)) {
		// KYTY_BINDLESS_INCOMPATIBLE_LOG=<n> (diagnostic): the first n keys left on the placeholder
		// because their T# does not fit the binding's view type.
		static const uint32_t log_limit = [] {
			const char* env = std::getenv("KYTY_BINDLESS_INCOMPATIBLE_LOG");
			return env != nullptr ? static_cast<uint32_t>(std::strtoul(env, nullptr, 10)) : 0u;
		}();
		static std::atomic<uint32_t> logged = 0;
		if (logged.fetch_add(1, std::memory_order_relaxed) < log_limit) {
			LOGF("Bindless: key %u of heap 0x%016" PRIx64 " (binding %u) incompatible: T# base 0x%" PRIx64
			     " type %u format %u %ux%u depth %u null %d\n",
			     key, heap.base, heap.binding, descriptor.Base40(), static_cast<uint32_t>(descriptor.Type()),
			     static_cast<uint32_t>(descriptor.Format()), static_cast<uint32_t>(descriptor.Width5()) + 1u,
			     static_cast<uint32_t>(descriptor.Height5()) + 1u, static_cast<uint32_t>(descriptor.Depth()),
			     descriptor.IsNull() ? 1 : 0);
		}
		watch_exit.outcome = one_d_heap ? "1D heap" : "incompatible";
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	// KYTY_BINDLESS_3D_AS_ARRAY (adapted from KytyPS5-Frank fed216ce8): the GPU addresses by the
	// T# type, so a 3D instruction's 2D or 2D-array texture is a 2D array at layer round(z). It
	// lives in the 2D-array binding; the translation entry's flag tells the shader.
	const bool as_array = heap.binding == ShaderRecompiler::IR::BindlessImages3D &&
	                      (descriptor.Type() == Prospero::ImageType::kColor2D ||
	                       descriptor.Type() == Prospero::ImageType::kColor2DArray);
	const uint32_t slot_binding =
	    as_array ? ShaderRecompiler::IR::BindlessImages2DArray : heap.binding;
	auto resource = heap.resource;
	if (as_array) {
		resource.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2DArray;
	}
	if (integer_views) {
		// KYTY_RUNTIME_INT_IMAGES: an integer T#'s view has its own component type.
		if (const auto numeric_class = Prospero::SampledTextureNumericClass(descriptor.Format());
		    numeric_class != Prospero::TextureNumericClass::Float) {
			resource.numeric_class = numeric_class;
			resource.depth_compare = false;
		}
	}
	auto& texture_cache = m_context.GetTextureCache();
	// KYTY_BINDLESS_HOLD_EXTENT=1 (default 0; on in the PPSA03671 preset; adapted from upstream
	// 95bbc9c65): heaps hold a descriptor per dynamic-resolution extent of a render target, and
	// only one extent at an address holds data. A key whose texture has no image while another
	// extent at its address is pinned by a heap creates no second image (which would be filled
	// from stale guest bytes and fight the render target over its memory): it stays pending and
	// is asked for again by the next draw that samples it, until such an image exists.
	static const bool hold_extent = [] {
		const char* env = std::getenv("KYTY_BINDLESS_HOLD_EXTENT");
		return env != nullptr && env[0] == '1';
	}();
	if (hold_extent) {
		const auto desc       = BuildTextureDescription(resource, descriptor);
		const bool conversion = TextureGetSurfaceFormatInfo(descriptor.Format()).conversion_format !=
		                        Prospero::BufferFormat::kInvalid;
		if (texture_cache.HeapExtentHeld(desc.info, conversion)) {
			heap.settled[key]            = 0;
			m_bindless_transient_failure = true;
			watch_exit.outcome           = "held extent";
			table.SetTranslation(heap, key, ShaderRecompiler::IR::BindlessPending);
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 8) {
				LOGF("Bindless: key %u of heap 0x%016" PRIx64 " (binding %u) held: another extent "
				     "at 0x%016" PRIx64 " is pinned (KYTY_BINDLESS_HOLD_EXTENT)\n",
				     key, heap.base, heap.binding, desc.info.data.address);
			}
			return false;
		}
	}
	// KYTY_BINDLESS_CLASS_LOG=<n> (diagnostic, default 0): the first n keys whose T# holds an
	// integer format while the heap's image resource samples float (the bindless arrays have one
	// numeric class per resource; Erlou93/KytyPS5-Yotei KYTY_PERF_BINDLESS_CLASSES: Ghost of Yotei's
	// terrain tile records hold 8_UINT material maps among float textures).
	{
		static const uint32_t class_log_limit = [] {
			const char* env = std::getenv("KYTY_BINDLESS_CLASS_LOG");
			return env != nullptr ? static_cast<uint32_t>(std::strtoul(env, nullptr, 10)) : 0u;
		}();
		if (class_log_limit != 0u) {
			const auto type = ShaderRecompiler::Format::GetFormatInfo(descriptor.Format()).type;
			const bool integer = type == ShaderRecompiler::Format::ComponentType::Uint ||
			                     type == ShaderRecompiler::Format::ComponentType::Sint;
			static std::atomic<uint32_t> class_logged = 0;
			if (integer && resource.numeric_class != Prospero::TextureNumericClass::Uint &&
			    resource.numeric_class != Prospero::TextureNumericClass::Sint &&
			    class_logged.fetch_add(1, std::memory_order_relaxed) < class_log_limit) {
				LOGF("Bindless class: key %u of heap 0x%016" PRIx64 " (binding %u, record pointer %u, table offset %u) "
				     "holds integer T# format %u %ux%u; the resource samples numeric class %u\n",
				     key, heap.base, heap.binding, heap.record_pointer, heap.table_offset,
				     static_cast<uint32_t>(descriptor.Format()), static_cast<uint32_t>(descriptor.Width5()) + 1u,
				     static_cast<uint32_t>(descriptor.Height5()) + 1u,
				     static_cast<uint32_t>(resource.numeric_class));
			}
		}
	}
	TextureBinding binding;
	ResolveTexture(resource, value, binding);
	auto* image         = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || image->info.data.Empty()) {
		watch_exit.outcome = "no image";
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	BindImage(binding.image_id, false);
	const auto view = texture_cache.FindTexture(binding.image_id, binding.desc);
	image           = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image != nullptr && !image->registered) {
		// Dropped by the cache (its destruction waits for the GPU): a slot holding its view would
		// outlive it, since its unregistration is over. The key is asked for again next frame.
		heap.settled[key]            = 0;
		m_bindless_transient_failure = true;
		watch_exit.outcome           = "image dropped";
		table.SetTranslation(heap, key, ShaderRecompiler::IR::BindlessPending);
		return false;
	}
	if (image == nullptr || view == nullptr) {
		watch_exit.outcome = "no view";
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	const auto slot = table.AllocateSlot(slot_binding);
	if (slot == 0) {
		m_bindless_transient_failure = true;
		watch_exit.outcome           = "no slot";
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 8) {
			LOGF("Bindless: binding %u has no free slot (%u per array); key %u of heap 0x%016" PRIx64
			     " samples the placeholder\n",
			     slot_binding, table.ImagesPerArray(), key, heap.base);
		}
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	const auto layout = image->info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                                          : vk::ImageLayout::eShaderReadOnlyOptimal;
	table.WriteSlot(slot_binding, slot, view, layout);
	{
		// KYTY_BINDLESS_KEY_LOG=<n> (diagnostic): the first n keys resolved, with their texture.
		static const uint32_t log_limit = [] {
			const char* env = std::getenv("KYTY_BINDLESS_KEY_LOG");
			return env != nullptr ? static_cast<uint32_t>(std::strtoul(env, nullptr, 10)) : 0u;
		}();
		static std::atomic<uint32_t> logged = 0;
		if (logged.fetch_add(1, std::memory_order_relaxed) < log_limit) {
			LOGF("Bindless key: heap 0x%016" PRIx64 " key %u binding %u slot %u: T# base 0x%" PRIx64
			     " type %u format %u %ux%u depth %u mips %u-%u -> image %u (%ux%u vk%u layers %u)\n",
			     heap.base, key, slot_binding, slot, descriptor.Base40(), static_cast<uint32_t>(descriptor.Type()),
			     static_cast<uint32_t>(descriptor.Format()), static_cast<uint32_t>(descriptor.Width5()) + 1u,
			     static_cast<uint32_t>(descriptor.Height5()) + 1u, static_cast<uint32_t>(descriptor.Depth()),
			     static_cast<uint32_t>(descriptor.BaseLevel()), static_cast<uint32_t>(descriptor.LastLevel()),
			     binding.image_id.index, image->info.extent.width, image->info.extent.height,
			     static_cast<uint32_t>(image->backing.format), image->info.resources.layers);
		}
	}
	const uint32_t entry = as_array ? (slot | ShaderRecompiler::IR::BindlessSlotArrayFlag) : slot;
	heap.slots[key]         = entry;
	heap.images[key]        = binding.image_id;
	image->bindless_pinned  = true;
	image->bindless_self_id = binding.image_id;
	image->usage.texture    = true;
	if (std::find(heap.resolved.begin(), heap.resolved.end(), binding.image_id) ==
	    heap.resolved.end()) {
		heap.resolved.push_back(binding.image_id);
	}
	heap.unchecked.push_back(binding.image_id);
	table.AddImageReference(binding.image_id, heap, key);
	table.SetTranslation(heap, key, entry);
	if (BindlessShareSlots() && !heap.runtime) {
		table.NoteSharedSlot(heap, heap.descriptors[key], entry, binding.image_id);
	}
	return true;
}

// KYTY_BINDLESS_PREFILL=1 (default 0; =2 also logs the first heaps and every 1024th): a heap of at
// most KYTY_BINDLESS_PREFILL_MAX keys (default 256) has every key settled at its first use, before
// the draw or dispatch that reads it, instead of sampling the placeholder until the feedback of a
// later frame asks for it. A T# that a prefilled key of any heap already resolved shares that slot
// (BindlessTable::FindSharedSlot), so a new heap of known textures takes no new slots or uploads.
// Ghost of Yotei writes its terrain tile records (56 records of 440 bytes, the height map T# at
// offset 0) to a new address every frame: each frame's heap was new, its keys never resolved, and
// "Generate Terrain Index Buffers" (0041b6a0) read the placeholder's heights, culled every tile
// and left the terrain without instances (measured, RenderDoc 2026-11-04; Erlou93/KytyPS5-Yotei
// resolves new heaps the same way, PrefillBindlessHeap and descriptor_slots).
static uint32_t BindlessPrefillMode() {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_BINDLESS_PREFILL");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	return mode;
}

static uint32_t BindlessPrefillMax() {
	static const uint32_t max = [] {
		const char* value = std::getenv("KYTY_BINDLESS_PREFILL_MAX");
		return value != nullptr && value[0] != '\0' ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10))
		                                            : 256u;
	}();
	return max;
}

// KYTY_BINDLESS_REFILL (default on with KYTY_BINDLESS_PREFILL; =0 off, the old behaviour; adapted
// from Erlou93/KytyPS5-Yotei's table path, which translates a texture table as a whole when a draw
// that reads it is prepared and again whenever its bytes changed): when a draw or dispatch that
// reads a heap is prepared, the heap's key table (at most KYTY_BINDLESS_REFILL_MAX keys, default
// 4096) is hashed once per command-buffer tick, and when its bytes changed, its unresolved keys are
// settled now (PrefillBindlessHeap), instead of being sampled as the pending placeholder until a
// later frame's feedback asks for them. Ghost of Yotei rewrites the T#s of its terrain material
// tables (2664 keys of 96 bytes, two tables in alternate frames) while virtual-texture tiles
// stream; the keys new in a frame read transparent black and the terrain showed white snow
// patches on the mountain and white sparkles on the water for a few frames (41-80 times per 300 s
// standing at the flower field; KYTY_BINDLESS_DEBUG_COLORS=1 hid them, Erlou93/KytyPS5-Yotei has
// none). Settling at the draw, not when the guest's write is noticed: the guest may still be
// writing a table for a later frame then (that variant, resolving half-written T#s, made it worse).
static std::atomic<uint64_t> g_bindless_refills {0}; // tables whose bytes changed, re-prefilled

static bool BindlessRefillEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_REFILL");
		return BindlessPrefillMode() != 0u && (value == nullptr || std::strcmp(value, "0") != 0);
	}();
	return enabled;
}

static uint32_t BindlessRefillMax() {
	static const uint32_t max = [] {
		const char* value = std::getenv("KYTY_BINDLESS_REFILL_MAX");
		return value != nullptr && value[0] != '\0' ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 4096u;
	}();
	return max;
}

void RenderExecutor::RefillBindlessHeap(BindlessHeap& heap, uint32_t entries) {
	const auto tick = m_context.GetCommandScheduler().CurrentTick();
	if (heap.refill_tick == tick || entries == 0 || heap.stride == 0) {
		return;
	}
	heap.refill_tick  = tick;
	const auto bytes  = uint64_t {entries - 1u} * heap.stride + 32u;
	m_bindless_refill_bytes.resize(bytes);
	if (!LibKernel::Memory::TryReadBacking(heap.base + heap.table_offset, m_bindless_refill_bytes.data(), bytes)) {
		return;
	}
	const auto digest = XXH3_64bits(m_bindless_refill_bytes.data(), bytes);
	if (digest == heap.refill_digest) {
		return;
	}
	heap.refill_digest = digest;
	g_bindless_refills.fetch_add(1, std::memory_order_relaxed);
	PrefillBindlessHeap(heap, entries);
}

void RenderExecutor::PrefillBindlessHeap(BindlessHeap& heap, uint32_t entries) {
	auto&          table   = m_context.GetBindlessTable();
	auto&          cache   = m_context.GetTextureCache();
	const uint32_t keys    = std::min({entries, heap.entries, static_cast<uint32_t>(heap.settled.size())});
	uint32_t       shared  = 0;
	uint32_t       made    = 0;
	uint32_t       failed  = 0;
	uint32_t       skipped = 0;
	// New textures per heap (each may upload and detile); the rest wait for feedback as before.
	constexpr uint32_t Budget = 64;
	for (uint32_t key = 0; key < keys; key++) {
		if (heap.settled[key] != 0) {
			continue;
		}
		std::array<uint32_t, 8> words {};
		if (!ReadBindlessEntry(heap, key, words.data()) ||
		    std::ranges::all_of(words, [](uint32_t w) { return w == 0u; })) {
			skipped++;
			continue;
		}
		if (const auto* found = table.FindSharedSlot(heap.binding, words, heap.view_key); found != nullptr) {
			const auto id    = found->image;
			const auto entry = found->entry;
			auto*      image = cache.m_slot_images.try_get(id);
			if (image != nullptr && image->registered && image->bindless_pinned) {
				heap.slots[key]       = entry;
				heap.images[key]      = id;
				heap.settled[key]     = 1;
				heap.descriptors[key] = words;
				if (std::find(heap.resolved.begin(), heap.resolved.end(), id) == heap.resolved.end()) {
					heap.resolved.push_back(id);
				}
				heap.unchecked.push_back(id);
				table.AddImageReference(id, heap, key);
				table.AddSharedSlotRef(BindlessTable::SlotBinding(heap, entry), BindlessTable::SlotIndex(entry));
				table.SetTranslation(heap, key, entry);
				shared++;
				continue;
			}
		}
		if (made + failed >= Budget) {
			skipped++;
			continue;
		}
		if (ResolveBindlessKey(heap, key)) {
			table.NoteSharedSlot(heap, heap.descriptors[key], heap.slots[key], heap.images[key]);
			made++;
		} else {
			failed++;
			if (BindlessPrefillMode() >= 2u) {
				static std::atomic<uint32_t> failures_logged {0};
				if (failures_logged.fetch_add(1, std::memory_order_relaxed) < 400u) {
					ShaderRecompiler::IR::DescriptorValue value {};
					value.dword_count = 8u;
					std::copy_n(words.begin(), 8, value.dwords.begin());
					const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
					LOGF("Bindless prefill failed: frame=%" PRIu64 " heap 0x%" PRIx64 "+%u key %u: T# base 0x%" PRIx64
					     " type %u format %u %ux%u transient %d slot 0x%x\n",
					     BindlessFrame(), heap.base, heap.table_offset, key, descriptor.Base40(),
					     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(descriptor.Format()),
					     static_cast<uint32_t>(descriptor.Width5()) + 1u, static_cast<uint32_t>(descriptor.Height5()) + 1u,
					     m_bindless_transient_failure ? 1 : 0, heap.slots[key]);
				}
			}
		}
	}
	if (BindlessPrefillMode() >= 2u) {
		static std::atomic<uint32_t> logged {0};
		const auto                   n = logged.fetch_add(1, std::memory_order_relaxed);
		// Mode 3: every heap.
		if (n < 64u || (n & 1023u) == 0u || BindlessPrefillMode() >= 3u) {
			LOGF("Bindless prefill: frame=%" PRIu64 " heap 0x%" PRIx64 "+%u stride %u binding %u keys %u: shared %u "
			     "resolved %u failed %u skipped %u forgotten %" PRIu64 " (#%u)\n",
			     BindlessFrame(), heap.base, heap.table_offset, heap.stride, heap.binding, keys, shared, made,
			     failed, skipped, table.SharedSlotsForgotten(), n);
		}
	}
}

static uint64_t BindlessRecordRetryId(const BindlessHeap& heap, uint32_t key) {
	return (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&heap)) << 20u) ^ key;
}

void RenderExecutor::QueueBindlessRecordRetry(BindlessHeap& heap, uint32_t key, uint32_t attempt) {
	constexpr uint32_t Attempts = 5; // after 4, 8, 16, 32 and 64 frames
	if (attempt >= Attempts || !m_bindless_record_queued.insert(BindlessRecordRetryId(heap, key)).second) {
		return;
	}
	m_bindless_record_retry.push_back({&heap, key, attempt, BindlessFrame() + (4ull << attempt)});
	g_bindless_engine_stats.record_retries.fetch_add(1, std::memory_order_relaxed);
}

bool RenderExecutor::ResettleBindlessKey(BindlessHeap& heap, uint32_t key) {
	auto& table         = m_context.GetBindlessTable();
	auto& texture_cache = m_context.GetTextureCache();
	// IDXTRI: resolve the new T# before giving up the old slot. The guest rewrites an entry mostly
	// to step its min LOD while a texture streams; if the new view cannot be made yet (the image
	// is being dropped, the slot array is full), the key keeps sampling the old view and is tried
	// again next frame, instead of sampling the placeholder for frames. KYTY_BINDLESS_KEEP_OLD=0
	// gives the old slot up at once.
	static const bool keep_old = [] {
		const char* value = std::getenv("KYTY_BINDLESS_KEEP_OLD");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	const auto old_image      = heap.images[key];
	const auto old_slot       = heap.slots[key];
	const auto old_descriptor = heap.descriptors[key];
	const bool keep           = keep_old && old_slot != 0 && old_image;
	const bool no_refs        = table.ReleaseKey(heap, key, !keep);
	const bool resolved       = ResolveBindlessKey(heap, key);
	auto*      image          = texture_cache.m_slot_images.try_get(old_image);
	if (keep && !resolved && m_bindless_transient_failure && image != nullptr &&
	    image->registered && image->bindless_pinned && heap.slots[key] == 0) {
		heap.slots[key]   = old_slot;
		heap.images[key]  = old_image;
		heap.settled[key] = 1;
		// The old T#: the key stays "changed" for the retry below. An unfilled record keeps zeros
		// (revalidation sees the T# once written) and is read again by m_bindless_record_retry.
		heap.descriptors[key] = m_bindless_record_unfilled ? std::array<uint32_t, 8> {} : old_descriptor;
		if (std::find(heap.resolved.begin(), heap.resolved.end(), old_image) == heap.resolved.end()) {
			heap.resolved.push_back(old_image);
		}
		table.AddImageReference(old_image, heap, key);
		table.SetTranslation(heap, key, old_slot);
		if (!m_bindless_record_unfilled && std::find(m_bindless_retry.begin(), m_bindless_retry.end(),
		              std::pair<BindlessHeap*, uint32_t> {&heap, key}) == m_bindless_retry.end()) {
			m_bindless_retry.emplace_back(&heap, key);
		}
		g_bindless_engine_stats.kept_old.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	if (keep) {
		table.FreeSlot(BindlessTable::SlotBinding(heap, old_slot), BindlessTable::SlotIndex(old_slot));
	}
	if (no_refs && image != nullptr && image->bindless_pinned && old_image != heap.images[key]) {
		// No key refers to it any more: the texture cache may collect it again.
		NoteBindlessStateChange(*image);
		image->bindless_pinned = false;
		// KYTY_BINDLESS_GUEST_DROP_FRAMES: and frees it once unused for a while.
		texture_cache.ReleaseDroppedByGuest(old_image);
	}
	return resolved;
}

void RenderExecutor::SyncBindlessHeaps() {
	auto& table = m_context.GetBindlessTable();
	if (!table.Enabled() || !BindlessEngineEnabled() || !table.HasWrittenRanges()) {
		return;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	if (!scheduler.Active() || scheduler.Current().IsInvalid()) {
		return; // the written pages stay recorded for the next submission
	}
	RenderLockGuard lock(m_context.GetMutex());
	SyncWrittenBindlessEntries();
}

// KYTY_BINDLESS_SYNC_INDEX (live, default off): see SyncWrittenBindlessEntries.
static Live::Switch g_bindless_sync_index("KYTY_BINDLESS_SYNC_INDEX", Live::ParseDefaultOff);
// KYTY_BINDLESS_SYNC_RETIRE=<frames> (live, default 0): see SyncWrittenBindlessEntries.
static Live::Switch g_bindless_sync_retire("KYTY_BINDLESS_SYNC_RETIRE", [](const char* value) -> int64_t {
	return value != nullptr ? static_cast<int64_t>(std::min(std::strtoull(value, nullptr, 10), 1000000ull)) : 0;
});
static std::atomic<uint64_t> g_bindless_retired_skips {0};
static std::atomic<uint64_t> g_bindless_retired_rechecks {0};

void RenderExecutor::RecheckBindlessKeys(BindlessHeap& heap, uint32_t first, uint32_t last,
                                         uint32_t& view_changes, uint32_t& resource_changes,
                                         uint32_t& resettled) {
	const auto heap_begin = heap.base + heap.table_offset;
	const auto stride     = heap.stride;
	m_bindless_window.resize(last - first);
	if (!ReadHeapDescriptors(heap_begin, stride, first, last - first, m_bindless_window.data())) {
		return;
	}
	for (auto& words: m_bindless_window) {
		MaskR128HeapWords(heap, words.data());
	}
	for (uint32_t key = first; key < last; key++) {
		const auto& words = m_bindless_window[key - first];
		if (heap.settled[key] == 0 || heap.descriptors[key] == words) {
			continue;
		}
		bool resource = false;
		for (uint32_t i = 0; i < 8; i++) {
			resource |= ((heap.descriptors[key][i] ^ words[i]) & ~BindlessViewMask[i]) != 0;
		}
		(resource ? resource_changes : view_changes)++;
		resettled += ResettleBindlessKey(heap, key) ? 1u : 0u;
	}
}

void RenderExecutor::SyncWrittenBindlessEntries() {
	auto& table = m_context.GetBindlessTable();
	KYTY_PROFILER_FUNCTION();
	// Engine method, E2: the pages the guest wrote since the last submission, already watched
	// again. Every entry in them is compared with the T# its key was settled from.
	table.TakeWrittenRanges(m_bindless_written);
	uint32_t view_changes = 0, resource_changes = 0, resettled = 0;
	// KYTY_BINDLESS_SYNC_INDEX=1 (live, default 0; PPSA26344 preset): only the heaps whose key tables can
	// reach a written range are compared, found through 1 MiB address buckets of the heaps' key
	// spans (filled incrementally: new heaps, and heaps whose region was reallocated). Ghost of
	// Yotei creates ~100k small heaps in 10 minutes (KYTY_BINDLESS_SMALL_HEAPS, never freed) and
	// every sync walked all of them for every written range (CPU profile: ~24% of the command
	// processor thread at the village). Same comparisons, same order per heap.
	const bool sync_index = g_bindless_sync_index.Get() != 0;
	auto& heaps = table.Heaps();
	auto& candidates = m_bindless_heap_candidates;
	candidates.clear();
	const bool use_index = sync_index;
	if (use_index) {
		constexpr uint32_t BucketShift = 20u;
		const auto add = [&](uint32_t serial) {
			const auto& heap  = heaps[serial];
			if (heap.record_pointer != UINT32_MAX || heap.runtime) {
				return; // not compared here (see below)
			}
			const auto  begin = heap.base + heap.table_offset;
			const auto  end   = begin + uint64_t {heap.entries} * std::max(heap.stride, 32u) + 32u;
			for (uint64_t bucket = begin >> BucketShift; bucket <= (end - 1u) >> BucketShift; bucket++) {
				m_bindless_heap_buckets[bucket].push_back(serial);
			}
		};
		if (m_bindless_heap_indexed > heaps.size()) {
			m_bindless_heap_buckets.clear();
			m_bindless_heap_indexed = 0;
		}
		// A region (re)allocation may have widened an indexed heap's span: index it again over
		// its new span (duplicate serials are removed below).
		table.TakeResizedHeaps(m_bindless_heap_resized);
		for (const auto serial: m_bindless_heap_resized) {
			if (serial < m_bindless_heap_indexed) {
				add(serial);
			}
		}
		for (; m_bindless_heap_indexed < heaps.size(); m_bindless_heap_indexed++) {
			add(static_cast<uint32_t>(m_bindless_heap_indexed));
		}
		// Each heap once, in table order (as the full walk visits them): a bit per heap.
		auto& marks = m_bindless_heap_marks;
		marks.assign((heaps.size() + 63u) / 64u, 0u);
		auto& buckets = m_bindless_heap_resized; // reused as scratch: the bucket ids, each once
		buckets.clear();
		for (const auto& [begin, end]: m_bindless_written) {
			if (end <= begin) {
				continue;
			}
			for (uint64_t bucket = begin >> BucketShift; bucket <= (end - 1u) >> BucketShift; bucket++) {
				buckets.push_back(static_cast<uint32_t>(bucket));
			}
		}
		std::sort(buckets.begin(), buckets.end());
		buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
		for (const auto bucket: buckets) {
			if (const auto found = m_bindless_heap_buckets.find(bucket); found != m_bindless_heap_buckets.end()) {
				for (const auto serial: found->second) {
					marks[serial >> 6u] |= uint64_t {1} << (serial & 63u);
				}
			}
		}
		for (size_t word = 0; word < marks.size(); word++) {
			for (auto bits = marks[word]; bits != 0; bits &= bits - 1u) {
				candidates.push_back(static_cast<uint32_t>(word * 64u + static_cast<size_t>(std::countr_zero(bits))));
			}
		}
	} else {
		table.TakeResizedHeaps(m_bindless_heap_resized);
		m_bindless_heap_buckets.clear();
		m_bindless_heap_indexed = 0;
	}
	// KYTY_BINDLESS_SYNC_RETIRE=<frames> (live, default 0 = off; PPSA26344 preset): a heap no draw
	// prepared in that many frames is not compared here; it is marked stale and its next user
	// compares all its keys before the draw (the same comparison, later). Ghost of Yotei leaves
	// tens of thousands of dead heaps over memory it keeps rewriting (material records at moving
	// addresses), and comparing them made each sync slower with play time (uProf: the sync and
	// its guest reads grew from 330 s to 670 s of one boot).
	const auto retire  = static_cast<uint64_t>(g_bindless_sync_retire.Get());
	const auto frame   = BindlessFrame();
	uint64_t   retired = 0;
	const auto heap_count = use_index ? candidates.size() : heaps.size();
	for (size_t heap_index = 0; heap_index < heap_count; heap_index++) {
		auto& heap = heaps[use_index ? candidates[heap_index] : heap_index];
		if (heap.record_pointer != UINT32_MAX || heap.runtime) {
			continue; // compared key by key in RevalidateBindlessKeys (runtime heaps: never)
		}
		const auto heap_begin = heap.base + heap.table_offset;
		const auto keys = std::min(heap.guest_entries, static_cast<uint32_t>(heap.settled.size()));
		const auto stride   = heap.stride; // 32, or a record's (KYTY_BINDLESS_STRIDE)
		const auto heap_end =
		    heap_begin + (keys == 0 ? 0 : uint64_t {keys - 1u} * stride + 32u);
		for (const auto& [begin, end]: m_bindless_written) {
			const auto from = std::max(begin, heap_begin);
			const auto to   = std::min(end, heap_end);
			if (from >= to) {
				continue;
			}
			const auto first = static_cast<uint32_t>((from - heap_begin) / stride);
			const auto last =
			    std::min(keys, static_cast<uint32_t>((to - heap_begin + stride - 1u) / stride));
			if (last <= first) {
				continue;
			}
			if (retire != 0 && frame > heap.sync_used_frame + retire) {
				// KYTY_BINDLESS_SYNC_RETIRE: no draw used the heap for a while; its next user
				// compares every key first (PrepareBindlessHeaps).
				heap.sync_stale = true;
				retired++;
				break;
			}
			RecheckBindlessKeys(heap, first, last, view_changes, resource_changes, resettled);
		}
	}
	g_bindless_retired_skips.fetch_add(retired, std::memory_order_relaxed);
	if (retire != 0 || sync_index) {
		// Every 10 s while either switch is on: what the sync did (log only).
		static auto last_line = std::chrono::steady_clock::now();
		static uint64_t syncs = 0, candidate_sum = 0;
		syncs++;
		candidate_sum += use_index ? candidates.size() : heaps.size();
		if (const auto now = std::chrono::steady_clock::now(); now - last_line >= std::chrono::seconds(10)) {
			last_line = now;
			LOGF("BindlessSync 10s: syncs=%" PRIu64 " heaps=%zu compared_per_sync=%.0f retired_skips=%" PRIu64
			     " stale_rechecks=%" PRIu64 " refills=%" PRIu64 "\n",
			     syncs, heaps.size(), syncs != 0 ? static_cast<double>(candidate_sum) / static_cast<double>(syncs) : 0.0,
			     g_bindless_retired_skips.exchange(0), g_bindless_retired_rechecks.exchange(0),
			     g_bindless_refills.exchange(0));
			syncs = candidate_sum = 0;
		}
	}
	uint64_t bytes = 0;
	for (const auto& [begin, end]: m_bindless_written) {
		bytes += end - begin;
	}
	auto& stats = g_bindless_engine_stats;
	stats.syncs.fetch_add(1, std::memory_order_relaxed);
	stats.written_bytes.fetch_add(bytes, std::memory_order_relaxed);
	stats.view_changes.fetch_add(view_changes, std::memory_order_relaxed);
	stats.resource_changes.fetch_add(resource_changes, std::memory_order_relaxed);
	stats.resettled.fetch_add(resettled, std::memory_order_relaxed);
}

// KYTY_BINDLESS_RECORD_LAZY=0 (default 1): record pointer heaps are handled with the others once a
// frame (at its first bindless dispatch). On, each is handled at its own first use in the frame
// (ProcessRecordHeap): Miles Morales alternates two call tables and rebuilds the idle one while the
// GPU works through the other, so reading the idle table's records then saw records half written
// (all-zero T#s behind new pointers) and settled keys from them.
static bool BindlessRecordLazy() {
	static const bool lazy = [] {
		const char* env = std::getenv("KYTY_BINDLESS_RECORD_LAZY");
		return env == nullptr || env[0] != '0';
	}();
	return lazy;
}

uint32_t RenderExecutor::RevalidateBindlessKeys(uint32_t budget) {
	KYTY_PROFILER_FUNCTION();
	// With the engine method only record pointer heaps are compared here: the write watch sees
	// their records, not the T#s they point at.
	const bool pointers_only = BindlessEngineEnabled() && !BindlessEngineVerify();
	if (pointers_only &&
	    std::ranges::none_of(m_context.GetBindlessTable().Heaps(), [](const auto& heap) {
		    return heap.record_pointer != UINT32_MAX;
	    })) {
		return 0; // the write watch reports rewritten entries (SyncBindlessHeaps)
	}
	// The guest rewrites heap entries as it streams textures out and others in. A key settled
	// from the old T# kept sampling the old texture, and kept it pinned in the texture cache for
	// the rest of the run. Each frame a window of every heap is read back in one go and compared;
	// a changed key is settled again into a new slot, and an image no key refers to any more is
	// unpinned, so the cache can collect it. A heap of 64 Ki keys is covered in 32 frames.
	auto&    table    = m_context.GetBindlessTable();
	uint32_t resolved = 0;
	for (auto& heap: table.Heaps()) {
		const bool pointers = heap.record_pointer != UINT32_MAX;
		if (heap.runtime || (pointers_only && !pointers) || (pointers && BindlessRecordLazy())) {
			continue; // record heaps: ProcessRecordHeap
		}
		resolved += RevalidateBindlessHeap(heap, budget - std::min(budget, resolved));
	}
	return resolved;
}

uint32_t RenderExecutor::RevalidateBindlessHeap(BindlessHeap& heap, uint32_t budget) {
	constexpr uint32_t Window   = 2048;
	uint32_t           resolved = 0;
	{
		const auto keys     = static_cast<uint32_t>(heap.settled.size());
		const bool pointers = heap.record_pointer != UINT32_MAX;
		if (keys == 0 || heap.runtime) {
			return 0;
		}
		const uint32_t begin = heap.revalidate_cursor < keys ? heap.revalidate_cursor : 0u;
		const uint32_t count = std::min(pointers ? Window / 8u : Window, keys - begin);
		m_bindless_window.resize(count);
		bool readable = true;
		if (pointers) {
			// Each key's T# lives elsewhere: an unsettled or unreadable key reads as unchanged.
			for (uint32_t i = 0; i < count; i++) {
				if (heap.settled[begin + i] == 0 ||
				    !ReadBindlessEntry(heap, begin + i, m_bindless_window[i].data())) {
					m_bindless_window[i] = heap.descriptors[begin + i];
				}
			}
		} else {
			readable = ReadHeapDescriptors(heap.base + heap.table_offset, heap.stride, begin, count,
			                               m_bindless_window.data());
			for (auto& words: m_bindless_window) {
				MaskR128HeapWords(heap, words.data());
			}
		}
		uint32_t   next     = begin + count;
		for (uint32_t i = 0; readable && i < count; i++) {
			const auto key = begin + i;
			if (heap.settled[key] == 0 || heap.descriptors[key] == m_bindless_window[i]) {
				continue;
			}
			if (resolved >= budget) {
				next = key; // look at it again next frame
				break;
			}
			if (BindlessEngineEnabled() && !pointers) {
				static std::atomic<uint32_t> missed = 0;
				if (missed.fetch_add(1) < 64) {
					LOGF("Bindless engine: write not reported, heap=0x%" PRIx64 "+0x%x binding=%u "
					     "key=%u\n",
					     heap.base, heap.table_offset, heap.binding, key);
				}
			}
			resolved += ResettleBindlessKey(heap, key) ? 1u : 0u;
		}
		heap.revalidate_cursor = next >= keys ? 0u : next;
	}
	return resolved;
}

// KYTY_BINDLESS_RECORD_RETRY: unfilled records whose time has come (of heap, or of every heap for
// nullptr) are read again; a T# written meanwhile is settled (the key keeps its old view if that
// fails), a record still unfilled waits twice as long, up to the last attempt.
uint32_t RenderExecutor::RetryBindlessRecords(BindlessHeap* heap, uint64_t frame, uint32_t budget) {
	uint32_t resolved = 0;
	if (m_bindless_record_retry.empty()) {
		return 0;
	}
	auto due = std::move(m_bindless_record_retry);
	m_bindless_record_retry.clear();
	for (const auto& retry: due) {
		if ((heap != nullptr && retry.heap != heap) || retry.due > frame || resolved >= budget) {
			m_bindless_record_retry.push_back(retry);
			continue;
		}
		m_bindless_record_queued.erase(BindlessRecordRetryId(*retry.heap, retry.key));
		std::array<uint32_t, 8> words {};
		const bool filled = ReadBindlessEntry(*retry.heap, retry.key, words.data()) &&
		                    std::ranges::any_of(words, [](uint32_t w) { return w != 0u; });
		if (!filled) {
			QueueBindlessRecordRetry(*retry.heap, retry.key, retry.attempt + 1u);
			continue;
		}
		if (words == retry.heap->descriptors[retry.key] && retry.heap->settled[retry.key] != 0) {
			continue; // settled from this T# meanwhile
		}
		m_bindless_record_attempt = retry.attempt + 1u;
		resolved += ResettleBindlessKey(*retry.heap, retry.key) ? 1u : 0u;
		m_bindless_record_attempt = 0;
		g_bindless_engine_stats.record_filled.fetch_add(1, std::memory_order_relaxed);
	}
	return resolved;
}

// The flagged keys of one heap (from the snapshot consumed this frame), within the budget.
uint32_t RenderExecutor::ResolveFlaggedKeys(BindlessHeap& heap, uint32_t budget, uint32_t* requested) {
	auto&    table    = m_context.GetBindlessTable();
	uint32_t resolved = 0;
	m_bindless_requests.clear();
	table.TakeRequests(heap, m_bindless_requests);
	*requested += static_cast<uint32_t>(m_bindless_requests.size());
	// KYTY_BINDLESS_REQ_LOG=<n> (diagnostic, default off): the first n lines "Bindless requests": per
	// heap and frame, how many unresolved keys shaders sampled (feedback), with the heap's identity.
	static const uint32_t req_log = [] {
		const char* value = std::getenv("KYTY_BINDLESS_REQ_LOG");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	if (req_log != 0 && !m_bindless_requests.empty()) {
		static std::atomic<uint32_t> lines {0};
		if (lines.fetch_add(1, std::memory_order_relaxed) < req_log) {
			uint32_t unsettled = 0;
			for (const auto key: m_bindless_requests) {
				unsettled += key < heap.settled.size() && heap.settled[key] == 0 ? 1u : 0u;
			}
			LOGF("Bindless requests: frame=%" PRIu64 " heap 0x%" PRIx64 "+%u stride %u binding %u entries %u record %d "
			     "runtime %d requested %zu unsettled %u first_key %u\n",
			     BindlessFrame(), heap.base, heap.table_offset, heap.stride, heap.binding, heap.entries,
			     heap.record_pointer != UINT32_MAX ? 1 : 0, heap.runtime ? 1 : 0, m_bindless_requests.size(),
			     unsettled, m_bindless_requests.front());
		}
	}
	for (const auto key: m_bindless_requests) {
		if (heap.settled[key] != 0) {
			continue;
		}
		if (resolved >= budget) {
			break; // still pending: the next frame flags it again
		}
		resolved += ResolveBindlessKey(heap, key) ? 1u : 0u;
	}
	return resolved;
}

void RenderExecutor::ProcessRecordHeap(BindlessHeap& heap) {
	const BindlessHostTimer timer;
	const auto              frame = BindlessFrame();
	constexpr uint32_t      Budget = 128;
	if (m_bindless_budget_frame != frame) {
		m_bindless_budget_frame = frame;
		if (m_bindless_frame != frame) {
			m_bindless_budget_left = Budget; // no snapshot consumed this frame (yet)
		}
	}
	// The flags of each consumed snapshot once: the two call tables are used in alternate frames,
	// and a snapshot is not consumed every frame.
	if (heap.processed_snapshot != m_bindless_snapshot_serial) {
		heap.processed_snapshot = m_bindless_snapshot_serial;
		uint32_t   requested    = 0;
		const auto spent        = ResolveFlaggedKeys(heap, m_bindless_budget_left, &requested);
		g_bindless_engine_stats.requested.fetch_add(requested, std::memory_order_relaxed);
		m_bindless_budget_left -= std::min(m_bindless_budget_left, spent);
	}
	if (heap.processed_frame != frame) {
		heap.processed_frame = frame;
		auto spent           = RetryBindlessRecords(&heap, frame, m_bindless_budget_left);
		m_bindless_budget_left -= std::min(m_bindless_budget_left, spent);
		spent = RevalidateBindlessHeap(heap, m_bindless_budget_left);
		m_bindless_budget_left -= std::min(m_bindless_budget_left, spent);
	}
}
// KYTY_BINDLESS_IDLE_RELEASE=<frames> (startup, default 0 = off; PPSA26344 preset 300; ported from
// Erlou93/KytyPS5-Yotei v0.1.4 ReleaseUnusedBindlessSlots): a guest heap no draw or dispatch used for
// that many frames releases its resolved keys: their slots go back to the free lists and their images
// lose the heap's references; an image no key refers to any more is unpinned, so the texture cache may
// collect it (KYTY_BINDLESS_GUEST_DROP_FRAMES frees it). Without it every resolved key of every heap
// stays: Ghost of Yotei creates ~100k small heaps per 10 minutes (per-frame terrain and material
// tables), each holding its images pinned for the rest of the session (VRAM growth over a long
// session). A released heap used again resolves its keys again (prefill, or feedback). Runtime and
// record pointer heaps are left alone. 512 heaps are looked at per frame, round robin.
void RenderExecutor::ReleaseIdleBindlessHeaps(uint64_t frame) {
	static const uint64_t idle = [] {
		const char* value = std::getenv("KYTY_BINDLESS_IDLE_RELEASE");
		return value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {0};
	}();
	if (idle == 0) {
		return;
	}
	auto&        table = m_context.GetBindlessTable();
	auto&        cache = m_context.GetTextureCache();
	auto&        heaps = table.Heaps();
	const size_t count = heaps.size();
	if (count == 0 || frame < idle) {
		return;
	}
	static uint64_t released_heaps = 0, released_keys = 0, unpinned = 0, logged_frame = 0;
	constexpr uint32_t HeapBudget = 512;
	for (uint32_t n = 0; n < HeapBudget && n < count; n++) {
		if (m_bindless_idle_cursor >= count) {
			m_bindless_idle_cursor = 0;
		}
		auto& heap = heaps[m_bindless_idle_cursor++];
		if (heap.runtime || heap.record_pointer != UINT32_MAX || heap.resolved.empty() ||
		    heap.sync_used_frame + idle > frame) {
			continue;
		}
		for (uint32_t key = 0; key < heap.slots.size(); key++) {
			const auto id = heap.images[key];
			if (heap.slots[key] == 0 && !id) {
				continue;
			}
			released_keys++;
			if (table.ReleaseKey(heap, key, true) && id) {
				if (auto* image = cache.m_slot_images.try_get(id); image != nullptr && image->bindless_pinned) {
					NoteBindlessStateChange(*image);
					image->bindless_pinned = false;
					cache.ReleaseDroppedByGuest(id);
					unpinned++;
				}
			}
		}
		heap.prefilled = false; // KYTY_BINDLESS_PREFILL settles it again at its next use
		heap.refresh_gpu_keys.clear();
		heap.refresh_frame = UINT64_MAX;
		released_heaps++;
	}
	if (released_heaps != 0 && frame >= logged_frame + 600u) {
		logged_frame = frame;
		LOGF("Bindless idle release: frame=%" PRIu64 " heaps %" PRIu64 " keys %" PRIu64 " images unpinned %" PRIu64
		     " (KYTY_BINDLESS_IDLE_RELEASE=%" PRIu64 ", %zu heaps)\n",
		     frame, released_heaps, released_keys, unpinned, idle, count);
	}
}

void RenderExecutor::ResolveBindlessRequests() {
	KYTY_PROFILER_FUNCTION();
	const BindlessHostTimer timer;
	const auto frame = BindlessFrame();
	if (frame == m_bindless_frame) {
		return;
	}
	auto& table = m_context.GetBindlessTable();
	// The flags come from a snapshot recorded in an earlier frame. Until the GPU has executed
	// it, check again at the next bindless draw instead of waiting.
	if (table.SnapshotRecorded() && !table.SnapshotReady()) {
		return;
	}
	m_bindless_frame = frame;
	if (!table.SnapshotRecorded()) {
		table.RecordFeedbackSnapshot();
		m_bindless_snapshot_frame = frame;
		return;
	}
	table.ConsumeSnapshot();
	m_bindless_snapshot_serial++;
	table.DebugWatchReport(frame);
	ReleaseIdleBindlessHeaps(frame);
	const auto snapshot_frame = m_bindless_snapshot_frame;
	// Each texture may upload and detile; spread first sight of a scene over a few frames.
	constexpr uint32_t Budget    = 128;
	uint32_t           resolved  = RevalidateBindlessKeys(Budget);
	uint32_t           requested = 0;
	if (!BindlessRecordLazy()) {
		resolved += RetryBindlessRecords(nullptr, frame, Budget - std::min(Budget, resolved));
	}
	// Keys that kept their old view (ResettleBindlessKey): their new T# is tried again.
	if (!m_bindless_retry.empty()) {
		auto retry = std::move(m_bindless_retry);
		m_bindless_retry.clear();
		for (const auto& [heap, key]: retry) {
			if (resolved >= Budget) {
				m_bindless_retry.emplace_back(heap, key);
				continue;
			}
			std::array<uint32_t, 8> words {};
			if (heap->settled[key] == 0 || !ReadBindlessEntry(*heap, key, words.data()) ||
			    words == heap->descriptors[key]) {
				continue; // settled anew meanwhile, or the guest wrote the old T# back
			}
			resolved += ResettleBindlessKey(*heap, key) ? 1u : 0u;
		}
	}
	static std::atomic<uint32_t> frames_logged = 0;
	const bool                   log_frame     = frames_logged.fetch_add(1) < 12;
	if (const auto word0 = table.TakeWordZero(); word0 != 0) {
		// KYTY_BINDLESS_OOR_LOG=<n> (default 64): how many of these lines are logged.
		static const uint32_t        keys_limit  = [] {
			const char* env = std::getenv("KYTY_BINDLESS_OOR_LOG");
			return env != nullptr ? static_cast<uint32_t>(std::strtoul(env, nullptr, 10)) : 64u;
		}();
		static std::atomic<uint32_t> keys_logged = 0;
		if (keys_logged.fetch_add(1) < keys_limit) {
			LOGF("Bindless out-of-range key: frame=%" PRIu64 " key=%u (0x%08x)\n", frame,
			     word0 & 0x7fffffffu, word0);
		}
	}
	// KYTY_RUNTIME_IMAGES (adapted from usa-ratt/KytyPS5-GT7 2fa6c6514): T#s shaders loaded at run
	// time and found missing from the runtime image table: each takes a key of its binding's
	// runtime heap and is resolved like a heap key. A key not resolved now stays pending in the
	// table, and the next draw that reads it flags it.
	if (BindlessTable::RuntimeImagesEnabled()) {
		m_runtime_image_requests.clear();
		table.TakeRuntimeRequests(m_runtime_image_requests);
		uint32_t runtime_requested = 0, runtime_resolved = 0;
		for (const auto& [dwords, binding]: m_runtime_image_requests) {
			auto* heap = table.RuntimeHeap(binding, nullptr);
			if (heap == nullptr) {
				continue;
			}
			const auto key = table.RuntimeKey(*heap, dwords);
			if (key == UINT32_MAX || heap->settled[key] != 0) {
				continue;
			}
			runtime_requested++;
			if (resolved >= Budget) {
				continue;
			}
			const bool ok = ResolveBindlessKey(*heap, key);
			resolved += ok ? 1u : 0u;
			runtime_resolved += ok ? 1u : 0u;
		}
		if (runtime_requested != 0) {
			static std::atomic<uint32_t> runtime_logged = 0;
			const auto                   logged_count   = runtime_logged.fetch_add(1);
			if (logged_count < 32 || (logged_count & (logged_count + 1u)) == 0u) {
				uint32_t keys = 0;
				for (uint32_t b = 0; b < BindlessTable::ImageArrays; b++) {
					if (const auto* heap = table.RuntimeHeap(b, nullptr); heap != nullptr) {
						keys += heap->runtime_next_key;
					}
				}
				LOGF("Bindless runtime images: frame=%" PRIu64 " requested=%u resolved=%u (%u T#s so "
				     "far)\n",
				     frame, runtime_requested, runtime_resolved, keys);
			}
		}
	}
	for (auto& heap: table.Heaps()) {
		if (heap.record_pointer != UINT32_MAX && !heap.runtime && BindlessRecordLazy()) {
			continue; // ProcessRecordHeap, at the heap's first use in the frame
		}
		const auto before = requested;
		resolved += ResolveFlaggedKeys(heap, Budget - std::min(Budget, resolved), &requested);
		if (log_frame) {
			LOGF("Bindless feedback: frame=%" PRIu64 " binding=%u region=%u flagged=%u\n", frame, heap.binding,
			     heap.region, requested - before);
		}
	}
	m_bindless_budget_left = Budget - std::min(Budget, resolved);
	g_bindless_engine_stats.requested.fetch_add(requested, std::memory_order_relaxed);
	static std::atomic<uint32_t> logged = 0;
	if (requested != 0 && logged.fetch_add(1) < 64) {
		LOGF("Bindless requests: frame=%" PRIu64 " requested=%u resolved=%u\n", frame, requested,
		     resolved);
	}
	LogBindlessHeapStats(table, frame);
	UpdateBindlessUsageProbe(frame, snapshot_frame);
	table.RecordFeedbackSnapshot();
	m_bindless_snapshot_frame = frame;
}

// KYTY_BINDLESS_USE_LOG=<trigger file>[|<count>] (diagnostic): once the trigger file exists, the
// next <count> (default 2000) bindless heap uses are logged ("Bindless use": frame, program, heap
// address and size, the region and entry count patched, or NO HEAP), then the file is removed.
static bool BindlessPatchLogTake() {
	struct Request {
		std::string path;
		uint32_t    count = 2000;
	};
	static const Request request = [] {
		Request parsed;
		if (const char* value = std::getenv("KYTY_BINDLESS_USE_LOG"); value != nullptr && value[0] != '\0') {
			std::string text(value);
			if (const auto bar = text.find('|'); bar != std::string::npos) {
				parsed.count = static_cast<uint32_t>(std::strtoul(text.c_str() + bar + 1, nullptr, 10));
				text.resize(bar);
			}
			parsed.path = text;
		}
		return parsed;
	}();
	if (request.path.empty()) return false;
	static std::atomic<uint32_t> left {0};
	static std::atomic<uint32_t> polls {0};
	if (left.load(std::memory_order_relaxed) == 0) {
		if ((polls.fetch_add(1, std::memory_order_relaxed) & 255u) != 0u) return false;
		std::error_code error;
		if (!std::filesystem::exists(request.path, error)) return false;
		std::filesystem::remove(request.path, error);
		LOGF("Bindless use log: start, %u uses\n", request.count);
		left.store(request.count, std::memory_order_relaxed);
	}
	auto value = left.load(std::memory_order_relaxed);
	while (value != 0 && !left.compare_exchange_weak(value, value - 1, std::memory_order_relaxed)) {
	}
	return value != 0;
}

// KYTY_BINDLESS_REFRESH=1 (live, default off; ported from Erlou93/KytyPS5-Yotei v0.1.4
// PrepareBindlessHeaps / RefreshBindlessDescriptor): a bound texture is looked up again by every
// draw, so the texture cache uploads CPU writes into it and follows GPU-rendered or re-aliased data
// to the image that holds it now. A settled bindless key was looked up only when its T# words
// changed or its image was dropped: its slot kept sampling the image (and contents) it was settled
// to. With the switch, every resolved key of a heap a draw uses is looked up again once per frame
// (once per slot: keys sharing a slot share the lookup), and by every draw while its image is
// GPU-written (render targets, depth pyramids: a draw samples what earlier draws of this frame
// wrote). A key that now maps to another image gets that image's view in its slot.
// =2: once per frame only (no per-draw lookups of GPU-written images).
// KYTY_BINDLESS_REFRESH_LOG=1 (diagnostic): lookups / remaps / GPU-written slots every 600 frames.
static Live::Switch g_bindless_refresh("KYTY_BINDLESS_REFRESH", [](const char* value) -> int64_t {
	return value != nullptr ? std::strtol(value, nullptr, 10) : 0;
});

// KYTY_BINDLESS_EVICT=1 (default off; adapted from Senaxx 8585096e6, on by default there):
// bindless textures stay pinned in the texture cache while a heap key refers to them, and the
// guest's heaps refer to every texture it has streamed in (2-5 GB on 4-8 GB cards in the fork's
// title). Under memory pressure (TextureCache::BindlessUnderPressure) a probe marks every resident
// key's feedback word; draws that sample a key overwrite the mark, and a snapshot recorded
// ProbeFrames later lists the images whose keys all kept it. While the cache stays under
// pressure, TextureCache::RunGarbageCollector frees those (clean, unbound ones only); their keys
// go back to pending, and a draw that samples one again requests it anew. The fork's slot reuse
// and placeholder-type fix are in this line already (FreeSlot, OnImageUnregistered).
void RenderExecutor::UpdateBindlessUsageProbe(uint64_t frame, uint64_t snapshot_frame) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_EVICT");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled) {
		return;
	}
	constexpr uint64_t ProbeFrames   = 16; // frames a texture must go unsampled
	constexpr uint64_t ProbeInterval = 32; // frames between the end of one probe and the next
	auto&              table         = m_context.GetBindlessTable();
	auto&              texture_cache = m_context.GetTextureCache();
	if (m_bindless_probe_frame != 0) {
		// The snapshot read here must have been recorded ProbeFrames after the marks.
		if (snapshot_frame < m_bindless_probe_frame + ProbeFrames) {
			return;
		}
		m_bindless_unused.clear();
		table.CollectUnusedImages(m_bindless_unused);
		const auto bytes       = texture_cache.SetBindlessEvictable(m_bindless_unused);
		m_bindless_probe_frame = 0;
		m_bindless_next_probe  = frame + ProbeInterval;
		static std::atomic<uint32_t> logged = 0;
		if (logged.fetch_add(1) < 32) {
			LOGF("Bindless probe: frame=%" PRIu64 " unused=%zu (%" PRIu64
			     " MB) may be evicted (KYTY_BINDLESS_EVICT)\n",
			     frame, m_bindless_unused.size(), bytes >> 20u);
		}
		return;
	}
	if (frame < m_bindless_next_probe) {
		return;
	}
	if (!texture_cache.BindlessUnderPressure()) {
		// No pressure: an older list must not free textures sampled since.
		m_bindless_unused.clear();
		(void)texture_cache.SetBindlessEvictable(m_bindless_unused);
		m_bindless_next_probe = frame + ProbeInterval;
		return;
	}
	table.ArmUsageProbe();
	m_bindless_probe_frame = std::max<uint64_t>(frame, 1u); // 0 = idle
}

// KYTY_BINDLESS_SAMPLERS (a minimal port of Senaxx's mirrored sampler heaps, 59cb8a83e): the S#
// records of each sampler heap a stage indexes are mirrored into the sampler array of set 1, once
// per frame per heap and kind of use, and the shader gets the region base and record count. Keys
// outside it, heaps the GPU has written or that cannot be read, and a full array, use the stand-in
// sampler in slot 0 (the words stay 0, or the heap keeps its previous mirror).
void RenderExecutor::PrepareBindlessSamplers(const ShaderStageRuntime& runtime,
                                             PreparedBindings&         prepared) {
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	auto&       table    = m_context.GetBindlessTable();
	auto&       cache    = m_context.GetSamplerCache();
	{
		ShaderSamplerResource stand_in;
		std::copy_n(ShaderRecompiler::IR::BindlessDefaultSampler, 4u, stand_in.fields);
		table.WriteDefaultSampler(cache.GetSampler(stand_in, false));
	}
	const auto         frame      = BindlessFrame();
	constexpr uint64_t MaxRecords = 1024;
	constexpr uint64_t MaxBytes   = 256u * 1024u;
	std::vector<std::array<uint32_t, 4>> records;
	for (const auto& use: snapshot.bindless_sampler_heaps) {
		if (use.sampler >= program.info.samplers.size() || use.stride < 16u) {
			continue;
		}
		const auto&    sampler = program.info.samplers[use.sampler];
		const bool     cube    = CubeEdgeClamp() && SamplesCubesOnly(program, use.sampler);
		const uint32_t flags   = (sampler.depth_compare ? 1u : 0u) |
		                       (sampler.force_point_filtering ? 2u : 0u) |
		                       (sampler.integer_border ? 4u : 0u) | (cube ? 8u : 0u);
		auto& heap = table.FindOrCreateSamplerHeap(use.base, use.table_offset, use.stride, flags);
		if (heap.checked_frame != frame) {
			heap.checked_frame       = frame;
			const uint64_t available = use.size > use.table_offset
			                               ? (use.size - use.table_offset + use.stride - 16u) / use.stride
			                               : 0u;
			records.resize(std::min({available, MaxRecords, MaxBytes / use.stride}));
			const uint64_t address = use.base + use.table_offset;
			const uint64_t bytes =
			    records.empty() ? 0u : (records.size() - 1u) * use.stride + sizeof(records[0]);
			bool read = !records.empty() &&
			            !m_context.GetBufferCache().HasGpuDirtyBytes(address, bytes);
			if (read && use.stride == sizeof(records[0])) {
				read = LibKernel::Memory::TryReadBacking(address, records.data(), bytes);
			} else {
				for (size_t key = 0; read && key < records.size(); key++) {
					read = LibKernel::Memory::TryReadBacking(address + key * use.stride,
					                                         records[key].data(),
					                                         sizeof(records[key]));
				}
			}
			if (read) {
				const auto make = [&](const std::array<uint32_t, 4>& record) {
					ShaderRecompiler::IR::DescriptorValue value;
					value.dword_count = 4u;
					std::copy(record.begin(), record.end(), value.dwords.begin());
					return cache.GetSampler(NativeSamplerDescriptor(program, use.sampler, value),
					                        sampler.integer_border);
				};
				(void)table.MirrorSamplerHeap(heap, records, make);
			}
		}
		const uint32_t entries = heap.region != 0u ? static_cast<uint32_t>(heap.records.size()) : 0u;
		prepared.bindless_patches.push_back({use.mapping_offset, heap.region, entries});
	}
}

void RenderExecutor::PrepareBindlessHeaps(const ShaderStageRuntime& runtime,
                                          PreparedBindings&         prepared) {
	prepared.bindless_patches.clear();
	prepared.bindless_heaps.clear();
	const auto& snapshot = *runtime.resources;
	if (snapshot.bindless_heaps.empty() && snapshot.bindless_sampler_heaps.empty()) [[likely]] {
		return;
	}
	const auto& program = *runtime.program;
	auto&       table   = m_context.GetBindlessTable();
	if (!snapshot.bindless_sampler_heaps.empty() && table.EnsureCreated()) {
		PrepareBindlessSamplers(runtime, prepared);
	}
	for (const auto& use: snapshot.bindless_heaps) {
		if (use.record_pointer == UINT32_MAX && !use.runtime &&
		    m_bindless_surveyed.insert(use.base ^ (static_cast<uint64_t>(use.table_offset) << 48u))
		        .second) {
			SurveyBindlessHeap(use);
		}
	}
	if (!table.EnsureCreated()) {
		return; // region 0, count 0: every key samples the placeholder
	}
	ResolveBindlessRequests();
	// KYTY_BINDLESS_REPLACED_RESOLVE: keys whose image the texture cache replaced (ExpandImage)
	// are resolved again now, before this draw or dispatch samples them.
	if (table.HasReplacedKeys()) {
		m_bindless_replaced_keys.clear();
		table.TakeReplacedKeys(m_bindless_replaced_keys);
		for (const auto& [heap, key]: m_bindless_replaced_keys) {
			if (key < heap->settled.size() && heap->settled[key] == 0) {
				(void)ResolveBindlessKey(*heap, key);
			}
		}
	}
	for (const auto& use: snapshot.bindless_heaps) {
		if (use.image >= program.info.images.size()) {
			continue;
		}
		const auto&    resource = program.info.images[use.image];
		if (use.runtime) {
			// KYTY_RUNTIME_IMAGES: the runtime heap of the image's binding; its resolved images
			// must be readable. Nothing to patch.
			auto* heap = table.RuntimeHeap(BindlessBindingFor(resource), &resource);
			if (heap != nullptr && std::find(prepared.bindless_heaps.begin(),
			                                 prepared.bindless_heaps.end(),
			                                 heap) == prepared.bindless_heaps.end()) {
				prepared.bindless_heaps.push_back(heap);
			}
			continue;
		}
		const bool     pointers = use.record_pointer != UINT32_MAX;
		const uint64_t span     = pointers                    ? use.size
		                          : use.size > use.table_offset ? use.size - use.table_offset
		                                                        : 0;
		const uint64_t keys_per_record =
		    use.nested_vsharp != UINT32_MAX && use.nested_keys != 0 ? use.nested_keys : 1u;
		// KYTY_BINDLESS_RECORD_KEYS=<n> (default 0 = the whole table, up to 512 Ki keys): the keys of a
		// heap read through record pointers (a call table's material records). Each (table, T# offset,
		// view type) is a heap of its own; Miles Morales' ray-tracing callees use about fifty of them
		// over two call tables of 273064 records, and sample material keys up to at least 151749
		// (H view over the river): with 65536 keys per heap those samples read the out-of-range
		// placeholder (black), in some frames and not others. The whole table needs the larger
		// translation buffer (KYTY_BINDLESS_TRANSLATION_MI, default 16).
		static const uint64_t record_keys = [] {
			const char* value = std::getenv("KYTY_BINDLESS_RECORD_KEYS");
			return value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 0) : uint64_t {0};
		}();
		const uint64_t key_cap   = pointers ? uint64_t {1u << 19u} : uint64_t {1u << 18u};
		const uint64_t key_limit = pointers && record_keys != 0 ? std::min(record_keys, key_cap) : key_cap;
		const auto     entries   = static_cast<uint32_t>(std::min<uint64_t>(
            (pointers ? span / std::max(use.stride, 1u) : HeapEntriesInSpan(span, use.stride)) * keys_per_record,
            key_limit));
		auto* heap = table.FindOrCreateHeap(use.base, use.table_offset, BindlessBindingFor(resource),
		                                    entries, resource, use.stride, use.record_pointer,
		                                    use.nested_vsharp, use.nested_heap, use.nested_keys);
		if (BindlessPatchLogTake()) {
			LOGF("Bindless use: frame=%" PRIu64 " hash=0x%016" PRIx64 " image=%u base=0x%" PRIx64
			     " size=0x%" PRIx64 " offset=%u stride=%u record_pointer=%d entries=%u -> %s region=%u "
			     "heap_entries=%u\n",
			     BindlessFrame(), program.shader_hash, use.image, use.base, use.size, use.table_offset,
			     use.stride, use.record_pointer == UINT32_MAX ? -1 : static_cast<int>(use.record_pointer),
			     entries, heap == nullptr ? "NO HEAP" : "heap", heap != nullptr ? heap->region : 0u,
			     heap != nullptr ? heap->entries : 0u);
		}
		if (heap == nullptr) {
			continue; // region 0, count 0: every key samples the placeholder
		}
		// KYTY_BINDLESS_TOUCH (default on): the images this heap's keys resolved are in use this frame.
		// Bindless sampling never reached TextureCache::TouchImage, so such an image kept its creation
		// frame as its last access: the idle collectors (KYTY_VRAM_IDLE_FRAMES, the partially resident
		// retire) freed it while draws sampled it, its slots went to the placeholder and its keys
		// re-resolved, over and over (Ghost of Yotei's menu: black foliage, grey floor, red hilt).
		if (BindlessTouchImages() && heap->touched_frame != BindlessFrame()) {
			heap->touched_frame = BindlessFrame();
			auto& cache         = m_context.GetTextureCache();
			for (const auto id: heap->resolved) {
				if (auto* image = cache.m_slot_images.try_get(id); image != nullptr && image->registered) {
					cache.TouchImage(*image);
				}
			}
		}
		// KYTY_BINDLESS_SYNC_RETIRE (SyncWrittenBindlessEntries): in use again; a skipped write is
		// caught up by comparing every key now, before this draw.
		heap->sync_used_frame = BindlessFrame();
		if (heap->sync_stale) {
			heap->sync_stale = false;
			const auto keys  = std::min(heap->guest_entries, static_cast<uint32_t>(heap->settled.size()));
			uint32_t   view = 0, resource_changes = 0, resettled = 0;
			if (keys != 0 && heap->record_pointer == UINT32_MAX && !heap->runtime) {
				RecheckBindlessKeys(*heap, 0, keys, view, resource_changes, resettled);
			}
			g_bindless_retired_rechecks.fetch_add(1, std::memory_order_relaxed);
		}
		if (BindlessEngineEnabled() && table.NeedsWatch(*heap)) {
			table.WatchHeap(*heap);
		}
		if (pointers && BindlessRecordLazy()) {
			ProcessRecordHeap(*heap);
		}
		// KYTY_BINDLESS_PREFILL: a new small heap's keys, before this draw or dispatch reads them.
		if (BindlessPrefillMode() != 0u && !heap->prefilled && !pointers && !heap->runtime &&
		    heap->nested_vsharp == UINT32_MAX && entries <= BindlessPrefillMax()) {
			heap->prefilled = true;
			PrefillBindlessHeap(*heap, std::min(entries, heap->entries));
		}
		// KYTY_BINDLESS_REFILL: a heap's table whose bytes changed has its new keys settled before
		// this draw or dispatch reads them.
		if (BindlessRefillEnabled() && !pointers && !heap->runtime && heap->nested_vsharp == UINT32_MAX &&
		    entries <= BindlessRefillMax()) {
			RefillBindlessHeap(*heap, std::min(entries, heap->entries));
		}
		// KYTY_BINDLESS_REFRESH: the heap's resolved keys follow the texture cache (CPU writes, aliases).
		if (g_bindless_refresh.Get() != 0 && !heap->runtime) {
			RefreshBindlessHeap(*heap, std::min(entries, heap->entries));
		}
		prepared.bindless_patches.push_back(
		    {use.mapping_offset, heap->region, std::min(entries, heap->entries)});
		prepared.bindless_heaps.push_back(heap);
	}
}

namespace {
struct BindlessRefreshStats {
	std::atomic<uint64_t> lookups {0};
	std::atomic<uint64_t> draw_lookups {0};
	std::atomic<uint64_t> remapped {0};
	std::atomic<uint64_t> view_changed {0};
	std::atomic<uint64_t> unpinned {0};
	std::atomic<uint64_t> failed {0};
	uint64_t              logged_frame = 0;
} g_bindless_refresh_stats;
} // namespace

void RenderExecutor::RefreshBindlessHeap(BindlessHeap& heap, uint32_t keys) {
	const uint64_t frame    = BindlessFrame();
	const uint64_t draw     = ++m_bindless_refresh_draw;
	const bool     per_draw = g_bindless_refresh.Get() == 1;
	keys                    = std::min(keys, static_cast<uint32_t>(heap.slots.size()));
	const auto state_of     = [&](uint32_t entry) -> BindlessRefreshState& {
		const auto binding = BindlessTable::SlotBinding(heap, entry);
		const auto slot    = BindlessTable::SlotIndex(entry);
		auto&      states  = m_bindless_refresh[binding & 3u];
		if (states.size() <= slot) {
			states.resize(static_cast<size_t>(slot) + 1u);
		}
		return states[slot];
	};
	if (heap.refresh_frame != frame) {
		heap.refresh_frame = frame;
		heap.refresh_gpu_keys.clear();
		for (uint32_t key = 0; key < keys; key++) {
			const auto entry = heap.slots[key];
			if (entry == 0 || !heap.images[key]) {
				continue;
			}
			auto& state = state_of(entry);
			if (state.frame != frame) {
				state.frame = frame;
				state.draw  = draw;
				g_bindless_refresh_stats.lookups.fetch_add(1, std::memory_order_relaxed);
				if (!RefreshBindlessSlot(heap, key, state)) {
					continue;
				}
			}
			if (state.gpu_written && per_draw) {
				heap.refresh_gpu_keys.push_back(key);
			}
		}
	} else if (per_draw) {
		for (const auto key: heap.refresh_gpu_keys) {
			if (key >= keys) {
				continue;
			}
			const auto entry = heap.slots[key];
			if (entry == 0 || !heap.images[key]) {
				continue;
			}
			auto& state = state_of(entry);
			if (!state.gpu_written || state.draw == draw) {
				continue;
			}
			state.frame = frame;
			state.draw  = draw;
			g_bindless_refresh_stats.draw_lookups.fetch_add(1, std::memory_order_relaxed);
			RefreshBindlessSlot(heap, key, state);
		}
	}
	static const bool log = std::getenv("KYTY_BINDLESS_REFRESH_LOG") != nullptr;
	if (log && frame >= g_bindless_refresh_stats.logged_frame + 600u) {
		auto& st        = g_bindless_refresh_stats;
		st.logged_frame = frame;
		uint32_t gpu    = 0;
		for (const auto& states: m_bindless_refresh) {
			for (const auto& state: states) {
				gpu += state.gpu_written ? 1u : 0u;
			}
		}
		LOGF("Bindless refresh: frame=%" PRIu64 " lookups %" PRIu64 " per-draw %" PRIu64 " remapped %" PRIu64
		     " view-changed %" PRIu64 " unpinned %" PRIu64 " failed %" PRIu64 " gpu-written slots %u\n",
		     frame, st.lookups.exchange(0), st.draw_lookups.exchange(0), st.remapped.exchange(0),
		     st.view_changed.exchange(0), st.unpinned.exchange(0), st.failed.exchange(0), gpu);
	}
}

// One lookup of a resolved key (KYTY_BINDLESS_REFRESH). False when the key has no usable lookup now
// (left as it is: OnImageUnregistered and revalidation deal with such keys).
bool RenderExecutor::RefreshBindlessSlot(BindlessHeap& heap, uint32_t key, BindlessRefreshState& state) {
	auto&       table         = m_context.GetBindlessTable();
	auto&       texture_cache = m_context.GetTextureCache();
	const auto  entry         = heap.slots[key];
	const auto  old_id        = heap.images[key];
	auto*       old_image     = texture_cache.m_slot_images.try_get(old_id);
	const auto& words         = heap.descriptors[key];
	if (old_image == nullptr || !old_image->registered ||
	    std::ranges::all_of(words, [](uint32_t w) { return w == 0u; })) {
		return false;
	}
	ShaderRecompiler::IR::DescriptorValue value {};
	value.dword_count = 8u;
	std::copy_n(words.begin(), 8, value.dwords.begin());
	auto resource = heap.resource;
	if ((entry & ShaderRecompiler::IR::BindlessSlotFlagMask) == ShaderRecompiler::IR::BindlessSlotArrayFlag) {
		resource.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2DArray; // KYTY_BINDLESS_3D_AS_ARRAY
	}
	TextureBinding binding;
	ResolveTexture(resource, value, binding);
	auto* image = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || image->info.data.Empty() || !image->registered) {
		g_bindless_refresh_stats.failed.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	if (image->IsGpuModified()) {
		state.gpu_written = true;
	}
	if (state.gpu_written) {
		BindImage(binding.image_id, false);
	}
	// Uploads CPU writes (RefreshImage) and makes the view's levels resident.
	const auto view = texture_cache.FindTexture(binding.image_id, binding.desc);
	image           = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || view == nullptr || !image->registered) {
		g_bindless_refresh_stats.failed.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	const auto slot_binding = BindlessTable::SlotBinding(heap, entry);
	const auto slot         = BindlessTable::SlotIndex(entry);
	if (binding.image_id == old_id && view == table.SlotView(slot_binding, slot)) {
		return true;
	}
	if (binding.image_id == old_id) {
		g_bindless_refresh_stats.view_changed.fetch_add(1, std::memory_order_relaxed);
	} else {
		g_bindless_refresh_stats.remapped.fetch_add(1, std::memory_order_relaxed);
	}
	image->bindless_pinned  = true;
	image->bindless_self_id = binding.image_id;
	image->usage.texture    = true;
	const auto layout       = image->info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
	const bool no_refs = table.RemapSlotImage(slot_binding, slot, old_id, binding.image_id, view, layout);
	if (no_refs) {
		if (auto* old = texture_cache.m_slot_images.try_get(old_id); old != nullptr && old->bindless_pinned) {
			// No key refers to it any more: the texture cache may collect it again (ResettleBindlessKey).
			NoteBindlessStateChange(*old);
			old->bindless_pinned = false;
			texture_cache.ReleaseDroppedByGuest(old_id);
			g_bindless_refresh_stats.unpinned.fetch_add(1, std::memory_order_relaxed);
		}
	}
	return true;
}

void RenderExecutor::MakeBindlessImagesReadable(std::span<BindlessHeap* const> heaps,
                                                vk::CommandBuffer              target) {
	auto&      cache  = m_context.GetTextureCache();
	const bool verify = BindlessCommitVerify();
	// Bindless textures are sampled through set 1 in a read-only layout; one this command binds
	// as a render target or for writes keeps the layout that use needs.
	const auto readable_layout = [](const Image& image) {
		return image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
		                            : vk::ImageLayout::eShaderReadOnlyOptimal;
	};
	const auto skipped = [](const Image& image) {
		return !image.registered || image.binding.is_target || image.info.data.Empty() ||
		       (image.binding.is_bound && (image.binding.shader_write || image.binding.force_general));
	};
	const auto needs_transition = [&](const Image& image) {
		return image.backing.state.layout != readable_layout(image) ||
		       !image.backing.subresource_states.empty();
	};
	if (BindlessEngineEnabled() && !verify) {
		// Engine method, E5: only images settled since the last bindless draw, and pinned images
		// whose state changed (NoteBindlessStateChange), are made readable again; a heap of
		// thousands of images is not walked whenever any one of them changed.
		const auto make_readable = [&](ImageId id) {
			auto* image = cache.m_slot_images.try_get(id);
			if (image == nullptr || !image->bindless_pinned || skipped(*image) ||
			    !needs_transition(*image)) {
				return;
			}
			image->Transit(readable_layout(*image), vk::AccessFlagBits2::eShaderRead, {}, target, true);
		};
		for (auto* heap: heaps) {
			for (const auto id: heap->unchecked) {
				make_readable(id);
			}
			heap->unchecked.clear();
		}
		{
			std::scoped_lock lock {g_bindless_state_changes.mutex};
			m_bindless_changed.swap(g_bindless_state_changes.ids);
		}
		for (const auto id: m_bindless_changed) {
			make_readable(id);
		}
		m_bindless_changed.clear();
		return;
	}
	// A heap whose images' states did not change since it was last checked
	// (g_bindless_state_generation) needs a look only at the images resolved since.
	// KYTY_BINDLESS_VERIFY=1 checks every image and stops if a skipped one needed it.
	for (auto* heap: heaps) {
		heap->unchecked.clear();
		const auto& resolved = heap->resolved;
		const auto  skip     = heap->checked_generation ==
		                          g_bindless_state_generation.load(std::memory_order_relaxed) &&
		                      heap->checked_count <= resolved.size()
		                          ? heap->checked_count
		                          : size_t {0};
		for (size_t i = verify ? 0 : skip; i < resolved.size(); i++) {
			auto* image = cache.m_slot_images.try_get(resolved[i]);
			if (image == nullptr || skipped(*image) || !needs_transition(*image)) {
				continue;
			}
			if (verify && i < skip) {
				EXIT("bindless: image %u of heap 0x%016" PRIx64
				     " changed layout without a state generation change\n",
				     static_cast<uint32_t>(i), heap->base);
			}
			image->Transit(readable_layout(*image), vk::AccessFlagBits2::eShaderRead, {}, target, true);
		}
		// After the transitions above, which bump the generation themselves.
		heap->checked_generation = g_bindless_state_generation.load(std::memory_order_relaxed);
		heap->checked_count      = resolved.size();
	}
	// The engine's change list is not read on this path; keep it from growing.
	std::scoped_lock lock {g_bindless_state_changes.mutex};
	g_bindless_state_changes.ids.clear();
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared, DrawPrep::StagePlan* plan,
                                     bool keep_images) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	prepared.runtime = &runtime;
	prepared.xframe_verify_views = false;
	prepared.plan             = plan;
	prepared.plan_shader_data = false;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory      = {};
	prepared.buffer_sources.clear();
	prepared.buffers.clear();
	if (!keep_images) {
		prepared.samplers.clear();
	}
	prepared.shader_data.clear();
	const bool verify = plan != nullptr && DrawPrep::BindingsVerifyMode() != 0;
	// KYTY_DRAW_RUN continuation: the texture and sampler bindings are the previous draw's (the
	// same program, T# and S# words; RenderExecutor::DrawRunCandidate).
	CommitStats::SubScope texture_scope(CommitStats::Sub::Textures);
	if (!keep_images && !RepeatStageTextures(program, snapshot, prepared)) {
		// KYTY_DRAW_PREP_BINDINGS textures: the memo hashes the preparing worker computed.
		const auto count  = static_cast<uint32_t>(program.info.images.size());
		const bool hashes = plan != nullptr && plan->texture_hashes_valid &&
		                    plan->texture_hashes.size() == count;
		// KYTY_DRAW_PREP_BINDINGS texturememo (P4b-2): the memo entries the preparing worker found
		// for the keys (their tags). Runs of hits take the texture-cache lock once
		// (TextureBindingMemo::TryResolveRun); the first binding a run does not hit is resolved as
		// before, and the next run starts after it, so the order of effects is unchanged.
		const bool tags = hashes && plan->texture_tags_valid && plan->texture_tags.size() == count &&
		                  TextureBindingMemo::Enabled();
		const auto storage = [](const TextureBinding& binding) {
			return binding.desc.type == TextureCache::BindingType::Storage;
		};
		auto& texture_cache = m_context.GetTextureCache();
		prepared.images.resize(count);
		uint32_t run_end = 0; // verify: [i, run_end) predicted as run hits
		for (uint32_t i = 0; i < count;) {
			if (tags) {
				const auto rest = count - i;
				const std::span<const uint64_t> run_hashes(plan->texture_hashes.data() + i, rest);
				const std::span<const uint64_t> run_tags(plan->texture_tags.data() + i, rest);
				const std::span<TextureBinding> run_bindings(prepared.images.data() + i, rest);
				if (!verify) {
					const auto run = m_texture_memo.TryResolveRun(texture_cache, run_hashes, run_tags,
					                                              run_bindings);
					for (uint32_t j = i; j < i + run; j++) {
						// ResolveTexture's work besides the memo hit.
						auto& binding      = prepared.images[j];
						binding.image_view = nullptr;
						binding.layout     = vk::ImageLayout::eUndefined;
						binding.mip_views.clear();
						if (HangTrace::Enabled()) {
							const auto descriptor =
							    DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[j]);
							if (!descriptor.IsNull()) {
								HangTrace::RecordTexture(descriptor.fields);
							}
						}
						BindImage(binding.image_id, storage(binding));
					}
					if (run != 0) {
						OwnedCounters::Add(DrawPrep::GetBindingTotals().texture_run_hits, run);
						Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingTextureRunHits,
						                          run);
					}
					i += run;
					if (i == count) {
						break;
					}
				} else if (i >= run_end) {
					run_end = i + m_texture_memo.TryResolveRun(texture_cache, run_hashes, run_tags,
					                                           run_bindings, false);
				}
			}
			auto& binding = prepared.images[i];
			SetNullCompareFuncs(ImageCompareFuncs(program, snapshot, i));
			ResolveTexture(program.info.images[i], snapshot.images[i], binding,
			               hashes ? &plan->texture_hashes[i] : nullptr,
			               tags ? plan->texture_tags[i] : 0);
			if (verify && i < run_end) {
				// A predicted run hit is TryResolve's plain hit on the hinted entry.
				DrawPrep::CountBindingVerifyCheck();
				if (binding.memo_tag != plan->texture_tags[i] || m_texture_memo.LastHitRevalidated()) {
					DrawPrep::ReportBindingMismatch("texture memo run", i);
				}
			}
			BindImage(binding.image_id, storage(binding));
			i++;
		}
		prepared.texture_program = &program;
		prepared.texture_words.assign(snapshot.images.begin(), snapshot.images.end());
		if (program.stage == ShaderType::Pixel && TextureCache::LightGridProbeEnabled()) {
			for (uint32_t i = 0; i < count; i++) {
				const auto& resource = program.info.images[i];
				const auto& binding  = prepared.images[i];
				if (!resource.written && binding.image_id &&
				    binding.desc.info.pixel_format == vk::Format::eR32G32Uint &&
				    binding.desc.info.resources.layers > 1) {
					TextureCache::ProbeLightGrid(binding.desc.info, program.shader_hash);
				}
			}
		}
		if (program.stage == ShaderType::Pixel && TextureCache::LightGridCapEnabled()) {
			// KYTY_LIGHT_GRID_CAP (usa-ratt/KytyPS5-GT7 ecef51e59): a pixel shader's light grid
			// has its later uploads checked (TextureCache::NoteLightGridShape).
			for (uint32_t i = 0; i < count; i++) {
				const auto& resource = program.info.images[i];
				const auto& binding  = prepared.images[i];
				const auto& info     = binding.desc.info;
				if (!resource.written && binding.image_id &&
				    info.pixel_format == vk::Format::eR32G32Uint && info.resources.layers > 1 &&
				    (resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray ||
				     resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D)) {
					TextureCache::NoteLightGridShape(info);
				}
			}
		}
	}
	texture_scope.Stop();
	CommitStats::SubScope sampler_scope(CommitStats::Sub::Samplers);
	// KYTY_DRAW_PREP_BINDINGS samplers: the handles the worker found (never evicted); a null one is
	// resolved here as before.
	const bool plan_samplers = plan != nullptr && plan->samplers_valid &&
	                           !prepared.normalize_unnormalized &&
	                           plan->samplers.size() == program.info.samplers.size();
	if (!keep_images) {
		prepared.samplers.reserve(program.info.samplers.size());
	}
	uint32_t planned_samplers = 0;
	for (uint32_t i = 0; !keep_images && i < program.info.samplers.size(); i++) {
		auto sampler = plan_samplers ? plan->samplers[i] : vk::Sampler {};
		planned_samplers += sampler != nullptr ? 1u : 0u;
		if (sampler == nullptr || verify) {
			// Native filtering/border variants share their source sampler's snapshot entry.
			const auto serial =
			    NativeSampler(program, i, snapshot.samplers[program.info.samplers[i].snapshot_index],
			                  prepared.normalize_unnormalized);
			if (sampler != nullptr) {
				DrawPrep::CountBindingVerifyCheck();
				if (sampler != serial) {
					DrawPrep::ReportBindingMismatch("sampler handle", i);
				}
			}
			sampler = serial;
		}
		prepared.samplers.push_back(sampler);
	}
	if (planned_samplers != 0 && !verify) {
		OwnedCounters::Add(DrawPrep::GetBindingTotals().samplers_used, planned_samplers);
	}
	sampler_scope.Stop();
	// KYTY_DRAW_PREP_BINDINGS userdata: the worker's shader data (user dwords and mip-statistics
	// fields, memory offsets zero) is taken as it is; RebindBuffers only packs the offsets in.
	const auto dwords = program.bindings.ShaderDataDwords();
	if (plan != nullptr && plan->shader_data_valid && plan->shader_data.size() == dwords &&
	    !verify) {
		prepared.shader_data.swap(plan->shader_data);
		prepared.plan_shader_data = true;
		OwnedCounters::Add(DrawPrep::GetBindingTotals().shader_data_used, 1);
	} else {
		prepared.shader_data.reserve(dwords);
		AppendUserShaderData(program, snapshot, prepared.shader_data);
		prepared.shader_data.resize(dwords);
		if (verify && plan->shader_data_valid) {
			// The worker's vector is the serial one after RebindBuffers' reset and fields.
			static thread_local std::vector<uint32_t> serial;
			serial.assign(prepared.shader_data.begin(), prepared.shader_data.end());
			std::fill(serial.begin() + program.bindings.memory_offset_dword, serial.end(), 0);
			const bool active = WriteMipStatsFields(program, snapshot, serial);
			DrawPrep::CountBindingVerifyCheck();
			if (serial != plan->shader_data || active != plan->mip_stats_active) {
				DrawPrep::ReportBindingMismatch("shader data", static_cast<uint32_t>(program.stage));
			}
		}
	}
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
		NoteGdsProgram(program, snapshot);
	}
	// Bindless images (Senaxx): every draw and dispatch, also a draw-run continuation (the heap
	// regions go into the flattened SRT). Nothing for a program without bindless heaps.
	PrepareBindlessHeaps(runtime, prepared);
}

// The binding range is stride * NUM_RECORDS (NUM_RECORDS bytes without a stride) whatever the
// V# OOB_SELECT (RDNA2 ISA 8.1.5, Table 35: 0 also checks offset < stride, 1 checks the index
// only, 2 disables the check, 3 counts NUM_RECORDS in bytes). Log each OOB_SELECT / stride /
// swizzle / ADD_TID combination once so a run shows which modes a title uses.
static std::atomic<uint32_t> g_buffer_out_of_bounds_seen {0};

uint32_t BufferOutOfBoundsCombination(const ShaderBufferResource& descriptor) {
	return static_cast<uint32_t>(descriptor.OutOfBounds()) | (descriptor.Stride() != 0 ? 4u : 0u) |
	       (descriptor.SwizzleEnabled() ? 8u : 0u) | (descriptor.AddTid() ? 16u : 0u);
}

// KYTY_BUFFER_RANGE_FIXES=1 (default off): upstream buffer-range fixes for other games, each
// logged once when it changes a binding:
//  - fde0b98e6: a zero-stride V# with OOB_SELECT=2 tests only NUM_RECORDS != 0, so a non-scalar
//    fetch keeps the shader's whole payload (max_byte_extent) when NUM_RECORDS is smaller;
//  - cd4eab9d6: a written unbounded V# (NUM_RECORDS = ~0) is clamped to the storage-buffer limit
//    as reads are (PPSA15246);
//  - e5f0ee7fc: a finite V# below 1 GiB keeps its size across sparse PRT holes instead of being
//    cut at the first unmapped page.
// The draw-prep binding plan abstains from such bindings, so the serial path decides them.
bool BufferRangeFixesEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BUFFER_RANGE_FIXES");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

static void NoteBufferRangeFix(const char* what, uint64_t address, uint64_t before, uint64_t after) {
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::printf("KYTY_BUFFER_RANGE_FIXES: %s at 0x%016" PRIx64 ": 0x%" PRIx64 " -> 0x%" PRIx64 " bytes\n",
		            what, address, before, after);
		std::fflush(stdout);
	}
}

static void NoteBufferOutOfBoundsMode(const ShaderBufferResource& descriptor, uint64_t address) {
	auto&      seen        = g_buffer_out_of_bounds_seen;
	const auto combination = BufferOutOfBoundsCombination(descriptor);
	const auto bit = 1u << combination;
	if ((seen.load(std::memory_order_relaxed) & bit) != 0u ||
	    (seen.fetch_or(bit, std::memory_order_relaxed) & bit) != 0u) {
		return;
	}
	LOGF("Buffer: V# OOB_SELECT=%u stride=%u swizzle=%u add_tid=%u num_records=%u first bound at "
	     "0x%016" PRIx64 "\n",
	     static_cast<uint32_t>(descriptor.OutOfBounds()), static_cast<uint32_t>(descriptor.Stride()),
	     descriptor.SwizzleEnabled() ? 1u : 0u, descriptor.AddTid() ? 1u : 0u,
	     descriptor.NumRecords(), address);
}

// KYTY_BUFFER_PEEK=<shader hash hex>[:<begin hex>-<end hex>] (diagnostic, off by default): when
// that shader binds its buffers, the guest (CPU-side) bytes at each V# base: the first 8
// bindings, then every 300th. With a range: every binding of a call that binds a buffer
// overlapping it (at most 200 lines).
struct BufferPeekConfig {
	uint64_t hash = 0, begin = 0, end = 0;
};
static const BufferPeekConfig& GetBufferPeekConfig() {
	static const BufferPeekConfig config = [] {
		BufferPeekConfig c;
		const char* v = std::getenv("KYTY_BUFFER_PEEK");
		if (v == nullptr) {
			return c;
		}
		char* end = nullptr;
		c.hash    = std::strtoull(v, &end, 16);
		if (end != nullptr && *end == ':') {
			c.begin = std::strtoull(end + 1, &end, 16);
			if (end != nullptr && *end == '-') {
				c.end = std::strtoull(end + 1, nullptr, 16);
			}
		}
		return c;
	}();
	return config;
}

static void BufferPeek(uint64_t shader_hash, uint32_t index, uint64_t address, uint64_t size) {
	const auto& config = GetBufferPeekConfig();
	if (config.hash == 0 || shader_hash != config.hash) {
		return;
	}
	static std::atomic<uint32_t> calls {0};
	const auto n = calls.fetch_add(1, std::memory_order_relaxed);
	if (config.end != 0) {
		if (n >= 200) {
			return;
		}
	} else if (n >= 8 && (n % 300u) != 0u) {
		return;
	}
	const auto bytes = std::min<uint64_t>(size, 32);
	std::string text;
	if (Libs::LibKernel::Memory::ClampRangeSizeQuiet(address, bytes) == bytes) {
		for (uint64_t k = 0; k + 4 <= bytes; k += 4) {
			uint32_t word = 0;
			std::memcpy(&word, reinterpret_cast<const void*>(address + k), 4);
			float f = 0.0f;
			std::memcpy(&f, &word, 4);
			text += fmt::format(" {:08x}({:.5g})", word, f);
		}
	} else {
		text = " unreadable";
	}
	LOGF("BufferPeek: #%u CS 0x%016" PRIx64 " buffer %u 0x%016" PRIx64 "+0x%" PRIx64 ":%s\n", n,
	     shader_hash, index, address, size, text.c_str());
}

// KYTY_SPARSE_WRITES=1 (default off, read once; adapted from usa-ratt/KytyPS5-GT7 cb543d499): a
// written buffer binding over memory with holes in its mappings (sparse memory: 64 KB pages mapped
// only where needed) is bound from its base to the end of its last mapped byte, instead of up to
// its first hole (or as empty when its first page is a hole). The guest GPU's stores land in every
// mapped page; those to a hole stay in the cached buffer and are never written back. Up to 1 GB and
// the device's storage-buffer range; `clamped` otherwise. GT7 clears a sparse depth map with a
// compute fill over the whole range.
static uint64_t SparseWriteSize(uint64_t shader, uint64_t address, uint64_t requested,
                                uint64_t clamped, uint64_t limit) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SPARSE_WRITES");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled || address == 0) {
		return clamped;
	}
	const auto extent = Libs::LibKernel::Memory::CommittedExtent(address, requested);
	if (extent <= clamped || extent > std::min(limit, uint64_t {1} << 30u)) {
		return clamped;
	}
	static std::mutex                              mutex;
	static std::set<std::pair<uint64_t, uint64_t>> logged;
	std::lock_guard                                lock(mutex);
	if (logged.size() < 32 && logged.insert({shader, address}).second) {
		LOGF("RenderExecutor: shader %016" PRIx64 " writes 0x%016" PRIx64 "+0x%" PRIx64
		     " over memory with holes: bound to its last mapped byte, 0x%" PRIx64
		     " (up to the first hole it was 0x%" PRIx64 "; KYTY_SPARSE_WRITES)\n",
		     shader, address, requested, extent, clamped);
	}
	return extent;
}

void RenderExecutor::FindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       cache    = m_context.GetBufferCache();
	const BufferLookupStats::PartScope stats_scope(BufferLookupStats::Part::FindBuffer);

	prepared.buffer_sources.clear();
	prepared.buffer_sources.reserve(program.info.buffers.size());
	// KYTY_DRAW_PREP_BINDINGS buffers: the ranges the worker decoded and clamped, while the guest
	// virtual ranges are the ones it clamped against. FindBuffer (it may create and join buffers)
	// runs here, in the same order, either way.
	const auto* plan = prepared.plan;
	const bool  use_plan = plan != nullptr && plan->ranges_valid &&
	                      plan->ranges.size() == program.info.buffers.size() &&
	                      m_binding_plan != nullptr &&
	                      m_binding_plan->vm_generation == LibKernel::Memory::VirtualRangesGeneration();
	if (plan != nullptr && plan->ranges_valid && !use_plan) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingFallbackVm);
	}
	if (use_plan) {
		OwnedCounters::Add(DrawPrep::GetBindingTotals().ranges_used, 1);
	}
	const bool verify = use_plan && DrawPrep::BindingsVerifyMode() != 0;
	const auto& graphics = m_context.GetGraphics();
	const uint64_t unbounded_read_limit = std::min<uint64_t>(
	    uint64_t {256} * 1024 * 1024,
	    uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
	        (graphics.StorageMinAlignment() - 1));
	struct PeekItem {
		uint32_t index;
		uint64_t address, size;
	};
	std::vector<PeekItem> peek; // KYTY_BUFFER_PEEK
	const auto            flush_peek = [&] {
		const auto& config = GetBufferPeekConfig();
		bool        hit    = config.end == 0;
		for (const auto& p: peek) {
			hit = hit || (p.address < config.end && p.address + p.size > config.begin);
		}
		if (hit) {
			for (const auto& p: peek) {
				BufferPeek(program.shader_hash, p.index, p.address, p.size);
			}
		}
	};
	struct FlushPeekAtExit {
		const decltype(flush_peek)& f;
		~FlushPeekAtExit() { f(); }
	} flush_peek_at_exit {flush_peek};
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		uint64_t address = 0;
		uint64_t size    = 0;
		if (use_plan) {
			const auto& range = plan->ranges[i];
			if ((g_buffer_out_of_bounds_seen.load(std::memory_order_relaxed) &
			     (1u << range.out_of_bounds_mode)) == 0u) {
				const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
				NoteBufferOutOfBoundsMode(descriptor, descriptor.Base48());
			}
			address = range.address;
			size    = range.size;
		}
		if (!use_plan || verify) {
			auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
			const auto serial_address = descriptor.Base48();
			auto       requested_size = descriptor.GetSize();
			if (!use_plan) {
				NoteBufferOutOfBoundsMode(descriptor, serial_address);
			}
			const bool range_fixes = BufferRangeFixesEnabled();
			if (range_fixes && requested_size != 0 && descriptor.Stride() == 0 &&
			    descriptor.OutOfBounds() == 2 && !program.info.buffers[i].scalar &&
			    program.info.buffers[i].max_byte_extent > requested_size) {
				NoteBufferRangeFix("zero-stride payload", serial_address, requested_size,
				                   program.info.buffers[i].max_byte_extent);
				requested_size = program.info.buffers[i].max_byte_extent;
			}
			uint64_t serial_size = 0;
			uint64_t serial_base = 0;
			if (serial_address != 0 && requested_size != 0) {
				serial_base = serial_address;
				serial_size = Libs::LibKernel::Memory::ClampRangeSizeQuiet(serial_address, requested_size);
				if (range_fixes && serial_size != 0 && serial_size != requested_size &&
				    requested_size < (uint64_t {1} << 30u) && descriptor.NumRecords() != UINT32_MAX) {
					NoteBufferRangeFix("finite range across sparse holes", serial_address, serial_size,
					                   requested_size);
					serial_size = requested_size;
				}
				if (serial_size != requested_size && program.info.buffers[i].written) {
					serial_size = SparseWriteSize(program.shader_hash, serial_address, requested_size,
					                              serial_size, unbounded_read_limit);
				}
				if (serial_size == 0) {
					// An unmapped V# (typically left over from a skipped shader's output): bind
					// no buffer instead of ending the emulator.
					static std::atomic_bool unmapped_warned = false;
					if (!unmapped_warned.exchange(true, std::memory_order_relaxed)) {
						std::printf("Warning: buffer descriptor %u points at unmapped memory "
						            "(addr=0x%016" PRIx64 " size=0x%" PRIx64 "); bound as empty\n",
						            static_cast<uint32_t>(i), serial_address, requested_size);
					}
					serial_base = 0;
				}
			}
			if (verify) {
				DrawPrep::CountBindingVerifyCheck();
				if (serial_base != address || serial_size != size ||
				    BufferOutOfBoundsCombination(descriptor) != plan->ranges[i].out_of_bounds_mode) {
					DrawPrep::ReportBindingMismatch("buffer range", i);
				}
			}
			address = serial_base;
			size    = serial_size;
		}
		if (address == 0 && size == 0) {
			prepared.buffer_sources.push_back({});
			continue;
		}
		if (GetBufferPeekConfig().hash == program.shader_hash) {
			peek.push_back({i, address, size});
		}
		if (size > unbounded_read_limit &&
		    (!program.info.buffers[i].written || BufferRangeFixesEnabled()) &&
		    DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]).NumRecords() ==
		        UINT32_MAX) {
			if (program.info.buffers[i].written) {
				NoteBufferRangeFix("written unbounded V# clamp", address, size, unbounded_read_limit);
			}
			// Upstream #1066 (e5dcfdb87): a read through an unbounded V# (NUM_RECORDS = ~0) binds
			// at most 256 MiB, within the storage-buffer range limit. Only sizes past the limit
			// decode the V# here, so the plan and serial ranges above stay as they are.
			size = unbounded_read_limit;
		}
		if (const auto& resource = program.info.buffers[i]; !resource.written && !resource.formatted) {
			// KYTY_CP_BINDING_BATCH_PREFETCH=find (BufferCache::PrefetchReadBinding).
			cache.PrefetchReadBinding(address, size, true);
		}
		BufferLookupStats::PartScope find_scope(BufferLookupStats::Part::FindCall);
		prepared.buffer_sources.push_back({address, size, cache.FindBuffer(address, size)});
	}
}

void RenderExecutor::FindBuffers(std::span<PreparedBindings* const> stages) {
	for (auto* stage: stages) {
		FindBuffers(*stage);
	}
}

namespace {

struct WriteRangeScratch {
	ShaderRecompiler::IR::WriteRangeEvaluator         evaluator;
	std::vector<ShaderRecompiler::IR::WriteRangeSpan> spans;
	std::vector<GuestRange>                           ranges;
};

WriteRangeScratch& ThreadWriteRangeScratch() {
	thread_local WriteRangeScratch scratch;
	return scratch;
}

// The descriptor's stride/swizzle/ADD_TID bits normalized exactly as the resource
// specialization that selected the compiled program (BuildResourceSpecialization).
uint32_t SpecializedPackedStride(const ShaderBufferResource& descriptor) {
	auto       packed  = descriptor.PackedStride();
	const auto stride  = packed & 0x3fffu;
	const bool swizzle = stride != 0u && ((packed >> 14u) & 1u) != 0u;
	if (stride == 0u) {
		packed &= ~((1u << 14u) | (3u << 16u));
	} else if (!swizzle) {
		packed &= ~(3u << 16u);
	}
	return packed;
}

} // namespace

// KYTY_SRT_GPU_FILL: whether the write-range program reads a flat dword the refresh left to the
// GPU (its snapshot word is a placeholder).
static bool WriteRangesReadGpuFills(const ShaderRecompiler::IR::WriteRangeProgram& ranges,
                                    std::span<const ShaderRecompiler::IR::SrtGpuFill> fills) {
	for (const auto& node: ranges.nodes) {
		if (node.op != ShaderRecompiler::IR::WriteRangeNode::Op::FlatSrt) {
			continue;
		}
		for (const auto& fill: fills) {
			if (fill.flat_offset == node.a) {
				return true;
			}
		}
	}
	return false;
}

// KYTY_SRT_GPU_FILL_DIAG=1: console counts of the write ranges above, every 10 s while they happen.
static void NoteGpuFillWriteRange(bool whole) {
	if (!SrtGpuFillDiag()) {
		return;
	}
	static std::atomic<uint64_t> whole_count {0};
	static std::atomic<uint64_t> host_count {0};
	static std::atomic<int64_t>  last_ms {0};
	(whole ? whole_count : host_count).fetch_add(1, std::memory_order_relaxed);
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	auto last = last_ms.load(std::memory_order_relaxed);
	if (last == 0) {
		last_ms.store(now, std::memory_order_relaxed);
	} else if (now - last >= 10000 && last_ms.compare_exchange_strong(last, now)) {
		std::printf("SrtGpuFill write ranges %.0fs: %" PRIu64 " bindings taken whole, %" PRIu64
		            " evaluations read the filled dwords on the host\n",
		            static_cast<double>(now - last) / 1000.0, whole_count.exchange(0),
		            host_count.exchange(0));
	}
}

// A binding up to this size whose write range reads a GPU-filled dword is taken as written
// whole; a larger one reads the dword on the host for an exact range, as without the fill.
constexpr uint64_t GpuFillWholeBindingLimit = 16ull * 1024 * 1024;

// Guest ranges the shader can write through writable buffer `index`, or nullptr when the whole
// binding must be treated as written (no proof, proof covers everything, or disabled).
static const std::vector<GuestRange>* ResolveWrittenRanges(RenderContext&            context,
                                                           const ShaderStageRuntime& runtime,
                                                           const PreparedBindings&   prepared,
                                                           uint32_t index, bool& evaluated,
                                                           WriteRangeScratch& scratch) {
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	const auto& resource = program.info.buffers[index];
	const auto& source   = prepared.buffer_sources[index];
	if (!resource.written || source.address == 0 || source.size == 0) {
		return nullptr;
	}
	const auto* entry  = program.write_ranges.Find(index);
	bool        proven = false;
	const char* result = "unprovable";
	// KYTY_SRT_GPU_FILL: the program reads a dword the GPU fills (a store offset the GPU wrote).
	const bool reads_fills = !snapshot.gpu_fills.empty() && PreciseWriteRangesEnabled() &&
	                         entry != nullptr && entry->bounded &&
	                         WriteRangesReadGpuFills(program.write_ranges, snapshot.gpu_fills);
	if (!PreciseWriteRangesEnabled()) {
		result = "disabled";
	} else if (entry == nullptr || !entry->bounded) {
		result = "unprovable";
	} else if (reads_fills && source.size <= GpuFillWholeBindingLimit) {
		result = "gpu-fill";
		NoteGpuFillWriteRange(true);
	} else if (index >= snapshot.buffers.size() ||
	           SpecializedPackedStride(DecodeNativeDescriptor<ShaderBufferResource>(
	               snapshot.buffers[index])) != entry->packed_stride) {
		result = "stride-mismatch";
	} else {
		if (!evaluated) {
			std::span<const uint32_t> flat = snapshot.flattened_srt;
			if (reads_fills) {
				// Exact bounds need the dwords' values: read them on the host (waiting for the
				// GPU, as without the fill). They are then no longer GPU-written, so the flat
				// upload reads the same values on the host too.
				thread_local std::vector<uint32_t> patched;
				patched.assign(flat.begin(), flat.end());
				for (const auto& fill: snapshot.gpu_fills) {
					if (fill.flat_offset < patched.size()) {
						std::memcpy(&patched[fill.flat_offset],
						            reinterpret_cast<const void*>(fill.address), sizeof(uint32_t));
					}
				}
				flat = patched;
				NoteGpuFillWriteRange(false);
			}
			const ShaderRecompiler::IR::WriteRangeInputs inputs {
			    .user_data     = snapshot.user_data,
			    .flattened_srt = flat,
			    .groups        = prepared.dispatch_groups,
			    .has_groups    = prepared.has_dispatch_groups,
			};
			scratch.evaluator.Evaluate(program.write_ranges, inputs);
			evaluated = true;
		}
		proven = scratch.evaluator.Spans(program.write_ranges, index, source.size, scratch.spans);
		result = proven ? "narrowed" : "unbounded";
	}
	uint64_t written_bytes = source.size;
	scratch.ranges.clear();
	if (proven) {
		written_bytes = 0;
		for (const auto& span: scratch.spans) {
			scratch.ranges.push_back({source.address + span.begin, span.end - span.begin});
			written_bytes += span.end - span.begin;
		}
	}
	const bool narrowed = proven && written_bytes < source.size;
	if (proven && !narrowed) {
		result = "covers-binding";
	}
	uint32_t spared = 0;
	if (narrowed) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::WriteRangeNarrowed);
		Profiler::CountFrameEvent(Profiler::FrameEvent::WriteRangeBytesAvoided,
		                          source.size - written_bytes);
		if (WriteRangeImageStatsEnabled()) {
			spared = context.GetTextureCache().CountImagesOutsideGpuWrite(source.address,
			                                                              source.size, scratch.ranges);
			Profiler::CountFrameEvent(Profiler::FrameEvent::WriteRangeImagesSpared, spared);
		}
	} else {
		Profiler::CountFrameEvent(Profiler::FrameEvent::WriteRangeWhole);
	}

	if (const auto limit = WriteRangeLogLimit(); limit != 0u) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < limit) {
			std::string spans;
			for (const auto& range: scratch.ranges) {
				spans += fmt::format(" 0x{:x}+0x{:x}", range.address, range.size);
			}
			const auto detail =
			    entry != nullptr && evaluated
			        ? scratch.evaluator.Describe(program.write_ranges, index)
			        : std::string(entry == nullptr ? "no-proof" : entry->bounded ? "" : "unprovable");
			LOGF("WriteRange: stage=%s shader=0x%016" PRIx64 " slot=%u guest=0x%016" PRIx64
			     " size=0x%" PRIx64 " read=%d atomic=%d groups=%s%ux%ux%u result=%s written=0x%" PRIx64
			     " images_spared=%u spans:%s detail: %s\n",
			     ShaderStageResourceName(program.stage), program.shader_hash, index, source.address,
			     source.size, resource.read ? 1 : 0, resource.atomic ? 1 : 0,
			     prepared.has_dispatch_groups ? "" : "unknown:", prepared.dispatch_groups[0],
			     prepared.dispatch_groups[1], prepared.dispatch_groups[2], result, written_bytes,
			     spared, spans.c_str(), detail.c_str());
		}
	}
	return narrowed ? &scratch.ranges : nullptr;
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	prepared.write_preparation_tick = BufferCache::ShaderWriteRetickEnabled()
	                                      ? m_context.GetCommandScheduler().CurrentTick()
	                                      : UINT64_MAX;
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	EXIT_IF(prepared.buffer_sources.size() != program.info.buffers.size());

	prepared.buffers.clear();
	prepared.buffers.reserve(program.info.buffers.size());
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	// A plan's shader data (PrepareBindings) holds zero offsets and the fields until the offsets
	// are packed below; a later rebind of these bindings starts from the reset again.
	const bool plan_shader_data = prepared.plan_shader_data;
	prepared.plan_shader_data   = false;
	if (!plan_shader_data) {
		std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
		          prepared.shader_data.end(), 0);
	}
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	auto& write_scratch = ThreadWriteRangeScratch();
	bool  write_ranges_evaluated = false;
	// All bindings' uploads share one barrier pair (KYTY_UPLOAD_BATCH).
	const BufferCache::UploadBatch upload_batch(m_context.GetBufferCache());
	// KYTY_CP_BINDING_BATCH_PREFETCH: the memo slots of every read binding first, so their cache
	// misses overlap (a hint only; the lookups below are unchanged).
	BufferLookupStats::PartScope prefetch_scope(BufferLookupStats::Part::Prefetch);
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& source   = prepared.buffer_sources[i];
		const auto& resource = program.info.buffers[i];
		if (source.address != 0 && source.size != 0 && !resource.written && !resource.formatted) {
			m_context.GetBufferCache().PrefetchReadBinding(source.address, source.size);
		}
	}
	prefetch_scope.Stop();
	CommitStats::StepMark(CommitStats::Step::RbHead);
	const auto* uniform_binding = ShaderRecompiler::IR::FindBinding(
	    program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::UniformBuffers);
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		uint32_t   buffer_offset = 0;
		const auto ranges_start  = program.info.buffers[i].written ? BufferLookupStats::Now() : 0;
		const auto* written      = ResolveWrittenRanges(m_context, *prepared.runtime, prepared, i,
		                                                write_ranges_evaluated, write_scratch);
		if (ranges_start != 0) {
			BufferLookupStats::AddPart(BufferLookupStats::Part::WriteRanges,
			                           BufferLookupStats::Now() - ranges_start);
		}
		const bool uniform_view = uniform_binding != nullptr &&
		                          std::ranges::binary_search(uniform_binding->resources, i);
		prepared.buffers.push_back(NativeStorageBuffer(m_context, prepared.buffer_sources[i],
		                                               program.info.buffers[i], program.stage, i,
		                                               buffer_offset, written, uniform_view));
		pack_memory_offset(i, buffer_offset);
	}
	CommitStats::StepMark(CommitStats::Step::RbBindings);
	prepared.mip_stats_canary = false;
	BufferLookupStats::PartScope mip_scope(BufferLookupStats::Part::MipStats);
	prepared.mip_stats_active = plan_shader_data
	                                ? prepared.plan->mip_stats_active
	                                : WriteMipStatsFields(program, snapshot, prepared.shader_data);
	mip_scope.Stop();
	CommitStats::StepMark(CommitStats::Step::RbMipStats);
	const BufferLookupStats::PartScope tables_scope(BufferLookupStats::Part::Tables);
	// Upload sites: stage type and table kind (the dedup checks the site's last entry first).
	const auto site = static_cast<uint32_t>(program.stage) * 2u;
	prepared.fresh_upload = false;
	if (ShaderRecompiler::IR::FindBinding(
	        layout, ShaderRecompiler::IR::DescriptorBindingKind::FlattenedSrt) != nullptr) {
		bool                      fresh = false;
		std::span<const uint32_t> flat  = snapshot.flattened_srt;
		if (!prepared.bindless_patches.empty()) [[unlikely]] {
			// Bindless heaps (Senaxx): each image's two words get its heap's translation region and
			// entry count (PrepareBindlessHeaps); zeros sample the placeholder.
			m_bindless_srt.assign(snapshot.flattened_srt.begin(), snapshot.flattened_srt.end());
			for (const auto& [offset, region, entries]: prepared.bindless_patches) {
				if (static_cast<size_t>(offset) + 1u < m_bindless_srt.size()) {
					m_bindless_srt[offset]      = region;
					m_bindless_srt[offset + 1u] = entries;
				}
				// A GPU fill of a patched word would overwrite the region or entry count.
				for (const auto& fill: snapshot.gpu_fills) {
					if (fill.flat_offset == offset || fill.flat_offset == offset + 1u) {
						static std::atomic<uint32_t> reported {0};
						if (reported.fetch_add(1, std::memory_order_relaxed) < 32) {
							LOGF("Bindless patch: hash=0x%016" PRIx64 " offset=%u is overwritten by a GPU fill "
							     "(word %u from 0x%" PRIx64 ")\n",
							     program.shader_hash, offset, fill.flat_offset, fill.address);
						}
					}
				}
				// The first 256 distinct patches are logged; after that no draw takes the lock.
				static std::atomic<bool>            logging_done = false;
				static std::mutex                   logged_mutex;
				static std::unordered_set<uint64_t> logged;
				if (logging_done.load(std::memory_order_relaxed)) {
					continue;
				}
				std::scoped_lock lock(logged_mutex);
				if (logged.size() >= 256) {
					logging_done.store(true, std::memory_order_relaxed);
				} else if (logged.insert(program.shader_hash ^ offset).second) {
					LOGF("Bindless patch: stage=%u hash=0x%016" PRIx64
					     " offset=%u region=%u entries=%u srt_words=%zu\n",
					     static_cast<uint32_t>(program.stage), program.shader_hash, offset, region,
					     entries, m_bindless_srt.size());
				}
			}
			flat = m_bindless_srt;
		}
		if (!snapshot.gpu_fills.empty()) [[unlikely]] {
			prepared.flattened_srt = UploadFlatWithGpuFills(m_context, snapshot, flat);
			fresh                  = true;
		} else {
			prepared.flattened_srt = UploadShaderData(flat, site, &fresh);
		}
		prepared.fresh_upload |= fresh;
	}
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::ShaderData) != nullptr) {
		bool fresh                  = false;
		prepared.shader_data_buffer = UploadShaderData(prepared.shader_data, site + 1u, &fresh);
		prepared.fresh_upload |= fresh;
	}
	if (BufferLookupStats::Enabled()) {
		BufferLookupStats::NoteStage(SyncEpoch::Current());
	}
	CommitStats::StepMark(CommitStats::Step::RbTables);
}

// KYTY_SPARSE_HOLES=1 (TextureCache::ResetUnmappedDepth): after a stage's views are acquired (and
// their images refreshed), each image it samples with a depth comparison has its unmapped blocks
// reset to the depth that passes the comparison: 1.0 for LESS/LEQUAL, 0.0 for GREATER/GEQUAL. None
// for other functions, bindless samplers, or samplers of one image that disagree.
static void ResetSparseDepthHoles(TextureCache& texture_cache, const PreparedBindings& prepared) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SPARSE_HOLES");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled || prepared.runtime == nullptr || !*prepared.runtime) {
		return;
	}
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	const auto& images   = prepared.images;
	for (uint32_t i = 0; i < program.info.images.size() && i < images.size(); i++) {
		if (!program.info.images[i].depth_compare ||
		    images[i].desc.type == TextureCache::BindingType::Storage) {
			continue;
		}
		std::optional<float> depth;
		bool                 usable = true;
		for (const auto& pair: program.info.sampled_pairs) {
			if (pair.image != i || pair.sampler >= program.info.samplers.size()) {
				continue;
			}
			const auto& sampler = program.info.samplers[pair.sampler];
			if (!sampler.depth_compare) {
				continue;
			}
			if (sampler.bindless || sampler.snapshot_index >= snapshot.samplers.size()) {
				usable = false;
				break;
			}
			float passing = 0.0f;
			switch (DecodeNativeDescriptor<ShaderSamplerResource>(snapshot.samplers[sampler.snapshot_index])
			            .DepthCompareFunc()) {
				case 1: // LESS
				case 3: // LEQUAL
					passing = 1.0f;
					break;
				case 4: // GREATER
				case 6: // GEQUAL
					passing = 0.0f;
					break;
				default: usable = false; break;
			}
			if (!usable || (depth.has_value() && *depth != passing)) {
				usable = false;
				break;
			}
			depth = passing;
		}
		if (usable && depth.has_value()) {
			texture_cache.ResetUnmappedDepth(images[i].image_id, *depth);
		}
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF(images.size() != program.info.images.size());
	auto& texture_cache = m_context.GetTextureCache();
	// KYTY_SPARSE_HOLES: once the views are acquired (and their images refreshed), on every exit,
	// for draws and dispatches alike.
	struct SparseDepthHolesOnExit {
		TextureCache&           cache;
		const PreparedBindings& bindings;
		~SparseDepthHolesOnExit() { ResetSparseDepthHoles(cache, bindings); }
	} sparse_depth_holes {texture_cache, prepared};
	// KYTY_DRAW_SEQUENCE_FAST (textures): when every binding would get TryAcquireView's hit and
	// nothing else (TextureBindingMemo::TryRepeatViews, one texture-cache lock for the stage).
	bool verify_views = false;
	if (DrawSequenceEnabled(DrawSequencePart::Textures) && TextureBindingMemo::Enabled() &&
	    !images.empty() &&
	    std::ranges::none_of(program.info.images, [](const auto& resource) {
		    return resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic;
	    })) {
		const bool verify = DrawSequenceVerifyMode() != 0 || prepared.xframe_verify_views;
		if (m_texture_memo.TryRepeatViews(texture_cache, images, !verify)) {
			m_draw_sequence_totals.view_repeats++;
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawSequenceViewRepeats);
			if (!verify) {
				for (auto& binding: images) {
					binding.mip_views.clear();
					// Sampled bindings only (TryRepeatViews).
					auto& image         = texture_cache.GetImage(binding.image_id);
					image.usage.texture = true;
					texture_cache.NoteExactRangeStaleUse(image, false);
				}
				return;
			}
			// KYTY_DRAW_SEQUENCE_VERIFY: the loops below must acquire the views claimed here (a
			// guest write racing the check may make them refresh an image, not change a view).
			m_draw_sequence_totals.verify_checks++;
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawSequenceVerifyChecks);
			m_claimed_views.clear();
			for (const auto& binding: images) {
				m_claimed_views.push_back(m_texture_memo.EntryView(binding));
			}
			verify_views = true;
		}
	}
	const auto count        = static_cast<uint32_t>(program.info.images.size());
	const auto needs_rebind = [&](uint32_t i) {
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		return old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		       old_image->binding.needs_rebind;
	};
	const auto rebind = [&](uint32_t i) {
		if (const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		    old_image != nullptr) {
			if (old_image->binding.is_target) {
				NoteBindlessStateChange(*old_image);
			}
			old_image->binding = {};
		}
		SetNullCompareFuncs(ImageCompareFuncs(program, snapshot, i));
		ResolveTexture(program.info.images[i], snapshot.images[i], images[i]);
		BindImage(images[i].image_id, images[i].desc.type == TextureCache::BindingType::Storage);
	};
	const auto acquire_view = [&](uint32_t i) {
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u);
			// usa-ratt/KytyPS5-GT7 3d2a02104: a T# bound as a null texture (KYTY_SKIP_INVALID) has
			// one level where the shader has a view for each of its T#'s: with KYTY_SKIP_INVALID
			// the views past the binding's levels repeat its last one.
			const uint32_t levels = std::max(binding.desc.view_info.level_count, 1u);
			if (levels != resource.mip_count) {
				EXIT_IF(!Common::SkipInvalidEnabled());
				static std::atomic<uint32_t> reported {0};
				if (reported.fetch_add(1, std::memory_order_relaxed) < 16u) {
					LOGF("Dynamic mips: shader %016" PRIx64 " tex[%u] 0x%012" PRIx64
					     " has %u levels bound where the shader selects among %u: the views past "
					     "them repeat its last level\n",
					     program.shader_hash, i, binding.desc.info.data.address, levels,
					     resource.mip_count);
				}
			}
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += std::min(mip, levels - 1u);
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else if (!m_texture_memo.TryAcquireView(texture_cache, binding)) {
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
			m_texture_memo.RecordView(binding, binding.image_view);
		}
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		image.usage.storage |= storage;
		image.usage.texture |= !storage;
		texture_cache.NoteExactRangeStaleUse(image, storage);
	};
	// KYTY_DRAW_SEQUENCE_VERIFY / KYTY_XFRAME_REUSE=verify: the views claimed above.
	const auto check_claimed_views = [&] {
		if (!verify_views) {
			return;
		}
		for (uint32_t i = 0; i < count; i++) {
			if (prepared.xframe_verify_views) {
				// KYTY_XFRAME_REUSE=verify: the views of a set taken from the cross-frame store.
				XFrameReuse::GetTotals().verify_checks.fetch_add(1, std::memory_order_relaxed);
				if (images[i].image_view != m_claimed_views[i]) {
					XFrameReuse::ReportMismatch("view", i);
				}
			} else if (images[i].image_view != m_claimed_views[i]) {
				m_draw_sequence_totals.verify_mismatches++;
				ReportDrawSequenceMismatch("a repeated stage texture view");
			}
		}
	};
	uint32_t first_rebind = count;
	for (uint32_t i = 0; i < count; i++) {
		if (needs_rebind(i)) {
			first_rebind = i;
			break;
		}
	}
	if (first_rebind != count) {
		// Upstream a2f851788: a binding's rebind can replace the image an earlier binding of the
		// stage resolved to, so each view is acquired before the next binding rebinds. Only a stage
		// with a binding to rebind (an image changed) takes this order, without view runs.
		for (uint32_t i = 0; i < count; i++) {
			if (i >= first_rebind && needs_rebind(i)) {
				rebind(i);
			}
			acquire_view(i);
		}
		check_claimed_views();
		return;
	}
	// KYTY_DRAW_PREP_BINDINGS texturememo (P4b-2): runs of TryAcquireView hits take the
	// texture-cache lock once (TextureBindingMemo::TryAcquireViewRun); the first binding a run does
	// not hit takes the path below, and the next run starts after it.
	const bool view_runs = prepared.plan != nullptr && prepared.plan->texture_tags_valid &&
	                       TextureBindingMemo::Enabled();
	const bool verify_runs  = view_runs && DrawPrep::BindingsVerifyMode() != 0;
	uint32_t   view_run_end = 0; // verify: [i, view_run_end) predicted as run hits
	const auto run_length   = [&](uint32_t first) {
        uint32_t end = first;
        while (end < count &&
               program.info.images[end].mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic) {
            end++;
        }
        return end - first;
	};
	for (uint32_t i = 0; i < count; i++) {
		if (view_runs && !verify_runs) {
			const auto run = m_texture_memo.TryAcquireViewRun(
			    texture_cache, std::span<TextureBinding>(images.data() + i, run_length(i)));
			for (uint32_t j = i; j < i + run; j++) {
				images[j].mip_views.clear();
				// Sampled bindings only (TryAcquireView).
				auto& image         = texture_cache.GetImage(images[j].image_id);
				image.usage.texture = true;
				texture_cache.NoteExactRangeStaleUse(image, false);
			}
			if (run != 0) {
				OwnedCounters::Add(DrawPrep::GetBindingTotals().view_run_hits, run);
				Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingViewRunHits, run);
			}
			i += run;
			if (i == count) {
				break;
			}
		} else if (verify_runs && i >= view_run_end) {
			const std::span<TextureBinding> rest(images.data() + i, run_length(i));
			view_run_end = i + m_texture_memo.TryAcquireViewRun(texture_cache, rest, false);
			m_claimed_run_views.clear();
			for (uint32_t j = i; j < view_run_end; j++) {
				m_claimed_run_views.push_back(m_texture_memo.EntryView(images[j]));
			}
		}
		const auto claimed_view = verify_runs && i < view_run_end
		                              ? m_claimed_run_views[i - (view_run_end - m_claimed_run_views.size())]
		                              : vk::ImageView {};
		acquire_view(i);
		if (verify_runs && i < view_run_end) {
			// A predicted run hit acquires the entry's view.
			DrawPrep::CountBindingVerifyCheck();
			if (images[i].image_view != claimed_view) {
				DrawPrep::ReportBindingMismatch("texture view run", i);
			}
		}
	}
	check_claimed_views();
}

// KYTY_REBIND_AFTER_JOIN=1 (default off; chenxiao07/KytyPS5 ed6f063e1): a draw or dispatch whose
// resolved buffers lost one to a join (BufferCache::UnregisterEpoch moved) resolves them again
// before it is recorded. The first 16 occurrences are logged.
bool RebindAfterJoinEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_REBIND_AFTER_JOIN");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

void NoteRebindAfterJoin(const char* kind, const char* name) {
	static std::atomic<uint32_t> count {0};
	const auto n = count.fetch_add(1, std::memory_order_relaxed);
	if (n < 16 || (n & (n - 1)) == 0) {
		std::printf("Kyty rebind after join: a %s's buffers resolved again (%s; %u so far)\n", kind,
		            name != nullptr ? name : "-", n + 1);
	}
}

void PrefaultDmaPointers(BufferCache& buffers, const ShaderStageRuntime& stage) {
	static const uint64_t span = [] {
		const char* value = std::getenv("KYTY_DMA_PREFAULT_KB");
		return (value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 10) : 0ull)
		       << 10u;
	}();
	if (span == 0 || stage.program == nullptr || stage.resources == nullptr) {
		return;
	}
	const auto& program = *stage.program;
	const auto& user    = stage.resources->user_data;
	// A 64-bit SGPR pair starts at an even register; a pointer a scalar load reads through is
	// 4-byte aligned. Pages that are not committed are left alone (MakeResident).
	for (size_t reg = program.user_data_base % 2u; reg + 1u < user.size(); reg += 2u) {
		const uint32_t low  = user[reg];
		const uint32_t high = user[reg + 1u];
		if ((low & 3u) != 0 || high == 0 || high > 0xffffu) {
			continue;
		}
		const uint64_t address = low | (uint64_t {high} << 32u);
		const auto     made    = buffers.MakeResident(address, span);
		if (made == 0) {
			continue;
		}
		static std::atomic<uint64_t> total {0};
		const auto count = total.fetch_add(made, std::memory_order_relaxed) + made;
		if (count <= 32 || (count & (count - 1u)) == 0) {
			LOGF("BufferCache: %u pages shader %016" PRIx64 " reads through user data s[%zu:%zu] "
			     "0x%012" PRIx64 " had no cached buffer: cached before it runs (%" PRIu64
			     " so far; KYTY_DMA_PREFAULT_KB)\n",
			     made, program.shader_hash, program.user_data_base + reg,
			     program.user_data_base + reg + 1u, address, count);
		}
	}
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors, bool keep_images) {
	bool uses_dma = false;
	bool uses_bvh = false;
	for (auto* stage: stages) {
		if (stage->runtime->program->info.uses_dma) {
			PrefaultDmaPointers(m_context.GetBufferCache(), *stage->runtime);
		}
	}
	for (auto* stage: stages) {
		FindBuffers(*stage);
		uses_dma |= stage->runtime->program->info.uses_dma;
		uses_bvh |= stage->runtime->program->info.uses_bvh;
	}
	CommitStats::StepMark(CommitStats::Step::FindBuffers);
	if (uses_dma) {
		const BufferLookupStats::PartScope stats_scope(BufferLookupStats::Part::PrepareBda);
		m_context.PrepareBda();
	}
	// KYTY_GRAPHICS_BDA_WRITES: a stage that writes through BDA; settled right after the draw
	// (ExecutePreparedDraw). As for a dispatch, PrepareBda above uploaded every CPU-dirty page.
	for (auto* stage: stages) {
		if (stage->runtime->program->info.bda_writes) {
			EXIT_IF(!uses_dma);
			m_context.GetBufferCache().PrepareBdaWrites();
			break;
		}
	}
	CommitStats::StepMark(CommitStats::Step::Bda);
	CommitStats::Mark(CommitStats::Phase::FindBuffers);
	// KYTY_DRAW_RUN continuation: the views and targets are the previous draw's (checked against the
	// image state after the buffer work, RenderExecutor::DrawRunImagesUnchanged).
	if (uses_bvh) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::RtStubDraws);
	}
	{
		const CommitStats::SubScope views(CommitStats::Sub::Views);
		for (auto* stage: stages) {
			if (keep_images) {
				break;
			}
			RebindImages(*stage);
		}
	}
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		if (keep_images) {
			break;
		}
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				if (old_image->binding.is_target) {
					NoteBindlessStateChange(*old_image);
				}
				old_image->binding = {};
			}
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	CommitStats::StepMark(CommitStats::Step::Images);
	CommitStats::Mark(CommitStats::Phase::RebindImages);
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	// The uploads of every stage share one barrier pair (KYTY_UPLOAD_BATCH).
	{
		const BufferCache::UploadBatch upload_batch(m_context.GetBufferCache());
		// KYTY_REBIND_AFTER_JOIN: the count the draw checks after its per-draw ranges.
		m_rebind_epoch = m_context.GetBufferCache().UnregisterEpoch();
		for (auto* stage: stages) {
			RebindBuffers(*stage);
		}
	}
	CommitStats::StepMark(CommitStats::Step::RbBatchEnd);
	CommitStats::Mark(CommitStats::Phase::RebindBuffers);
}

// KYTY_PUSH_SHADOW_FRESH_SKIP=1 (default off): a push-descriptor update whose stages include a shader
// data or flattened-SRT upload made for this command (PreparedBindings::fresh_upload) skips the
// comparison with the update in effect (CommandBuffer::PushDescriptors) and the copy kept for the
// next comparison: it cannot be equal, since no earlier descriptor of the recording refers to that
// upload. At the Sky Garden start every one of about 4,000 pushes per flip missed anyway
// (DescriptorPushMissBuffer/Layout, no DescriptorPushesAvoided), after copying about 1-2 KB of
// writes and infos for the next comparison.
// Live switch (common/liveSwitch.h): a per-push choice between two valid behaviours; a push made
// without a copy leaves the shadow invalid, so the next comparison after switching off starts over.
static int64_t ParseOnlyOne(const char* value) {
	return value != nullptr && std::strcmp(value, "1") == 0 ? 1 : 0;
}
static Live::Switch g_push_shadow_fresh_skip("KYTY_PUSH_SHADOW_FRESH_SKIP", ParseOnlyOne);

static bool PushShadowFreshSkipEnabled() {
	return g_push_shadow_fresh_skip.On();
}

// Why a renderer push-descriptor update was recorded (FrameEvent.DescriptorPushMiss*).
static void CountDescriptorPushMiss(int32_t result, std::span<const vk::WriteDescriptorSet> writes) {
	using Event = Profiler::FrameEvent;
	if (result == CommandBuffer::PushAvoided) {
		return;
	}
	if (result == CommandBuffer::PushMissState) {
		Profiler::CountFrameEvent(Event::DescriptorPushMissLayout);
		return;
	}
	if (result == CommandBuffer::PushMissFresh) {
		Profiler::CountFrameEvent(Event::DescriptorPushMissUpload);
		return;
	}
	if (result == CommandBuffer::PushMissShape || result < 0 ||
	    static_cast<size_t>(result) >= writes.size()) {
		Profiler::CountFrameEvent(Event::DescriptorPushMissShape);
		return;
	}
	// NativeBinding(stage, kind) = kind + stage group * DescriptorBindingKind::Count.
	const auto kind = static_cast<BindingKind>(writes[static_cast<size_t>(result)].dstBinding %
	                                           static_cast<uint32_t>(BindingKind::Count));
	if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) !=
	    ShaderRecompiler::IR::ImageResourceClass::None) {
		Profiler::CountFrameEvent(Event::DescriptorPushMissImage);
		return;
	}
	switch (kind) {
		case BindingKind::Samplers: Profiler::CountFrameEvent(Event::DescriptorPushMissSampler); break;
		case BindingKind::Buffers: Profiler::CountFrameEvent(Event::DescriptorPushMissBuffer); break;
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData: Profiler::CountFrameEvent(Event::DescriptorPushMissUpload); break;
		default: Profiler::CountFrameEvent(Event::DescriptorPushMissOther); break;
	}
}

// KYTY_DESCRIPTOR_OFFSET_AUDIT=1 (P4b-D0, a measurement; P4B-WORKER-LOOKUPS.md 2.8): how many
// descriptor sets written and push-descriptor updates would repeat the previous one, or any
// earlier one, of the same command buffer if the per-draw buffer descriptors were dynamic (their
// offsets passed at bind time). Those are the flattened SRT and shader-data tables of every stage
// (new stream-ring ranges each draw), then the V# storage buffers in write order, as many as
// maxDescriptorSetStorageBuffersDynamic allows: their buffer offsets are left out of the digest,
// their buffers and ranges are not. Push layouts cannot hold dynamic descriptors, so their counts
// are the potential of splitting them (a pushed set and a dynamic one). Counted as
// DescriptorOffsetAudit* frame events (with a connected profiler).
static bool DescriptorOffsetAuditEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_DESCRIPTOR_OFFSET_AUDIT");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}));
	return enabled;
}

namespace {

class DescriptorOffsetAudit {
public:
	struct Result {
		bool previous   = false; // the previous one of its kind in this command buffer
		bool any        = false; // any earlier one of its kind in this command buffer
		bool over_limit = false; // more per-draw buffer descriptors than the dynamic limit
	};

	Result Note(uint64_t tick, vk::DescriptorSetLayout layout,
	            std::span<const vk::WriteDescriptorSet> writes, uint32_t dynamic_limit,
	            bool push) {
		if (tick != m_tick) {
			m_tick = tick;
			m_sets = {};
			m_pushes = {};
		}
		Result result;
		const auto digest = MaskedDigest(layout, writes, dynamic_limit, result.over_limit);
		auto&      track  = push ? m_pushes : m_sets;
		result.previous   = track.has_previous && track.previous == digest;
		result.any        = !track.seen.insert(digest).second;
		track.previous     = digest;
		track.has_previous = true;
		return result;
	}

private:
	struct Track {
		uint64_t                     previous     = 0;
		bool                         has_previous = false;
		std::unordered_set<uint64_t> seen;
	};

	template <typename T>
	static uint64_t HandleBits(T handle) {
		return static_cast<uint64_t>(
		    reinterpret_cast<uintptr_t>(static_cast<typename T::CType>(handle)));
	}

	static uint64_t MaskedDigest(vk::DescriptorSetLayout                  layout,
	                             std::span<const vk::WriteDescriptorSet> writes,
	                             uint32_t dynamic_limit, bool& over_limit) {
		// The leading elements of each write that would be dynamic: the per-draw tables first,
		// then V# buffers, in write order, up to the limit.
		thread_local std::vector<uint32_t> masked;
		thread_local std::vector<uint64_t> words;
		masked.assign(writes.size(), 0);
		const auto kind_of = [](const vk::WriteDescriptorSet& write) {
			// NativeBinding(stage, kind) = kind + stage group * DescriptorBindingKind::Count.
			return static_cast<BindingKind>(write.dstBinding %
			                                static_cast<uint32_t>(BindingKind::Count));
		};
		uint32_t budget = dynamic_limit;
		for (const bool tables: {true, false}) {
			for (size_t i = 0; i < writes.size(); i++) {
				const auto& write = writes[i];
				const auto  kind  = kind_of(write);
				const bool  table = kind == BindingKind::FlattenedSrt || kind == BindingKind::ShaderData;
				if (write.pBufferInfo == nullptr || (tables ? !table : kind != BindingKind::Buffers)) {
					continue;
				}
				masked[i] = std::min(write.descriptorCount, budget);
				budget -= masked[i];
				over_limit |= masked[i] < write.descriptorCount;
			}
		}
		words.clear();
		words.push_back(HandleBits(layout));
		for (size_t w = 0; w < writes.size(); w++) {
			const auto& write = writes[w];
			words.push_back((static_cast<uint64_t>(write.dstBinding) << 32u) | write.descriptorCount);
			words.push_back((static_cast<uint64_t>(write.descriptorType) << 32u) |
			                write.dstArrayElement);
			for (uint32_t i = 0; i < write.descriptorCount; i++) {
				if (write.pBufferInfo != nullptr) {
					const auto& info = write.pBufferInfo[i];
					words.push_back(HandleBits(info.buffer));
					words.push_back(i < masked[w] ? UINT64_MAX : info.offset);
					words.push_back(info.range);
				} else if (write.pImageInfo != nullptr) {
					const auto& info = write.pImageInfo[i];
					words.push_back(HandleBits(info.sampler));
					words.push_back(HandleBits(info.imageView));
					words.push_back(static_cast<uint64_t>(info.imageLayout));
				}
			}
		}
		return XXH3_64bits(words.data(), words.size() * sizeof(uint64_t));
	}

	uint64_t m_tick = UINT64_MAX;
	Track    m_sets;
	Track    m_pushes;
};

} // namespace

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings,
                                    bool                               keep_images) {
	KYTY_PROFILER_FUNCTION();
	// Storage writes were reserved during preparation. A later binding/table upload may have
	// submitted that recording before this command was emitted. Its readbacks must wait for
	// the final recording, not the already submitted preparation tick. No dirty range expands.
	if (BufferCache::ShaderWriteRetickEnabled()) {
		const auto tick = m_context.GetCommandScheduler().CurrentTick();
		for (auto* prepared: prepared_bindings) {
			EXIT_IF(prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime);
			if (prepared->write_preparation_tick == tick) continue;
			const auto& resources = prepared->runtime->program->info.buffers;
			EXIT_IF(prepared->buffer_sources.size() != resources.size());
			for (size_t i = 0; i < resources.size(); ++i) {
				if (!resources[i].written) continue;
				const auto& source = prepared->buffer_sources[i];
				m_context.GetBufferCache().RetagShaderWrite(source.address, source.size,
				                                          prepared->write_preparation_tick);
			}
			prepared->write_preparation_tick = tick;
		}
	}
	// Run after resource discovery so an unbounded address writer cannot retain a
	// metadata-inspection memo created during preparation of this same command.
	// Read-only BDA access leaves contents unchanged; tracked descriptor writes carry
	// their own canonical-buffer revisions.
	if (std::ranges::any_of(prepared_bindings, [](const auto* prepared) {
		    return prepared->runtime->program->has_address_writes;
	    })) {
		m_context.GetBufferCache().InvalidateContentRevisions();
	}
	// State commands only: image transitions and the GDS dependency below go through the barrier
	// batcher and are recorded at the caller's next flush point (the draw's BeginRendering, the
	// dispatch's Handle()), before any command that accesses the resources. Image::Transit takes
	// this command buffer only as the batch target (identity; KYTY_CP_RECORDER records nothing
	// through it).
	const auto vk_buffer    = buffer.Identity();
	size_t descriptor_count = 0;
	size_t write_count      = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	constexpr auto                 GraphicsStages =
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eMeshEXT |
	    vk::ShaderStageFlagBits::eTessellationControl |
	    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment;
	vk::ShaderStageFlags push_stages = pipeline_bind_point == vk::PipelineBindPoint::eGraphics
	                                       ? vk::ShaderStageFlagBits::eFragment
	                                       : vk::ShaderStageFlags {};
	bool mesh_stage = false;
	for (const auto* prepared: prepared_bindings) {
		EXIT_IF(prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime);
		const auto& program = *prepared->runtime->program;
		mesh_stage |= program.stage == ShaderType::Mesh;
		write_count += program.bindings.descriptors.size();
		for (const auto& binding: program.bindings.descriptors) {
			descriptor_count += NativeDescriptorCount(binding);
		}
		const auto shader_stage = NativeShaderStage(program.stage);
		push_stages |= shader_stage;
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	m_descriptor_buffers.clear();
	m_descriptor_images.clear();
	m_descriptor_writes.clear();
	m_descriptor_buffers.reserve(descriptor_count);
	m_descriptor_images.reserve(descriptor_count);
	m_descriptor_writes.reserve(write_count);
	// KYTY_DRAW_RUN_PUSH: a continuation pushes only its per-draw descriptors; its image and sampler
	// descriptors are the previous draw's, still in effect (RenderExecutor::DrawRunPartialPush).
	const bool partial = keep_images && pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
	                     DrawRunPartialPush(buffer, pipeline);
	if (partial) {
		OwnedCounters::Add(DrawRun::GetTotals().partial_pushes, 1);
	}
	CommitStats::StepMark(CommitStats::Step::CbHead);

	for (auto* prepared: prepared_bindings) {
		const auto& program       = *prepared->runtime->program;
		auto&       descriptors   = *prepared;
		const auto  shader_stage  = NativeShaderStage(program.stage);
		const auto  shader_stages = ShaderPipelineStages(shader_stage);
		if (descriptors.gds.buffer != nullptr) {
			KYTY_GPU_OP_SITE("descriptors.gds_barrier");
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			constexpr auto source_stages =
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			    vk::PipelineStageFlagBits::eAllGraphics | vk::PipelineStageFlagBits::eComputeShader;
			if (BarrierBatchEnabled()) {
				// Sync1 masks share their bit values with the synchronization2 ones.
				vk::BufferMemoryBarrier2 barrier2 {};
				barrier2.srcStageMask = vk::PipelineStageFlags2(
				    static_cast<VkPipelineStageFlags2>(VkPipelineStageFlags(source_stages)));
				barrier2.srcAccessMask = vk::AccessFlags2(
				    static_cast<VkAccessFlags2>(VkAccessFlags(barrier.srcAccessMask)));
				barrier2.dstStageMask = vk::PipelineStageFlags2(
				    static_cast<VkPipelineStageFlags2>(VkPipelineStageFlags(shader_stages)));
				barrier2.dstAccessMask = vk::AccessFlags2(
				    static_cast<VkAccessFlags2>(VkAccessFlags(barrier.dstAccessMask)));
				barrier2.srcQueueFamilyIndex = barrier.srcQueueFamilyIndex;
				barrier2.dstQueueFamilyIndex = barrier.dstQueueFamilyIndex;
				barrier2.buffer              = barrier.buffer;
				barrier2.offset              = barrier.offset;
				barrier2.size                = barrier.size;
				buffer.RequestBufferBarrier(barrier2, BarrierOrigin::Gds);
			} else {
				buffer.EndRendering();
				buffer.StateHandle().pipelineBarrier(source_stages, shader_stages,
				                                     vk::DependencyFlags {}, 0, nullptr, 1, &barrier,
				                                     0, nullptr);
			}
		}

		// KYTY_GUEST_STORAGE_REPEAT: these are the guest draw's or dispatch's own transitions.
		const Image::GuestTransitScope guest_transitions;
		CommitStats::SubScope          transitions_scope(CommitStats::Sub::Transitions);
		// Bindless textures (Senaxx, E5): sampled through set 1 in a read-only layout.
		if (!descriptors.bindless_heaps.empty()) [[unlikely]] {
			MakeBindlessImagesReadable(descriptors.bindless_heaps, vk_buffer);
		}
		// KYTY_DRAW_RUN continuation: every image is in the state the previous draw's transitions
		// left it in (RenderExecutor::DrawRunImagesUnchanged), which the bindings' layouts record.
		for (uint32_t i = 0; !keep_images && i < program.info.images.size(); i++) {
			auto& image   = m_context.GetTextureCache().GetImage(descriptors.images[i].image_id);
			auto& binding = descriptors.images[i];
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.info.data.Empty()) {
				image.Transit(vk::ImageLayout::eGeneral,
				              storage ? vk::AccessFlagBits2::eShaderRead |
				                            vk::AccessFlagBits2::eShaderWrite
				                      : vk::AccessFlagBits2::eShaderRead,
				              range, vk_buffer, true);
			} else if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				Profiler::CountFrameEvent(image.info.IsDepth()
				                              ? Profiler::FrameEvent::SampledDepthAttachmentBindings
				                              : Profiler::FrameEvent::SampledColorAttachmentBindings);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer, true);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer, true);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer, true);
			} else {
				image.Transit(image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                   : vk::ImageLayout::eShaderReadOnlyOptimal,
				              vk::AccessFlagBits2::eShaderRead, range, vk_buffer, true);
			}
			binding.layout = image.backing.state.layout;
		}

		transitions_scope.Stop();
		CommitStats::StepMark(CommitStats::Step::CbTransitions);
		const CommitStats::SubScope build_scope(CommitStats::Sub::DescBuild);
		m_image_occurrences.assign(descriptors.images.size(), 0);
		for (const auto& binding: program.bindings.descriptors) {
			if (partial && (binding.kind == BindingKind::Samplers ||
			                ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) !=
			                    ShaderRecompiler::IR::ImageResourceClass::None)) {
				continue;
			}
			vk::WriteDescriptorSet write {};
			write.dstBinding     = ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind);
			write.descriptorType = NativeDescriptorType(binding.kind);
			write.descriptorCount   = NativeDescriptorCount(binding);
			const auto buffer_start = m_descriptor_buffers.size();
			const auto image_start  = m_descriptor_images.size();
			if (ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				for (const auto resource: binding.resources) {
					m_descriptor_images.push_back(MakeImageInfo(
					    descriptors.images.at(resource), m_image_occurrences.at(resource)++));
				}
			} else {
				switch (binding.kind) {
					case BindingKind::Buffers:
						for (const auto resource: binding.resources) {
							const auto& view = descriptors.buffers.at(resource);
							EXIT_IF(view.buffer == nullptr);
							m_descriptor_buffers.push_back(view);
						}
						break;
					case BindingKind::UniformBuffers: {
						// KYTY_UNIFORM_BUFFERS: the storage view, its range cut to the uniform
						// array the shader declares (constant reads stay inside it).
						const auto max_range = std::min<vk::DeviceSize>(
						    ShaderRecompiler::IR::UniformBufferBytes,
						    m_context.GetGraphics()
						        .GetPhysicalDeviceProperties()
						        .limits.maxUniformBufferRange);
						for (const auto resource: binding.resources) {
							auto view = descriptors.buffers.at(resource);
							EXIT_IF(view.buffer == nullptr);
							view.range = std::min(view.range, max_range);
							m_descriptor_buffers.push_back(view);
						}
						break;
					}
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto&       cache      = m_context.GetBufferCache();
						const auto* bda_buffer = binding.kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						m_descriptor_buffers.emplace_back(bda_buffer->Handle(), 0,
						                                  bda_buffer->Size());
						break;
					}
					case BindingKind::MipStats: {
						auto& lod_stats = m_context.GetLodStats();
						auto& counters  = descriptors.mip_stats_canary ? lod_stats.CanaryBuffer()
						                                               : lod_stats.CounterBuffer();
						m_descriptor_buffers.emplace_back(counters.Handle(), 0, counters.Size());
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (binding.kind == BindingKind::FlattenedSrt ||
						    binding.kind == BindingKind::ShaderData) {
							view = binding.kind == BindingKind::FlattenedSrt
							           ? &descriptors.flattened_srt
							           : &descriptors.shader_data_buffer;
							if (ShaderRecompiler::Spirv::SrtUniformEnabled()) {
								// KYTY_SRT_UNIFORM: at most the uniform array the shader declares.
								EXIT_IF(view->buffer == nullptr);
								auto uniform  = *view;
								uniform.range = std::min<vk::DeviceSize>(
								    uniform.range,
								    std::min<vk::DeviceSize>(ShaderRecompiler::IR::UniformBufferBytes,
								                             m_context.GetGraphics()
								                                 .GetPhysicalDeviceProperties()
								                                 .limits.maxUniformBufferRange));
								if (uniform.range < view->range) {
									static std::atomic<uint32_t> reported {0};
									if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
										LOGF("KYTY_SRT_UNIFORM: table (kind %u) of %" PRIu64
										     " bytes cut to the uniform range\n",
										     static_cast<uint32_t>(binding.kind),
										     static_cast<uint64_t>(view->range));
									}
								}
								m_descriptor_buffers.push_back(uniform);
								break;
							}
						} else if (binding.kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						m_descriptor_buffers.push_back(*view);
						break;
					}
					case BindingKind::Samplers:
						for (const auto resource: binding.resources) {
							const auto sampler = descriptors.samplers.at(resource);
							EXIT_IF(sampler == nullptr);
							m_descriptor_images.emplace_back(sampler, nullptr,
							                                 vk::ImageLayout::eUndefined);
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (m_descriptor_buffers.size() != buffer_start) {
				write.pBufferInfo = m_descriptor_buffers.data() + buffer_start;
			}
			if (m_descriptor_images.size() != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; !partial && i < descriptors.images.size(); i++) {
			const auto expected =
			    descriptors.images[i].mip_views.empty()
			        ? 1u
			        : static_cast<uint32_t>(descriptors.images[i].mip_views.size());
			EXIT_IF(m_image_occurrences[i] != expected);
		}

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF(prepared->shader_data.size() != shader_data_dwords);
		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
		CommitStats::StepMark(CommitStats::Step::CbBuild);
	}

	if (has_push_data) {
		const CommitStats::SubScope push_scope(CommitStats::Sub::PushConst);
		buffer.PushConstants(pipeline.pipeline_layout, push_stages, sizeof(push_data),
		                     push_data.dwords.data());
	}
	if (pipeline_bind_point == vk::PipelineBindPoint::eGraphics) {
		CommitStats::AddItems(CommitStats::Item::DescWrites, m_descriptor_writes.size());
	}
	CommitStats::StepMark(CommitStats::Step::CbPushConst);
	if (mesh_stage) {
		// The mesh draw path then records its draw arguments over the first push-constant dwords
		// through StateHandle(); the shadow no longer describes what is in effect.
		buffer.InvalidatePushConstants();
	}

	if (!m_descriptor_writes.empty()) {
		EXIT_IF(pipeline.descriptor_set_layout == nullptr);
		if (DescriptorOffsetAuditEnabled()) [[unlikely]] {
			// Draws and dispatches are recorded under the render mutex, one at a time.
			static DescriptorOffsetAudit audit;
			using E           = Profiler::FrameEvent;
			const bool push   = pipeline.uses_push_descriptors;
			const auto result = audit.Note(
			    m_context.GetCommandScheduler().CurrentTick(), pipeline.descriptor_set_layout,
			    m_descriptor_writes,
			    m_context.GetGraphics()
			        .GetPhysicalDeviceProperties()
			        .limits.maxDescriptorSetStorageBuffersDynamic,
			    push);
			Profiler::CountFrameEvent(push ? E::DescriptorOffsetAuditPushes
			                               : E::DescriptorOffsetAuditSets);
			if (result.previous) {
				Profiler::CountFrameEvent(push ? E::DescriptorOffsetAuditPushRepeatsPrevious
				                               : E::DescriptorOffsetAuditSetRepeatsPrevious);
			}
			if (result.any) {
				Profiler::CountFrameEvent(push ? E::DescriptorOffsetAuditPushRepeatsAny
				                               : E::DescriptorOffsetAuditSetRepeatsAny);
			}
			if (result.over_limit) {
				Profiler::CountFrameEvent(E::DescriptorOffsetAuditOverLimit);
			}
		}
		if (pipeline.uses_push_descriptors) {
			const CommitStats::SubScope desc_scope(CommitStats::Sub::DescPush);
			// KYTY_PUSH_SHADOW_FRESH_SKIP: a stage's shader data or flattened SRT was uploaded for
			// this command, so its descriptor differs from every earlier push of this recording.
			const bool known_miss =
			    PushShadowFreshSkipEnabled() &&
			    std::ranges::any_of(prepared_bindings,
			                        [](const auto* prepared) { return prepared->fresh_upload; });
			const auto result = buffer.PushDescriptors(
			    pipeline_bind_point, pipeline.pipeline_layout, 0,
			    static_cast<uint32_t>(m_descriptor_writes.size()), m_descriptor_writes.data(),
			    known_miss);
			CountDescriptorPushMiss(result, m_descriptor_writes);
		} else {
			const CommitStats::SubScope desc_scope(CommitStats::Sub::DescSet);
			const bool fresh = std::ranges::any_of(
			    prepared_bindings, [](const auto* prepared) { return prepared->fresh_upload; });
			CommitDescriptorSet(buffer, pipeline_bind_point, pipeline, fresh);
		}
	}
	if (pipeline.uses_bindless) [[unlikely]] {
		// Set 1: the bindless table (Senaxx). Binding it leaves set 0 as it is (the same layout).
		auto& table = m_context.GetBindlessTable();
		if (!table.EnsureCreated()) {
			EXIT("bindless: the descriptor table for set 1 could not be created\n");
		}
		const auto set = table.Set();
		buffer.StateSink().bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 1, 1, &set,
		                                      0, nullptr);
	}
	CommitStats::StepMark(CommitStats::Step::CbDescriptors);
}

// KYTY_SET_REUSE_FRESH (default on; 0 off; verify): a descriptor set whose writes refer to a
// shader-data or flattened-SRT range UploadShaderData allocated for this command skips the reuse
// lookup and is not remembered. The reuse cache holds sets written during the current tick only,
// and the stream ring never hands out a range twice during one tick (a wrap first submits and
// completes the tick), so no remembered set refers to that range: the lookup cannot hit. Only a
// later command whose upload the dedup resolves to this same range could have reused the set;
// such commands carry byte-identical tables, which the desert stamps (a new SRT and user data per
// draw, 933 sets written per flip) and U52's captures (no reuse hit in any scene) never showed.
// verify: the lookup still runs and a hit is reported (it would contradict the argument above).
static int SetReuseFreshMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_SET_REUSE_FRESH");
		if (value == nullptr || value[0] == '\0') {
			return 1;
		}
		if (std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "verify") == 0 || std::strcmp(value, "exit") == 0
		           ? (std::strcmp(value, "exit") == 0 ? 3 : 2)
		           : 1;
	}();
	return mode;
}

// KYTY_RECORDER_DESCRIPTOR_SETS=1 (default off; BryanKAdams/KytyPS5 8d2e74c): with the CP recorder
// (KYTY_CP_RECORDER), a descriptor set written for a layout beyond maxPushDescriptors is written by
// the recorder thread (an UpdateDescriptorSets packet, commandStream.h) in stream order before the
// packet that binds it, instead of by vkUpdateDescriptorSets on the CP (about 1,000 sets per flip
// at the Sky Garden start). The CP still allocates the set, so the reuse cache and the bind shadow
// see the same handles; the set is new (or a reuse of one written earlier in the same recording,
// whose packet precedes), and its pool is reset only after the tick completes. Without the
// recorder, and in a direct window, the CP writes it at once as before.
// Live switch (common/liveSwitch.h): a per-commit choice. Either way the set is written before the
// bind that follows it (in the stream, or at once), so sets of both kinds may share a recording.
static Live::Switch g_recorder_descriptor_sets("KYTY_RECORDER_DESCRIPTOR_SETS", ParseOnlyOne);

static bool RecorderDescriptorSetsEnabled() {
	return g_recorder_descriptor_sets.On();
}

void RenderExecutor::CommitDescriptorSet(CommandBuffer& buffer, vk::PipelineBindPoint point,
                                         const PipelineCache::Pipeline& pipeline, bool fresh) {
	const auto        layout     = pipeline.descriptor_set_layout;
	const bool        reuse      = DescriptorSetReuseEnabled();
	const auto        fresh_mode = fresh && reuse ? SetReuseFreshMode() : 0;
	const bool        lookup     = reuse && fresh_mode != 1;
	const auto        tick       = m_context.GetCommandScheduler().CurrentTick();
	uint64_t          hash       = 0;
	vk::DescriptorSet set        = nullptr;
	if (fresh_mode == 1) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorSetReuseSkippedFresh);
	}
	if (lookup) {
		hash = DescriptorSetReuse::Hash(layout, m_descriptor_writes);
		set  = m_descriptor_set_reuse.Find(tick, layout, m_descriptor_writes, hash);
		if (set != nullptr && fresh_mode >= 2) {
			m_set_reuse_fresh_mismatches++;
			std::fprintf(stderr, "SetReuseFreshVerify: a set referring to a fresh upload was "
			                     "found in the reuse cache\n");
			if (fresh_mode == 3) {
				EXIT("SetReuseFreshVerify: a set referring to a fresh upload was reused\n");
			}
		}
	}
	if (DescriptorSetReuseAuditEnabled()) {
		const auto audit = m_descriptor_set_reuse.Audit(
		    tick, DescriptorSetReuse::Digest(layout, m_descriptor_writes));
		Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorSetAuditCommits);
		if (audit.repeat) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorSetAuditRepeats);
		}
		if (audit.digest_slot_hit) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorSetAuditDigestSlotHits);
		}
	}
	if (set != nullptr) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorSetsReused);
	} else {
		set = m_context.GetDescriptorHeap().Commit(layout);
		for (auto& write: m_descriptor_writes) {
			write.dstSet = set;
		}
		const auto count = static_cast<uint32_t>(m_descriptor_writes.size());
		if (RecorderDescriptorSetsEnabled()) {
			// Encoded before the bind below: the recorder writes the set, then binds it.
			buffer.StateSink().updateDescriptorSets(m_context.GetGraphics().device, set, count,
			                                        m_descriptor_writes.data());
		} else {
			m_context.GetGraphics().device.updateDescriptorSets(count, m_descriptor_writes.data(),
			                                                    0, nullptr);
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorSetsWritten);
		if (lookup) {
			m_descriptor_set_reuse.Insert(tick, layout, m_descriptor_writes, hash, set);
		}
	}
	buffer.BindDescriptorSet(point, pipeline.pipeline_layout, set);
}

} // namespace Libs::Graphics
