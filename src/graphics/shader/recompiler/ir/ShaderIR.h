#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"
#include "graphics/shader/shader.h"

#include <array>
#include <atomic>
#include <bit>
#include <deque>
#include <list>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Emulated 64-bit FLAT apertures. Each contains 4 GiB of byte offsets and is outside guest global
// VA space. Keep queries and FLAT routing consistent.
inline constexpr uint32_t SharedApertureHigh  = 0x70000000u;
inline constexpr uint32_t PrivateApertureHigh = 0x80000000u;

enum class ResourceKind {
	None,
	ScalarBuffer,
	ScalarAddress,
	Buffer,
	IndirectBuffer,
	Flat,
	Global,
	Scratch,
	Lds,
	Gds,
	Image,
	Sampler
};

[[nodiscard]] constexpr bool IsAddressResourceKind(ResourceKind kind) {
	return kind == ResourceKind::ScalarAddress || kind == ResourceKind::Flat ||
	       kind == ResourceKind::Global || kind == ResourceKind::Scratch;
}

// MemoryInfo::resource of an IndirectBuffer access that KYTY_SRT_VARIANT_READS created: it has no
// bound buffer and no descriptor source.
inline constexpr uint32_t NoIndirectBufferResource = UINT32_MAX;

struct MemoryInfo {
	ResourceKind            kind                     = ResourceKind::None;
	uint32_t                resource                 = 0;
	uint32_t                sampler                  = 0;
	uint32_t                offset                   = 0;
	uint32_t                secondary_offset         = 0;
	uint32_t                dmask                    = 0;
	uint32_t                data_dwords              = 1;
	uint32_t                data_bits                = 32;
	uint32_t                component_index          = 0;
	uint32_t                component_count          = 1;
	uint32_t                data_format              = 0;
	uint32_t                number_format            = 0;
	uint32_t                image_sample_flags       = 0;
	Decoder::ImageDimension image_dimension          = Decoder::ImageDimension::Unknown;
	uint32_t                image_address_components = 0;
	bool                    address_is_full                                       = false;
	bool                    data_signed                                           = false;
	bool                    typed                                                 = false;
	bool                    formatted                                             = false;
	bool                    image_has_mip                                         = false;
	bool                    image_r128                                            = false;
	bool                    image_cube                                            = false; // MIMG DIM is CUBE.
	bool                    idxen                                                 = false;
	bool                    offen                                                 = false;
	bool                    coherent                                              = false;
	bool                    planning_only                                         = false;

	[[nodiscard]] bool IsCoherentBufferPairLoad(ValueOpcode opcode) const {
		return kind == ResourceKind::Buffer && opcode == ValueOpcode::LoadBufferU32x2 && coherent &&
		       !typed && !formatted && data_bits == 32u;
	}

	// KYTY_BUFFER_PAIR_ATOMIC: a raw GLC/DLC BUFFER_LOAD_DWORDX2 or BUFFER_STORE_DWORDX2 through a
	// bound V# or one the shader computed, which may run as one 64-bit access.
	[[nodiscard]] bool IsCoherentBufferPair(ValueOpcode opcode) const {
		return (kind == ResourceKind::Buffer || kind == ResourceKind::IndirectBuffer) &&
		       (opcode == ValueOpcode::LoadBufferU32x2 || opcode == ValueOpcode::StoreBufferU32x2) &&
		       coherent && !typed && !formatted && data_bits == 32u;
	}

	[[nodiscard]] bool SupportsIndirectBufferLoad(ValueOpcode opcode) const {
		// ReadConstBuffer: one dword of an S_BUFFER_LOAD (no formats, RDNA2 ISA 7.2.1). A formatted
		// X load decodes the V#'s format in the shader (upstream a55a28380, LoadIndirectFormattedX):
		// Miles' hit callbacks read vertex data with BUFFER_LOAD_FORMAT_X through a material's V#.
		return !typed && data_bits == 32u &&
		       (formatted ? opcode == ValueOpcode::LoadBufferU32 ||
		                        opcode == ValueOpcode::LoadBufferU32x2 ||
		                        opcode == ValueOpcode::LoadBufferU32x3 ||
		                        opcode == ValueOpcode::LoadBufferU32x4
		                  : opcode == ValueOpcode::LoadBufferU32x2 ||
		                        opcode == ValueOpcode::LoadBufferU32x3 ||
		                        opcode == ValueOpcode::LoadBufferU32x4 ||
		                        opcode == ValueOpcode::ReadConstBuffer);
	}
	// KYTY_BDA_WRITES: raw (unformatted) stores and every buffer atomic can write through a V# the
	// shader computes.
	[[nodiscard]] bool SupportsIndirectRawWrite(ValueOpcode opcode) const {
		if (formatted || typed) {
			return false;
		}
		switch (opcode) {
			case ValueOpcode::StoreBufferU8: return data_bits == 8u;
			case ValueOpcode::StoreBufferU16: return data_bits == 16u;
			case ValueOpcode::StoreBufferU32:
			case ValueOpcode::StoreBufferU32x2:
			case ValueOpcode::StoreBufferU32x3:
			case ValueOpcode::StoreBufferU32x4: return data_bits == 32u;
			case ValueOpcode::BufferAtomicSwap32:
			case ValueOpcode::BufferAtomicCmpSwap32:
			case ValueOpcode::BufferAtomicIAdd32:
			case ValueOpcode::BufferAtomicISub32:
			case ValueOpcode::BufferAtomicSMin32:
			case ValueOpcode::BufferAtomicUMin32:
			case ValueOpcode::BufferAtomicSMax32:
			case ValueOpcode::BufferAtomicUMax32:
			case ValueOpcode::BufferAtomicAnd32:
			case ValueOpcode::BufferAtomicOr32:
			case ValueOpcode::BufferAtomicXor32:
			case ValueOpcode::BufferAtomicInc32:
			case ValueOpcode::BufferAtomicDec32:
			case ValueOpcode::BufferAtomicFMin32:
			case ValueOpcode::BufferAtomicFMax32:
			case ValueOpcode::BufferAtomicSwap64:
			case ValueOpcode::BufferAtomicCmpSwap64:
			case ValueOpcode::BufferAtomicIAdd64:
			case ValueOpcode::BufferAtomicISub64:
			case ValueOpcode::BufferAtomicSMin64:
			case ValueOpcode::BufferAtomicUMin64:
			case ValueOpcode::BufferAtomicSMax64:
			case ValueOpcode::BufferAtomicUMax64:
			case ValueOpcode::BufferAtomicAnd64:
			case ValueOpcode::BufferAtomicOr64:
			case ValueOpcode::BufferAtomicXor64: return true;
			default: return false;
		}
	}
	// KYTY_SRT_VARIANT_READS: every raw (unformatted) vector load can read through a V# the shader
	// computes, BUFFER_LOAD_UBYTE/USHORT/DWORD included, not only DWORDX2-X4.
	[[nodiscard]] bool SupportsIndirectRawLoad(ValueOpcode opcode) const {
		if (formatted || typed) {
			return false;
		}
		switch (opcode) {
			case ValueOpcode::LoadBufferU8: return data_bits == 8u;
			case ValueOpcode::LoadBufferU16: return data_bits == 16u;
			case ValueOpcode::LoadBufferU32:
			case ValueOpcode::LoadBufferU32x2:
			case ValueOpcode::LoadBufferU32x3:
			case ValueOpcode::LoadBufferU32x4: return data_bits == 32u;
			default: return false;
		}
	}

	bool operator==(const MemoryInfo& other) const = default;
};

enum class ExportTargetKind { Unknown, Null, Position, Primitive, Parameter, Mrt, MrtZ };

struct ExportInfo {
	ExportTargetKind kind   = ExportTargetKind::Unknown;
	uint32_t         target = 0;
	uint32_t         index  = 0;
	uint32_t         en     = 0;
	bool             done   = false;
	bool             compr  = false;
	bool             vm     = false;

	bool operator==(const ExportInfo& other) const = default;
};

// KYTY_BUFFER_OOB_MODES: BufferResource::packed_stride flag (above ShaderBufferResource::PackedStride's
// bits): the V# has OOB_SELECT 0 and a stride (no swizzle), so an offset at or past the stride is out
// of range (RDNA2 ISA Table 35).
inline constexpr uint32_t BufferOobOffsetCheck = 1u << 23u;

struct BufferResource {
	static constexpr uint32_t NoImageAlias = UINT32_MAX;
	static constexpr uint32_t NoIndirectBuffer = UINT32_MAX;

	uint32_t               source             = 0;
	uint32_t               first_use_pc       = 0;
	uint32_t               max_byte_extent    = 0;
	uint32_t               packed_stride      = 0;
	Prospero::BufferFormat descriptor_format  = Prospero::BufferFormat::kInvalid;
	uint32_t               descriptor_swizzle = DstSel(4, 5, 6, 7);
	uint32_t               image_alias        = NoImageAlias;
	bool                   read               = false;
	bool                   written            = false;
	bool                   atomic             = false;
	bool                   formatted          = false;
	bool                   scalar             = false;
	uint32_t               indirect_root              = NoIndirectBuffer;
	uint32_t               indirect_mapping_offset    = 0;
	uint32_t               indirect_search_iterations = 0;
	std::vector<uint32_t>  indirect_resources;

	bool operator==(const BufferResource& other) const = default;
};

enum class ImageMipMode { None, Dynamic };

// ImageResource::table_access bits (recorded with KYTY_SAMPLED_TABLE_READS only).
enum ImageTableAccess : uint32_t {
	ImageTableAccessRead        = 1u << 0u, // ImageRead (IMAGE_LOAD*)
	ImageTableAccessSample      = 1u << 1u, // ImageSampleRaw without LOD, clamp, gradients or offset
	ImageTableAccessSampleExtra = 1u << 2u, // ImageSampleRaw with an LOD or clamp after its coordinates
	ImageTableAccessWrite       = 1u << 3u, // ImageWrite
	ImageTableAccessOther       = 1u << 4u, // gathers, queries, atomics, samples with gradients/offsets
};

constexpr uint32_t ShaderImageIdentitySwizzle = 0x00000facu;

struct ImageResource {
	static constexpr uint32_t NoIndirectImage = UINT32_MAX;
	// max_array_index when no bound of an array image's layer coordinate was shown.
	static constexpr uint32_t NoArrayIndexBound = UINT32_MAX;

	uint32_t                      source            = 0;
	uint32_t                      first_use_pc      = 0;
	ImageResourceClass            resource_class    = ImageResourceClass::None;
	Prospero::TextureNumericClass numeric_class     = Prospero::TextureNumericClass::Unsupported;
	Decoder::ImageDimension       dimension         = Decoder::ImageDimension::Unknown;
	ImageMipMode                  mip_mode          = ImageMipMode::None;
	uint32_t                      mip_count         = 1;
	Prospero::BufferFormat        conversion_format = Prospero::BufferFormat::kInvalid;
	uint32_t                      shader_swizzle    = ShaderImageIdentitySwizzle;
	bool                          read              = false;
	bool                          written           = false;
	bool                          atomic            = false;
	bool                          atomic64          = false;
	bool                          depth_compare     = false;
	bool                          cube              = false;
	bool                          r128              = false;
	// Sampled through the bindless image arrays (descriptor set 1) at the slot the translation
	// table gives for the handle's key; the mapping offset locates the heap's region and size.
	bool                          bindless          = false;
	// KYTY_RUNTIME_IMAGES, with bindless: the handle holds the eight T# dwords the shader loaded at
	// run time (from a material record, say). The shader looks the T# up in the runtime image
	// table (set 1) and decodes its size from the dwords; one resource serves every such T#.
	bool                          runtime           = false;
	// KYTY_FLOAT_STORE_CLAMP: a storage image of a 16-bit float format whose float stores clamp
	// finite values beyond the half range to the largest half, as the guest GPU's conversion.
	bool                          clamp_half_float  = false;
	// KYTY_SAMPLED_TABLE_READS: the kinds of instruction that access the image (ImageTableAccess
	// bits, ResourceTracking). A GPU-selected table lets its candidates differ in dimension only
	// when every access takes its own candidate's coordinates. 0 without the switch.
	uint32_t                      table_access      = 0;
	uint32_t                      indirect_root     = NoIndirectImage;
	uint32_t                      indirect_mapping_offset   = 0;
	uint32_t                      indirect_search_iterations = 0;
	// A sampled array image's largest layer the shader can read, counted from the view's base
	// layer (ResourceTracking: every sample and gather of it takes its layer coordinate from
	// values of a known range). Its view needs no layer past it (KYTY_TEXTURE_LAYER_BOUND,
	// descriptors.cpp TextureViewInfo).
	uint32_t                      max_array_index = NoArrayIndexBound;
	std::vector<uint32_t>         indirect_resources;

	bool operator==(const ImageResource& other) const = default;
};

// KYTY_DEPTH_BITS_READ (upstream aafe014d0): a raw load of a kDepth k16UInt T# reads a D16
// surface's codes. Vulkan cannot view D16 as R16Uint, so the host binds it as UNorm and the shader
// converts back. Only ResourceMaterialization sets this conversion format (with the switch on).
[[nodiscard]] constexpr bool ReadsDepthBits(const ImageResource& image) {
	return image.conversion_format == Prospero::BufferFormat::k16UInt;
}

struct SamplerResource {
	uint32_t source                = 0;
	uint32_t first_use_pc          = 0;
	// Native filtering/border variants share the original sampler's runtime descriptor.
	uint32_t snapshot_index        = 0;
	bool     force_point_filtering = false;
	bool     depth_compare         = false;
	bool     integer_border        = false;
	bool     gather_lod            = false;
	// KYTY_BINDLESS_SAMPLERS: selected per lane by a key into a guest sampler heap
	// (DescriptorSource::bindless_sampler). The shader indexes the bindless sampler array (set 1)
	// with the region base and record count at this flattened-SRT offset, which the host patches.
	bool     bindless                = false;
	uint32_t bindless_mapping_offset = 0;

	bool operator==(const SamplerResource& other) const = default;
};

// The specialization constant a pre-rasterization stage clamps its exported render target array
// index with (KYTY_LAYER_CLAMP): the bound layer count (0: no clamp).
inline constexpr uint32_t LayerLimitSpecId = 0x4c00u;

struct SampledResourcePair {
	uint32_t image        = 0;
	uint32_t sampler      = 0;
	uint32_t first_use_pc = 0;

	bool operator==(const SampledResourcePair& other) const = default;
};

enum class TessellationAttribute {
	LocalOutput,
	ControlInput,
	ControlOutput,
	EvaluationInput,
	PatchOutput,
	PatchInput,
	Factor,
	// KYTY_TESS_MERGED_CARRY: the VGPRs a split merged LS+HS program carries from its LS part into
	// its HS part (ShaderTessellationInputInfo::carried_vgprs): the LS stage's per-vertex output and
	// the HS stage's per-control-point input, one u32 each, packed four per vec4 after the LS
	// outputs; address = 4 * the register's rank in the set.
	CarriedOutput,
	CarriedInput
};
constexpr uint32_t NumTessellationAttributes =
    static_cast<uint32_t>(TessellationAttribute::CarriedInput) + 1u;

enum class StageInputKind {
	VertexIndex,
	InvocationId,
	PrimitiveId,
	TessCoord,
	InstanceIndex,
	BaseInstance,
	BaseVertex,
	FragCoord,
	FrontFacing,
	PackedAncillary,
	Layer,
	SampleId,
	BaryCoordSmooth,
	BaryCoordSmoothCentroid,
	BaryCoordNoPerspective,
	WorkgroupId,
	NumWorkgroups,
	LocalInvocationId,
	LocalInvocationIndex,
	GlobalInvocationId,
	Parameter,
	// Further SPI_PS_INPUT I/J pairs (components 0/1 are I/J).
	BaryCoordSmoothSample,
	BaryCoordNoPerspectiveCentroid,
	BaryCoordNoPerspectiveSample,
};

// How a V_INTERP_P2 read is interpolated, from the I/J pair it uses (GetAttribute flags).
// Unknown: the I/J operand is not one of the hardware-provided pairs (or the switch is off); the
// input keeps the shader-wide interpolation.
enum class InterpolationMode : uint32_t {
	Unknown,
	PerspectiveCenter,
	PerspectiveCentroid,
	PerspectiveSample,
	LinearCenter,
	LinearCentroid,
	LinearSample,
};

enum class StageOutputKind {
	Position,
	Parameter,
	Mrt,
	Depth,
	SampleMask,
	PointSize,
	ClipDistance,
	CullDistance,
	Layer,
	ViewportIndex,
	// KYTY_OBJPRIM_ID: the 32-bit ObjPrimID the hardware writes to a colour slot the pixel shader
	// does not export (location = that slot).
	ObjPrimId
};

struct PositionExportComponent {
	uint32_t clip_distance = UINT32_MAX;
	uint32_t cull_distance = UINT32_MAX;
	bool     point_size     = false;
	bool     layer          = false;
	bool     viewport       = false;
};

inline PositionExportComponent DecodePositionExportComponent(uint32_t control,
	                                                           uint32_t pos_index,
	                                                           uint32_t component) {
	PositionExportComponent result;
	if (pos_index == 0 || component >= 4) {
		return result;
	}

	uint32_t slot   = pos_index - 1;
	uint32_t vector = 3;
	for (uint32_t i = 0; i < 3; i++) {
		if ((control & (1u << (21u + i))) != 0) {
			if (slot == 0) {
				vector = i;
				break;
			}
			slot--;
		}
	}
	if (vector == 3) {
		return result;
	}

	if (vector == 0) {
		result.point_size = component == 0 && (control & (1u << 16u)) != 0;
		result.layer      = component == 2 && (control & (1u << 18u)) != 0;
		result.viewport   = component == 2 && (control & (1u << 19u)) != 0;
		return result;
	}

	const auto scalar = (vector - 1) * 4 + component;
	const auto lower  = (1u << scalar) - 1u;
	const auto clip   = control & 0xffu;
	const auto cull   = (control >> 8u) & 0xffu;
	if ((clip & (1u << scalar)) != 0) {
		result.clip_distance = std::popcount(clip & lower);
	}
	if ((cull & (1u << scalar)) != 0) {
		result.cull_distance = std::popcount(cull & lower);
	}
	return result;
}

struct StageInput {
	StageInputKind kind            = StageInputKind::VertexIndex;
	uint32_t       location        = 0;
	uint32_t       component_count = 1;
	std::string    debug_name;
	bool           per_vertex = false;

	bool operator==(const StageInput& other) const = default;
};

struct StageOutput {
	StageOutputKind kind     = StageOutputKind::Parameter;
	uint32_t        index    = 0;
	uint32_t        location = 0;
	std::string     debug_name;

	bool operator==(const StageOutput& other) const = default;
};

inline constexpr uint32_t FirstImageBinding           = 1u;
inline constexpr uint32_t FirstComparisonImageBinding = 22u;
inline constexpr uint32_t FirstStorageImageBinding    = 29u;
inline constexpr uint32_t ImageBindingCount           = 48u;

enum class DescriptorBindingKind : uint32_t {
	Buffers  = 0u,
	Samplers = FirstImageBinding + ImageBindingCount,
	Gds,
	BdaPagetable,
	FaultBuffer,
	FlattenedSrt,
	ShaderData,
	MipStats,
	SharedMemory,
	// Buffers only ever read by ReadConstBuffer at immediate offsets (UniformBufferResources):
	// bound as uniform buffers instead of in Buffers, so the reads go through the constant cache.
	UniformBuffers,
	Count,
};

// The uniform buffer range a UniformBuffers element binds (maxUniformBufferRange is at least
// 64 KiB on the GPUs that run the emulator). Constant reads stay below it with the base
// adjustment (< UniformBufferBaseSlack bytes).
inline constexpr uint32_t UniformBufferBytes     = 65536u;
inline constexpr uint32_t UniformBufferBaseSlack = 256u;

static_assert(static_cast<uint32_t>(DescriptorBindingKind::Samplers) == 49u);
static_assert(static_cast<uint32_t>(DescriptorBindingKind::Count) == 58u);

struct PushData {
	static constexpr uint32_t DwordCount = 32;
	static constexpr uint32_t MeshDrawDwordCount = 6;
	// Upstream abc509d33: the task payload of an indirect mesh draw (draw parameters, then the
	// record's first instance). Referenced by the emitter only.
	static constexpr uint32_t MeshTaskPayloadDwordCount = MeshDrawDwordCount + 1;
	// Mesh draw dword 6, only for programs with ShaderMeshInputInfo::split_groups: the first
	// workgroup of this part of a draw split past the host's X group limit (0 when unsplit; a
	// native indirect draw's parameter block holds 0 there).
	static constexpr uint32_t MeshFirstGroupDword = MeshDrawDwordCount;
	[[nodiscard]] static constexpr uint32_t MeshDrawDwords(bool split_groups) {
		return MeshDrawDwordCount + (split_groups ? 1u : 0u);
	}
	// PA_CL_UCP user clip planes (upstream feed075a9): planes 0..N of UCP_ENA, four float dwords
	// each, after the mesh draw dwords (also for vertex shaders, whose draws push nothing else).
	static constexpr uint32_t UserClipPlaneDword = MeshDrawDwordCount + 1u;
	[[nodiscard]] static constexpr uint32_t GraphicsPushStart(uint32_t mesh_dwords,
	                                                          uint32_t user_clip_planes) {
		uint32_t planes_end = 0;
		for (uint32_t mask = user_clip_planes; mask != 0; mask >>= 1u) {
			planes_end += 4u;
		}
		return user_clip_planes != 0 && UserClipPlaneDword + planes_end > mesh_dwords
		           ? UserClipPlaneDword + planes_end
		           : mesh_dwords;
	}
	// Mesh draw dword 3 (the index size: 0, 1, 2 or 4 when pushed by the CPU) marking a native
	// indirect mesh draw: dwords 0-1 then hold the device address of the dispatch's parameter
	// block, whose first MeshDrawDwordCount dwords replace the pushed ones (CodegenOptions::
	// mesh_indirect_params, renderer/meshIndirect.h).
	static constexpr uint32_t MeshIndirectSentinel = 0xffffffffu;
	static constexpr uint32_t NoStart    = UINT32_MAX;
	std::array<uint32_t, DwordCount> dwords {};

	[[nodiscard]] static constexpr bool CanFit(uint32_t start, uint32_t size) {
		return size != 0 && start <= DwordCount && size <= DwordCount - start;
	}
	[[nodiscard]] static constexpr uint32_t StartFor(uint32_t cursor, uint32_t size) {
		return CanFit(cursor, size) ? cursor : NoStart;
	}
};

static_assert(sizeof(PushData) == 128);
constexpr uint32_t NativePushConstantSize = sizeof(PushData);

[[nodiscard]] constexpr uint32_t NativeBinding(ShaderType stage, DescriptorBindingKind kind) {
	const uint32_t group = stage == ShaderType::Pixel                    ? 1u
	                       : stage == ShaderType::TessellationControl    ? 2u
	                       : stage == ShaderType::TessellationEvaluation ? 3u
	                                                                     : 0u;
	return static_cast<uint32_t>(kind) +
	       group * static_cast<uint32_t>(DescriptorBindingKind::Count);
}

[[nodiscard]] constexpr ImageResourceClass ImageBindingResourceClass(DescriptorBindingKind kind) {
	const auto value = static_cast<uint32_t>(kind);
	if (value >= FirstImageBinding && value < FirstStorageImageBinding) {
		return ImageResourceClass::Sampled;
	}
	if (value >= FirstStorageImageBinding &&
	    value < static_cast<uint32_t>(DescriptorBindingKind::Samplers)) {
		return ImageResourceClass::Storage;
	}
	return ImageResourceClass::None;
}

[[nodiscard]] constexpr uint32_t ImageBindingIndex(DescriptorBindingKind kind) {
	return static_cast<uint32_t>(kind) - FirstImageBinding;
}

[[nodiscard]] constexpr std::optional<DescriptorBindingKind>
DescriptorBindingForImage(const ImageResource& image) {
	constexpr uint32_t SampledFloatBinding = 1u;
	constexpr uint32_t SampledUintBinding  = 8u;
	constexpr uint32_t SampledSintBinding  = 15u;
	constexpr uint32_t StorageFloatBinding = FirstStorageImageBinding;
	constexpr uint32_t StorageUintBinding  = StorageFloatBinding + 5u;
	constexpr uint32_t AtomicUintBinding   = StorageUintBinding + 5u;

	uint32_t base    = 0;
	bool     sampled = false;
	if (image.resource_class == ImageResourceClass::Sampled) {
		if (image.atomic) {
			return std::nullopt;
		}
		sampled = true;
		switch (image.numeric_class) {
			case Prospero::TextureNumericClass::Float:
				base = image.depth_compare ? FirstComparisonImageBinding : SampledFloatBinding;
				break;
			case Prospero::TextureNumericClass::Uint: base = SampledUintBinding; break;
			case Prospero::TextureNumericClass::Sint: base = SampledSintBinding; break;
			case Prospero::TextureNumericClass::Unsupported: return std::nullopt;
			default: return std::nullopt;
		}
		if (image.depth_compare && image.numeric_class != Prospero::TextureNumericClass::Float) {
			return std::nullopt;
		}
	} else if (image.resource_class == ImageResourceClass::Storage) {
		if (image.atomic) {
			if (image.numeric_class != Prospero::TextureNumericClass::Uint) {
				return std::nullopt;
			}
			base = AtomicUintBinding + (image.atomic64 ? 5u : 0u);
		} else {
			switch (image.numeric_class) {
				case Prospero::TextureNumericClass::Float: base = StorageFloatBinding; break;
				case Prospero::TextureNumericClass::Uint: base = StorageUintBinding; break;
				case Prospero::TextureNumericClass::Sint:
				case Prospero::TextureNumericClass::Unsupported: return std::nullopt;
				default: return std::nullopt;
			}
		}
	} else {
		return std::nullopt;
	}

	uint32_t dimension = 0;
	switch (image.dimension) {
		case Decoder::ImageDimension::Dim1D: break;
		case Decoder::ImageDimension::Dim1DArray: dimension = 1u; break;
		case Decoder::ImageDimension::Dim2D: dimension = 2u; break;
		case Decoder::ImageDimension::Dim2DArray: dimension = 3u; break;
		case Decoder::ImageDimension::Dim2DMsaa:
			if (!sampled) {
				return std::nullopt;
			}
			dimension = 4u;
			break;
		case Decoder::ImageDimension::Dim2DMsaaArray:
			if (!sampled) {
				return std::nullopt;
			}
			dimension = 5u;
			break;
		case Decoder::ImageDimension::Dim3D: dimension = sampled ? 6u : 4u; break;
		case Decoder::ImageDimension::Unknown: return std::nullopt;
		default: return std::nullopt;
	}
	return static_cast<DescriptorBindingKind>(base + dimension);
}

struct DescriptorBinding {
	DescriptorBindingKind kind = DescriptorBindingKind::Buffers;
	std::vector<uint32_t> resources;

	bool operator==(const DescriptorBinding& other) const = default;
};

// KYTY_CALL_PARTITIONS (ShaderCalls.h): a partition program's call-state push parameters (the
// report set's device address, the dispatch's target pair) and the goto variable that holds the
// lanes its drop markers took out of the wave (IREmitter::SetExec leaves them out of EXEC).
constexpr uint32_t CallPartitionParamReportLo = 0;
constexpr uint32_t CallPartitionParamReportHi = 1;
constexpr uint32_t CallPartitionParamTargetLo = 2;
constexpr uint32_t CallPartitionParamTargetHi = 3;
constexpr uint32_t CallPartitionReportSlots   = 512;
constexpr uint32_t PartitionDropVariable      = 0x7ffffff0u;
// Call-split callee programs (CallSplitPlan.cpp BuildCode): the lanes whose call matched the callee the
// dispatch runs (SplitTargetCompare). A shaded-only SaveChanged stores just these lanes.
constexpr uint32_t SplitShadedVariable        = 0x7ffffff1u;
// Set by the translation of a program with partition markers (thread of that translation).
void SetPartitionDropLanes(bool value) noexcept;
[[nodiscard]] bool PartitionDropLanes() noexcept;

struct BindingLayout {
	uint32_t                       push_data_start_dword = PushData::NoStart;
	uint32_t                       dispatch_thread_dword = PushData::NoStart;
	// Call-split programs (CallSplit.h): CallStateDwords push dwords (state base, wave list and wave
	// header addresses, wave stride, callee address), after the dispatch thread extents.
	uint32_t                       call_state_dword = PushData::NoStart;
	static constexpr uint32_t      CallStateDwords  = 9;
	uint32_t                       memory_offset_dword = 0;
	uint32_t                       memory_offset_count = 0;
	// Per-draw GET_LOD_STATS field of each image (one dword each, LodStatsReport::ImageField:
	// counter id, base level, no-counter flag, counting threshold), after the memory offsets.
	// Non-zero only for shaders instrumented for mip statistics.
	uint32_t                       mip_stats_count = 0;
	std::vector<uint32_t>          user_data_registers;
	std::vector<DescriptorBinding> descriptors;

	[[nodiscard]] uint32_t MipStatsOffsetDword() const {
		return memory_offset_dword + (memory_offset_count + 3u) / 4u;
	}
	[[nodiscard]] uint32_t ShaderDataDwords() const {
		return MipStatsOffsetDword() + mip_stats_count;
	}
	[[nodiscard]] bool UsesPushData() const {
		return push_data_start_dword != PushData::NoStart;
	}
	void AdvancePushData(uint32_t& cursor) const {
		if (UsesPushData()) {
			cursor = push_data_start_dword + ShaderDataDwords();
		}
	}

	bool operator==(const BindingLayout& other) const = default;
};

struct ShaderInfo {
	// usa-ratt/KytyPS5-GT7 8b76a992a: raised from 64/64/32 for expanded table calls (each inlined
	// GT7 material variant brings its own statically resolved descriptors). A shader past a limit
	// fails translation (its draws are skipped), so only such shaders change; the fixed arrays
	// sized by these are small (ResourceMaterialization, the emitter's buffer offsets).
	static constexpr uint32_t MaxBuffers      = 256;
	static constexpr uint32_t MaxImages       = 512;
	static constexpr uint32_t MaxSamplers     = 128;

	std::vector<BufferResource>      buffers;
	std::vector<ImageResource>       images;
	std::vector<SamplerResource>     samplers;
	std::vector<SampledResourcePair> sampled_pairs;
	// Flattened-SRT slots (low, high) of global-load bases read straight from the SRT (upstream
	// c174ade18; renderCompute.cpp KYTY_BACK_ADDRESS_BASES backs their pages).
	std::vector<std::array<uint32_t, 2>> address_bases;
	std::vector<StageInput>          inputs;
	std::vector<StageOutput>         outputs;
	std::array<uint8_t, 32>          vertex_fetch_components {};
	int32_t                          vertex_offset_sgpr = -1;
	int32_t                          instance_offset_sgpr = -1;
	bool                             has_bitwise_xor    = false;
	bool                             uses_dma           = false;
	// IMAGE_BVH*_INTERSECT_RAY translated (KYTY_RT_STUB); counts the program's draws/dispatches.
	bool                             uses_bvh           = false;
	// KYTY_BDA_WRITES: the program stores or does atomics through a V# it computes (IndirectBuffer
	// writes through BDA). Implies uses_dma and has_address_writes; the renderer settles each of
	// its dispatches synchronously.
	bool                             bda_writes         = false;
	// KYTY_POPS_INTERLOCK: a pixel shader that writes hwreg(POPS_PACKER) with S_SETREG_B32, the start
	// of a primitive-ordered pixel shading section (RDNA2 ISA Table 16; REPORT-ISA-2 M1).
	bool                             pops               = false;
	// KYTY_RT_HW (RayTracingPatch.h): the native ray tracing sites of the program (bit per site)
	// and their guest TLAS: the header address is the 64-bit value at the base of the V# at user
	// data pointer `ray_query_user_sgpr` + `ray_query_vsharp_offset`, plus
	// `ray_query_header_offset`; the root node `ray_query_root`.
	uint32_t                         ray_query_sites         = 0;
	uint32_t                         ray_query_user_sgpr     = 0;
	uint32_t                         ray_query_vsharp_offset = 0;
	uint32_t                         ray_query_header_offset = 0;
	uint32_t                         ray_query_root          = 0;

	bool operator==(const ShaderInfo& other) const = default;
};

struct BlockInfo {
	uint32_t        id       = 0;
	uint32_t        start_pc = 0;
	uint32_t        end_pc   = 0;
	CFG::Terminator terminator;
	Value           condition;
	Value           indirect_target;
};

struct DescriptorSource {
	struct IndirectDescriptor {
		struct SelectorRead {
			// KYTY_SELECTOR_MASK_RANKS (upstream 99f4eba22): the mask words whose set bits are the
			// records a bit scan selects; every word at offset + k * stride of this buffer counts.
			struct MaskRanks {
				uint32_t source = UINT32_MAX;
				uint32_t stride = 0;
				uint32_t offset = 0;

				bool operator==(const MaskRanks& other) const = default;
			};
			uint32_t source = UINT32_MAX;
			uint32_t stride = 0;
			uint32_t offset = 0;
			// KYTY_SELECTOR_BITSET (upstream 145578ac0, 914a3bb4b): stride / 4 bitmap words per
			// record (four or eight); the keys are first + key_count - 1 - each set-bit rank of the
			// words OR-ed over the buffer.
			bool     bitset = false;
			// KYTY_SELECTOR_EXCLUDED (upstream f297c0bfe): a guard keeps this key from the read.
			std::optional<uint32_t>  excluded;
			std::optional<MaskRanks> ranks;

			bool operator==(const SelectorRead& other) const = default;
		};
		std::optional<SelectorRead> selector;
		uint32_t table_source    = 0;
		uint32_t table_offset    = 0;
		uint32_t table_immediate = 0;
		uint32_t table_stride    = 0;
		uint32_t table_record_bytes = 0;
		bool     table_scalar = false;
		// No enumeration: the shader looks the key up in the bindless translation table.
		bool     bindless     = false;
		// KYTY_RUNTIME_IMAGES: no table either; the T# is a value the shader loads
		// (ImageResource::runtime).
		bool     runtime      = false;
		// KYTY_WORKGROUP_IMAGES (upstream de3ed905c): nonzero, the workgroup keys are
		// (selector_first + ID) * workgroup_multiplier >> workgroup_shift (shift 32..63).
		uint8_t  workgroup_shift = 0;
		uint32_t workgroup_axis  = UINT32_MAX;
		// Bindless through a record pointer: the key selects a table_stride-byte record of the
		// table, whose 64-bit word at this byte offset points at the T# (table_offset bytes past
		// it). UINT32_MAX: the T# itself is in the table, 32 bytes per key.
		uint32_t record_pointer  = UINT32_MAX;
		// Nested heap (with a record pointer): the T# is in a heap whose V# is read at byte
		// nested_heap of the buffer whose V# is at byte nested_vsharp past the record pointer; the
		// shader's key is record * nested_keys + the key into that heap (table_offset and 32-byte
		// T#s from its base). nested_vsharp UINT32_MAX: no nesting.
		uint32_t nested_vsharp   = UINT32_MAX;
		uint32_t nested_heap     = 0;
		uint32_t nested_keys     = 0;
		uint32_t workgroup_multiplier = 0;
		// Address tables: keys 0 .. key_count - 1 (from selector_first when set, KYTY_SELECTOR_BITSET's
		// reversed ranges). A scalar material selector (KYTY_SELECTOR_BITSCAN /
		// KYTY_SELECTOR_SHARED_LIST): the records 0 .. key_count - 1 instead of every offset.
		Value    key_count;
		// With workgroup_axis: a runtime base the shader adds to the workgroup ID (empty: none).
		// Without it, on an address table: the first key (KYTY_SELECTOR_BITSET).
		Value                 selector_first;
		Value    selector_mask;

		bool operator==(const IndirectDescriptor& other) const = default;
	};

	std::array<Value, 8>         dwords {};
	uint32_t                     dword_count = 0;
	std::optional<IndirectDescriptor> indirect_descriptor;
	// KYTY_BINDLESS_SAMPLERS: an S# read from a sampler heap at a key the shader computes. Dwords
	// 0-2 are the heap's V# (dword 3 is 0); key k's S# is at table_offset + k * stride bytes.
	struct BindlessSampler {
		uint32_t table_offset = 0;
		uint32_t stride       = 16;

		bool operator==(const BindlessSampler& other) const = default;
	};
	std::optional<BindlessSampler> bindless_sampler;

	bool operator==(const DescriptorSource& other) const = default;
};

struct SrtRead {
	Value    value;
	uint32_t flat_offset = 0;

	bool operator==(const SrtRead& other) const = default;
};

struct ResourceBlock {
	// Conditional successors are ordered true, false; an empty condition follows every edge.
	Value                 condition;
	std::vector<uint32_t> successors;
	// Descriptor sources the block reads; with KYTY_SRT_IMAGE_REACH an entry
	// descriptor_sources.size() + i also marks image i (ImageActiveSlot, SrtWalker.h).
	std::vector<uint32_t> sources;
};

// Static bounds of the bytes a shader may write through its descriptor-bound storage buffers
// (passes/WriteRangeAnalysis.h). Nodes form a tape in dependency order: operands (a, b, c) index
// earlier nodes. Leaves are compile-time ranges or record-time inputs (user data, flattened SRT
// dwords, dispatch size); every node evaluates to an unsigned interval that contains each value
// the shader can compute for it, and [0, UINT32_MAX] stands for "unknown".
struct WriteRangeNode {
	enum class Op : uint8_t {
		Range,              // [lo, hi]
		UserData,           // snapshot user data dword a
		FlatSrt,            // flattened SRT dword a
		WorkgroupId,        // [0, groups[a] - 1]
		GlobalInvocationId, // [0, groups[a] * hi - 1]
		Add,
		Sub,
		Mul,
		Shl,
		Shr,
		Sar,
		And,
		Or,
		Xor,
		UMin,
		UMax,
		SMin,
		SMax,
		UDiv,
		UMulHi,
		BitExtract, // (value a, offset b, count c)
		Union,
	};
	Op       op = Op::Range;
	uint32_t a  = 0;
	uint32_t b  = 0;
	uint32_t c  = 0;
	uint32_t lo = 0;
	uint32_t hi = UINT32_MAX;

	bool operator==(const WriteRangeNode& other) const = default;
};

// One buffer store or atomic: its bytes start at index * stride + offset + immediate + k + soffset
// (swizzled when the specialized descriptor enables it) for k in [0, extent - 4].
struct WriteRangeAccess {
	uint32_t index     = 0; // node
	uint32_t offset    = 0; // node
	uint32_t soffset   = 0; // node
	uint32_t immediate = 0;
	uint32_t extent    = 4;

	bool operator==(const WriteRangeAccess& other) const = default;
};

struct BufferWriteRange {
	uint32_t                      buffer        = 0;
	uint32_t                      packed_stride = 0; // Specialized ShaderBufferResource::PackedStride
	bool                          bounded       = false;
	std::vector<WriteRangeAccess> accesses;

	bool operator==(const BufferWriteRange& other) const = default;
};

// The guest memory one read through a computed address (BDA) can reach (KYTY_BDA_REACH, see
// WriteRangeEvaluator::DmaSpans). Node fields index WriteRangeProgram::nodes.
// - Window: an address load at base + offset + immediate + [0, extent), base = (base_low,
//   base_high) (both bits 1:0 cleared for a scalar address read, as the emitter does), the
//   offset zero-extended from 32 bits.
// - IndirectScalar / IndirectVector: a load through a V# the shader assembles from words[0..3];
//   word k comes from table dword k when bit k of table_words is set: the 16 bytes at
//   (base & ~3) + (o & ~3) + table_immediate for every o of node table_offset whose set bits
//   lie in table_bits. A scalar load reads [V# base, V# base + V# size). A vector load reads at
//   index * stride + offset + immediate + soffset (+ lane with ADD_TID) and extent bytes past it.
struct DmaReadAccess {
	// MeshIndex: the mesh prologue's index fetch (Translate.cpp: base = draw dwords 4-5, read only for
	// input vertices below the dispatch's count): the reach is the draw's index range, from the host.
	enum class Kind : uint8_t { Window, IndirectScalar, IndirectVector, MeshIndex };
	Kind     kind          = Kind::Window;
	bool     scalar        = false; // Window: ResourceKind::ScalarAddress (bits 1:0 cleared)
	uint8_t  table_words   = 0;
	uint32_t base_low      = 0;
	uint32_t base_high     = 0;
	uint32_t offset        = 0;
	int32_t  immediate     = 0;
	uint32_t extent        = 0;
	uint32_t table_offset  = 0;
	uint32_t table_bits    = 0;
	int32_t  table_immediate = 0;
	std::array<uint32_t, 4> words {};
	uint32_t index         = 0;
	uint32_t soffset       = 0;

	bool operator==(const DmaReadAccess& other) const = default;
};

struct DmaReadReach {
	// Every guest memory read of the program through a device address is one of `accesses`.
	bool                       bounded = false;
	// Why not (DmaUnbounded); 0 when bounded or when the program has no device-address access.
	uint32_t                   reason  = 0;
	std::vector<DmaReadAccess> accesses;

	bool operator==(const DmaReadReach& other) const = default;
};

enum class DmaUnbounded : uint32_t {
	None = 0,
	AddressWrites,   // stores or atomics through device addresses
	RayTracing,      // BVH or native ray tracing walks
	FullAddress,     // a 64-bit address computed per lane
	AddressHandle,   // an address load without a GetAddressResource base pair
	IndirectHandle,  // an indirect V# access without a four-word GetBufferResource
	OtherAccess,     // another kind of device-address access
	TooManyAccesses, // more distinct accesses than the host walks per draw
	NodeOverflow,    // the interval tape is full
	Count,
};

struct WriteRangeProgram {
	std::vector<WriteRangeNode>   nodes;
	std::vector<BufferWriteRange> buffers; // One entry per written buffer resource.
	DmaReadReach                  dma;

	[[nodiscard]] const BufferWriteRange* Find(uint32_t buffer) const {
		for (const auto& entry: buffers) {
			if (entry.buffer == buffer) {
				return &entry;
			}
		}
		return nullptr;
	}
	bool operator==(const WriteRangeProgram& other) const = default;
};

// Stable shader metadata consumed by the renderer after native IR has been discarded.
struct CompiledShaderInfo {
	ShaderType                    stage               = ShaderType::Unknown;
	uint64_t                      shader_hash         = 0;
	uint32_t                      wave_size           = 64;
	uint32_t                      user_data_base      = 0;
	uint32_t                      user_data_count     = 64;
	uint32_t                      scratch_dwords      = 0;
	uint32_t                      param_export_mask   = 0;
	bool                          has_address_writes  = false;
	ShaderInfo                    info;
	BindingLayout                 bindings;
	WriteRangeProgram             write_ranges;

	// Every member (the persistent program cache's verify mode compares whole objects).
	bool operator==(const CompiledShaderInfo& other) const = default;
};

struct UniformFillPlan {
	UniformFill          fill;
	std::array<Value, 4> values;
};

// Resource analysis retained by the shader cache. It owns immutable descriptor/SRT,
// condition and fill values without translated blocks. Evaluation scratch lives in
// EvaluationScratch so that a sealed plan can be evaluated by several threads at once.
struct ResourcePlan {
	// Immutable operand recipes use the same indices as the legacy evaluator's memo.
	// They contain no guest memory values and are rebuilt for each retained IR plan.
	struct EvaluationOperand {
		enum class Kind : uint32_t { Invalid, Immediate, Node };
		uint64_t value = 0;
		uint32_t index = 0;
		Kind kind = Kind::Invalid;
	};
	struct EvaluationRecipe {
		enum class Kind : uint32_t {
			Fallback, Operation, UserData, FlatRead, Forward, Extract, ExtractCarry,
			RawAddress, RawBuffer, Select,
		};
		const Inst* instruction = nullptr;
		std::array<EvaluationOperand, 5> operands {};
		Value selection_mask;
		int64_t offset = 0;
		uint32_t parameter = 0;
		Kind kind = Kind::Fallback;
	};
	struct ArithmeticTapeOperand {
		uint64_t value = 0; // Immediate bits, or an earlier tape instruction's result index.
		bool immediate = true;
	};
	struct ArithmeticTapeInstruction {
		enum class Kind : uint8_t { Boundary, Operation, Forward, Extract, ExtractCarry };
		std::array<ArithmeticTapeOperand, 4> operands {};
		const Inst* instruction = nullptr;
		uint32_t parameter = 0; // Boundary recipe index or extract component.
		Kind kind = Kind::Boundary;
		uint8_t operand_count = 0;
	};
	struct ArithmeticTape {
		uint32_t first = 0;
		uint32_t count = 0; // Zero keeps the original evaluator path.
	};
	struct EvaluationContext {
		struct Entry {
			uint64_t value      = 0;
			uint64_t generation = 0;
		};

		std::vector<Entry> values;
		// Nested tapes in this memo context use disjoint, stack-allocated slices.
		std::vector<uint64_t> tape_values;
		size_t               tape_values_used = 0;
		uint64_t           generation = 0;
	};

	ResourcePlan() = default;
	~ResourcePlan();

	ResourcePlan(const ResourcePlan&)            = delete;
	ResourcePlan& operator=(const ResourcePlan&) = delete;
	ResourcePlan(ResourcePlan&&) noexcept         = default;
	ResourcePlan& operator=(ResourcePlan&& other) noexcept;

	ShaderType                    stage           = ShaderType::Unknown;
	uint64_t                      shader_hash     = 0;
	uint32_t                      user_data_base  = 0;
	uint32_t                      user_data_count = 64;
	std::list<Inst>                     value_storage;
	std::vector<MemoryInfo>             memory_info;
	std::vector<DescriptorSource>       descriptor_sources;
	std::vector<ResourceBlock>          control_flow;
	std::vector<SrtRead>                srt_reads;
	// Exclusive run ends for adjacent raw flat reads; empty on untranslated plans.
	// Runs preserve read order, resolved handle, opcode and clean/ordinary context.
	std::vector<uint32_t>               srt_read_run_ends;
	std::vector<EvaluationRecipe>       evaluation_recipes;
	// Immutable entry points for the same evaluator, decoded once after cloning.
	// These contain indices/immediate constants only, never guest read results.
	std::vector<std::array<EvaluationOperand, 8>> descriptor_roots;
	std::vector<EvaluationOperand>      flat_read_roots;
	std::vector<EvaluationOperand>      condition_roots;
	std::vector<uint8_t>                initial_active_sources;
	// Compacted traversal skips only empty unconditional blocks; no guest reads
	// or source activations are removed. Separate from decoded expression roots.
	std::vector<uint32_t>               flow_aliases;
	std::vector<uint8_t>                flow_initial_sources;
	std::vector<ArithmeticTape>         arithmetic_tapes;
	std::vector<ArithmeticTapeInstruction> arithmetic_tape_instructions;
	std::vector<uint8_t>                clean_flat_slots;
	// Indexed by flat offset like clean_flat_slots: flat slots only the shader uses
	// (GpuFillSlots, ResourceMaterialization.cpp), which a compute refresh on the command
	// processor may leave to the GPU when their bytes are GPU-written (KYTY_SRT_GPU_FILL).
	std::vector<uint8_t>                gpu_fill_slots;
	bool                                requires_specialization_memory = false;
	bool                                has_address_writes = false;
	bool                                srt_plan_complete          = false;
	bool                                resource_tracking_complete = false;
	ShaderInfo                          info;
	UniformFillPlan                     uniform_fill;
	// Dense memo slot count. Unsealed programs still grow it lazily while evaluating;
	// a sealed plan assigned every slot when it was extracted and never writes it again.
	mutable uint32_t                    evaluation_value_count = 0;
	bool                                evaluation_sealed      = false;
	// Unique per sealed plan (AssignTraceIdentity, when sealed, cloned or decoded); keys the
	// per-thread SRT replay traces (KYTY_SRT_TRACE).
	uint64_t                            trace_identity         = 0;

	void AssignTraceIdentity() {
		static std::atomic<uint64_t> next_identity {0};
		trace_identity = next_identity.fetch_add(1, std::memory_order_relaxed) + 1;
	}
};

// Per-thread scratch for nested clean/EXEC memos, activity and material keys. Plans stay
// read-only during evaluation. Memo entries and visit tags are generation-stamped, so a
// scratch can serve different plans in turn without clearing; walkers release LIFO.
struct EvaluationScratch {
	EvaluationScratch() = default;

	EvaluationScratch(const EvaluationScratch&)            = delete;
	EvaluationScratch& operator=(const EvaluationScratch&) = delete;
	EvaluationScratch(EvaluationScratch&&) noexcept         = default;
	EvaluationScratch& operator=(EvaluationScratch&&) noexcept = default;

	std::deque<ResourcePlan::EvaluationContext> evaluation_contexts;
	uint32_t                                    evaluation_depth = 0;
	std::vector<uint32_t>                       flow_visit_tags;
	uint32_t                                    flow_visit_epoch = 0;
	std::vector<uint8_t>                        active_sources;
	std::vector<uint8_t>                        visited_blocks;
	std::vector<uint32_t>                       pending_blocks;
	std::vector<uint32_t>                       material_keys;
	std::vector<std::pair<uint64_t, uint64_t>>  specialization_reads;
};

struct Program: ResourcePlan {
	Program() = default;
	~Program();

	Program(const Program&)            = delete;
	Program& operator=(const Program&) = delete;
	Program(Program&&) noexcept         = default;
	Program& operator=(Program&& other) noexcept;
	CompiledShaderInfo TakeCompiledInfo() &&;

	std::vector<std::unique_ptr<Block>> block_storage;
	BlockList                           blocks;
	uint32_t                      wave_size      = 64;
	uint32_t                      scratch_dwords = 0;
	// KYTY_LANE_LOCAL: a wave64 workgroup program proven lane-local (Frontend::IsWaveLaneLocal) and
	// translated for a host with 32-wide subgroups: it runs one guest lane per host invocation.
	bool                          lane_local          = false;
	bool                          dispatcher_fallback = false;
	CFG::FailureKind              cfg_failure_kind    = CFG::FailureKind::None;
	std::string                   fallback_reason;
	std::vector<BlockInfo>        block_info;
	// Typed memory and export instructions reference shader-local metadata by dense index.
	// Decoder-only details (such as NSA register numbers) have already become IR operands.
	std::vector<ExportInfo>       export_info;
	std::vector<Value>            dynamic_reads;
	bool                          shader_info_complete = false;
	BindingLayout                 bindings;
	bool                          binding_layout_complete = false;
	WriteRangeProgram             write_ranges;

};

std::string ProgramToString(const Program& program);
bool        HasShaderMemoryWrites(const Program& program);

// Deep copy of a translated program (ir/ProgramClone.cpp); false when `source` references an
// instruction or block it does not own. The pipeline cache specializes such copies instead of
// translating a program source again (KYTY_TRANSLATION_CACHE). Adding a member to Program,
// ResourcePlan, Block or Inst requires updating CloneProgram (its layout checks enforce it), and
// adding one to ResourcePlan, Inst, CompiledShaderInfo or a type they contain requires updating
// ir/ProgramCodec.cpp (the persistent program cache's encoding; its layout checks enforce it).
[[nodiscard]] bool CloneProgram(const Program& source, Program& target);

void  ValidateProgram(const Program& program, bool require_ssa);
// Whether translations run ValidateProgram: debug builds, or with KYTY_IR_VALIDATE=1 in release.
// The checks only ever stop the emulator, and they were about a fifth of a release translation
// (first-encounter hitches, precompile time; chenxiao07 a28de66fe).
[[nodiscard]] bool ValidationEnabled();
void  ResolveControlFlowIdentities(Program& program);
// References from the resource-plan tables (descriptor sources, resource control flow, SRT reads,
// evaluation recipes and tapes, uniform fill, dynamic reads) to instructions the program no longer
// has. Specialization can leave such references: a zero-stride buffer read or an FMASK read it
// replaces takes its descriptor's last uses, and dead-code elimination then erases instructions
// the descriptor source still names. Drop clears them (an empty value, a null instruction) and
// returns how many it cleared; Count only counts them. Neither dereferences such a reference.
size_t CountDeadPlanReferences(const Program& program);
size_t DropDeadPlanReferences(Program& program);
// The instructions the resource-plan tables name, except dynamic_reads (a list of the walk's
// reads, not values the plan evaluates). Dead-code elimination after resource tracking keeps them:
// tracking can take a value's last IR use (a stand-in or bindless sampler plan rewires the S#
// handle, stranding the heap reads and with them the V# words of the heap's buffer source), and
// ExtractResourcePlan reads every such value and its arguments.
std::unordered_set<const Inst*> PlanTableInstructions(const Program& program);
// Points plan-table references to Identity instructions at what the identities resolve to.
// RemoveIdentities rewires only IR uses and erases the identities, so such a reference would be
// dead (DropDeadPlanReferences clears it) although its value is still live: Astro Bot's Sky Garden
// lost its sky and lighting that way. Before DropDeadPlanReferences, these references were read
// through the freed identity, which still held its argument. Returns how many it redirected.
size_t ResolvePlanIdentities(Program& program);
bool  EquivalentValue(const ResourcePlan& program, Value left, Value right);
Value ResolveInvariantPhi(const ResourcePlan& program, Value value);
Value ResolveActiveU32(Value value, Value active);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_ */
