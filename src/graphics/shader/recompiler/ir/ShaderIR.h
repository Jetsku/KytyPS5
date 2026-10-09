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
#include <bit>
#include <deque>
#include <list>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

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
	bool                    idxen                                                 = false;
	bool                    offen                                                 = false;
	bool                    coherent                                              = false;
	bool                    planning_only                                         = false;

	[[nodiscard]] bool SupportsIndirectBufferLoad(ValueOpcode opcode) const {
		// ReadConstBuffer: one dword of an S_BUFFER_LOAD (no formats, RDNA2 ISA 7.2.1).
		return !formatted && !typed && data_bits == 32u &&
		       (opcode == ValueOpcode::LoadBufferU32x2 || opcode == ValueOpcode::LoadBufferU32x3 ||
		        opcode == ValueOpcode::LoadBufferU32x4 || opcode == ValueOpcode::ReadConstBuffer);
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

struct BufferResource {
	static constexpr uint32_t NoImageAlias = UINT32_MAX;

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

	bool operator==(const BufferResource& other) const = default;
};

enum class ImageMipMode { None, Dynamic };

constexpr uint32_t ShaderImageIdentitySwizzle = 0x00000facu;

struct ImageResource {
	static constexpr uint32_t NoIndirectImage = UINT32_MAX;

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
	uint32_t                      indirect_root     = NoIndirectImage;
	uint32_t                      indirect_mapping_offset   = 0;
	uint32_t                      indirect_search_iterations = 0;
	std::vector<uint32_t>         indirect_resources;

	bool operator==(const ImageResource& other) const = default;
};

struct SamplerResource {
	uint32_t source                = 0;
	uint32_t first_use_pc          = 0;
	// Native filtering/border variants share the original sampler's runtime descriptor.
	uint32_t snapshot_index        = 0;
	bool     force_point_filtering = false;
	bool     depth_compare         = false;
	bool     integer_border        = false;
	bool     gather_lod            = false;

	bool operator==(const SamplerResource& other) const = default;
};

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
	Factor
};

enum class StageInputKind {
	VertexIndex,
	InvocationId,
	PrimitiveId,
	TessCoord,
	InstanceIndex,
	FragCoord,
	FrontFacing,
	PackedAncillary,
	Layer,
	SampleId,
	BaryCoordSmooth,
	BaryCoordSmoothCentroid,
	BaryCoordNoPerspective,
	WorkgroupId,
	LocalInvocationId,
	LocalInvocationIndex,
	GlobalInvocationId,
	Parameter,
	// Further SPI_PS_INPUT I/J pairs (components 0/1 are I/J).
	BaryCoordSmoothSample,
	BaryCoordNoPerspectiveCentroid,
	BaryCoordNoPerspectiveSample,
	// The dispatch's workgroup counts (compute; the LDS device buffer's per-workgroup regions).
	NumWorkgroups,
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
	ViewportIndex
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
	// KYTY_LDS_DEVICE_BUFFER: compute LDS kept in a device buffer (ShaderComputeInputInfo::
	// lds_storage). Compute only, so it sits past the per-stage binding range (NativeBinding).
	SharedMemory,
	Count,
};

// Native bindings per stage group; SharedMemory (compute, group 0 only) lies past it, so the
// graphics stages' bindings are those of the builds before it.
inline constexpr uint32_t NativeBindingGroupStride = 56u;

static_assert(static_cast<uint32_t>(DescriptorBindingKind::Samplers) == 49u);
static_assert(static_cast<uint32_t>(DescriptorBindingKind::SharedMemory) == NativeBindingGroupStride);
static_assert(static_cast<uint32_t>(DescriptorBindingKind::Count) == 57u);

struct PushData {
	static constexpr uint32_t DwordCount = 32;
	static constexpr uint32_t MeshDrawDwordCount = 6;
	// Mesh draw dword 6, only for programs with ShaderMeshInputInfo::split_groups: the first
	// workgroup of this part of a draw split past the host's X group limit (0 when unsplit; a
	// native indirect draw's parameter block holds 0 there).
	static constexpr uint32_t MeshFirstGroupDword = MeshDrawDwordCount;
	[[nodiscard]] static constexpr uint32_t MeshDrawDwords(bool split_groups) {
		return MeshDrawDwordCount + (split_groups ? 1u : 0u);
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
	return static_cast<uint32_t>(kind) + group * NativeBindingGroupStride;
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

struct BindingLayout {
	uint32_t                       push_data_start_dword = PushData::NoStart;
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
	static constexpr uint32_t MaxBuffers      = 64;
	static constexpr uint32_t MaxImages       = 64;
	static constexpr uint32_t MaxSamplers     = 32;
	static constexpr uint32_t MaxSampledPairs = 64;

	std::vector<BufferResource>      buffers;
	std::vector<ImageResource>       images;
	std::vector<SamplerResource>     samplers;
	std::vector<SampledResourcePair> sampled_pairs;
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
	struct IndirectImage {
		uint32_t material_source = UINT32_MAX;
		uint32_t table_source    = 0;
		uint32_t selector_stride = 0;
		uint32_t selector_offset = 0;
		uint32_t table_offset    = 0;
		Value    key_count;
		Value    selector_mask;

		bool operator==(const IndirectImage& other) const = default;
	};

	std::array<Value, 8>         dwords {};
	uint32_t                     dword_count = 0;
	std::optional<IndirectImage> indirect_image;

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

struct WriteRangeProgram {
	std::vector<WriteRangeNode>   nodes;
	std::vector<BufferWriteRange> buffers; // One entry per written buffer resource.

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
bool  EquivalentValue(const ResourcePlan& program, Value left, Value right);
Value ResolveInvariantPhi(const ResourcePlan& program, Value value);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_ */
