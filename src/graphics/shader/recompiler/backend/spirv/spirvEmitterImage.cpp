#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/recompiler/frontend/decode/ImageOps.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t DmaskComponentIndex(uint32_t dmask, uint32_t component) {
	uint32_t index = 0;
	for (uint32_t i = 0; i < component; i++) {
		index += (dmask >> i) & 1u;
	}
	return index;
}

uint32_t DmaskComponent(uint32_t dmask, uint32_t index) {
	for (uint32_t component = 0; component < 4u; component++) {
		if (((dmask >> component) & 1u) != 0u && index-- == 0u) return component;
	}
	return 0u;
}

uint32_t ImageGatherComponent(uint32_t dmask) {
	switch (dmask) {
		case 0x2u: return 1;
		case 0x4u: return 2;
		case 0x8u: return 3;
		default: return 0;
	}
}

bool HasFlag(const IR::MemoryInfo& mem, uint32_t flag) {
	return (mem.image_sample_flags & flag) != 0u;
}

uint32_t AddressU32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                    uint32_t component) {
	const auto layout = Decoder::ImageAddressComponentLayout(mem.image_sample_flags,
	                                                         mem.image_dimension, component);
	const auto packed = layout.bit_offset / 32u;
	if (packed >= address.NumArgs()) return ConstantU32(ctx.state, 0);
	auto value = ctx.Def(address.Arg(packed));
	if (layout.bit_width == 16u) {
		if ((layout.bit_offset & 31u) != 0u) {
			value = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), value,
			               ConstantU32(ctx.state, 16));
		}
		value = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), value,
		               ConstantU32(ctx.state, 0xffffu));
	}
	return value;
}

uint32_t AddressF32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                    uint32_t component) {
	const auto value  = AddressU32(ctx, mem, address, component);
	const auto layout = Decoder::ImageAddressComponentLayout(mem.image_sample_flags,
	                                                         mem.image_dimension, component);
	return layout.bit_width == 16u ? EmitF16BitsToF32(ctx.state, value)
	                               : Unary(ctx.state, spv::OpBitcast, TypeF32(ctx.state), value);
}

ImageSampleLayout Layout(const IR::MemoryInfo& mem) {
	ImageSampleLayout layout;
	uint32_t          cursor = 0;
	const auto&       info   = ImageDimensionInfoFor(mem.image_dimension);
	if (HasFlag(mem, Decoder::ImageSampleFlagOffset)) layout.offset = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagBias)) layout.bias = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagCompare)) layout.dref = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagDerivative)) {
		layout.grad_x = cursor;
		cursor += info.spatial_components;
		layout.grad_y = cursor;
		cursor += info.spatial_components;
	}
	layout.coord = cursor;
	cursor += info.coordinate_components;
	if (HasFlag(mem, Decoder::ImageSampleFlagLod)) layout.lod = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagLodClamp)) layout.clamp = cursor++;
	return layout;
}

uint32_t ZeroF32(EmitterState& state) {
	return ConstantF32(state, 0);
}

uint32_t GatherMip(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                   const IR::Inst& address, const ImageSampleLayout& layout, uint32_t mip_count) {
	auto& state = ctx.state;
	const auto Extract = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first),
		                           ConstantU32(state, count));
	};
	const auto LodFixed = [&](uint32_t word, uint32_t first) {
		return Binary(state, spv::OpFMul, TypeF32(state),
		              Unary(state, spv::OpConvertUToF, TypeF32(state), Extract(word, first, 12)),
		              ConstantF32Value(state, 1.0f / 256.0f));
	};
	const auto sampler_control = ctx.Arg(inst, 6);
	const auto mip_filter = Extract(sampler_control, 26, 2);
	const auto preclamp = Binary(state, spv::OpINotEqual, TypeBool(state),
	                            Extract(sampler_control, 28, 1), ConstantU32(state, 0));
	const auto zero = ZeroF32(state);
	// Point selects the nearest mip; linear gathers return the lower, primary mip only.
	const auto rounding = Select(state, TypeF32(state),
	    Binary(state, spv::OpIEqual, TypeBool(state), mip_filter, ConstantU32(state, 1)),
	    ConstantF32Value(state, 0.5f), zero);
	const auto base = Unary(state, spv::OpConvertUToF, TypeF32(state),
	                        Extract(ctx.Arg(inst, 4), 12, 4));
	const auto minimum = Binary(state, spv::OpFSub, TypeF32(state),
	                            LodFixed(ctx.Arg(inst, 3), 8), base);
	const auto last = ConstantF32Value(state, static_cast<float>(mip_count - 1u));
	auto lod = Binary(state, spv::OpFAdd, TypeF32(state),
	                  AddressF32(ctx, mem, address, layout.lod),
	                  Select(state, TypeF32(state), preclamp, rounding, zero));
	lod = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(
	    state, lod, LodFixed(ctx.Arg(inst, 5), 0), LodFixed(ctx.Arg(inst, 5), 12));
	// T# minimum LOD is in physical mip space and applies after the S# view-relative clamp.
	lod = EmitGlsl<GLSLstd450FMax, IR::Type::F32>(state, lod, minimum);
	lod = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(state, lod, zero, last);
	lod = Binary(state, spv::OpFAdd, TypeF32(state), lod,
	             Select(state, TypeF32(state), preclamp, zero, rounding));
	// The clamped value is nonnegative, so conversion performs the integer mip rounding.
	const auto mip = EmitUMin32(state, Unary(state, spv::OpConvertFToU, TypeU32(state), lod),
	                           ConstantU32(state, mip_count - 1u));
	const auto mip_none = Binary(state, spv::OpIEqual, TypeBool(state),
	                            mip_filter, ConstantU32(state, 0));
	return Select(state, TypeU32(state), mip_none, ConstantU32(state, 0), mip);
}

uint32_t CubeAxis(EmitterState& state, uint32_t value) {
	return Binary(state, spv::OpFSub, TypeF32(state), value, ConstantF32(state, 0x3f800000u));
}

uint32_t CubeLayer(EmitterState& state, uint32_t value) {
	const auto guest = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), guest, value);
	const auto padding = Binary(
	    state, spv::OpShiftLeftLogical, TypeU32(state),
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), guest, ConstantU32(state, 3)),
	    ConstantU32(state, 1));
	const auto host   = Binary(state, spv::OpISub, TypeU32(state), guest, padding);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), result, host);
	return result;
}

// `row` is the host y of a 1D image, inserted after x.
uint32_t CoordF32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                  uint32_t first, uint32_t components, bool cube = false, uint32_t row = 0) {
	auto&    state = ctx.state;
	uint32_t values[3] {AddressF32(ctx, mem, address, first)};
	uint32_t count = 1;
	if (row != 0u) values[count++] = row;
	for (uint32_t index = 1; index < components; index++) {
		values[count++] = mem.image_address_components > first + index
		                      ? AddressF32(ctx, mem, address, first + index)
		                      : ZeroF32(state);
	}
	if (cube) {
		values[0] = CubeAxis(state, values[0]);
		values[1] = CubeAxis(state, values[1]);
		if (count == 3u) values[2] = CubeLayer(state, values[2]);
	}
	if (count == 1u) return values[0];
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, count), result,
	                          std::span(values).first(count));
	return result;
}

// Texel center of a 1D image's single host row.
uint32_t RowCenter(EmitterState& state, ImageDimension dimension) {
	return ImageDimensionInfoFor(dimension).one_dimensional ? ConstantF32Value(state, 0.5f) : 0u;
}

uint32_t CoordU32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                  ImageDimension dimension) {
	const auto& info = ImageDimensionInfoFor(dimension);
	uint32_t    values[3] {AddressU32(ctx, mem, address, 0)};
	uint32_t    count = 1;
	if (info.one_dimensional) values[count++] = ConstantU32(ctx.state, 0);
	for (uint32_t index = 1; index < info.coordinate_components; index++) {
		values[count++] = mem.image_address_components > index
		                      ? AddressU32(ctx, mem, address, index)
		                      : ConstantU32(ctx.state, 0);
	}
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, count),
	                              result, std::span(values).first(count));
	return result;
}

uint32_t LodU32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                ImageDimension dimension) {
	const auto component = ImageDimensionInfoFor(dimension).coordinate_components;
	return mem.image_has_mip && mem.image_address_components > component
	           ? AddressU32(ctx, mem, address, component)
	           : ConstantU32(ctx.state, 0);
}

uint32_t FloatBits(ValueEmitContext& ctx, uint32_t value) {
	return Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), value);
}

uint32_t SampledComponentBits(ValueEmitContext& ctx, uint32_t value,
                              Prospero::TextureNumericClass numeric_class) {
	if (numeric_class == Prospero::TextureNumericClass::Uint) {
		return value;
	}
	return Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), value);
}

uint32_t SampledComponentZero(EmitterState& state, Prospero::TextureNumericClass numeric_class) {
	switch (numeric_class) {
		case Prospero::TextureNumericClass::Float: return ZeroF32(state);
		case Prospero::TextureNumericClass::Uint: return ConstantU32(state, 0);
		case Prospero::TextureNumericClass::Sint: return ConstantI32(state, 0);
		case Prospero::TextureNumericClass::Unsupported: break;
	}
	EXIT("invalid sampled image numeric class");
}

uint32_t ResultVector(ValueEmitContext& ctx, uint32_t value,
                      Prospero::TextureNumericClass numeric_class, bool dref,
                      const IR::MemoryInfo& mem, bool gather = false) {
	auto value_class = numeric_class;
	if (dref) {
		value_class = Prospero::TextureNumericClass::Float;
	}
	const bool integer = value_class == Prospero::TextureNumericClass::Uint ||
	                     value_class == Prospero::TextureNumericClass::Sint;
	if (mem.data_bits == 16u) {
		uint32_t   packed[4] = {ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0),
		                        ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0)};
		const auto scalar    = [&](uint32_t index) {
			if (dref) return value;
			const auto component = gather ? index : DmaskComponent(mem.dmask, index);
			const auto result    = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract,
			                              ImageScalarType(ctx.state, value_class), result, value,
			                              component);
			return result;
		};
		for (uint32_t word = 0; word < mem.data_dwords; word++) {
			const auto low_index  = word * 2u;
			const auto high_index = low_index + 1u;
			const auto low        = scalar(low_index);
			const auto high       = high_index < mem.component_count
			                            ? scalar(high_index)
			                            : SampledComponentZero(ctx.state, value_class);
			if (integer) {
				const auto low_bits  = SampledComponentBits(ctx, low, value_class);
				const auto high_bits = SampledComponentBits(ctx, high, value_class);
				const auto mask      = ConstantU32(ctx.state, 0xffffu);
				packed[word] =
				    Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state),
				           Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), low_bits, mask),
				           Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
				                  Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state),
				                         high_bits, mask),
				                  ConstantU32(ctx.state, 16u)));
			} else {
				const auto pair = ctx.state.builder.AllocateId();
				ctx.state.builder.AddFunction(spv::OpCompositeConstruct,
				                              TypeF32Vector(ctx.state, 2), pair, low, high);
				packed[word] = ctx.state.builder.AllocateId();
				ctx.state.builder.AddFunction(spv::OpExtInst, TypeU32(ctx.state), packed[word],
				                              GlslStd450(ctx.state), GLSLstd450PackHalf2x16, pair);
			}
		}
		const auto result = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4),
		                              result, packed[0], packed[1], packed[2], packed[3]);
		return result;
	}
	uint32_t component[4] {};
	for (uint32_t index = 0; index < 4u; index++) {
		if (dref) {
			component[index] = index == 0u ? FloatBits(ctx, value) : ConstantU32(ctx.state, 0);
			continue;
		}
		const auto scalar = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(
		    spv::OpCompositeExtract, ImageScalarType(ctx.state, value_class), scalar, value, index);
		component[index] = SampledComponentBits(ctx, scalar, value_class);
	}
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4), result,
	                              component[0], component[1], component[2], component[3]);
	return result;
}

uint32_t QueryDimensions(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                         const IR::Inst& address) {
	const auto  dimension = ImageResourceAt(ctx.state, mem.resource).dimension;
	const auto& info      = ImageDimensionInfoFor(dimension);
	const auto  image     = LoadImageDescriptor(ctx.state, mem.resource);
	const auto  size      = ctx.state.builder.AllocateId();
	if (info.multisampled != 0u) {
		ctx.state.builder.AddFunction(spv::OpImageQuerySize,
		                              ImageViewSizeType(ctx.state, dimension), size, image);
	} else {
		ctx.state.builder.AddFunction(spv::OpImageQuerySizeLod,
		                              ImageViewSizeType(ctx.state, dimension), size, image,
		                              AddressU32(ctx, mem, address, 0));
	}
	uint32_t result[4] = {ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0),
	                      ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0)};
	for (uint32_t index = 0; index < info.coordinate_components; index++) {
		// Skip a 1D image's host height.
		const auto component = info.one_dimensional && index != 0u ? index + 1u : index;
		result[index]        = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), result[index],
		                              size, component);
	}
	if (info.multisampled == 0u) {
		result[3] = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpImageQueryLevels, TypeU32(ctx.state), result[3],
		                              image);
	}
	const auto vector = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4), vector,
	                              result[0], result[1], result[2], result[3]);
	return vector;
}

uint32_t PackedOffset(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                      const ImageSampleLayout& layout, ImageDimension dimension) {
	const auto& info    = ImageDimensionInfoFor(dimension);
	const auto  spatial = info.spatial_components;
	// A 1D image is a host 2D image one texel high (upstream e6cc1ce30); its row takes no offset.
	const auto components = info.one_dimensional ? 2u : spatial;
	const auto zero       = ConstantI32(ctx.state, 0);
	if (layout.offset == NoImageComponent || mem.image_address_components <= layout.offset) {
		const auto result = ctx.state.builder.AllocateId();
		if (components == 3u) {
			ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(ctx.state, 3),
			                              result, zero, zero, zero);
		} else {
			ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(ctx.state, 2),
			                              result, zero, zero);
		}
		return result;
	}
	const auto packed    = Unary(ctx.state, spv::OpBitcast, TypeI32(ctx.state),
	                             AddressU32(ctx, mem, address, layout.offset));
	uint32_t   values[3] = {zero, zero, zero};
	for (uint32_t index = 0; index < spatial; index++) {
		values[index] = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(ctx.state), values[index],
		                              packed, ConstantU32(ctx.state, index * 8u),
		                              ConstantU32(ctx.state, 6));
	}
	const auto result = ctx.state.builder.AllocateId();
	if (components == 3u) {
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(ctx.state, 3),
		                              result, values[0], values[1], values[2]);
	} else {
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(ctx.state, 2),
		                              result, values[0], values[1]);
	}
	return result;
}

// IMAGE_SAMPLE*_O (RDNA2 ISA 8.2.5): one dword of six-bit signed texel offsets, x in bits 5:0,
// y in 13:8 and z in 21:16, added to the texel coordinates of the level that is read. Vulkan
// accepts only constant offsets on sample instructions (ConstOffset), within
// [minTexelOffset, maxTexelOffset]; every implementation supports -8..7. Other offsets move the
// normalized coordinate by offset / level size instead (ApplyDynamicSampleOffset).
struct SampleOffset {
	uint32_t const_offset = 0;     // ConstOffset operand id; 0 when there is none
	uint32_t packed       = 0;     // the packed offsets when they need ApplyDynamicSampleOffset
	bool     dynamic      = false;
};

int32_t SignExtend6(uint32_t value) {
	return static_cast<int32_t>((value & 0x3fu) ^ 0x20u) - 0x20;
}

// The condition masking every use of an image result: each use is a component extract whose
// every use is SelectU32(condition, component, old), i.e. the destination write under EXEC.
// Lanes where it is false discard the result, so an operand written under the same condition
// (Select(condition, value, old)) may be read as `value`.
std::optional<IR::Value> ResultExecGuard(const IR::Inst& inst) {
	std::optional<IR::Value> guard;
	if (!inst.HasUses()) {
		return std::nullopt;
	}
	for (const auto& use: inst.Uses()) {
		const auto* extract = use.user;
		if (extract == nullptr || extract->GetOpcode() != IR::ValueOpcode::CompositeExtractU32x4 ||
		    !extract->HasUses()) {
			return std::nullopt;
		}
		for (const auto& extract_use: extract->Uses()) {
			const auto* select = extract_use.user;
			if (select == nullptr || select->GetOpcode() != IR::ValueOpcode::SelectU32 ||
			    extract_use.operand != 1u) {
				return std::nullopt;
			}
			const auto condition = select->Arg(0).Resolve();
			if (guard.has_value() && !(*guard == condition)) {
				return std::nullopt;
			}
			guard = condition;
		}
	}
	return guard;
}

SampleOffset SampleOffsetFor(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                             const IR::Inst& address, const ImageSampleLayout& layout,
                             ImageDimension dimension, bool cube) {
	SampleOffset result;
	if (!GetCodegenOptions().sample_offsets || layout.offset == NoImageComponent ||
	    mem.image_address_components <= layout.offset) {
		return result;
	}
	if (cube) {
		static std::atomic_flag warned = ATOMIC_FLAG_INIT;
		if (!warned.test_and_set(std::memory_order_relaxed)) {
			std::fputs("Warning: IMAGE_SAMPLE*_O offsets on a cube map are ignored.\n", stderr);
		}
		return result;
	}
	auto&      state      = ctx.state;
	const auto components = ImageDimensionInfoFor(dimension).spatial_components;
	const auto offset_layout = Decoder::ImageAddressComponentLayout(mem.image_sample_flags,
	                                                              mem.image_dimension, layout.offset);
	const auto packed_index  = offset_layout.bit_offset / 32u;
	auto value =
	    packed_index < address.NumArgs() ? address.Arg(packed_index).Resolve() : IR::Value(0u);
	// Offsets are literals in guest code; under a non-uniform EXEC the register holds
	// Select(exec, literal, old), which the lanes that keep the result see as the literal.
	if (!value.IsImmediate()) {
		if (const auto guard = ResultExecGuard(inst); guard.has_value()) {
			while (!value.IsImmediate()) {
				const auto* select = value.TryInstruction();
				if (select == nullptr || select->GetOpcode() != IR::ValueOpcode::SelectU32 ||
				    !(select->Arg(0).Resolve() == *guard)) {
					break;
				}
				value = select->Arg(1).Resolve();
			}
		}
	}
	if (value.IsImmediate() &&
	    (value.GetType() == IR::Type::U32 || value.GetType() == IR::Type::F32)) {
		const auto bits = value.GetType() == IR::Type::U32 ? value.U32()
		                                                   : std::bit_cast<uint32_t>(value.F32Value());
		int32_t offsets[3] = {};
		bool    in_range   = true;
		bool    any        = false;
		for (uint32_t index = 0; index < components; index++) {
			offsets[index] = SignExtend6(bits >> (index * 8u));
			in_range       = in_range && offsets[index] >= -8 && offsets[index] <= 7;
			any            = any || offsets[index] != 0;
		}
		if (!any) {
			return result;
		}
		if (in_range) {
			uint32_t ids[3] = {};
			for (uint32_t index = 0; index < components; index++) {
				ids[index] = ConstantI32(state, offsets[index]);
			}
			if (components == 1u) {
				// A 1D image's host row (upstream e6cc1ce30) takes offset 0.
				result.const_offset = ImageDimensionInfoFor(dimension).one_dimensional
				                          ? state.builder.Constant(spv::OpConstantComposite,
				                                                   TypeI32Vector(state, 2), ids[0],
				                                                   ConstantI32(state, 0))
				                          : ids[0];
			} else if (components == 2u) {
				result.const_offset = state.builder.Constant(
				    spv::OpConstantComposite, TypeI32Vector(state, 2), ids[0], ids[1]);
			} else {
				result.const_offset = state.builder.Constant(
				    spv::OpConstantComposite, TypeI32Vector(state, 3), ids[0], ids[1], ids[2]);
			}
			return result;
		}
	}
	result.dynamic = true;
	result.packed  = AddressU32(ctx, mem, address, layout.offset);
	return result;
}

// Moves `coord` by the packed texel offsets divided by the size of the level `level_f32`
// (relative to the view's base, clamped to the view's levels). Exact for single-level reads up
// to float rounding; with linear mip filtering the coarser level moves by half the offset.
uint32_t ApplyDynamicSampleOffset(EmitterState& state, uint32_t resource, uint32_t coord,
                                  ImageDimension dimension, uint32_t packed, uint32_t level_f32) {
	static std::atomic_flag warned = ATOMIC_FLAG_INIT;
	if (!warned.test_and_set(std::memory_order_relaxed)) {
		std::fputs("Warning: IMAGE_SAMPLE*_O offsets that are not constant or lie outside -8..7 "
		           "move the coordinate by offset / level size.\n",
		           stderr);
	}
	state.builder.RequireCapability(spv::CapabilityImageQuery);
	const auto& info       = ImageDimensionInfoFor(dimension);
	const auto  components = info.spatial_components;
	const auto  image      = LoadImageDescriptor(state, resource);
	const auto  levels     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQueryLevels, TypeU32(state), levels, image);
	const auto floored = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), floored, GlslStd450(state),
	                          GLSLstd450FMax, level_f32, ConstantF32Value(state, 0.0f));
	const auto level_float = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), level_float, GlslStd450(state),
	                          GLSLstd450Floor, floored);
	const auto level_u32 = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), level_u32, level_float);
	const auto last  = Binary(state, spv::OpISub, TypeU32(state), levels, ConstantU32(state, 1));
	const auto level = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), level, GlslStd450(state),
	                          GLSLstd450UMin, level_u32, last);
	const auto size = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQuerySizeLod, ImageViewSizeType(state, dimension), size,
	                          image, level);
	const auto packed_i32 = Unary(state, spv::OpBitcast, TypeI32(state), packed);
	uint32_t   moved[4]   = {};
	// A 1D image's coordinate and size carry the host row as component 1 (upstream e6cc1ce30).
	const uint32_t host_count = info.coordinate_components + (info.one_dimensional ? 1u : 0u);
	for (uint32_t host = 0; host < host_count; host++) {
		const bool     row   = info.one_dimensional && host == 1u;
		const uint32_t index = info.one_dimensional && host > 1u ? host - 1u : host;
		auto           component = coord;
		if (host_count > 1u) {
			component = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, coord,
			                          host);
		}
		if (!row && index < components) {
			const auto offset = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), offset, packed_i32,
			                          ConstantU32(state, index * 8u), ConstantU32(state, 6));
			const auto offset_f32 = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertSToF, TypeF32(state), offset_f32, offset);
			auto extent = size;
			if (host_count > 1u) {
				extent = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), extent, size,
				                          host);
			}
			const auto extent_f32 = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), extent_f32, extent);
			component = Binary(state, spv::OpFAdd, TypeF32(state), component,
			                   Binary(state, spv::OpFDiv, TypeF32(state), offset_f32, extent_f32));
		}
		moved[host] = component;
	}
	if (host_count == 1u) {
		return moved[0];
	}
	const auto result = state.builder.AllocateId();
	if (host_count == 2u) {
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), result,
		                          moved[0], moved[1]);
	} else {
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 3), result,
		                          moved[0], moved[1], moved[2]);
	}
	return result;
}

uint32_t HorizontalOffsets(EmitterState& state) {
	const auto count      = ConstantU32(state, 4);
	const auto array_type = state.builder.Type(spv::OpTypeArray, TypeI32Vector(state, 2), count);
	uint32_t   offsets[4] {};
	for (uint32_t index = 0; index < 4u; index++) {
		offsets[index] = state.builder.Constant(spv::OpConstantComposite, TypeI32Vector(state, 2),
		                                        ConstantI32(state, static_cast<int32_t>(index) - 1),
		                                        ConstantI32(state, 0));
	}
	return state.builder.Constant(spv::OpConstantComposite, array_type, offsets[0], offsets[1],
	                              offsets[2], offsets[3]);
}

uint32_t InverseSwizzle(uint32_t swizzle, uint32_t component) {
	for (uint32_t source = 0; source < 4u; source++) {
		if (((swizzle >> (source * 3u)) & 7u) == 4u + component) return source;
	}
	return UINT32_MAX;
}

Format::BufferFormatInfo ImageConversionFormat(const EmitterState&   state,
                                               const IR::MemoryInfo& mem) {
	const auto format = ImageResourceAt(state, mem.resource).conversion_format;
	if (format == Prospero::BufferFormat::kInvalid) return {};
	const auto info = Format::GetFormatInfo(format);
	if (IR::ReadsDepthBits(ImageResourceAt(state, mem.resource)) || Format::IsSrgbFormat(format)) {
		return info;
	}
	EXIT_IF(Prospero::RemapTextureFormat(format) == format || info.component_count == 0u ||
	        info.component_count > 4u);
	// Non-packed: scaled formats on a normalized backing. Packed: the raw word in a k32UInt
	// backing (KYTY_PACKED_TEXTURES decodes UNorm/UScaled ones too).
	EXIT_IF(info.packed_bitfield
	            ? info.type == Format::ComponentType::Float || info.byte_size != sizeof(uint32_t)
	            : info.type != Format::ComponentType::Uscaled &&
	                  info.type != Format::ComponentType::Sscaled);
	return info;
}

bool PackedFloatResult(const Format::BufferFormatInfo& info) {
	return info.type != Format::ComponentType::Uint && info.type != Format::ComponentType::Sint;
}

// Bits of a decoded value as ResultVector consumes them for a uint image: 32-bit words, or the
// half in the low 16 bits for a D16 read.
uint32_t PackedResultBits(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t value_f32) {
	if (mem.data_bits != 16u) {
		return Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), value_f32);
	}
	const auto pair = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(ctx.state, 2), pair,
	                              value_f32, ZeroF32(ctx.state));
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpExtInst, TypeU32(ctx.state), result, GlslStd450(ctx.state),
	                              GLSLstd450PackHalf2x16, pair);
	return result;
}

uint32_t PackedOneBits(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                       const Format::BufferFormatInfo& info) {
	if (!PackedFloatResult(info)) {
		return ConstantU32(ctx.state, 1u);
	}
	return ConstantU32(ctx.state, mem.data_bits == 16u ? 0x3c00u : 0x3f800000u);
}

// One component of a packed word, decoded to the guest's number format.
uint32_t DecodePackedComponent(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                               const Format::BufferFormatInfo& info, uint32_t packed,
                               uint32_t component) {
	using Format::ComponentType;
	const auto bits = info.component_bits[component];
	const bool sign = info.type == ComponentType::Snorm || info.type == ComponentType::Sscaled ||
	                  info.type == ComponentType::Sint;
	const auto value = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(
	    sign ? spv::OpBitFieldSExtract : spv::OpBitFieldUExtract, TypeU32(ctx.state), value, packed,
	    ConstantU32(ctx.state, info.component_bit_offset[component]), ConstantU32(ctx.state, bits));
	if (!PackedFloatResult(info)) {
		return value;
	}
	auto converted =
	    Unary(ctx.state, sign ? spv::OpConvertSToF : spv::OpConvertUToF, TypeF32(ctx.state), value);
	if (info.type == ComponentType::Unorm) {
		converted = Binary(ctx.state, spv::OpFDiv, TypeF32(ctx.state), converted,
		                   ConstantF32Value(ctx.state, float((1u << bits) - 1u)));
	} else if (info.type == ComponentType::Snorm) {
		converted          = Binary(ctx.state, spv::OpFDiv, TypeF32(ctx.state), converted,
		                            ConstantF32Value(ctx.state, float((1u << (bits - 1u)) - 1u)));
		const auto clamped = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpExtInst, TypeF32(ctx.state), clamped,
		                              GlslStd450(ctx.state), GLSLstd450FMax, converted,
		                              ConstantF32Value(ctx.state, -1.0f));
		converted = clamped;
	}
	return PackedResultBits(ctx, mem, converted);
}

// Normalized backing value back to the scaled integer it stands for.
float ScaledFactor(const Format::BufferFormatInfo& info) {
	const auto bits = info.component_bits[0];
	return info.type == Format::ComponentType::Sscaled ? float((1u << (bits - 1u)) - 1u)
	                                                   : float((1u << bits) - 1u);
}

uint32_t ImageGatherSource(const EmitterState& state, const IR::MemoryInfo& mem) {
	const auto component = ImageGatherComponent(mem.dmask);
	const auto info      = ImageConversionFormat(state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return component;
	if (info.packed_bitfield) return 0u;
	const auto selector =
	    (ImageResourceAt(state, mem.resource).shader_swizzle >> (component * 3u)) & 7u;
	return selector >= 4u ? (selector - 4u) % info.component_count : 0u;
}

// The sRGB EOTF of the PS5 sampler for one UNorm-read value (Erlou93/KytyPS5-Yotei SrgbToLinear;
// 8 / 8_8 sRGB textures, Format::IsSrgbFormat, KYTY_SRGB8_DECODE).
uint32_t SrgbToLinear(EmitterState& state, uint32_t value) {
	const auto type   = TypeF32(state);
	const auto linear = Binary(state, spv::OpFMul, type, value, ConstantF32Value(state, 1.0f / 12.92f));
	const auto base   = Binary(state, spv::OpFMul, type,
	                           Binary(state, spv::OpFAdd, type, value, ConstantF32Value(state, 0.055f)),
	                           ConstantF32Value(state, 1.0f / 1.055f));
	const auto curve  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, type, curve, GlslStd450(state), GLSLstd450Pow, base,
	                          ConstantF32Value(state, 2.4f));
	const auto low    = Binary(state, spv::OpFOrdLessThanEqual, TypeBool(state), value,
	                           ConstantF32Value(state, 0.04045f));
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, type, result, low, linear, curve);
	return result;
}

uint32_t UnpackImageTexel(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t texel) {
	const auto info = ImageConversionFormat(ctx.state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return texel;
	if (Format::IsSrgbFormat(info.format)) {
		// UNorm backing read as float: decode the colour channels, then the guest swizzle.
		uint32_t components[4] {};
		for (uint32_t component = 0; component < info.component_count; component++) {
			const auto value = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(ctx.state), value, texel,
			                              component);
			components[component] = SrgbToLinear(ctx.state, value);
		}
		const auto swizzle = ImageResourceAt(ctx.state, mem.resource).shader_swizzle;
		uint32_t   selected[4] {};
		for (uint32_t component = 0; component < 4u; component++) {
			const auto selector = (swizzle >> (component * 3u)) & 7u;
			selected[component] = selector == 1u ? ConstantF32Value(ctx.state, 1.0f)
			                      : selector >= 4u && selector - 4u < info.component_count
			                          ? components[selector - 4u]
			                          : ZeroF32(ctx.state);
		}
		const auto result = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(ctx.state, 4), result,
		                              selected[0], selected[1], selected[2], selected[3]);
		return result;
	}

	const auto numeric_class = ImageResourceAt(ctx.state, mem.resource).numeric_class;
	const auto scalar_type   = ImageScalarType(ctx.state, numeric_class);
	// KYTY_DEPTH_BITS_READ (upstream aafe014d0): the UNorm D16 value back to its stored code.
	const bool depth_bits    = IR::ReadsDepthBits(ImageResourceAt(ctx.state, mem.resource));
	uint32_t   packed        = 0;
	if (info.packed_bitfield) {
		packed = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, scalar_type, packed, texel, 0u);
	}
	uint32_t components[4] {};
	for (uint32_t component = 0; component < info.component_count; component++) {
		if (info.packed_bitfield && PackedFloatResult(info)) {
			components[component] = DecodePackedComponent(ctx, mem, info, packed, component);
			continue;
		}
		components[component] = ctx.state.builder.AllocateId();
		if (info.packed_bitfield) {
			ctx.state.builder.AddFunction(spv::OpBitFieldUExtract, scalar_type,
			                              components[component], packed,
			                              ConstantU32(ctx.state, info.component_bit_offset[component]),
			                              ConstantU32(ctx.state, info.component_bits[component]));
		} else {
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, scalar_type,
			                              components[component], texel, component);
			// Normalized backing preserves the guest's filtering; scaling precedes swizzle
			// constants.
			components[component] =
			    Binary(ctx.state, spv::OpFMul, scalar_type, components[component],
			           ConstantF32Value(ctx.state, ScaledFactor(info)));
			if (depth_bits) {
				// Round back to the stored UNorm code; its bits travel in the float vector.
				components[component] = Unary(
				    ctx.state, spv::OpBitcast, TypeF32(ctx.state),
				    Unary(ctx.state, spv::OpConvertFToU, TypeU32(ctx.state),
				          Binary(ctx.state, spv::OpFAdd, scalar_type, components[component],
				                 ConstantF32Value(ctx.state, 0.5f))));
			}
		}
	}
	for (uint32_t component = info.component_count; component < 4u; component++) {
		components[component] = components[component % info.component_count];
	}

	const auto swizzle = ImageResourceAt(ctx.state, mem.resource).shader_swizzle;
	uint32_t   selected[4] {};
	for (uint32_t component = 0; component < 4u; component++) {
		const auto selector = (swizzle >> (component * 3u)) & 7u;
		if (selector == 1u) {
			selected[component] = info.packed_bitfield ? PackedOneBits(ctx, mem, info)
			                      : depth_bits ? Unary(ctx.state, spv::OpBitcast, TypeF32(ctx.state),
			                                           ConstantU32(ctx.state, 1u))
			                                   : ConstantF32Value(ctx.state, 1.0f);
		} else if (selector >= 4u) {
			selected[component] = components[selector - 4u];
		} else {
			selected[component] = info.packed_bitfield ? ConstantU32(ctx.state, 0u)
			                                          : ZeroF32(ctx.state);
		}
	}
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct,
	                              ImageVectorType(ctx.state, numeric_class, 4), result, selected[0],
	                              selected[1], selected[2], selected[3]);
	return result;
}

uint32_t UnpackImageGather(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t gathered) {
	const auto info = ImageConversionFormat(ctx.state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return gathered;

	const auto component = ImageGatherComponent(mem.dmask);
	const auto selector =
	    (ImageResourceAt(ctx.state, mem.resource).shader_swizzle >> (component * 3u)) & 7u;
	const auto numeric_class = ImageResourceAt(ctx.state, mem.resource).numeric_class;
	const auto vector_type   = ImageVectorType(ctx.state, numeric_class, 4);
	if (selector < 4u) {
		uint32_t value;
		if (!info.packed_bitfield) {
			value = ConstantF32Value(ctx.state, selector == 1u ? 1.0f : 0.0f);
		} else {
			value = selector == 1u ? PackedOneBits(ctx, mem, info) : ConstantU32(ctx.state, 0u);
		}
		return ctx.state.builder.Constant(spv::OpConstantComposite, vector_type, value, value,
		                                  value, value);
	}
	if (Format::IsSrgbFormat(info.format)) {
		// One stored channel of four texels, each decoded from sRGB (every 8 / 8_8 channel is colour).
		uint32_t texels[4] {};
		for (uint32_t index = 0; index < 4u; index++) {
			const auto value = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(ctx.state), value, gathered,
			                              index);
			texels[index] = SrgbToLinear(ctx.state, value);
		}
		const auto result = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, vector_type, result, texels[0],
		                              texels[1], texels[2], texels[3]);
		return result;
	}
	if (!info.packed_bitfield) {
		const auto scale  = ConstantF32Value(ctx.state, ScaledFactor(info));
		const auto scales = ctx.state.builder.Constant(spv::OpConstantComposite, vector_type,
		                                                scale, scale, scale, scale);
		return Binary(ctx.state, spv::OpFMul, vector_type, gathered, scales);
	}

	uint32_t values[4] {};
	for (uint32_t lane = 0; lane < 4u; lane++) {
		const auto packed = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), packed, gathered,
		                              lane);
		const auto physical = (selector - 4u) % info.component_count;
		if (PackedFloatResult(info)) {
			values[lane] = DecodePackedComponent(ctx, mem, info, packed, physical);
			continue;
		}
		values[lane] = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(ctx.state), values[lane],
		                              packed,
		                              ConstantU32(ctx.state, info.component_bit_offset[physical]),
		                              ConstantU32(ctx.state, info.component_bits[physical]));
	}
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4), result,
	                              values[0], values[1], values[2], values[3]);
	return result;
}

uint32_t PackImageTexel(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t texel) {
	const auto info = ImageConversionFormat(ctx.state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return texel;
	EXIT_IF(info.type != Format::ComponentType::Uint || !info.packed_bitfield);

	auto packed = ConstantU32(ctx.state, 0u);
	for (uint32_t component = 0; component < info.component_count; component++) {
		const auto value = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), value, texel,
		                              component);
		const auto maximum =
		    ConstantU32(ctx.state, info.component_bits[component] == 32u
		                               ? UINT32_MAX
		                               : (1u << info.component_bits[component]) - 1u);
		const auto within =
		    Binary(ctx.state, spv::OpULessThan, TypeBool(ctx.state), value, maximum);
		const auto clamped = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpSelect, TypeU32(ctx.state), clamped, within, value,
		                              maximum);
		const auto shifted =
		    info.component_bit_offset[component] == 0u
		        ? clamped
		        : Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state), clamped,
		                 ConstantU32(ctx.state, info.component_bit_offset[component]));
		packed = Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state), packed, shifted);
	}
	const auto zero   = ConstantU32(ctx.state, 0u);
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4), result,
	                              packed, zero, zero, zero);
	return result;
}

uint32_t StoreTexel(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t data, bool integer) {
	const auto swizzle = ImageResourceAt(ctx.state, mem.resource).shader_swizzle;
	uint32_t   values[4] {};
	const auto dmask = mem.dmask != 0u ? mem.dmask : 1u;
	for (uint32_t component = 0; component < 4u; component++) {
		const auto source = InverseSwizzle(swizzle, component);
		uint32_t   raw    = ConstantU32(ctx.state, 0);
		if (source < 4u && ((dmask >> source) & 1u) != 0u) {
			const auto packed_index = DmaskComponentIndex(dmask, source);
			raw                     = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), raw, data,
			                              mem.data_bits == 16u ? packed_index / 2u : packed_index);
			if (mem.data_bits == 16u) {
				if ((packed_index & 1u) != 0u) {
					raw = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), raw,
					             ConstantU32(ctx.state, 16u));
				}
				raw = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), raw,
				             ConstantU32(ctx.state, 0xffffu));
			}
		}
		if (!integer && mem.data_bits != 16u &&
		    ImageResourceAt(ctx.state, mem.resource).clamp_half_float) {
			// KYTY_FLOAT_STORE_CLAMP: a 16-bit float storage image keeps the guest's range.
			raw = EmitClampToHalfRangeBits(ctx.state, raw);
		}
		values[component] = integer ? raw
		                    : mem.data_bits == 16u
		                        ? EmitF16BitsToF32(ctx.state, raw)
		                        : Unary(ctx.state, spv::OpBitcast, TypeF32(ctx.state), raw);
	}
	const auto texel = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct,
	                              integer ? TypeU32Vector(ctx.state, 4)
	                                      : TypeF32Vector(ctx.state, 4),
	                              texel, values[0], values[1], values[2], values[3]);
	return PackImageTexel(ctx, mem, texel);
}

spv::Op ImageAtomicOpcode(IR::ValueOpcode opcode) {
	switch (opcode) {
		case IR::ValueOpcode::ImageAtomicSwap32:
		case IR::ValueOpcode::ImageAtomicSwap64: return spv::OpAtomicExchange;
		case IR::ValueOpcode::ImageAtomicIAdd32:
		case IR::ValueOpcode::ImageAtomicIAdd64: return spv::OpAtomicIAdd;
		case IR::ValueOpcode::ImageAtomicSMin32: return spv::OpAtomicSMin;
		case IR::ValueOpcode::ImageAtomicUMin32:
		case IR::ValueOpcode::ImageAtomicUMin64: return spv::OpAtomicUMin;
		case IR::ValueOpcode::ImageAtomicSMax32: return spv::OpAtomicSMax;
		case IR::ValueOpcode::ImageAtomicUMax32:
		case IR::ValueOpcode::ImageAtomicUMax64: return spv::OpAtomicUMax;
		case IR::ValueOpcode::ImageAtomicAnd32:
		case IR::ValueOpcode::ImageAtomicAnd64: return spv::OpAtomicAnd;
		case IR::ValueOpcode::ImageAtomicOr32:
		case IR::ValueOpcode::ImageAtomicOr64: return spv::OpAtomicOr;
		case IR::ValueOpcode::ImageAtomicXor32:
		case IR::ValueOpcode::ImageAtomicXor64: return spv::OpAtomicXor;
		case IR::ValueOpcode::ImageAtomicISub32: return spv::OpAtomicISub;
		// R32ui texel pointers: SMin/SMax interpret the unsigned bits as signed.
		// Compare-exchange and wrapping inc/dec are expanded by EmitImage.
		case IR::ValueOpcode::ImageAtomicCmpSwap32: return spv::OpAtomicCompareExchange;
		case IR::ValueOpcode::ImageAtomicInc32:
		case IR::ValueOpcode::ImageAtomicDec32: return spv::OpAtomicCompareExchangeWeak;
		default: return spv::OpNop;
	}
}

} // namespace

// GET_LOD_STATS feedback. The per-draw shader data holds one field per image
// (LodStatsReport::ImageField). The mip_stats buffer holds 257 finest-mip words followed by 257
// counts; entry 256 absorbs images without a counter. The field is uniform per draw, so one lane
// per subgroup records the subgroup's finest level and whether any lane counts.
constexpr uint32_t MipStatsEntries = 257;

// 32-bit field: counter id in bits 0..7, T# BASE_LEVEL in bits 8..11, bit 15 = no counter, U4.8
// count threshold in bits 16..27.
uint32_t LoadMipStatsId(EmitterState& state, uint32_t resource) {
	return EmitShaderDataDwordLoad(state, state.program.bindings.MipStatsOffsetDword() + resource);
}

struct MipStatsSample {
	uint32_t finest  = 0; // subgroup minimum of the absolute mip level
	uint32_t counted = 0; // 1 when a lane's level is below the field's threshold
};

// The finest mip a sample wanted, as an absolute level: floor of the unclamped LOD plus the view's
// BASE_LEVEL, limited to 0..14, reduced to the subgroup minimum. A lane counts when that LOD is
// below the field's threshold: the T# MIN_LOD, i.e. the resident-level clamp raised the sample
// (reported in bits 0..23, "MipClamp" in Astro Bot's streamer), or every sample with
// KYTY_LOD_STATS_COUNT=samples (threshold beyond level 14).
MipStatsSample EmitMipStatsSample(EmitterState& state, uint32_t id, uint32_t lod) {
	state.builder.RequireCapability(spv::CapabilityGroupNonUniform);
	state.builder.RequireCapability(spv::CapabilityGroupNonUniformArithmetic);
	const auto base_level = Binary(
	    state, spv::OpBitwiseAnd, TypeU32(state),
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), id, ConstantU32(state, 8)),
	    ConstantU32(state, 0xfu));
	const auto base_f32 = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), base_f32, base_level);
	// The query LOD is relative to the view's base level (T# BASE_LEVEL); report absolute mips.
	const auto absolute = Binary(state, spv::OpFAdd, TypeF32(state), lod, base_f32);
	const auto clamped  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), clamped, GlslStd450(state),
	                          GLSLstd450FClamp, absolute, ConstantF32Value(state, 0.0f),
	                          ConstantF32Value(state, 14.0f));
	const auto floored = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), floored, GlslStd450(state),
	                          GLSLstd450Floor, clamped);
	const auto level = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), level, floored);
	MipStatsSample sample;
	sample.finest = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformUMin, TypeU32(state), sample.finest,
	                          ConstantU32(state, spv::ScopeSubgroup), spv::GroupOperationReduce,
	                          level);
	// Threshold in U4.8. Compared with the level clamped to 0..14, so that magnification (a LOD
	// below 0) is not a MIN_LOD clamp of a texture whose MIN_LOD is 0.
	const auto threshold_fixed = Binary(
	    state, spv::OpBitwiseAnd, TypeU32(state),
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), id, ConstantU32(state, 16)),
	    ConstantU32(state, 0xfffu));
	const auto threshold_f32 = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), threshold_f32, threshold_fixed);
	const auto threshold = Binary(state, spv::OpFMul, TypeF32(state), threshold_f32,
	                              ConstantF32Value(state, 1.0f / 256.0f));
	const auto below = Binary(state, spv::OpFOrdLessThan, TypeBool(state), clamped, threshold);
	const auto below_u32 = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), below_u32, below,
	                          ConstantU32(state, 1), ConstantU32(state, 0));
	sample.counted = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformUMax, TypeU32(state), sample.counted,
	                          ConstantU32(state, spv::ScopeSubgroup), spv::GroupOperationReduce,
	                          below_u32);
	return sample;
}

uint32_t MipStatsElement(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.mip_stats_variable, ConstantU32(state, 0), index);
	return pointer;
}

// Adds 1 to entry `index`'s count when a lane of the subgroup counted (one per subgroup and
// sample instruction; the guest only tests the count for zero).
void EmitMipStatsCount(EmitterState& state, uint32_t index, uint32_t counted) {
	const auto any =
	    Binary(state, spv::OpINotEqual, TypeBool(state), counted, ConstantU32(state, 0));
	EmitIfCondition(state, any, [&]() {
		const auto count_index =
		    Binary(state, spv::OpIAdd, TypeU32(state), index, ConstantU32(state, MipStatsEntries));
		const auto add_result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), add_result,
		                          MipStatsElement(state, count_index),
		                          ConstantU32(state, spv::ScopeDevice),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone),
		                          ConstantU32(state, 1));
	});
}

// KYTY_LOD_STATS_GATE (default): recording for an image whose T# has a counter (checked by the
// caller, uniformly per draw). Per subgroup, one lane
//  - issues the finest-level AtomicUMin only when a relaxed atomic load shows a larger value.
//    The finest words only ever decrease between resets (UMin is the only writer; the resets
//    are transfer fills ordered before and after every draw by pipeline barriers), so a value
//    at or below `finest` stays at or below it and the skipped UMin would not change the word.
//    A stale larger value just issues the UMin as before.
//  - adds 1 to the count when a lane counted (EmitMipStatsCount).
// Images without a counter recorded into entry 256, which the host never reads; they now record
// nothing.
void EmitGatedMipStatsRecord(EmitterState& state, uint32_t id, uint32_t lod) {
	const auto counter =
	    Binary(state, spv::OpBitwiseAnd, TypeU32(state), id, ConstantU32(state, 0xffu));
	const auto sample  = EmitMipStatsSample(state, id, lod);
	const auto elected = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformElect, TypeBool(state), elected,
	                          ConstantU32(state, spv::ScopeSubgroup));
	EmitIfCondition(state, elected, [&]() {
		const auto finest_pointer = MipStatsElement(state, counter);
		const auto current        = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicLoad, TypeU32(state), current, finest_pointer,
		                          ConstantU32(state, spv::ScopeDevice),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone));
		const auto lower =
		    Binary(state, spv::OpULessThan, TypeBool(state), sample.finest, current);
		EmitIfCondition(state, lower, [&]() {
			const auto min_result = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAtomicUMin, TypeU32(state), min_result, finest_pointer,
			                          ConstantU32(state, spv::ScopeDevice),
			                          ConstantU32(state, spv::MemorySemanticsMaskNone),
			                          sample.finest);
		});
		EmitMipStatsCount(state, counter, sample.counted);
	});
}

void EmitMipStatsRecord(EmitterState& state, uint32_t resource, uint32_t lod) {
	if (state.mip_stats_variable == 0 || !state.mip_stats_records ||
	    resource >= state.program.bindings.mip_stats_count) {
		return;
	}
	constexpr uint32_t Entries = MipStatsEntries;
	const auto id = LoadMipStatsId(state, resource);
	const auto disabled = Binary(state, spv::OpBitwiseAnd, TypeU32(state), id,
	                             ConstantU32(state, 0x8000u));
	const auto has_counter =
	    Binary(state, spv::OpIEqual, TypeBool(state), disabled, ConstantU32(state, 0));
	const auto counter =
	    Binary(state, spv::OpBitwiseAnd, TypeU32(state), id, ConstantU32(state, 0xffu));
	const auto slot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), slot, has_counter, counter,
	                          ConstantU32(state, Entries - 1u));
	const auto sample  = EmitMipStatsSample(state, id, lod);
	const auto elected = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformElect, TypeBool(state), elected,
	                          ConstantU32(state, spv::ScopeSubgroup));
	EmitIfCondition(state, elected, [&]() {
		const auto min_result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicUMin, TypeU32(state), min_result,
		                          MipStatsElement(state, slot), ConstantU32(state, spv::ScopeDevice),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone), sample.finest);
		EmitMipStatsCount(state, slot, sample.counted);
	});
}

// The bindless slot of the image handle's key: translation[region + key] when the key is inside
// the heap's entry count (both from the flattened SRT, patched by the host), otherwise and while
// the texture is pending, a placeholder slot. A pending key is flagged in the feedback buffer for
// the host to resolve; every lane that reads an entry sees the same translation, so the
// unconditional store never races with a different value.
static uint32_t BindlessSlot(ValueEmitContext& ctx, const IR::ImageResource& image,
                             IR::Value image_arg) {
	auto&       state  = ctx.state;
	const auto* handle = image_arg.ResolveInstruction();
	EXIT_IF(handle == nullptr || handle->NumArgs() == 0u || state.flattened_srt_variable == 0 ||
	        state.bindless_translation_variable == 0);
	const auto key  = ctx.Def(handle->Arg(0));
	const auto Load = [&](uint32_t variable, uint32_t index) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		                          variable == state.flattened_srt_variable
		                              ? FlattenedSrtElementPointerType(state)
		                              : TypeStorageBufferElementPointer(state),
		                          pointer, variable, ConstantU32(state, 0), index);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	const auto base     = Load(state.flattened_srt_variable,
	                           ConstantU32(state, image.indirect_mapping_offset));
	const auto count    = Load(state.flattened_srt_variable,
	                           ConstantU32(state, image.indirect_mapping_offset + 1u));
	const auto in_range = Binary(state, spv::OpULessThan, TypeBool(state), key, count);
	const auto entry    = Binary(state, spv::OpIAdd, TypeU32(state), base, key);
	const auto index    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), index, in_range, entry,
	                          ConstantU32(state, 0u));
	const auto raw     = Load(state.bindless_translation_variable, index);
	const auto pending = Binary(state, spv::OpIEqual, TypeBool(state), raw,
	                            ConstantU32(state, IR::BindlessPending));
	if (state.bindless_feedback_variable != 0) {
		const auto pending_flag = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), pending_flag, pending,
		                          ConstantU32(state, 1u), ConstantU32(state, 0u));
		// Diagnostics: a key outside its heap leaves itself (with the top bit set) in
		// feedback word 0, which belongs to no heap.
		const auto marked_key = Binary(state, spv::OpBitwiseOr, TypeU32(state), key,
		                               ConstantU32(state, 0x80000000u));
		const auto flag = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), flag, in_range, pending_flag,
		                          marked_key);
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.bindless_feedback_variable, ConstantU32(state, 0),
		                          index);
		state.builder.AddFunction(spv::OpStore, pointer, flag);
		// A raised flag also marks its 1024-entry chunk in the summary words after word 0
		// (IR::BindlessFeedbackSummary), so the host copies and scans only marked chunks.
		const auto raised = Binary(state, spv::OpINotEqual, TypeBool(state), flag, ConstantU32(state, 0u));
		EmitIfCondition(state, raised, [&]() {
			const auto chunk = Binary(state, spv::OpShiftRightLogical, TypeU32(state), index,
			                          ConstantU32(state, IR::BindlessFeedbackChunkShift));
			const auto word  = Binary(state, spv::OpIAdd, TypeU32(state), chunk,
			                          ConstantU32(state, IR::BindlessFeedbackSummary));
			const auto mark  = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), mark,
			                          state.bindless_feedback_variable, ConstantU32(state, 0), word);
			state.builder.AddFunction(spv::OpStore, mark, ConstantU32(state, 1u));
		});
	}
	state.bindless_translation_raw = raw;
	const auto slot = state.builder.AllocateId();
	// Pending keys sample a placeholder while their texture is loaded.
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), slot, pending,
	                          ConstantU32(state, IR::BindlessPendingSlot), raw);
	state.builder.AddAnnotation(spv::OpDecorate, slot, spv::DecorationNonUniform);
	return slot;
}

// KYTY_BINDLESS_SAMPLERS: the sampler array slot of the sampler handle's key: region + key while
// the key is inside the heap's record count (both from the flattened SRT, patched by the host),
// otherwise slot 0, the stand-in sampler.
static uint32_t BindlessSamplerSlot(ValueEmitContext& ctx, const IR::SamplerResource& sampler,
                                    IR::Value sampler_arg) {
	auto&       state  = ctx.state;
	const auto* handle = sampler_arg.ResolveInstruction();
	EXIT_IF(handle == nullptr || handle->NumArgs() == 0u || state.flattened_srt_variable == 0 ||
	        state.bindless_sampler_variable == 0);
	const auto key  = ctx.Def(handle->Arg(0));
	const auto Load = [&](uint32_t index) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.flattened_srt_variable, ConstantU32(state, 0),
		                          ConstantU32(state, index));
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	const auto base     = Load(sampler.bindless_mapping_offset);
	const auto count    = Load(sampler.bindless_mapping_offset + 1u);
	const auto in_range = Binary(state, spv::OpULessThan, TypeBool(state), key, count);
	const auto entry    = Binary(state, spv::OpIAdd, TypeU32(state), base, key);
	const auto slot     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), slot, in_range, entry,
	                          ConstantU32(state, 0u));
	state.builder.AddAnnotation(spv::OpDecorate, slot, spv::DecorationNonUniform);
	return slot;
}

// KYTY_RUNTIME_IMAGES (adapted from usa-ratt/KytyPS5-GT7 2fa6c6514): runtime images
// (ImageResource::runtime); the handle holds the eight T# dwords the shader loaded.
static std::array<uint32_t, 8> RuntimeImageDwords(ValueEmitContext& ctx, IR::Value image_arg) {
	const auto* handle = image_arg.ResolveInstruction();
	EXIT_IF(handle == nullptr || handle->NumArgs() != 8u);
	std::array<uint32_t, 8> dwords {};
	for (uint32_t dword = 0; dword < 8u; dword++) {
		dwords[dword] = ctx.Def(handle->Arg(dword));
	}
	return dwords;
}

// The set-1 array a runtime image's instruction reads (as DefineBindlessImages assigns it).
static uint32_t RuntimeImageBinding(const IR::ImageResource& image) {
	if (image.cube) return IR::BindlessImagesCube;
	if (image.dimension == ImageDimension::Dim3D) return IR::BindlessImages3D;
	if (image.dimension == ImageDimension::Dim2DArray ||
	    image.dimension == ImageDimension::Dim2DMsaaArray)
		return IR::BindlessImages2DArray;
	return IR::BindlessImages2D;
}

static uint32_t RuntimeLoadWord(EmitterState& state, uint32_t variable, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          variable, ConstantU32(state, 0), index);
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

static void RuntimeStoreWord(EmitterState& state, uint32_t variable, uint32_t index,
                             uint32_t value) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          variable, ConstantU32(state, 0), index);
	state.builder.AddFunction(spv::OpStore, pointer, value);
}

static uint32_t RuntimeAtomicWord(EmitterState& state, spv::Op op, uint32_t variable,
                                  uint32_t index, uint32_t value) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          variable, ConstantU32(state, 0), index);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(op, TypeU32(state), result, pointer,
	                          ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0), value);
	return result;
}

// Whether the T#'s format samples as float, from a table of the 512 formats (one bit each),
// selected with a switch on the word. The bindless arrays hold float views only; another class
// samples the pending placeholder.
static uint32_t RuntimeImageFloat(EmitterState& state, const std::array<uint32_t, 8>& dwords) {
	static const std::array<uint32_t, 16> table = [] {
		std::array<uint32_t, 16> words {};
		for (uint32_t format = 0; format < 512u; format++) {
			if (Prospero::SampledTextureNumericClass(static_cast<Prospero::BufferFormat>(format)) ==
			    Prospero::TextureNumericClass::Float) {
				words[format / 32u] |= 1u << (format % 32u);
			}
		}
		return words;
	}();
	const auto u32    = TypeU32(state);
	const auto format = EmitBitFieldUExtract(state, dwords[1], ConstantU32(state, 20),
	                                         ConstantU32(state, 9));
	const auto word   = EmitIndexSwitch(
	    state, Binary(state, spv::OpShiftRightLogical, u32, format, ConstantU32(state, 5)), 16u, u32,
	    [&](uint32_t index) { return ConstantU32(state, table[index]); });
	const auto bit = Binary(state, spv::OpBitwiseAnd, u32,
	                        Binary(state, spv::OpShiftRightLogical, u32, word,
	                               Binary(state, spv::OpBitwiseAnd, u32, format,
	                                      ConstantU32(state, 31))),
	                        ConstantU32(state, 1));
	return Binary(state, spv::OpINotEqual, TypeBool(state), bit, ConstantU32(state, 0));
}

// KYTY_RUNTIME_INT_IMAGES: the component class of the T#'s format (0 float, 1 uint, 2 sint, 3
// none), two bits per format from a table of the 512 formats, selected with a switch on the word.
static uint32_t RuntimeImageNumericClass(EmitterState& state, const std::array<uint32_t, 8>& dwords) {
	static const std::array<uint32_t, 32> table = [] {
		std::array<uint32_t, 32> words {};
		for (uint32_t format = 0; format < 512u; format++) {
			uint32_t value = 3u;
			switch (Prospero::SampledTextureNumericClass(static_cast<Prospero::BufferFormat>(format))) {
				case Prospero::TextureNumericClass::Float: value = 0u; break;
				case Prospero::TextureNumericClass::Uint: value = 1u; break;
				case Prospero::TextureNumericClass::Sint: value = 2u; break;
				case Prospero::TextureNumericClass::Unsupported: value = 3u; break;
			}
			words[format / 16u] |= value << ((format % 16u) * 2u);
		}
		return words;
	}();
	const auto u32    = TypeU32(state);
	const auto format = EmitBitFieldUExtract(state, dwords[1], ConstantU32(state, 20),
	                                         ConstantU32(state, 9));
	const auto word   = EmitIndexSwitch(
	    state, Binary(state, spv::OpShiftRightLogical, u32, format, ConstantU32(state, 4)), 32u, u32,
	    [&](uint32_t index) { return ConstantU32(state, table[index]); });
	const auto shift = Binary(state, spv::OpShiftLeftLogical, u32,
	                          Binary(state, spv::OpBitwiseAnd, u32, format, ConstantU32(state, 15)),
	                          ConstantU32(state, 1));
	return Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpShiftRightLogical, u32, word, shift),
	              ConstantU32(state, 3));
}

// The bindless slot of a runtime image's T#. The runtime image table maps the eight dwords and
// the binding to a translation entry, which holds the slot as a heap key's entry does: pending
// entries flag the feedback buffer and sample the pending placeholder. A T# the table lacks
// samples it too, and is requested in the request buffer by the first invocation that misses its
// home entry, for the host to resolve.
static uint32_t RuntimeImageSlot(ValueEmitContext& ctx, const IR::ImageResource& image,
                                 const std::array<uint32_t, 8>& dwords) {
	auto& state = ctx.state;
	EXIT_IF(state.runtime_image_table_variable == 0 || state.runtime_image_requests_variable == 0 ||
	        state.bindless_translation_variable == 0);
	const auto u32      = TypeU32(state);
	const auto boolean  = TypeBool(state);
	const auto binding  = ConstantU32(state, RuntimeImageBinding(image));
	const auto multiply = [&](uint32_t value) {
		return Binary(state, spv::OpIMul, u32, value, ConstantU32(state, 0x01000193u));
	};
	auto hash = ConstantU32(state, 0x811c9dc5u);
	for (const auto dword: dwords) {
		hash = multiply(Binary(state, spv::OpBitwiseXor, u32, hash, dword));
	}
	hash = multiply(Binary(state, spv::OpBitwiseXor, u32, hash, binding));
	hash = Binary(state, spv::OpBitwiseXor, u32, hash,
	              Binary(state, spv::OpShiftRightLogical, u32, hash, ConstantU32(state, 15)));
	const auto mask  = ConstantU32(state, IR::RuntimeImageEntries - 1u);
	const auto home  = Binary(state, spv::OpBitwiseAnd, u32, hash, mask);
	auto       found = ConstantBool(state, false);
	auto       entry = ConstantU32(state, 0);
	for (uint32_t probe = 0; probe < IR::RuntimeImageProbes; probe++) {
		const auto index =
		    probe == 0u ? home
		                : Binary(state, spv::OpBitwiseAnd, u32,
		                         Binary(state, spv::OpIAdd, u32, home, ConstantU32(state, probe)),
		                         mask);
		const auto base =
		    Binary(state, spv::OpIMul, u32, index, ConstantU32(state, IR::RuntimeImageEntryWords));
		const auto word = [&](uint32_t offset) {
			return RuntimeLoadWord(state, state.runtime_image_table_variable,
			                       Binary(state, spv::OpIAdd, u32, base, ConstantU32(state, offset)));
		};
		const auto target = word(8u);
		auto       match  = Binary(state, spv::OpLogicalAnd, boolean,
		                           Binary(state, spv::OpINotEqual, boolean, target, ConstantU32(state, 0)),
		                           Binary(state, spv::OpIEqual, boolean, word(9u), binding));
		for (uint32_t dword = 0; dword < 8u; dword++) {
			match = Binary(state, spv::OpLogicalAnd, boolean, match,
			               Binary(state, spv::OpIEqual, boolean, word(dword), dwords[dword]));
		}
		const auto take = Binary(state, spv::OpLogicalAnd, boolean,
		                         Unary(state, spv::OpLogicalNot, boolean, found), match);
		entry           = Select(state, u32, take,
		                         Binary(state, spv::OpISub, u32, target, ConstantU32(state, 1)), entry);
		found           = Binary(state, spv::OpLogicalOr, boolean, found, match);
	}
	const auto raw = RuntimeLoadWord(state, state.bindless_translation_variable,
	                                 Select(state, u32, found, entry, ConstantU32(state, 0)));
	const auto pending =
	    Binary(state, spv::OpLogicalOr, boolean, Unary(state, spv::OpLogicalNot, boolean, found),
	           Binary(state, spv::OpIEqual, boolean, raw, ConstantU32(state, IR::BindlessPending)));
	if (state.bindless_feedback_variable != 0) {
		// As for a heap key: 1 asks the host to resolve the entry again, 0 marks it sampled.
		EmitIfCondition(state, found, [&]() {
			RuntimeStoreWord(
			    state, state.bindless_feedback_variable, entry,
			    Select(state, u32, pending, ConstantU32(state, 1), ConstantU32(state, 0)));
			// The chunk's summary word (IR::BindlessFeedbackSummary), as for a heap key.
			EmitIfCondition(state, pending, [&]() {
				const auto chunk = Binary(state, spv::OpShiftRightLogical, u32, entry,
				                          ConstantU32(state, IR::BindlessFeedbackChunkShift));
				RuntimeStoreWord(state, state.bindless_feedback_variable,
				                 Binary(state, spv::OpIAdd, u32, chunk, ConstantU32(state, IR::BindlessFeedbackSummary)),
				                 ConstantU32(state, 1));
			});
		});
	}
	EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, found), [&]() {
		const auto bit_word =
		    Binary(state, spv::OpIAdd, u32,
		           Binary(state, spv::OpShiftRightLogical, u32, home, ConstantU32(state, 5)),
		           ConstantU32(state, IR::RuntimeImageBitsOffset));
		const auto bit =
		    Binary(state, spv::OpShiftLeftLogical, u32, ConstantU32(state, 1),
		           Binary(state, spv::OpBitwiseAnd, u32, home, ConstantU32(state, 31)));
		const auto old = RuntimeAtomicWord(state, spv::OpAtomicOr,
		                                   state.runtime_image_requests_variable, bit_word, bit);
		const auto first =
		    Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, old, bit),
		           ConstantU32(state, 0));
		EmitIfCondition(state, first, [&]() {
			const auto request =
			    RuntimeAtomicWord(state, spv::OpAtomicIAdd, state.runtime_image_requests_variable,
			                      ConstantU32(state, 0), ConstantU32(state, 1));
			EmitIfCondition(
			    state,
			    Binary(state, spv::OpULessThan, boolean, request,
			           ConstantU32(state, IR::RuntimeImageRequests)),
			    [&]() {
				    const auto record =
				        Binary(state, spv::OpIAdd, u32,
				               Binary(state, spv::OpIMul, u32, request,
				                      ConstantU32(state, IR::RuntimeImageRequestWords)),
				               ConstantU32(state, IR::RuntimeImageRingOffset));
				    for (uint32_t dword = 0; dword < 8u; dword++) {
					    RuntimeStoreWord(
					        state, state.runtime_image_requests_variable,
					        Binary(state, spv::OpIAdd, u32, record, ConstantU32(state, dword)),
					        dwords[dword]);
				    }
				    RuntimeStoreWord(
				        state, state.runtime_image_requests_variable,
				        Binary(state, spv::OpIAdd, u32, record, ConstantU32(state, 8)),
				        Binary(state, spv::OpIAdd, u32, binding, ConstantU32(state, 1)));
			    });
		});
	});
	state.bindless_translation_raw = raw;
	const auto slot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, u32, slot, pending,
	                          ConstantU32(state, IR::BindlessPendingSlot), raw);
	state.builder.AddAnnotation(spv::OpDecorate, slot, spv::DecorationNonUniform);
	return slot;
}

// IMAGE_GET_RESINFO of a runtime image, from its T#: {width, height, depth or layers, levels} of
// the level BASE_LEVEL + lod, laid out as QueryDimensions lays out a view's size.
static uint32_t RuntimeQueryDimensions(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                       const IR::Inst& address, ImageDimension dimension,
                                       const std::array<uint32_t, 8>& dwords) {
	auto&      state = ctx.state;
	const auto u32   = TypeU32(state);
	const auto field = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, dwords[word], ConstantU32(state, first),
		                            ConstantU32(state, count));
	};
	const auto add   = [&](uint32_t a, uint32_t b) { return Binary(state, spv::OpIAdd, u32, a, b); };
	const auto one   = ConstantU32(state, 1);
	const auto width = add(
	    Binary(state, spv::OpBitwiseOr, u32, field(1, 30, 2),
	           Binary(state, spv::OpShiftLeftLogical, u32, field(2, 0, 12), ConstantU32(state, 2))),
	    one);
	const auto height     = add(field(2, 14, 14), one);
	const auto depth      = add(field(4, 0, 13), one);
	const auto base_array = field(4, 16, 13);
	const auto base_level = field(3, 12, 4);
	const auto last_level = field(3, 16, 4);
	const auto level =
	    Binary(state, spv::OpBitwiseAnd, u32, add(base_level, AddressU32(ctx, mem, address, 0)),
	           ConstantU32(state, 31));
	const auto at_level = [&](uint32_t size) {
		return EmitUMax32(state, Binary(state, spv::OpShiftRightLogical, u32, size, level), one);
	};
	const auto& info      = ImageDimensionInfoFor(dimension);
	uint32_t    result[4] = {ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0),
	                         ConstantU32(state, 0)};
	const auto  layers = EmitUMax32(state, Binary(state, spv::OpISub, u32, depth, base_array), one);
	result[0]          = at_level(width);
	if (dimension == ImageDimension::Dim1DArray) {
		result[1] = layers;
	} else if (info.coordinate_components >= 2u) {
		result[1] = at_level(height);
	}
	if (dimension == ImageDimension::Dim3D) {
		result[2] = at_level(depth);
	} else if (dimension == ImageDimension::Dim2DArray) {
		result[2] = layers;
	}
	result[3] =
	    Select(state, u32, Binary(state, spv::OpULessThan, TypeBool(state), last_level, base_level),
	           one, add(Binary(state, spv::OpISub, u32, last_level, base_level), one));
	const auto vector = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), vector, result[0],
	                          result[1], result[2], result[3]);
	return vector;
}

static void EmitImageOperation(ValueEmitContext& ctx, const IR::Inst& inst);
static void EmitBindlessVolumeOrArray(ValueEmitContext& ctx, const IR::Inst& inst);

// KYTY_RUNTIME_INT_IMAGES (adapted from usa-ratt/KytyPS5-GT7 2fa6c6514): a load of a runtime image
// reads its slot through the alias of the T#'s component class: float through the image's own
// array, uint / sint through the aliases DefineBindlessImages declares. A T# whose class has no
// view, or an integer one still pending, reads zeros (the pending placeholder is a float view).
// The load is a fetch (no derivatives), so the class selects a branch.
static void EmitRuntimeImageRead(ValueEmitContext& ctx, const IR::Inst& inst,
                                 const IR::ImageResource& image, const std::array<uint32_t, 8>& dwords) {
	auto&      state         = ctx.state;
	const auto& mem          = ctx.Memory(inst);
	const auto numeric_class = RuntimeImageNumericClass(state, dwords);
	const auto entry         = RuntimeImageSlot(ctx, image, dwords);
	const auto u32           = TypeU32(state);
	const auto boolean       = TypeBool(state);
	const auto pending =
	    Binary(state, spv::OpIEqual, boolean, entry, ConstantU32(state, IR::BindlessPendingSlot));
	// An existing definition (a value used before this point) is re-pointed at the selection.
	std::optional<uint32_t> predefined;
	if (const auto found = ctx.definitions.find(&inst); found != ctx.definitions.end()) {
		predefined = found->second;
		ctx.definitions.erase(found);
	}
	const auto read = [&](Prospero::TextureNumericClass alias_class, uint32_t slot) {
		if (alias_class != image.numeric_class) {
			state.image_override_index        = mem.resource;
			state.image_override              = state.program.info.images.at(mem.resource);
			state.image_override.numeric_class = alias_class;
			state.image_override.depth_compare = false;
		}
		state.bindless_slot = slot;
		EmitImageOperation(ctx, inst);
		state.image_override_index = UINT32_MAX;
		const auto found           = ctx.definitions.find(&inst);
		EXIT_IF(found == ctx.definitions.end());
		const auto value = found->second;
		ctx.definitions.erase(found);
		return value;
	};
	const auto vector = TypeU32Vector(state, 4);
	const auto value  = EmitIndexSwitch(state, numeric_class, 4u, vector, [&](uint32_t which) {
		if (which == 0u || which == 3u) {
			// Float (or no view: the placeholder, which reads as the float path did before).
			const auto slot = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpSelect, u32, slot,
			                          Binary(state, spv::OpIEqual, boolean, numeric_class,
			                                 ConstantU32(state, 0)),
			                          entry, ConstantU32(state, IR::BindlessPendingSlot));
			state.builder.AddAnnotation(spv::OpDecorate, slot, spv::DecorationNonUniform);
			return read(image.numeric_class, slot);
		}
		const auto alias_class =
		    which == 1u ? Prospero::TextureNumericClass::Uint : Prospero::TextureNumericClass::Sint;
		return EmitValueOrDefaultIfCondition(
		    state, Unary(state, spv::OpLogicalNot, boolean, pending), vector,
		    ConstantU32CompositeZero(state, 4), [&]() { return read(alias_class, entry); });
	});
	if (predefined) {
		ctx.definitions.emplace(&inst, *predefined);
	}
	ctx.Define(inst, value);
}

void EmitImage(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state     = ctx.state;
	const auto& mem       = ctx.Memory(inst);
	const auto  image_arg = inst.Arg(0);
	ctx.ResourceIndex(image_arg, IR::ValueOpcode::GetImageResource);
	const auto& image   = ImageResourceAt(state, mem.resource);
	state.bindless_sampler_slot = 0u;
	if (IR::ImageOpcodeInfoOf(inst.GetOpcode()).needs_sampler && inst.NumArgs() > 1u &&
	    mem.sampler < state.program.info.samplers.size() &&
	    state.program.info.samplers[mem.sampler].bindless) {
		state.bindless_sampler_slot =
		    BindlessSamplerSlot(ctx, state.program.info.samplers[mem.sampler], inst.Arg(1));
	}
	if (image.runtime) {
		const auto dwords = RuntimeImageDwords(ctx, image_arg);
		if (inst.GetOpcode() == IR::ValueOpcode::ImageQueryDimensions) {
			const auto* address = ctx.ImageAddress(inst.Arg(1));
			ctx.Define(inst, RuntimeQueryDimensions(ctx, mem, *address, image.dimension, dwords));
			return;
		}
		if (inst.GetOpcode() == IR::ValueOpcode::ImageRead && GetCodegenOptions().runtime_int_images &&
		    image.resource_class == IR::ImageResourceClass::Sampled &&
		    ImageDimensionInfoFor(image.dimension).multisampled == 0u) {
			EmitRuntimeImageRead(ctx, inst, image, dwords);
			return;
		}
		const auto is_float = RuntimeImageFloat(state, dwords);
		const auto entry    = RuntimeImageSlot(ctx, image, dwords);
		const auto slot     = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), slot, is_float, entry,
		                          ConstantU32(state, IR::BindlessPendingSlot));
		state.builder.AddAnnotation(spv::OpDecorate, slot, spv::DecorationNonUniform);
		state.bindless_slot = slot;
		EmitImageOperation(ctx, inst);
		return;
	}
	state.bindless_slot = image.bindless ? BindlessSlot(ctx, image, image_arg) : 0u;
	if (BindlessVolumeMayBeArray(image) && inst.GetOpcode() != IR::ValueOpcode::ImageWrite) {
		EmitBindlessVolumeOrArray(ctx, inst);
		return;
	}
	EmitImageOperation(ctx, inst);
}

// KYTY_BINDLESS_3D_AS_ARRAY (adapted from KytyPS5-Frank fed216ce8): a bindless 3D image
// instruction whose key may name a 2D or 2D-array texture. The GPU addresses by the T# type, so
// such a texture is read as a 2D array at layer round(z) (Vulkan's array layer rule is the same).
// Both reads are emitted, each with the other view's placeholder slot, and the flag selects: no
// divergent branch, so implicit-LOD derivatives stay defined.
static void EmitBindlessVolumeOrArray(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state       = ctx.state;
	const auto& mem         = ctx.Memory(inst);
	const auto  raw         = state.bindless_translation_raw;
	const auto  slot        = state.bindless_slot;
	const auto  flags       = Binary(state, spv::OpBitwiseAnd, TypeU32(state), raw,
	                                 ConstantU32(state, IR::BindlessSlotFlagMask));
	const auto  as_array    = Binary(state, spv::OpIEqual, TypeBool(state), flags,
	                                 ConstantU32(state, IR::BindlessSlotArrayFlag));
	const auto  array_index = Binary(state, spv::OpBitwiseAnd, TypeU32(state), raw,
	                                 ConstantU32(state, ~IR::BindlessSlotFlagMask));
	const auto  NonUniformSelect = [&](uint32_t condition, uint32_t yes, uint32_t no) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), value, condition, yes, no);
		state.builder.AddAnnotation(spv::OpDecorate, value, spv::DecorationNonUniform);
		return value;
	};
	const auto array_slot  = NonUniformSelect(as_array, array_index, ConstantU32(state, 0u));
	const auto volume_slot = NonUniformSelect(as_array, ConstantU32(state, 0u), slot);
	// An existing definition (a value used before this point) is re-pointed at the selection.
	std::optional<uint32_t> predefined;
	if (const auto found = ctx.definitions.find(&inst); found != ctx.definitions.end()) {
		predefined = found->second;
		ctx.definitions.erase(found);
	}
	const auto Emit = [&](bool array_view) {
		if (array_view) {
			state.image_override_index     = mem.resource;
			state.image_override           = state.program.info.images.at(mem.resource);
			state.image_override.dimension = ImageDimension::Dim2DArray;
		}
		state.bindless_slot = array_view ? array_slot : volume_slot;
		EmitImageOperation(ctx, inst);
		state.image_override_index = UINT32_MAX;
		state.bindless_slot        = slot;
		const auto found           = ctx.definitions.find(&inst);
		EXIT_IF(found == ctx.definitions.end());
		const auto value = found->second;
		ctx.definitions.erase(found);
		return value;
	};
	const auto array_value  = Emit(true);
	const auto volume_value = Emit(false);
	const auto value        = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeId(state, inst.GetType()), value, as_array,
	                          array_value, volume_value);
	if (predefined) {
		ctx.definitions.emplace(&inst, *predefined);
	}
	ctx.Define(inst, value);
}

static void EmitImageOperation(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op         = inst.GetOpcode();
	const auto  image_info = IR::ImageOpcodeInfoOf(op);
	auto&       state      = ctx.state;
	const auto& mem        = ctx.Memory(inst);
	const auto& image      = ImageResourceAt(state, mem.resource);
	const auto  image_arg  = inst.Arg(0);
	const auto* address = ctx.ImageAddress(inst.Arg(image_info.needs_sampler ? 2 : 1));
	if (op == IR::ValueOpcode::ImageQueryDimensions) {
		state.builder.RequireCapability(spv::CapabilityImageQuery);
		ctx.Define(inst, QueryDimensions(ctx, mem, *address));
		return;
	}
	if (op == IR::ValueOpcode::ImageQueryLod) {
		state.builder.RequireCapability(spv::CapabilityImageQuery);
		const auto dimension = image.dimension;
		const auto sampled = MakeSampledImage(state, mem.resource,
		                                     LoadSamplerDescriptor(state, mem.sampler));
		const auto lod       = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpImageQueryLod, TypeF32Vector(state, 2), lod, sampled,
		    CoordF32(ctx, mem, *address, 0, ImageDimensionInfoFor(dimension).spatial_components,
		             image.cube, RowCenter(state, dimension)));
		uint32_t values[4] = {ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0),
		                      ConstantU32(state, 0)};
		for (uint32_t index = 0; index < 2u; index++) {
			const auto component = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, lod,
			                          index);
			values[index] = FloatBits(ctx, component);
		}
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result,
		                          values[0], values[1], values[2], values[3]);
		ctx.Define(inst, result);
		return;
	}
	if (op == IR::ValueOpcode::ImageRead && image.indirect_root == mem.resource &&
	    !image.bindless && image.indirect_resources.size() >= 2u) {
		// KYTY_SAMPLED_TABLE_READS (adapted from Erlou93/KytyPS5-Yotei EmitImageAccess): a load
		// through a table the GPU indexes reads the candidate its key selects, with the candidate's
		// own coordinates. The instruction's address holds as many coordinates as its own
		// dimension; a candidate that has more reads zero for the rest (the mip follows them).
		EXIT_IF(!GetCodegenOptions().sampled_table_reads);
		const auto* handle = image_arg.ResolveInstruction();
		if (handle == nullptr || handle->NumArgs() == 0u || state.flattened_srt_variable == 0) {
			ctx.Fail(inst, "has invalid indirect image key provenance");
		}
		const auto numeric_class = image.numeric_class;
		const auto instruction_dimension =
		    mem.image_dimension != ImageDimension::Unknown ? mem.image_dimension : image.dimension;
		const auto instruction_components =
		    ImageDimensionInfoFor(instruction_dimension).coordinate_components;
		const auto condition = ctx.Arg(inst, 2);
		ctx.Define(
		    inst,
		    EmitValueOrDefaultIfCondition(
		        state, condition, TypeU32Vector(state, 4), ConstantU32CompositeZero(state, 4),
		        [&]() {
			        // Everything the cases share is defined before the switch, so it dominates them.
			        const auto key = ctx.Def(handle->Arg(0));
			        std::array<uint32_t, 3> address_words {};
			        for (uint32_t component = 0; component < address_words.size(); component++) {
				        address_words[component] =
				            component < instruction_components &&
				                    mem.image_address_components > component
				                ? AddressU32(ctx, mem, *address, component)
				                : ConstantU32(state, 0);
			        }
			        const auto lod = LodU32(ctx, mem, *address, instruction_dimension);
			        const auto selected = EmitIndirectResourceIndex(
			            state, key, image.indirect_mapping_offset, image.indirect_search_iterations,
			            0u);
			        const auto resources = image.indirect_resources;
			        const auto color = EmitIndexSwitch(
			            state, selected, static_cast<uint32_t>(resources.size()),
			            ImageVectorType(state, numeric_class, 4), [&](uint32_t ordinal) {
			            	const auto  resource       = resources[ordinal];
			            	const auto& candidate      = ImageResourceAt(state, resource);
			            	const auto& candidate_info = ImageDimensionInfoFor(candidate.dimension);
			            	EXIT_IF(candidate.numeric_class != numeric_class);
			            	// A 1D candidate is a host 2D image one texel high (upstream e6cc1ce30): its host row 0
			            	// follows x, as CoordU32 builds it for a plain read.
			            	std::array<uint32_t, 3> words {};
			            	uint32_t                count = 0;
			            	words[count++]                = address_words[0];
			            	if (candidate_info.one_dimensional) {
			            		words[count++] = ConstantU32(state, 0);
			            	}
			            	for (uint32_t c = 1; c < candidate_info.coordinate_components; c++) {
			            		words[count++] = address_words[c];
			            	}
			            	uint32_t coord = words[0];
			            	if (count > 1u) {
			            		coord = state.builder.AllocateId();
			            		state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, count), coord,
			            		                          std::span(words).first(count));
			            	}
			            	const auto descriptor = LoadImageDescriptor(state, resource);
			            	const auto fetched    = state.builder.AllocateId();
			            	if (candidate_info.multisampled != 0u) {
			            		state.builder.AddFunction(
			            			spv::OpImageFetch, ImageVectorType(state, numeric_class, 4),
			            			fetched, descriptor, coord, spv::ImageOperandsSampleMask,
			            			AddressU32(ctx, mem, *address, instruction_components));
			            	} else {
			            		state.builder.AddFunction(
			            			spv::OpImageFetch, ImageVectorType(state, numeric_class, 4),
			            			fetched, descriptor, coord, spv::ImageOperandsLodMask, lod);
			            	}
			            	return fetched;
			            });
			        return ResultVector(ctx, UnpackImageTexel(ctx, mem, color), numeric_class,
			                            false, mem);
		        }));
		return;
	}
	if (op == IR::ValueOpcode::ImageRead) {
		const auto  dimension      = image.dimension;
		const auto& dimension_info = ImageDimensionInfoFor(dimension);
		const auto  numeric_class  = image.numeric_class;
		const auto  condition      = ctx.Arg(inst, 2);
		ctx.Define(
		    inst,
		    EmitValueOrDefaultIfCondition(
		        state, condition, TypeU32Vector(state, 4), ConstantU32CompositeZero(state, 4),
		        [&]() {
			        const auto descriptor = LoadImageDescriptor(state, mem.resource);
			        const auto color      = state.builder.AllocateId();
			        const auto coord      = CoordU32(ctx, mem, *address, dimension);
			        if (dimension_info.multisampled != 0u) {
				        state.builder.AddFunction(
				            spv::OpImageFetch, ImageVectorType(state, numeric_class, 4), color,
				            descriptor, coord, spv::ImageOperandsSampleMask,
				            AddressU32(ctx, mem, *address, dimension_info.coordinate_components));
			        } else {
				        state.builder.AddFunction(spv::OpImageFetch,
				                                  ImageVectorType(state, numeric_class, 4), color,
				                                  descriptor, coord, spv::ImageOperandsLodMask,
				                                  LodU32(ctx, mem, *address, dimension));
			        }
			        return ResultVector(ctx, UnpackImageTexel(ctx, mem, color), numeric_class,
			                            false, mem);
		        }));
		return;
	}
	if (op == IR::ValueOpcode::ImageReadStorage) {
		// KYTY_IMAGE_GLC_COHERENT / KYTY_POPS_INTERLOCK: IMAGE_LOAD* through the storage image, its
		// Coherent alias when the access is coherent.
		if (image.indirect_root == mem.resource && !image.bindless &&
		    image.indirect_resources.size() >= 2u) {
			ctx.Fail(inst, "reads a storage image table (not supported)");
			return;
		}
		state.builder.RequireCapability(spv::CapabilityStorageImageReadWithoutFormat);
		const auto  dimension      = image.dimension;
		const auto& dimension_info = ImageDimensionInfoFor(dimension);
		const auto  numeric_class  = image.numeric_class;
		const auto  condition      = ctx.Arg(inst, 2);
		ctx.Define(
		    inst,
		    EmitValueOrDefaultIfCondition(
		        state, condition, TypeU32Vector(state, 4), ConstantU32CompositeZero(state, 4),
		        [&]() {
			        const auto coord     = CoordU32(ctx, mem, *address, dimension);
			        const auto read_type = ImageVectorType(state, numeric_class, 4);
			        const auto EmitRead  = [&](uint32_t mip) {
				        const CoherentImageScope scope(state, ImageAccessCoherent(state, mem));
				        const auto descriptor = LoadImageDescriptor(state, mem.resource, mip);
				        const auto color      = state.builder.AllocateId();
				        if (dimension_info.multisampled != 0u) {
					        state.builder.AddFunction(
					            spv::OpImageRead, read_type, color, descriptor, coord,
					            spv::ImageOperandsSampleMask,
					            AddressU32(ctx, mem, *address, dimension_info.coordinate_components));
				        } else {
					        state.builder.AddFunction(spv::OpImageRead, read_type, color, descriptor,
					                                  coord);
				        }
				        return color;
			        };
			        const auto color =
			            image.mip_mode == IR::ImageMipMode::Dynamic
			                ? EmitIndexSwitch(state, LodU32(ctx, mem, *address, dimension),
			                                  image.mip_count, read_type, EmitRead)
			                : EmitRead(0);
			        return ResultVector(ctx, UnpackImageTexel(ctx, mem, color), numeric_class, false,
			                            mem);
		        }));
		return;
	}
	if (op == IR::ValueOpcode::ImageWrite) {
		const CoherentImageScope coherent_write(state, ImageAccessCoherent(state, mem));
		const bool uint_image = image.numeric_class == Prospero::TextureNumericClass::Uint;
		const auto dimension  = image.dimension;
		// KYTY_WORKGROUP_IMAGES: a table of enumerated storage images (one per workgroup); the
		// key in the handle selects the candidate, each case writes its own binding.
		const bool indirect = image.indirect_root == mem.resource && !image.bindless &&
		                      image.indirect_resources.size() >= 2u;
		EmitIfCondition(state, ctx.Arg(inst, 3), [&]() {
			if (!indirect) {
				const auto mip_lod =
				    ImageResourceAt(state, mem.resource).mip_mode == IR::ImageMipMode::Dynamic
				        ? LodU32(ctx, mem, *address, dimension)
				        : 0u;
				const auto coord = CoordU32(ctx, mem, *address, dimension);
				const auto texel = StoreTexel(ctx, mem, ctx.Arg(inst, 2), uint_image);
				EmitStorageImageWrite(state, mem.resource, mip_lod, coord, texel);
				return;
			}
			const auto* handle = image_arg.ResolveInstruction();
			if (handle == nullptr || handle->NumArgs() == 0u || state.flattened_srt_variable == 0) {
				ctx.Fail(inst, "has invalid indirect image key provenance");
				return;
			}
			// Everything the cases share is defined before the switch, so it dominates them.
			const auto key   = ctx.Def(handle->Arg(0));
			const auto coord = CoordU32(ctx, mem, *address, dimension);
			const auto data  = ctx.Arg(inst, 2);
			const bool dynamic_mip =
			    std::ranges::any_of(image.indirect_resources, [&](uint32_t resource) {
				    return ImageResourceAt(state, resource).mip_mode == IR::ImageMipMode::Dynamic;
			    });
			const auto lod = dynamic_mip ? LodU32(ctx, mem, *address, dimension) : 0u;
			const auto selected = EmitIndirectResourceIndex(
			    state, key, image.indirect_mapping_offset, image.indirect_search_iterations, 0u);
			const auto resources = image.indirect_resources;
			EmitIndexSwitch(state, selected, static_cast<uint32_t>(resources.size()), 0u,
			                [&](uint32_t ordinal) {
				                auto target     = mem;
				                target.resource = resources[ordinal];
				                const auto& candidate = ImageResourceAt(state, target.resource);
				                // KYTY_STORAGE_TABLE_LOOSE: a candidate of another plain dimension
				                // takes its coordinates (and mip) as its own type needs them.
				                const bool own_dimension = GetCodegenOptions().storage_table_loose &&
				                                           candidate.dimension != dimension;
				                const auto mip_lod =
				                    candidate.mip_mode == IR::ImageMipMode::Dynamic
				                        ? (own_dimension ? LodU32(ctx, target, *address, candidate.dimension)
				                                         : lod)
				                        : 0u;
				                const auto texel = StoreTexel(ctx, target, data, uint_image);
				                EmitStorageImageWrite(
				                    state, target.resource, mip_lod,
				                    own_dimension ? CoordU32(ctx, target, *address, candidate.dimension)
				                                  : coord,
				                    texel);
			                });
		});
		return;
	}
	if (op == IR::ValueOpcode::ImageSampleRaw || op == IR::ValueOpcode::ImageGatherRaw) {
		const auto  dimension      = image.dimension;
		const auto& dimension_info = ImageDimensionInfoFor(dimension);
		const auto  layout         = Layout(mem);
		const auto  numeric_class  = image.numeric_class;
		const bool  dref           = HasFlag(mem, Decoder::ImageSampleFlagCompare);
		if (dref && ImageResourceAt(state, mem.resource).conversion_format !=
		                Prospero::BufferFormat::kInvalid) {
			ctx.Fail(inst, "uses depth comparison with a converted image");
			return;
		}
		if (op == IR::ValueOpcode::ImageGatherRaw) {
			// Upstream 23ed47ec6: 1D and 1D-array images (host 2D one texel high, e6cc1ce30) gather
			// natively from the centre of their host row 0.
			const auto coord = CoordF32(ctx, mem, *address, layout.coord,
			                            dimension_info.coordinate_components, image.cube,
			                            RowCenter(state, dimension));
			const auto result_numeric_class = dref ? Prospero::TextureNumericClass::Float
			                                       : numeric_class;
			const auto result_type = ImageVectorType(state, result_numeric_class, 4);
			uint32_t component_or_dref;
			if (dref) {
				component_or_dref = layout.dref != NoImageComponent
				                        ? AddressF32(ctx, mem, *address, layout.dref)
				                        : ZeroF32(state);
			} else {
				component_or_dref = ConstantU32(state, ImageGatherSource(state, mem));
			}
			uint32_t operand_mask = 0;
			uint32_t offset = 0;
			if (HasFlag(mem, Decoder::ImageSampleFlagGatherHorizontal)) {
				operand_mask = spv::ImageOperandsConstOffsetsMask;
				offset = HorizontalOffsets(state);
			} else if (layout.offset != NoImageComponent) {
				operand_mask = spv::ImageOperandsOffsetMask;
				offset = PackedOffset(ctx, mem, *address, layout, dimension);
			}
			const auto sampler_id = LoadSamplerDescriptor(state, mem.sampler);
			const auto EmitGather = [&](uint32_t mip) {
				const auto sampled = MakeSampledImage(state, mem.resource, sampler_id, mip);
				const auto sample = state.builder.AllocateId();
				std::vector<uint32_t> words {
				    static_cast<uint32_t>(dref ? spv::OpImageDrefGather : spv::OpImageGather), result_type,
				    sample, sampled, coord, component_or_dref};
				if (operand_mask != 0u) {
					words.push_back(operand_mask);
					words.push_back(offset);
				}
				state.builder.AddFunction(words);
				// Host row 0 holds a 1D image's texels in w and z; x and y read row 1, which is out
				// of range. Horizontal gathers read row 0 in every component.
				if (!dimension_info.one_dimensional ||
				    HasFlag(mem, Decoder::ImageSampleFlagGatherHorizontal)) {
					return sample;
				}
				const auto shuffled = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVectorShuffle, result_type, shuffled, sample, sample,
				                          3u, 2u, 2u, 3u);
				return shuffled;
			};
			const auto sample = image.mip_mode == IR::ImageMipMode::Dynamic && image.mip_count > 1u
			                        ? EmitIndexSwitch(
			                              state, GatherMip(ctx, inst, mem, *address, layout,
			                                               image.mip_count),
			                              image.mip_count, result_type, EmitGather)
			                        : EmitGather(0);
			ctx.Define(inst, ResultVector(ctx, UnpackImageGather(ctx, mem, sample),
			                              result_numeric_class, false, mem, true));
			return;
		}
		const bool own_lod = HasFlag(mem, Decoder::ImageSampleFlagDerivative) ||
		                     HasFlag(mem, Decoder::ImageSampleFlagLod) ||
		                     HasFlag(mem, Decoder::ImageSampleFlagLevelZero);
		// KYTY_IMPLICIT_LOD_STAGES: compute shaders keep implicit LOD with lane-quad derivatives
		// (ComputeImplicitLod); other stages without derivatives take LOD 0, or the bias of an
		// IMAGE_SAMPLE_B (forced_lod).
		const bool implicit_here = state.program.stage == ShaderType::Pixel || ComputeImplicitLod(state);
		const bool explicit_lod  = own_lod || !implicit_here;
		const bool forced_lod    = !own_lod && !implicit_here;
		auto       opcode       = spv::OpImageSampleImplicitLod;
		if (explicit_lod) {
			opcode = dref ? spv::OpImageSampleDrefExplicitLod : spv::OpImageSampleExplicitLod;
		} else if (dref) {
			opcode = spv::OpImageSampleDrefImplicitLod;
		}
		uint32_t result_type = ImageVectorType(state, numeric_class, 4);
		uint32_t dref_value  = 0;
		if (dref) {
			result_type = TypeF32(state);
			dref_value  = ZeroF32(state);
			if (layout.dref != NoImageComponent) {
				dref_value = AddressF32(ctx, mem, *address, layout.dref);
			}
		}
		// IMAGE_SAMPLE*_CL: the last address component is a minimum LOD (D3D's Sample clamp).
		// Implicit and gradient samples take it as the MinLod operand; explicit-LOD forms (every
		// sample outside pixel shaders) raise their LOD to it.
		uint32_t lod_clamp = 0;
		if (layout.clamp != NoImageComponent && mem.image_address_components > layout.clamp &&
		    GetCodegenOptions().sample_lod_clamp) {
			lod_clamp = AddressF32(ctx, mem, *address, layout.clamp);
		}
		const bool gradient    = HasFlag(mem, Decoder::ImageSampleFlagDerivative);
		const bool min_lod_op  = lod_clamp != 0u && (gradient || !explicit_lod) &&
		                        GetHostImageFeatures().min_lod;
		uint32_t   explicit_lod_value = 0;
		uint32_t              operand_mask = 0;
		std::vector<uint32_t> operands;
		if (gradient) {
			operand_mask |= spv::ImageOperandsGradMask;
			// A 1D image's host row does not vary (upstream e6cc1ce30).
			const auto row = dimension_info.one_dimensional ? ZeroF32(state) : 0u;
			operands.push_back(CoordF32(ctx, mem, *address, layout.grad_x,
			                            dimension_info.spatial_components, false, row));
			operands.push_back(CoordF32(ctx, mem, *address, layout.grad_y,
			                            dimension_info.spatial_components, false, row));
		} else if (explicit_lod) {
			operand_mask |= spv::ImageOperandsLodMask;
			auto lod = ZeroF32(state);
			if (HasFlag(mem, Decoder::ImageSampleFlagLod) && layout.lod != NoImageComponent) {
				lod = AddressF32(ctx, mem, *address, layout.lod);
			} else if (forced_lod && layout.bias != NoImageComponent &&
			           GetCodegenOptions().implicit_lod_stages) {
				// IMAGE_SAMPLE_B outside pixel shaders: LOD 0 plus its bias.
				lod = AddressF32(ctx, mem, *address, layout.bias);
			}
			if (lod_clamp != 0u) {
				const auto clamped = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpExtInst, TypeF32(state), clamped, GlslStd450(state),
				                          GLSLstd450FMax, lod, lod_clamp);
				lod = clamped;
			}
			explicit_lod_value = lod;
			operands.push_back(lod);
		} else if (layout.bias != NoImageComponent) {
			operand_mask |= spv::ImageOperandsBiasMask;
			operands.push_back(AddressF32(ctx, mem, *address, layout.bias));
		}
		const auto sample_offset =
		    SampleOffsetFor(ctx, inst, mem, *address, layout, dimension, image.cube);
		if (sample_offset.const_offset != 0u) {
			operand_mask |= spv::ImageOperandsConstOffsetMask;
			operands.push_back(sample_offset.const_offset);
		}
		if (min_lod_op) {
			state.builder.RequireCapability(spv::CapabilityMinLod);
			operand_mask |= spv::ImageOperandsMinLodMask;
			operands.push_back(lod_clamp);
		} else if (lod_clamp != 0u && !explicit_lod) {
			static std::atomic_flag warned = ATOMIC_FLAG_INIT;
			if (!warned.test_and_set(std::memory_order_relaxed)) {
				std::fputs("Warning: IMAGE_SAMPLE*_CL clamp ignored: the device lacks "
				           "shaderResourceMinLod.\n",
				           stderr);
			}
		}
		const auto sampler_id = LoadSamplerDescriptor(state, mem.sampler);
		const auto EmitSample = [&](uint32_t resource, uint32_t array_index = 0u) {
			const auto& candidate = ImageResourceAt(state, resource);
			auto        coord =
			    CoordF32(ctx, mem, *address, layout.coord,
			             ImageDimensionInfoFor(candidate.dimension).coordinate_components,
			             candidate.cube, RowCenter(state, candidate.dimension));
			if (sample_offset.dynamic) {
				uint32_t level = ZeroF32(state);
				if (explicit_lod_value != 0u) {
					level = explicit_lod_value;
				} else if (!HasFlag(mem, Decoder::ImageSampleFlagLevelZero) && implicit_here) {
					// The level an implicit or gradient sample reads, relative to the view base.
					state.builder.RequireCapability(spv::CapabilityImageQuery);
					const auto query = state.builder.AllocateId();
					state.builder.AddFunction(
					    spv::OpImageQueryLod, TypeF32Vector(state, 2), query,
					    MakeSampledImage(state, resource, sampler_id),
					    CoordF32(ctx, mem, *address, layout.coord,
					             ImageDimensionInfoFor(candidate.dimension).spatial_components,
					             candidate.cube, RowCenter(state, candidate.dimension)));
					level = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), level, query,
					                          0u);
				}
				coord = ApplyDynamicSampleOffset(state, resource, coord, candidate.dimension,
				                                 sample_offset.packed, level);
			}
			const auto sampled = MakeSampledImage(state, resource, sampler_id, 0u, array_index);
			const auto            sample  = state.builder.AllocateId();
			std::vector<uint32_t> sample_operands {result_type, sample, sampled, coord};
			if (dref) {
				sample_operands.push_back(dref_value);
			}
			if (operand_mask != 0u) {
				sample_operands.push_back(operand_mask);
				sample_operands.insert(sample_operands.end(), operands.begin(), operands.end());
			}
			state.builder.AddFunction(opcode, sample_operands);
			if (state.mip_stats_variable != 0 && state.mip_stats_records) {
				const auto sample_lod = [&](uint32_t sampled_image) {
					uint32_t lod = ZeroF32(state);
					if (HasFlag(mem, Decoder::ImageSampleFlagLevelZero)) {
						// Level 0 explicitly.
					} else if (HasFlag(mem, Decoder::ImageSampleFlagLod) &&
					           layout.lod != NoImageComponent) {
						lod = AddressF32(ctx, mem, *address, layout.lod);
					} else {
						// Implicit or gradient sampling: the unclamped LOD (before MIN_LOD and the
						// resident range), which is what the streamer needs to know.
						state.builder.RequireCapability(spv::CapabilityImageQuery);
						const auto query = state.builder.AllocateId();
						state.builder.AddFunction(
						    spv::OpImageQueryLod, TypeF32Vector(state, 2), query, sampled_image,
						    CoordF32(ctx, mem, *address, layout.coord,
						             ImageDimensionInfoFor(candidate.dimension).spatial_components,
						             candidate.cube, RowCenter(state, candidate.dimension)));
						lod = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), lod,
						                          query, 1u);
						// IMAGE_SAMPLE_B*: the shader bias adds to the computed LOD.
						if (!explicit_lod && layout.bias != NoImageComponent) {
							lod = Binary(state, spv::OpFAdd, TypeF32(state), lod,
							             AddressF32(ctx, mem, *address, layout.bias));
						}
						// IMAGE_SAMPLE*_CL: the shader's own clamp raises the LOD before the T#
						// MIN_LOD clamp does.
						if (min_lod_op) {
							const auto clamped = state.builder.AllocateId();
							state.builder.AddFunction(spv::OpExtInst, TypeF32(state), clamped,
							                          GlslStd450(state), GLSLstd450FMax, lod,
							                          lod_clamp);
							lod = clamped;
						}
					}
					return lod;
				};
				if (GetCodegenOptions().lod_stats_gate &&
				    resource < state.program.bindings.mip_stats_count) {
					// The counter id is uniform per draw: when the T# has no counter, skip the LOD
					// query and the atomics entirely instead of recording into the unused slot.
					const auto id          = LoadMipStatsId(state, resource);
					const auto has_counter = Binary(
					    state, spv::OpIEqual, TypeBool(state),
					    Binary(state, spv::OpBitwiseAnd, TypeU32(state), id, ConstantU32(state, 0x8000u)),
					    ConstantU32(state, 0));
					// OpSampledImage must sit in the block of its consumer, so the query inside the
					// branch combines the image and sampler again.
					EmitIfCondition(state, has_counter, [&]() {
						EmitGatedMipStatsRecord(
						    state, id, sample_lod(MakeSampledImage(state, resource, sampler_id)));
					});
				} else {
					EmitMipStatsRecord(state, resource, sample_lod(sampled));
				}
			}
			return sample;
		};
		if (image.indirect_root != mem.resource || image.bindless) {
			const auto sample = EmitSample(mem.resource);
			auto       result = sample;
			if (!dref) {
				result = UnpackImageTexel(ctx, mem, sample);
			}
			ctx.Define(inst, ResultVector(ctx, result, numeric_class, dref, mem));
			return;
		}
		const auto* handle = image_arg.ResolveInstruction();
		const auto* source = image.source < state.program.descriptor_sources.size()
		                         ? &state.program.descriptor_sources[image.source]
		                         : nullptr;
		if (handle == nullptr || source == nullptr || !source->indirect_descriptor.has_value() ||
		    handle->NumArgs() == 0u) {
			ctx.Fail(inst, "has invalid indirect image key provenance");
			return;
		}
		const auto key = ctx.Def(handle->Arg(0));
		if (state.flattened_srt_variable == 0 ||
		    image.indirect_resources.size() < 2u) {
			ctx.Fail(inst, "has no indirect image runtime mapping");
			return;
		}
		const auto selected = EmitIndirectResourceIndex(
		    state, key, image.indirect_mapping_offset, image.indirect_search_iterations, 0u);
		struct SampleRun {
			uint32_t first;
			uint32_t count;
			uint32_t resource;
			uint32_t slot_bias;
		};
		std::vector<SampleRun> runs;
		// Dynamic sample offsets and GET_LOD_STATS records query or count the candidate itself, so
		// those samples keep one case per candidate.
		const bool native_runs = !sample_offset.dynamic &&
		                         !(state.mip_stats_variable != 0 && state.mip_stats_records);
		for (uint32_t ordinal = 0; ordinal < image.indirect_resources.size(); ++ordinal) {
			const auto resource = image.indirect_resources[ordinal];
			const auto& candidate = ImageResourceAt(state, resource);
			const auto kind = *IR::DescriptorBindingForImage(candidate);
			const auto& binding = *IR::FindBinding(state.program.bindings, kind);
			if (native_runs && !runs.empty()) {
				auto& run = runs.back();
				const auto& first = ImageResourceAt(state, run.resource);
				const auto next_slot = run.slot_bias + ordinal;
				if (candidate.dimension == first.dimension && candidate.cube == first.cube &&
				    IR::DescriptorBindingForImage(first) == kind &&
				    candidate.mip_count == 1u && first.mip_count == 1u &&
				    (ordinal == 1u || (next_slot < binding.resources.size() &&
				                      binding.resources[next_slot] == resource))) {
					// Native roots precede appended children; only the root may have a slot gap.
					if (ordinal == 1u)
						run.slot_bias = ResourceForDescriptor(state, kind, resource) - ordinal;
					++run.count;
					continue;
				}
			}
			runs.push_back({ordinal, 1u, resource,
			                ResourceForDescriptor(state, kind, resource) - ordinal});
		}
		const auto EmitRun = [&](uint32_t index) {
			const auto& run = runs[index];
			if (run.count == 1u) return EmitSample(run.resource);
			auto slot = Binary(state, spv::OpIAdd, TypeU32(state), selected,
			                   ConstantU32(state, run.slot_bias));
			if (run.first == 0u) {
				const auto kind = *IR::DescriptorBindingForImage(image);
				const auto root_slot = ResourceForDescriptor(state, kind, mem.resource);
				if (root_slot != run.slot_bias) {
					const auto is_root = Binary(state, spv::OpIEqual, TypeBool(state), selected,
					                            ConstantU32(state, 0u));
					const auto root_or_child = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpSelect, TypeU32(state), root_or_child,
					                          is_root, ConstantU32(state, root_slot), slot);
					slot = root_or_child;
				}
			}
			return EmitSample(run.resource, slot);
		};
		auto run_index = ConstantU32(state, 0u);
		for (uint32_t index = 1; index < runs.size(); ++index) {
			const auto at_or_after = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state),
			                               selected, ConstantU32(state, runs[index].first));
			const auto next = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpSelect, TypeU32(state), next, at_or_after,
			                          ConstantU32(state, index), run_index);
			run_index = next;
		}
		auto result = runs.size() == 1u
		                  ? EmitRun(0u)
		                  : EmitIndexSwitch(state, run_index, static_cast<uint32_t>(runs.size()),
		                                    result_type, EmitRun);
		if (!dref) {
			result = UnpackImageTexel(ctx, mem, result);
		}
		ctx.Define(inst, ResultVector(ctx, result, numeric_class, dref, mem));
		return;
	}
	const auto atomic_opcode = ImageAtomicOpcode(op);
	if (image_info.access == IR::ImageAccess::Atomic) {
		const auto dimension = image.dimension;
		const auto exec_arg    = inst.NumArgs() - 1;
		const auto result_type = TypeId(state, inst.GetType());
		const auto zero        = image.atomic64 ? ConstantU64(state, 0) : ConstantU32(state, 0);
		ctx.Define(inst, EmitValueOrDefaultIfCondition(
		                     state, ctx.Arg(inst, exec_arg), result_type, zero, [&]() {
			           const auto pointer      = state.builder.AllocateId();
			           const auto pointer_type =
			               state.builder.Type(spv::OpTypePointer, spv::StorageClassImage,
			                                  image.atomic64 ? TypeU64(state) : TypeU32(state));
			           state.builder.AddFunction(spv::OpImageTexelPointer, pointer_type, pointer,
			                                     ImageDescriptorPointer(state, mem.resource),
			                                     CoordU32(ctx, mem, *address, dimension),
			                                     ConstantU32(state, 0));
			           if (op == IR::ValueOpcode::ImageAtomicInc32 ||
			               op == IR::ValueOpcode::ImageAtomicDec32) {
				           // AtomicUpdate ends with its own image-memory barrier.
				           const auto limit = ctx.Arg(inst, 2);
				           return AtomicUpdate(state, pointer, IR::ResourceKind::Image,
				                               [&](uint32_t old) {
					                               return op == IR::ValueOpcode::ImageAtomicInc32
					                                          ? AtomicIncrement(state, old, limit)
					                                          : AtomicDecrement(state, old, limit);
				                               });
			           }
			           if (op == IR::ValueOpcode::ImageAtomicFMin32 ||
			               op == IR::ValueOpcode::ImageAtomicFMax32) {
				           return AtomicUpdate(state, pointer, IR::ResourceKind::Image,
				                               [&](uint32_t old) {
					                               return EmitFloatAtomicReplacement(
					                                   state, old, ctx.Arg(inst, 2),
					                                   op == IR::ValueOpcode::ImageAtomicFMax32);
				                               });
			           }
			           const auto old = state.builder.AllocateId();
			           if (image.atomic64) {
				           // R64ui texel pointer; the IR's U64 is a native 64-bit integer.
				           state.builder.AddFunction(atomic_opcode, TypeU64(state), old, pointer,
				                                     ConstantU32(state, spv::ScopeDevice),
				                                     ConstantU32(state, spv::MemorySemanticsMaskNone),
				                                     ctx.Arg(inst, 2));
				           state.builder.AddFunction(
				               spv::OpMemoryBarrier, ConstantU32(state, spv::ScopeDevice),
				               ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask |
				                                     spv::MemorySemanticsImageMemoryMask));
				           return old;
			           }
			           if (op == IR::ValueOpcode::ImageAtomicCmpSwap32) {
				           // DATA[0] is stored when the texel equals the comparator in DATA[1].
				           state.builder.AddFunction(
				               spv::OpAtomicCompareExchange, TypeU32(state), old, pointer,
				               ConstantU32(state, spv::ScopeDevice),
				               ConstantU32(state, spv::MemorySemanticsMaskNone),
				               ConstantU32(state, spv::MemorySemanticsMaskNone), ctx.Arg(inst, 2),
				               ctx.Arg(inst, 3));
			           } else {
				           state.builder.AddFunction(atomic_opcode, TypeU32(state), old, pointer,
				                                     ConstantU32(state, spv::ScopeDevice),
				                                     ConstantU32(state, spv::MemorySemanticsMaskNone),
				                                     ctx.Arg(inst, 2));
			           }
			           // This pointer names storage-image memory, not a storage buffer.
			           state.builder.AddFunction(
			               spv::OpMemoryBarrier, ConstantU32(state, spv::ScopeDevice),
			               ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask |
			                                     spv::MemorySemanticsImageMemoryMask));
			           return old;
		           }));
		return;
	}
	ctx.Fail(inst, "has no image SPIR-V emitter");
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
