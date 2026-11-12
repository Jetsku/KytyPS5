#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include <algorithm>
#include <bit>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t TessellationPointer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state          = ctx.state;
	using Attribute      = IR::TessellationAttribute;
	const auto  kind     = static_cast<Attribute>(inst.Arg(0).U32());
	const auto& tess     = state.input_info.vertex->tess;
	const auto  variable = state.tess_variables.at(static_cast<uint32_t>(kind));
	EXIT_IF(variable == 0);
	const auto input   = kind == Attribute::ControlInput || kind == Attribute::EvaluationInput ||
	                   kind == Attribute::CarriedInput ||
	                   (kind == Attribute::PatchOutput && state.program.stage == ShaderType::TessellationEvaluation);
	const auto storage = input ? spv::StorageClassInput : spv::StorageClassOutput;
	const auto pointer = state.builder.AllocateId();
	if (kind == Attribute::CarriedOutput || kind == Attribute::CarriedInput) {
		// KYTY_TESS_MERGED_CARRY: one u32 per carried VGPR, four per vec4; the HS invocation reads
		// its own control point's (the lane the LS part ran in for it).
		EXIT_NOT_IMPLEMENTED(!inst.Arg(1).IsImmediate());
		const auto rank      = inst.Arg(1).U32() / 4u;
		const auto attribute = ConstantU32(state, rank / 4u);
		const auto component = ConstantU32(state, rank % 4u);
		const auto type      = TypePointer(state, storage, TypeU32(state));
		if (kind == Attribute::CarriedOutput) {
			state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, attribute, component);
		} else {
			state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, EmitLaneId(state),
			                          attribute, component);
		}
		return pointer;
	}
	if (kind == Attribute::Factor) {
		EXIT_NOT_IMPLEMENTED(!inst.Arg(1).IsImmediate());
		// Guest factor dwords: triangles outer[3] inner[1], quads outer[4] inner[2], isolines
		// outer[2] in reverse order of the Vulkan builtin (as AMD drivers write them).
		const auto index   = inst.Arg(1).U32() / 4u;
		const auto outers  = tess.domain == 0u ? 2u : tess.domain == 1u ? 3u : 4u;
		const auto factors = tess.domain == 0u ? 2u : tess.domain == 1u ? 4u : 6u;
		const auto outer   = index < outers;
		EXIT_NOT_IMPLEMENTED(index >= factors);
		const auto element = !outer ? index - outers : tess.domain == 0u ? 1u - index : index;
		state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, storage, TypeF32(state)),
		                          pointer, outer ? variable : state.tess_inner_variable,
		                          ConstantU32(state, element));
		return pointer;
	}
	auto address = ctx.Arg(inst, 1);
	if (kind == Attribute::PatchOutput) {
		address =
		    EmitBinaryU32(state, spv::OpISub, address, ConstantU32(state, state.tess_patch_base));
	}
	const bool local  = kind == Attribute::LocalOutput || kind == Attribute::ControlInput;
	const auto stride = local ? tess.ls_stride : tess.hs_stride;
	const auto offset = kind == Attribute::PatchOutput ? address
	                                                   : EmitBinaryU32(state, spv::OpUMod, address,
	                                                                   ConstantU32(state, stride));
	const auto attribute =
	    EmitBinaryU32(state, spv::OpShiftRightLogical, offset, ConstantU32(state, 4));
	const auto component =
	    EmitBinaryU32(state, spv::OpBitwiseAnd,
	                  EmitBinaryU32(state, spv::OpShiftRightLogical, offset, ConstantU32(state, 2)),
	                  ConstantU32(state, 3));
	const auto type = TypePointer(state, storage, TypeU32(state));
	if (kind == Attribute::LocalOutput || kind == Attribute::PatchOutput) {
		state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, attribute,
		                          component);
	} else {
		const bool write = inst.GetOpcode() == IR::ValueOpcode::SetTessellationAttribute;
		auto       vertex =
		    kind == Attribute::ControlOutput && (write || !GetCodegenOptions().tess_indexed)
		              ? EmitLaneId(state)
		              : EmitBinaryU32(state, spv::OpUDiv, address, ConstantU32(state, stride));
		if (GetCodegenOptions().tess_indexed && !(kind == Attribute::ControlOutput && write)) {
			// KYTY_TESS_INDEXED (usa-ratt/KytyPS5-GT7 70341d775): an HS invocation writes its own
			// control point and reads any of them back; a read's control point (a loaded index)
			// is clamped to stay within the array.
			const auto vertices = kind == Attribute::ControlInput ? tess.input_control_points
			                                                      : tess.output_control_points;
			vertex = EmitUMin32(state, vertex, ConstantU32(state, vertices - 1u));
		}
		state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, vertex, attribute,
		                          component);
	}
	return pointer;
}

} // namespace

void DefineTessellationInterfaces(EmitterState& state) {
	using Attribute = IR::TessellationAttribute;
	// Tessellation lowering emits attribute accesses only in these stages.
	const auto stage = state.program.stage;
	if (stage != ShaderType::Local && stage != ShaderType::TessellationControl &&
	    stage != ShaderType::TessellationEvaluation) {
		return;
	}
	std::array<bool, IR::NumTessellationAttributes> used {};
	uint32_t            patch_begin = UINT32_MAX, patch_end = 0;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != IR::ValueOpcode::GetTessellationAttribute &&
			    inst.GetOpcode() != IR::ValueOpcode::SetTessellationAttribute)
				continue;
			const auto kind = inst.Arg(0).U32();
			used.at(kind)   = true;
			if (kind == static_cast<uint32_t>(Attribute::PatchOutput)) {
				EXIT_NOT_IMPLEMENTED(!inst.Arg(1).IsImmediate());
				patch_begin = std::min(patch_begin, inst.Arg(1).U32());
				patch_end   = std::max(patch_end, inst.Arg(1).U32() + 4u);
			}
		}
	}
	if (std::ranges::none_of(used, [](bool value) { return value; })) return;
	const auto& tess  = state.input_info.vertex->tess;
	const auto  array = [&](uint32_t type, uint32_t count) {
		return state.builder.Type(spv::OpTypeArray, type, ConstantU32(state, count));
	};
	for (uint32_t index = 0; index < used.size(); index++) {
		if (!used[index]) continue;
		const auto kind    = static_cast<Attribute>(index);
		const bool input   = kind == Attribute::ControlInput || kind == Attribute::EvaluationInput ||
		                   kind == Attribute::CarriedInput ||
		                   (kind == Attribute::PatchOutput &&
		                    state.program.stage == ShaderType::TessellationEvaluation);
		const auto storage = input ? spv::StorageClassInput : spv::StorageClassOutput;
		const bool carried = kind == Attribute::CarriedOutput || kind == Attribute::CarriedInput;
		uint32_t   type;
		if (kind == Attribute::Factor) {
			type = array(TypeF32(state), 4u);
		} else if (carried) {
			uint32_t count = 0;
			for (const auto word: tess.carried_vgprs) count += static_cast<uint32_t>(std::popcount(word));
			EXIT_IF(count == 0u);
			type = array(TypeU32Vector(state, 4), (count + 3u) / 4u);
			if (kind == Attribute::CarriedInput) type = array(type, tess.input_control_points);
		} else if (kind == Attribute::PatchOutput) {
			// KYTY_TESS_INDEXED: the range the analysis found the HS storing, so that the HS's
			// patch outputs and the TES's inputs agree whatever each of them touches.
			if (tess.patch_size != 0u) {
				EXIT_NOT_IMPLEMENTED(patch_begin < tess.patch_base ||
				                     patch_end > tess.patch_base + tess.patch_size);
				patch_begin = tess.patch_base;
				patch_end   = tess.patch_base + tess.patch_size;
			}
			state.tess_patch_base = patch_begin;
			type = array(TypeU32Vector(state, 4), (patch_end - patch_begin + 15u) / 16u);
		} else {
			const bool local  = kind == Attribute::LocalOutput || kind == Attribute::ControlInput;
			const auto stride = local ? tess.ls_stride : tess.hs_stride;
			type              = array(TypeU32Vector(state, 4), (stride + 15u) / 16u);
			if (kind != Attribute::LocalOutput) {
				type = array(type, local ? tess.input_control_points : tess.output_control_points);
			}
		}
		auto& variable = state.tess_variables[index];
		variable       = DefineInterfaceVariable(state, type, storage, "tess_attributes");
		if (kind == Attribute::Factor) {
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn,
			                            spv::BuiltInTessLevelOuter);
			state.tess_inner_variable =
			    DefineInterfaceVariable(state, array(TypeF32(state), 2u), storage, "tess_inner");
			state.builder.AddAnnotation(spv::OpDecorate, state.tess_inner_variable,
			                            spv::DecorationBuiltIn, spv::BuiltInTessLevelInner);
			state.builder.AddAnnotation(spv::OpDecorate, state.tess_inner_variable,
			                            spv::DecorationPatch);
		} else {
			state.builder.AddAnnotation(
			    spv::OpDecorate, variable, spv::DecorationLocation,
			    kind == Attribute::PatchOutput ? (tess.hs_stride + 15u) / 16u
			    : carried                      ? (tess.ls_stride + 15u) / 16u
			                                   : 0u);
		}
		if (kind == Attribute::Factor || kind == Attribute::PatchOutput) {
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationPatch);
		}
	}
}

void DefineTessellationExecutionModes(EmitterState& state) {
	const auto& tess = state.input_info.vertex->tess;
	// VGT_TF_PARAM: TYPE 0 isoline, 1 triangle, 2 quad; PARTITIONING 0 integer, 1 pow2,
	// 2 fractional odd, 3 fractional even; TOPOLOGY 0 point, 1 line, 2 triangle cw, 3 ccw.
	EXIT_NOT_IMPLEMENTED(tess.domain > 2u);
	// Lines only come from isolines and triangles only from the triangle and quad domains.
	EXIT_NOT_IMPLEMENTED(tess.output_topology != 0u &&
	                     (tess.output_topology == 1u) != (tess.domain == 0u));
	state.builder.RequireCapability(spv::CapabilityTessellation);
	if (state.program.stage == ShaderType::TessellationControl) {
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeOutputVertices,
		                               tess.output_control_points);
	} else {
		state.builder.AddExecutionMode(state.main_func, tess.domain == 0u ? spv::ExecutionModeIsolines
		                                                : tess.domain == 1u
		                                                    ? spv::ExecutionModeTriangles
		                                                    : spv::ExecutionModeQuads);
		// Vulkan has no power-of-two spacing; equal spacing is its nearest integer mode.
		state.builder.AddExecutionMode(state.main_func,
		                               tess.partitioning == 2u ? spv::ExecutionModeSpacingFractionalOdd
		                               : tess.partitioning == 3u
		                                   ? spv::ExecutionModeSpacingFractionalEven
		                                   : spv::ExecutionModeSpacingEqual);
		if (tess.output_topology == 0u) {
			state.builder.AddExecutionMode(state.main_func, spv::ExecutionModePointMode);
		} else if (tess.output_topology >= 2u) {
			state.builder.AddExecutionMode(state.main_func, tess.output_topology == 2u
			                                                    ? spv::ExecutionModeVertexOrderCw
			                                                    : spv::ExecutionModeVertexOrderCcw);
		}
	}
}

uint32_t EmitGetTessellationAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, 2), [&] {
		const auto value = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), value,
		                              TessellationPointer(ctx, inst));
		return value;
	});
}

void EmitSetTessellationAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, 3), [&] {
		auto value = ctx.Arg(inst, 2);
		if (inst.Arg(0).U32() == static_cast<uint32_t>(IR::TessellationAttribute::Factor)) {
			auto&      state    = ctx.state;
			const auto floating = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitcast, TypeF32(state), floating, value);
			value = floating;
			const auto& tess = state.input_info.vertex->tess;
			if (tess.max_tess_level != 0u && std::bit_cast<float>(tess.min_tess_level) <=
			                                     std::bit_cast<float>(tess.max_tess_level)) {
				// The tessellator clamps factors to VGT_HOS_MIN/MAX_TESS_LEVEL. Patches with a
				// factor <= 0 or NaN stay culled.
				const auto clamped = state.builder.AllocateId();
				state.builder.AddFunction(std::vector<uint32_t> {
				    spv::OpExtInst, TypeF32(state), clamped, GlslStd450(state), GLSLstd450FClamp,
				    value, ConstantF32Value(state, std::bit_cast<float>(tess.min_tess_level)),
				    ConstantF32Value(state, std::bit_cast<float>(tess.max_tess_level))});
				const auto positive = Binary(state, spv::OpFOrdGreaterThan, TypeBool(state), value,
				                             ConstantF32Value(state, 0.0f));
				value = Select(state, TypeF32(state), positive, clamped, value);
			}
			if (tess.partitioning == 1u) {
				// Power-of-two partitioning rounds each factor up to a power of two; with the
				// equal spacing it maps to, the generated vertices then match the guest's.
				// A factor <= 0 (or NaN) still culls the patch.
				const auto ext = [&](uint32_t opcode, uint32_t arg) {
					const auto result = state.builder.AllocateId();
					state.builder.AddFunction(std::vector<uint32_t> {
					    spv::OpExtInst, TypeF32(state), result, GlslStd450(state), opcode, arg});
					return result;
				};
				const auto one          = ConstantF32Value(state, 1.0f);
				const auto at_least_one = state.builder.AllocateId();
				state.builder.AddFunction(std::vector<uint32_t> {spv::OpExtInst, TypeF32(state),
				                                                 at_least_one, GlslStd450(state),
				                                                 GLSLstd450FMax, value, one});
				const auto rounded =
				    ext(GLSLstd450Exp2, ext(GLSLstd450Ceil, ext(GLSLstd450Log2, at_least_one)));
				const auto positive = Binary(state, spv::OpFOrdGreaterThan, TypeBool(state), value,
				                             ConstantF32Value(state, 0.0f));
				value = Select(state, TypeF32(state), positive, rounded, value);
			}
		}
		ctx.state.builder.AddFunction(spv::OpStore, TessellationPointer(ctx, inst), value);
	});
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
