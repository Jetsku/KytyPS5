#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/recompiler/CodegenOptions.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

bool UserDataDwordIndex(const EmitterState& state, IR::ScalarReg reg, uint32_t& dword_index) {
	const auto register_index = IR::RegIndex(reg);
	const auto& registers = state.program.bindings.user_data_registers;
	const auto  found     = std::lower_bound(registers.begin(), registers.end(), register_index);
	if (found == registers.end() || *found != register_index) {
		return false;
	}
	dword_index = static_cast<uint32_t>(found - registers.begin());
	return true;
}

uint32_t EmitBuiltinU32(EmitterState& state, IR::StageInputKind kind, uint32_t component) {
	if (kind == IR::StageInputKind::LocalInvocationIndex) {
		return EmitLocalInvocationIndex(state);
	}
	if (state.lane_count == 2 && (kind == IR::StageInputKind::LocalInvocationId ||
	                              kind == IR::StageInputKind::GlobalInvocationId)) {
		const auto* cs      = ShaderWorkgroupInput(state.program.stage, state.input_info);
		uint32_t    divisor = 1;
		for (uint32_t axis = 0; axis < component; axis++) {
			divisor *= std::max(cs->threads_num[axis], 1u);
		}
		const auto size    = std::max(cs->threads_num[component], 1u);
		const auto divided = EmitBinaryU32(state, spv::OpUDiv, EmitLocalInvocationIndex(state),
		                                   ConstantU32(state, divisor));
		const auto local   = EmitBinaryU32(state, spv::OpUMod, divided, ConstantU32(state, size));
		if (kind == IR::StageInputKind::LocalInvocationId) {
			return local;
		}
		const auto group = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, component);
		return EmitAddU32(state, local,
		                  EmitBinaryU32(state, spv::OpIMul, group, ConstantU32(state, size)));
	}
	const bool centroid = kind == IR::StageInputKind::BaryCoordSmoothCentroid ||
	                      kind == IR::StageInputKind::BaryCoordNoPerspectiveCentroid;
	const bool sample   = kind == IR::StageInputKind::BaryCoordSmoothSample ||
	                    kind == IR::StageInputKind::BaryCoordNoPerspectiveSample;
	const bool linear   = kind == IR::StageInputKind::BaryCoordNoPerspectiveCentroid ||
	                    kind == IR::StageInputKind::BaryCoordNoPerspectiveSample;
	const auto variable = InputVariableForKind(
	    state, centroid || sample ? (linear ? IR::StageInputKind::BaryCoordNoPerspective
	                                        : IR::StageInputKind::BaryCoordSmooth)
	                              : kind);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	if (sample) {
		// The sample I/J pair: the barycentrics at the current sample's position.
		const auto sample_id_variable = InputVariableForKind(state, IR::StageInputKind::SampleId);
		const auto sample_id          = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeI32(state), sample_id, sample_id_variable);
		const auto coordinates = state.builder.AllocateId();
		state.builder.RequireCapability(spv::CapabilityInterpolationFunction);
		state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 3), coordinates,
		                          GlslStd450(state), GLSLstd450InterpolateAtSample, variable,
		                          sample_id);
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), value, coordinates,
		                          component + 1u);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	if (kind == IR::StageInputKind::FrontFacing) {
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeBool(state), value, variable);
		// SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS selects uint masks or float +/-1.
		const bool all_bits =
		    state.input_info.pixel != nullptr && state.input_info.pixel->ps_front_face_all_bits;
		const auto front = ConstantU32(state, all_bits ? 0xffffffffu : 0x3f800000u);
		const auto back  = ConstantU32(state, all_bits ? 0u : 0xbf800000u);
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), bits, value, front, back);
		return bits;
	}
	if (kind == IR::StageInputKind::VertexIndex || kind == IR::StageInputKind::InstanceIndex ||
	    kind == IR::StageInputKind::BaseInstance || kind == IR::StageInputKind::BaseVertex ||
	    kind == IR::StageInputKind::InvocationId || kind == IR::StageInputKind::PrimitiveId ||
	    kind == IR::StageInputKind::Layer || kind == IR::StageInputKind::SampleId) {
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeI32(state), value, variable);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	if (kind == IR::StageInputKind::FragCoord || kind == IR::StageInputKind::TessCoord) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		const auto bits    = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
		                          pointer, variable, ConstantU32(state, component));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	if (centroid || kind == IR::StageInputKind::BaryCoordSmooth ||
	    kind == IR::StageInputKind::BaryCoordNoPerspective) {
		const auto value   = state.builder.AllocateId();
		const auto bits    = state.builder.AllocateId();
		if (centroid) {
			const auto coordinates = state.builder.AllocateId();
			state.builder.RequireCapability(spv::CapabilityInterpolationFunction);
			state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 3), coordinates,
			                          GlslStd450(state), GLSLstd450InterpolateAtCentroid, variable);
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), value,
			                          coordinates, component + 1u);
		} else {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain,
			                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
			                          pointer, variable, ConstantU32(state, component + 1u));
			state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		}
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	return EmitInputComponentU32(state, kind, component);
}

// Whether EXEC enables a source lane within the caller's 32-lane half.
uint32_t EmitSourceLaneActive(ValueEmitContext& ctx, uint32_t exec, uint32_t lane) {
	auto&      state  = ctx.state;
	const auto scope  = ConstantU32(state, spv::ScopeSubgroup);
	const auto ballot = EmitNative<spv::OpGroupNonUniformBallot, IR::Type::U32x4>(state, scope, exec);
	// A split wave64 (KYTY_WAVE64_WARPS) numbers its lanes 0-63 over two 32-lane host subgroups, and
	// DPP never leaves a half: the source's bit in this subgroup's ballot is lane mod 32
	// (KYTY_SPLIT_DPP_LANE).
	if (ctx.half != 0 || (state.split_wave && GetCodegenOptions().split_dpp_lane)) {
		lane = EmitAndConstant(state, lane, 31);
	}
	return EmitNative<spv::OpGroupNonUniformBallotBitExtract, IR::Type::U1>(state, scope, ballot,
	                                                                        lane);
}

uint32_t EmitDppWriteCondition(ValueEmitContext& ctx, const IR::DppMoveFlags& flags,
                               uint32_t exec) {
	auto&    state    = ctx.state;
	uint32_t writable = exec;
	uint32_t lane     = 0;
	// Each mask bit enables writes to one bank (4 lanes) or row (16 lanes); a full mask enables all.
	const auto enable = [&](uint32_t mask, uint32_t group_shift) {
		if (mask == 0xfu) return;
		lane = lane != 0 ? lane : EmitSubgroupLocalInvocationId(state);
		const auto group = EmitNative<spv::OpBitFieldUExtract, IR::Type::U32>(
		    state, lane, ConstantU32(state, group_shift), ConstantU32(state, 2));
		const auto bit = EmitNative<spv::OpBitFieldUExtract, IR::Type::U32>(
		    state, ConstantU32(state, mask), group, ConstantU32(state, 1));
		writable = Binary(state, spv::OpLogicalAnd, TypeBool(state), writable,
		                  Binary(state, spv::OpINotEqual, TypeBool(state), bit, ConstantU32(state, 0)));
	};
	enable(flags.bank_mask, 2u);
	enable(flags.row_mask, 4u);
	// Only row shifts (0x101-0x10f, 0x111-0x11f) can vacate a lane; other controls always read one.
	const bool row_shift = flags.control > 0x100u && flags.control < 0x120u &&
	                       flags.control != 0x110u;
	// DPP8 has no BOUND_CTRL; an inactive source reads zero there (handled by the read), not a skipped write.
	if (flags.dpp8 || flags.bound_control || (flags.fetch_inactive && !row_shift)) {
		return writable;
	}
	// Without BOUND_CTRL, a vacated source or, without FI, an EXEC-disabled one disables the write.
	const auto target = EmitDppTargetLane(state, flags);
	if (row_shift) {
		writable = Binary(state, spv::OpLogicalAnd, TypeBool(state), writable, target.valid);
	}
	// The EXEC-disabled source rule is KYTY_DPP_SKIP_INACTIVE's (EmitDppUpdateU32 applies it too); off,
	// the legacy zero read stays.
	if (!flags.fetch_inactive && GetCodegenOptions().dpp_skip_inactive) {
		// Vacated lanes target outside their row; wrapping keeps the ballot index in range.
		auto source = target.lane;
		if (row_shift) {
			lane   = lane != 0 ? lane : EmitSubgroupLocalInvocationId(state);
			source = EmitNative<spv::OpBitFieldInsert, IR::Type::U32>(
			    state, lane, source, ConstantU32(state, 0), ConstantU32(state, 4));
		}
		writable = Binary(state, spv::OpLogicalAnd, TypeBool(state), writable,
		                  EmitSourceLaneActive(ctx, exec, source));
	}
	return writable;
}

uint32_t LoadVertexParameter(EmitterState& state, const InputBinding& input, uint32_t vertex,
                             uint32_t chan) {
	const auto pointer = state.builder.AllocateId();
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassInput, TypeF32(state)), pointer,
	                          input.variable_id, ConstantU32(state, vertex),
	                          ConstantU32(state, chan & 3u));
	state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
	return value;
}

// Interpolates a pixel input channel to f32. A half (0 low, 1 high) selects one f16 of a packed
// FP16_INTERP_MODE dword before blending; smooth inputs read by halves are always per-vertex.
uint32_t EmitPixelAttribute(EmitterState& state, const InputBinding& input, uint32_t attr,
                            uint32_t chan, std::optional<uint32_t> half) {
	const auto select = [&](uint32_t value) {
		if (!half) {
			return value;
		}
		const auto bits     = state.builder.AllocateId();
		const auto unpacked = state.builder.AllocateId();
		const auto word     = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 2), unpacked,
		                          GlslStd450(state), GLSLstd450UnpackHalf2x16, bits);
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), word, unpacked, *half);
		return word;
	};
	if (!input.per_vertex) {
		const auto vector    = state.builder.AllocateId();
		const auto component = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeF32Vector(state, 4), vector, input.variable_id);
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, vector,
		                          chan & 3u);
		return select(component);
	}
	if (PixelParameterIsFlat(state, attr)) {
		const auto vertex =
		    state.input_info.pixel->parameter_mode == ShaderPixelParameterMode::LastVertex ? 2u : 0u;
		return select(LoadVertexParameter(state, input, vertex, chan));
	}
	const auto barycentric_kind = state.input_info.pixel->ps_no_perspective
	                                  ? IR::StageInputKind::BaryCoordNoPerspective
	                                  : IR::StageInputKind::BaryCoordSmooth;
	const auto barycentric      = InputVariableForKind(state, barycentric_kind);
	uint32_t   sum              = 0;
	for (uint32_t vertex = 0; vertex < 3u; vertex++) {
		const auto pointer = state.builder.AllocateId();
		const auto weight  = state.builder.AllocateId();
		const auto product = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
		                          pointer, barycentric, ConstantU32(state, vertex));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), weight, pointer);
		state.builder.AddFunction(spv::OpFMul, TypeF32(state), product,
		                          select(LoadVertexParameter(state, input, vertex, chan)), weight);
		if (vertex == 0u) {
			sum = product;
		} else {
			const auto next = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpFAdd, TypeF32(state), next, sum, product);
			sum = next;
		}
	}
	return sum;
}

uint32_t EmitAttribute(EmitterState& state, uint32_t attr, uint32_t chan,
                       IR::InterpolationMode mode = IR::InterpolationMode::Unknown) {
	const auto* input = InputBindingForParameter(state, attr);
	if (input == nullptr || input->variable_id == 0) {
		return ConstantU32(state, 0);
	}
	if (state.program.stage == ShaderType::Vertex || state.program.stage == ShaderType::Local) {
		return EmitVertexParameterComponentU32(state, *input, chan & 3u);
	}
	const auto load_per_vertex = [&](uint32_t vertex) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, spv::StorageClassInput, TypeF32(state)), pointer,
		    input->variable_id, ConstantU32(state, vertex), ConstantU32(state, chan & 3u));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		return value;
	};
	if (input->per_vertex) {
		if (PixelParameterIsFlat(state, attr)) {
			const auto value = load_per_vertex(
			    state.input_info.pixel->parameter_mode == ShaderPixelParameterMode::LastVertex ? 2u : 0u);
			const auto bits  = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
			return bits;
		}
		const auto barycentric_kind = state.input_info.pixel->ps_no_perspective
		                                  ? IR::StageInputKind::BaryCoordNoPerspective
		                                  : IR::StageInputKind::BaryCoordSmooth;
		const auto barycentric      = InputVariableForKind(state, barycentric_kind);
		uint32_t   sum              = 0;
		for (uint32_t vertex = 0; vertex < 3u; vertex++) {
			const auto pointer = state.builder.AllocateId();
			const auto weight  = state.builder.AllocateId();
			const auto product = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain,
			                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
			                          pointer, barycentric, ConstantU32(state, vertex));
			state.builder.AddFunction(spv::OpLoad, TypeF32(state), weight, pointer);
			state.builder.AddFunction(spv::OpFMul, TypeF32(state), product, load_per_vertex(vertex),
			                          weight);
			if (vertex == 0u) {
				sum = product;
			} else {
				const auto next = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFAdd, TypeF32(state), next, sum, product);
				sum = next;
			}
		}
		const auto bits = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, sum);
		return bits;
	}
	const auto vector    = state.builder.AllocateId();
	const auto component = state.builder.AllocateId();
	const auto bits      = state.builder.AllocateId();
	const auto decided   = state.input_interpolation.find(input->variable_id);
	if (decided != state.input_interpolation.end() &&
	    (decided->second == IR::InterpolationMode::PerspectiveCenter ||
	     decided->second == IR::InterpolationMode::LinearCenter) &&
	    (mode == IR::InterpolationMode::PerspectiveCentroid ||
	     mode == IR::InterpolationMode::LinearCentroid)) {
		// A centroid read of an input that other reads use at the pixel center.
		state.builder.RequireCapability(spv::CapabilityInterpolationFunction);
		state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 4), vector,
		                          GlslStd450(state), GLSLstd450InterpolateAtCentroid,
		                          input->variable_id);
	} else {
		state.builder.AddFunction(spv::OpLoad, TypeF32Vector(state, 4), vector,
		                          input->variable_id);
	}
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, vector,
	                          chan & 3u);
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, component);
	return bits;
}

uint32_t EmitInterpolationParameter(ValueEmitContext& ctx, uint32_t attr, uint32_t chan,
                                    uint32_t mode) {
	auto&       state = ctx.state;
	const auto* input = InputBindingForParameter(state, attr);
	if (!input->per_vertex) {
		return EmitAttribute(ctx.state, attr, chan);
	}
	uint32_t selected_vertex = (mode + 1u) % 3u;
	if (mode == 2u && PixelParameterIsFlat(state, attr)) {
		selected_vertex =
		    state.input_info.pixel->parameter_mode == ShaderPixelParameterMode::LastVertex ? 2u : 0u;
	}
	uint32_t value = LoadVertexParameter(state, *input, selected_vertex, chan);
	if (!PixelParameterIsCustom(state, attr) && mode < 2u) {
		const auto delta = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFSub, TypeF32(state), delta, value,
		                          LoadVertexParameter(state, *input, 0, chan));
		value = delta;
	}
	const auto bits = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
	return bits;
}

uint32_t MrtOutputMode(const EmitterState& state, const IR::ExportInfo& exp) {
	if (state.program.stage != ShaderType::Pixel || exp.kind != IR::ExportTargetKind::Mrt ||
	    exp.index >= std::size(state.input_info.pixel->target_output_mode)) {
		return 0;
	}
	return state.input_info.pixel->target_output_mode[exp.index];
}

uint32_t ExportRawComponent(ValueEmitContext& ctx, uint32_t vector, uint32_t component) {
	const auto value = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), value, vector,
	                              component);
	return value;
}

uint32_t ExportVector(ValueEmitContext& ctx, uint32_t data, const IR::ExportInfo& exp,
                      uint32_t vector_type) {
	auto&      state          = ctx.state;
	const bool integer_output = vector_type != TypeF32Vector(state, 4);
	if (exp.compr && !integer_output) {
		// SPI_SHADER_COL_FORMAT of a compressed export: 5 UNORM16_ABGR, 6 SNORM16_ABGR,
		// 8 SINT16_ABGR (not modelled: it would need an integer output), otherwise FP16_ABGR.
		const auto mode   = MrtOutputMode(state, exp);
		const auto unpack = mode == 5u   ? GLSLstd450UnpackUnorm2x16
		                    : mode == 6u ? GLSLstd450UnpackSnorm2x16
		                                 : GLSLstd450UnpackHalf2x16;
		if (mode == 8u) {
			static std::atomic_flag warned = ATOMIC_FLAG_INIT;
			if (!warned.test_and_set(std::memory_order_relaxed)) {
				std::fputs("Warning: SINT16_ABGR compressed color exports are unpacked as FP16.\n",
				           stderr);
			}
		}
		uint32_t f32[4] = {ConstantF32(state, 0), ConstantF32(state, 0), ConstantF32(state, 0),
		                   ConstantF32(state, 0x3f800000u)};
		for (uint32_t pair = 0; pair < 2u; pair++) {
			if ((exp.en & (3u << (pair * 2u))) == 0u) {
				continue;
			}
			const auto packed   = state.builder.AllocateId();
			const auto unpacked = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), packed, data, pair);
			state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 2), unpacked,
			                          GlslStd450(state), unpack, packed);
			for (uint32_t lane = 0; lane < 2u; lane++) {
				const auto component = pair * 2u + lane;
				if (((exp.en >> component) & 1u) != 0u) {
					f32[component] = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state),
					                          f32[component], unpacked, lane);
				}
			}
		}
		const auto vector = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), vector,
		                          f32[0], f32[1], f32[2], f32[3]);
		return vector;
	}
	if (exp.compr || exp.en != 0xfu) {
		uint32_t raw[4] = {
		    ConstantU32(state, 0),
		    ConstantU32(state, 0),
		    ConstantU32(state, 0),
		    ConstantU32(state, integer_output ? 1u : 0x3f800000u),
		};
		if (exp.compr) {
			const auto extract = vector_type == TypeU32Vector(state, 4) ? spv::OpBitFieldUExtract
			                                                            : spv::OpBitFieldSExtract;
			for (uint32_t pair = 0; pair < 2u; pair++) {
				if ((exp.en & (3u << (pair * 2u))) == 0u) {
					continue;
				}
				const auto packed = ExportRawComponent(ctx, data, pair);
				for (uint32_t lane = 0; lane < 2u; lane++) {
					const auto component = pair * 2u + lane;
					if (((exp.en >> component) & 1u) == 0u) {
						continue;
					}
					raw[component] = state.builder.AllocateId();
					state.builder.AddFunction(extract, TypeU32(state), raw[component], packed,
					                          ConstantU32(state, lane * 16u),
					                          ConstantU32(state, 16));
				}
			}
		} else {
			for (uint32_t component = 0; component < 4u; component++) {
				if (((exp.en >> component) & 1u) != 0u) {
					raw[component] = ExportRawComponent(ctx, data, component);
				}
			}
		}
		data = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), data,
		                          raw[0], raw[1], raw[2], raw[3]);
	}
	if (vector_type == TypeU32Vector(state, 4)) {
		return data;
	}
	return Unary(state, spv::OpBitcast, vector_type, data);
}

uint32_t DistancePointer(EmitterState& state, IR::StageOutputKind kind, uint32_t index) {
	if (state.program.stage == ShaderType::Mesh) {
		return MeshOutputPointer(state, kind, index);
	}
	const auto variable = kind == IR::StageOutputKind::ClipDistance ? state.clip_distance_variable
	                                                                : state.cull_distance_variable;
	const auto pointer  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassOutput, TypeF32(state)), pointer,
	                          variable, ConstantU32(state, index));
	return pointer;
}

// The clipper keeps the half space where dot(plane, position) >= 0 for each UCP_ENA plane
// (upstream feed075a9). The planes are push-constant dwords from PushData::UserClipPlaneDword.
void EmitUserClipDistances(EmitterState& state, uint32_t position) {
	uint32_t index = 0;
	for (uint32_t plane = 0; plane < 32u; plane++) {
		if ((state.user_clip_planes & (1u << plane)) == 0) {
			continue;
		}
		const auto dword = IR::PushData::UserClipPlaneDword + 4u * plane;
		uint32_t   components[4] {};
		for (uint32_t i = 0; i < 4; i++) {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state),
			                          pointer, state.push_constant_variable, ConstantU32(state, 0),
			                          ConstantU32(state, dword + i));
			const auto bits = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeU32(state), bits, pointer);
			components[i] = Unary(state, spv::OpBitcast, TypeF32(state), bits);
		}
		const auto plane_vector = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), plane_vector,
		                          components[0], components[1], components[2], components[3]);
		state.builder.AddFunction(
		    spv::OpStore, DistancePointer(state, IR::StageOutputKind::ClipDistance, index++),
		    Binary(state, spv::OpDot, TypeF32(state), plane_vector, position));
	}
}

void EmitAuxPositionExport(ValueEmitContext& ctx, uint32_t data, const IR::ExportInfo& exp) {
	auto& state = ctx.state;
	for (uint32_t component = 0; component < 4; component++) {
		if ((exp.en & (1u << component)) == 0) {
			continue;
		}
		const auto output = IR::DecodePositionExportComponent(
		    state.input_info.vertex->pa_cl_vs_out_cntl, exp.index, component);
		if (output.layer || output.viewport) {
			const auto raw = ExportRawComponent(ctx, data, component);
			if (output.layer) {
				auto layer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), layer, raw,
				                          ConstantU32(state, 0x7ffu));
				if (GetCodegenOptions().layer_clamp) {
					// KYTY_LAYER_CLAMP: an index past the bound layers (the pipeline's
					// LayerLimitSpecId; 0 keeps the index) selects layer 0.
					if (state.layer_limit_constant == 0) {
						state.layer_limit_constant =
						    state.builder.Constant(spv::OpSpecConstant, TypeU32(state), 0u);
						state.builder.AddAnnotation(spv::OpDecorate, state.layer_limit_constant,
						                            spv::DecorationSpecId, IR::LayerLimitSpecId);
					}
					const auto limit    = state.layer_limit_constant;
					const auto in_range = Binary(
					    state, spv::OpLogicalOr, TypeBool(state),
					    Binary(state, spv::OpIEqual, TypeBool(state), limit, ConstantU32(state, 0)),
					    Binary(state, spv::OpULessThan, TypeBool(state), layer, limit));
					layer = Select(state, TypeU32(state), in_range, layer, ConstantU32(state, 0));
				}
				const auto pointer = state.program.stage == ShaderType::Mesh
				                         ? MeshOutputPointer(state, IR::StageOutputKind::Layer)
				                         : state.layer_variable;
				state.builder.AddFunction(spv::OpStore, pointer, layer);
			}
			if (output.viewport) {
				// GFX10 MISC.z packs the viewport index in bits 16..19 alongside the layer.
				const auto viewport = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), viewport, raw,
				                          ConstantU32(state, 16), ConstantU32(state, 4));
				state.builder.AddFunction(spv::OpStore, state.viewport_index_variable, viewport);
			}
			continue;
		}
		if (!output.point_size && output.clip_distance == UINT32_MAX &&
		    output.cull_distance == UINT32_MAX) {
			continue;
		}

		const auto raw = ExportRawComponent(ctx, data, component);
		const auto f32 = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32, raw);
		if (output.point_size) {
			state.builder.AddFunction(spv::OpStore, state.point_size_variable, f32);
			continue;
		}
		auto StoreDistance = [&](IR::StageOutputKind kind, uint32_t index) {
			if (index != UINT32_MAX) {
				state.builder.AddFunction(spv::OpStore, DistancePointer(state, kind, index), f32);
			}
		};
		StoreDistance(IR::StageOutputKind::ClipDistance, output.clip_distance);
		StoreDistance(IR::StageOutputKind::CullDistance, output.cull_distance);
	}
}

uint32_t ConvertClipCoordinate(EmitterState& state, uint32_t coordinate, float scale,
                               float offset, float half_extent) {
	const auto window  = state.builder.AllocateId();
	const auto biased  = state.builder.AllocateId();
	const auto divided = state.builder.AllocateId();
	const auto ndc     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFMul, TypeF32(state), window, coordinate,
	                          ConstantF32Value(state, scale));
	state.builder.AddFunction(spv::OpFAdd, TypeF32(state), biased, window,
	                          ConstantF32Value(state, offset));
	state.builder.AddFunction(spv::OpFDiv, TypeF32(state), divided, biased,
	                          ConstantF32Value(state, half_extent));
	state.builder.AddFunction(spv::OpFSub, TypeF32(state), ndc, divided,
	                          ConstantF32Value(state, 1.0f));
	return ndc;
}

// KYTY_VTE_CNTL: the CLIP_DISABLE emulation's mapping in homogeneous form (the offset scaled by W,
// W subtracted); for W == 1 the same as ConvertClipCoordinate. Vulkan divides the result by W.
uint32_t ConvertClipCoordinateHomogeneous(EmitterState& state, uint32_t coordinate, uint32_t w,
                                          float scale, float offset, float half_extent) {
	const auto window   = state.builder.AllocateId();
	const auto offset_w = state.builder.AllocateId();
	const auto biased   = state.builder.AllocateId();
	const auto divided  = state.builder.AllocateId();
	const auto ndc      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFMul, TypeF32(state), window, coordinate,
	                          ConstantF32Value(state, scale));
	state.builder.AddFunction(spv::OpFMul, TypeF32(state), offset_w, w,
	                          ConstantF32Value(state, offset));
	state.builder.AddFunction(spv::OpFAdd, TypeF32(state), biased, window, offset_w);
	state.builder.AddFunction(spv::OpFDiv, TypeF32(state), divided, biased,
	                          ConstantF32Value(state, half_extent));
	state.builder.AddFunction(spv::OpFSub, TypeF32(state), ndc, divided, w);
	return ndc;
}

uint32_t ConvertPositionToClipSpace(EmitterState& state, uint32_t position) {
	const auto& transform = state.input_info.vertex->clip_space;
	uint32_t    components[4] {};
	for (uint32_t i = 0; i < 4; i++) {
		components[i] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), components[i], position,
		                          i);
	}
	if (GetCodegenOptions().vte_cntl) {
		// KYTY_VTE_CNTL (masterzhang233/KytyPS5-GT7 df7fc6108): a 1/W or pre-divided position is
		// made a plain clip position again.
		const auto format = ViewportTransform::UnpackPositionFormat(transform.vte_format);
		if ((format & ViewportTransform::IncomingW) == 0) {
			const auto w = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpFDiv, TypeF32(state), w, ConstantF32Value(state, 1.0f),
			                          components[3]);
			components[3] = w;
		}
		for (uint32_t i = 0; i < 3; ++i) {
			const auto predivided =
			    i < 2 ? ViewportTransform::PredividedXY : ViewportTransform::PredividedZ;
			if ((format & predivided) != 0) {
				const auto homogeneous = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFMul, TypeF32(state), homogeneous, components[i],
				                          components[3]);
				components[i] = homogeneous;
			}
		}
		if (transform.enabled) {
			components[0] =
			    ConvertClipCoordinateHomogeneous(state, components[0], components[3],
			                                     transform.scale[0], transform.offset[0],
			                                     transform.half_extent[0]);
			components[1] =
			    ConvertClipCoordinateHomogeneous(state, components[1], components[3],
			                                     transform.scale[1], transform.offset[1],
			                                     transform.half_extent[1]);
		}
	} else {
		components[0] = ConvertClipCoordinate(state, components[0], transform.scale[0],
		                                      transform.offset[0], transform.half_extent[0]);
		components[1] = ConvertClipCoordinate(state, components[1], transform.scale[1],
		                                      transform.offset[1], transform.half_extent[1]);
	}
	const auto converted = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), converted,
	                          components[0], components[1], components[2], components[3]);
	return converted;
}

} // namespace
uint32_t EmitWqmU64(EmitterState& state, uint32_t value) {
	const auto shifted_one = state.builder.AllocateId();
	const auto merged_one  = state.builder.AllocateId();
	const auto shifted_two = state.builder.AllocateId();
	const auto merged_two  = state.builder.AllocateId();
	const auto quad_bits   = state.builder.AllocateId();
	const auto result      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU64(state), shifted_one, value,
	                          ConstantU32(state, 1));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU64(state), merged_one, value, shifted_one);
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU64(state), shifted_two, merged_one,
	                          ConstantU32(state, 2));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU64(state), merged_two, merged_one,
	                          shifted_two);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU64(state), quad_bits, merged_two,
	                          ConstantU64(state, 0x1111111111111111ull));
	state.builder.AddFunction(spv::OpIMul, TypeU64(state), result, quad_bits,
	                          ConstantU64(state, 15));
	return result;
}

void EmitSetAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state = ctx.state;
	const auto& exp   = ctx.Export(inst);
	const auto  exec  = ctx.Arg(inst, 1);
	if (state.program.stage == ShaderType::Pixel && exp.vm && state.requirements.pixel_valid_mask &&
	    state.pixel_valid_mask_variable != 0) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), value, exec, ConstantU32(state, 1),
		                          ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpStore, state.pixel_valid_mask_variable, value);
	}
	if (exp.kind == IR::ExportTargetKind::Null || exp.en == 0u) {
		return;
	}
	// Skip dormant color exports after their valid mask; MRT1 is reserved for logical alpha.
	if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
	    exp.index != 0 && state.input_info.pixel->alpha_blend_source != ShaderAlphaBlendSource::None) {
		return;
	}
	EmitIfCondition(state, exec, [&]() {
		const auto data = ctx.Arg(inst, 0);
		if (exp.kind == IR::ExportTargetKind::Primitive) {
			if (state.program.stage == ShaderType::Mesh) {
				state.builder.AddFunction(spv::OpStore, MeshPrimitivePointer(state),
				                          ExportRawComponent(ctx, data, 0));
			}
			return;
		}
		if (exp.kind == IR::ExportTargetKind::Position && exp.index != 0) {
			EmitAuxPositionExport(ctx, data, exp);
			return;
		}
		if (exp.kind == IR::ExportTargetKind::MrtZ) {
			if ((exp.en & 1u) != 0u && state.depth_variable != 0) {
				const auto raw = ExportRawComponent(ctx, data, 0);
				const auto f32 = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32, raw);
				state.builder.AddFunction(spv::OpStore, state.depth_variable, f32);
			}
			if ((exp.en & 4u) != 0u && state.sample_mask_variable != 0) {
				const auto raw     = ExportRawComponent(ctx, data, 2);
				const auto value   = state.builder.AllocateId();
				const auto pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitcast, TypeI32(state), value, raw);
				state.builder.AddFunction(
				    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, TypeI32(state)),
				    pointer, state.sample_mask_variable, ConstantU32(state, 0));
				state.builder.AddFunction(spv::OpStore, pointer, value);
			}
			return;
		}
		const auto variable =
		    state.program.stage == ShaderType::Mesh ? 0u : OutputVariableForExport(state, exp);
		if (state.program.stage != ShaderType::Mesh && variable == 0) {
			return;
		}
		// KYTY_INT_MRT_OUTPUTS=1 (upstream 26ca60923): exports to UINT and SINT targets are typed
		// as integer vectors of the target's signedness. Off: only 32_ABGR exports (mode 7) are
		// unsigned, every other export a float vector (the behaviour before).
		const auto output_mode = MrtOutputMode(state, exp);
		const auto vector_type = GetCodegenOptions().int_mrt_outputs
		                             ? (output_mode != 0u ? MrtOutputType(state, exp.index)
		                                                  : TypeF32Vector(state, 4))
		                             : (output_mode == 7u ? TypeU32Vector(state, 4)
		                                                  : TypeF32Vector(state, 4));
		const bool uint_output = vector_type != TypeF32Vector(state, 4);
		auto       value       = ExportVector(ctx, data, exp, vector_type);
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    !uint_output && !exp.compr && exp.index < 8u &&
		    (state.input_info.pixel->target_half_float_mask & (1u << exp.index)) != 0u) {
			// KYTY_FLOAT_STORE_CLAMP: a 32-bit export to a 16-bit float target keeps the guest's
			// range (the largest half, not the host's infinity). Packed half exports are in range.
			uint32_t components[4] {};
			for (uint32_t component = 0; component < 4u; component++) {
				const auto bits = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), bits,
				                          Unary(state, spv::OpBitcast, TypeU32Vector(state, 4), value),
				                          component);
				components[component] = Unary(state, spv::OpBitcast, TypeF32(state),
				                              EmitClampToHalfRangeBits(state, bits));
			}
			value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, vector_type, value, components[0],
			                          components[1], components[2], components[3]);
		}
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    exp.index == 0 && !uint_output &&
		    state.input_info.pixel->alpha_blend_source != ShaderAlphaBlendSource::None) {
			// Broadcast logical alpha before swizzling the primary output.
			const auto blend_output =
			    OutputVariableForExport(state, {.kind = IR::ExportTargetKind::Mrt, .index = 1});
			if (blend_output != 0) {
				auto alpha = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVectorShuffle, vector_type, alpha, value, value,
				                          3u, 3u, 3u, 3u);
				uint32_t alpha_factor = 0;
				switch (state.input_info.pixel->alpha_blend_source) {
					case ShaderAlphaBlendSource::SourceAlphaOne:
						alpha_factor = ConstantF32Value(state, 1.0f);
						break;
					case ShaderAlphaBlendSource::SourceAlphaZero:
						alpha_factor = ConstantF32Value(state, 0.0f);
						break;
					default: break;
				}
				if (state.input_info.pixel->alpha_blend_source ==
				    ShaderAlphaBlendSource::OneAlphaZero) {
					// KYTY_UI_TEXT_BLEND (from Franseven): physical RGB 1 for logical colour, 0 for
					// logical alpha; physical A keeps Sa.
					const auto one_mapping = state.input_info.pixel->target_export_mapping[0];
					for (uint32_t component = 0; component < 3; component++) {
						const auto factors = state.builder.AllocateId();
						state.builder.AddFunction(
						    spv::OpCompositeInsert, vector_type, factors,
						    ConstantF32Value(state, one_mapping.Map(component) == 3u ? 0.0f : 1.0f),
						    alpha, component);
						alpha = factors;
					}
				}
				if (alpha_factor != 0) {
					const auto mapping = state.input_info.pixel->target_export_mapping[0];
					for (uint32_t component = 0; component < 4; component++) {
						if (mapping.Map(component) != 3u) continue;
						const auto factors = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpCompositeInsert, vector_type, factors,
						                          alpha_factor, alpha, component);
						alpha = factors;
						break;
					}
				}
				state.builder.AddFunction(spv::OpStore, blend_output, alpha);
			}
		}
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    exp.index < state.input_info.pixel->target_export_mapping.size()) {
			const auto mapping = state.input_info.pixel->target_export_mapping[exp.index];
			if (!mapping.IsIdentity()) {
				const auto mapped = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVectorShuffle, vector_type, mapped, value, value,
				                          mapping.Map(0), mapping.Map(1), mapping.Map(2),
				                          mapping.Map(3));
				value = mapped;
			}
		}
		if (exp.kind == IR::ExportTargetKind::Position &&
		    (state.input_info.vertex->clip_space.enabled ||
		     state.input_info.vertex->clip_space.vte_format != 0)) {
			value = ConvertPositionToClipSpace(state, value);
		}
		if (exp.kind == IR::ExportTargetKind::Position && state.user_clip_planes != 0) {
			EmitUserClipDistances(state, value);
		}
		if (state.program.stage == ShaderType::Mesh) {
			const auto kind = exp.kind == IR::ExportTargetKind::Position
			                      ? IR::StageOutputKind::Position
			                      : IR::StageOutputKind::Parameter;
			state.builder.AddFunction(spv::OpStore, MeshOutputPointer(state, kind, exp.index),
			                          value);
		} else if (exp.kind == IR::ExportTargetKind::Position) {
			if (state.invalid_position_clip_distance != UINT32_MAX) {
				const auto zero = state.builder.Constant(spv::OpConstantNull, TypeF32Vector(state, 4));
				const auto equal = state.builder.AllocateId();
				const auto invalid = state.builder.AllocateId();
				const auto distance = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFOrdEqual, TypeBoolVector(state, 4), equal,
				                          value, zero);
				state.builder.AddFunction(spv::OpAll, TypeBool(state), invalid, equal);
				// Zero at valid vertices makes a primitive containing an invalid position
				// collapse to its remaining edge, before the undefined 0/0 perspective divide.
				state.builder.AddFunction(spv::OpSelect, TypeF32(state), distance, invalid,
				                          ConstantF32Value(state, -1.0f),
				                          ConstantF32Value(state, 0.0f));
				state.builder.AddFunction(spv::OpStore,
				                          DistancePointer(state, IR::StageOutputKind::ClipDistance,
				                                          state.invalid_position_clip_distance),
				                          distance);
				static std::atomic_bool logged = false;
				if (!logged.exchange(true, std::memory_order_relaxed)) {
					Log::WriteToConsoleAndLog(
					    "Shader: emitted zero-position clip guard\n");
				}
			}
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(
			    spv::OpAccessChain,
			    TypePointer(state, spv::StorageClassOutput, TypeF32Vector(state, 4)), pointer,
			    variable, ConstantU32(state, 0));
			state.builder.AddFunction(spv::OpStore, pointer, value);
		} else {
			state.builder.AddFunction(spv::OpStore, variable, value);
		}
	});
}

uint32_t EmitIdentity(ValueEmitContext&, uint32_t value) {
	return value;
}

void EmitVoid(ValueEmitContext&) {}

void EmitStoreCompletion(EmitterState& state) {
	// A split wave spans two host subgroups of its workgroup.
	NoteWaveSync(state);
	state.builder.AddFunction(
	    spv::OpControlBarrier,
	    ConstantU32(state, state.split_wave ? spv::ScopeWorkgroup : spv::ScopeSubgroup),
	    ConstantU32(state, spv::ScopeDevice),
	    ConstantU32(state, spv::MemorySemanticsReleaseMask | spv::MemorySemanticsUniformMemoryMask |
	                          spv::MemorySemanticsImageMemoryMask));
}

void EmitDataCacheInvalidate(EmitterState& state) {
	// RDNA2 GL0/GL1 invalidation makes vector memory visible without a workgroup rendezvous.
	state.builder.AddFunction(
	    spv::OpMemoryBarrier, ConstantU32(state, spv::ScopeDevice),
	    ConstantU32(state, spv::MemorySemanticsAcquireMask | spv::MemorySemanticsUniformMemoryMask |
	                          spv::MemorySemanticsImageMemoryMask));
}

void EmitBarrier(EmitterState& state) {
	const auto tessellation = state.program.stage == ShaderType::TessellationControl;
	if (!tessellation && ShaderWorkgroupInput(state.program.stage, state.input_info) == nullptr) {
		// Independent graphics invocations have no native workgroup left to synchronize.
		return;
	}
	const auto memory_scope = tessellation ? spv::ScopeInvocation : spv::ScopeWorkgroup;
	auto memory = LdsMemorySemantics(state);
	// KYTY_BARRIER_ACQUIRE_STORES: waves read the other waves' buffer and image stores with plain
	// loads right after S_BARRIER (mainline ceb154d9e), so a program that stores also orders them.
	const auto written = [](const auto& resource) { return resource.written; };
	if (GetCodegenOptions().barrier_acquire_stores &&
	    (state.requirements.buffer_writes || state.requirements.image_writes ||
	     std::ranges::any_of(state.program.info.buffers, written) ||
	     std::ranges::any_of(state.program.info.images, written))) {
		memory = memory | spv::MemorySemanticsUniformMemoryMask | spv::MemorySemanticsImageMemoryMask;
	}
	const auto semantics = tessellation ? spv::MemorySemanticsMaskNone
	                                    : spv::MemorySemanticsAcquireReleaseMask | memory;
	NoteWaveSync(state);
	state.builder.AddFunction(spv::OpControlBarrier, ConstantU32(state, spv::ScopeWorkgroup),
	                          ConstantU32(state, memory_scope), ConstantU32(state, semantics));
}

void EmitSharedMemoryBarrier(EmitterState& state) {
	// S_WAITCNT lgkmcnt(0) after LDS writes: lanes of one guest wave live in different host
	// invocations of one subgroup (two lanes per invocation for a split wave64), so make their
	// LDS writes visible before the wave reads the exchanged data. Only compute-like stages own
	// workgroup memory; Vulkan rejects Workgroup memory scope elsewhere. The wait orders only the
	// wave's own operations, so the barrier stays at subgroup scope (KYTY_LDS_WAITCNT_BARRIER):
	// it may sit where other waves of the workgroup do not go, and the AMD driver makes a
	// workgroup-scope fence in a mesh shader a workgroup barrier, which then never completes.
	if (state.program.stage == ShaderType::TessellationControl ||
	    ShaderWorkgroupInput(state.program.stage, state.input_info) == nullptr) {
		return;
	}
	const auto scope = GetCodegenOptions().lds_waitcnt_barrier == LdsWaitcntBarrier::Workgroup
	                       ? spv::ScopeWorkgroup
	                       : spv::ScopeSubgroup;
	state.builder.AddFunction(spv::OpMemoryBarrier, ConstantU32(state, scope),
	                          ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask |
	                                                 LdsMemorySemantics(state)));
}

uint32_t EmitLaneId(EmitterState& state) {
	return state.program.stage == ShaderType::TessellationControl
	           ? EmitBuiltinU32(state, IR::StageInputKind::InvocationId, 0)
	           : EmitSubgroupLocalInvocationId(state);
}

uint32_t EmitMeshDrawParameter(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto index  = inst.Arg(0).U32();
	if (state.program.stage != ShaderType::Mesh ||
	    index >= IR::PushData::MeshDrawDwords(state.input_info.vertex->mesh.split_groups != 0)) {
		ctx.Fail(inst, "invalid mesh draw parameter");
	}
	const auto push_dword = [&](uint32_t dword) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state), pointer,
		                          state.push_constant_variable, ConstantU32(state, 0),
		                          ConstantU32(state, dword));
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	const auto pushed = push_dword(index);
	// KYTY_NATIVE_INDIRECT_MESH (renderer/meshIndirect.h): a native indirect mesh draw pushes
	// MeshIndirectSentinel as dword 3 (the index size, never that value on the CPU path) and the
	// device address of its parameter block, written on the GPU from the guest's argument record,
	// as dwords 0-1. The branch is uniform (push constants). With the option the module declares
	// 64-bit integers and physical addresses for every mesh program (UsesPhysicalAddresses).
	if (!GetCodegenOptions().mesh_indirect_params || !UsesPhysicalAddresses(state)) {
		return pushed;
	}
	const auto indirect = Binary(state, spv::OpIEqual, TypeBool(state), push_dword(3),
	                             ConstantU32(state, IR::PushData::MeshIndirectSentinel));
	return EmitValueOrDefaultIfCondition(state, indirect, TypeU32(state), pushed, [&]() {
		const auto u64 = TypeU64(state);
		const auto low = Unary(state, spv::OpUConvert, u64, push_dword(0));
		const auto high =
		    Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, push_dword(1)),
		           state.builder.Constant(spv::OpConstant, u64, 32u, 0u));
		const auto address =
		    Binary(state, spv::OpIAdd, u64, Binary(state, spv::OpBitwiseOr, u64, low, high),
		           state.builder.Constant(spv::OpConstant, u64, index * 4u, 0u));
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
		                          address);
		const auto         value     = state.builder.AllocateId();
		constexpr uint32_t alignment = sizeof(uint32_t);
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
		                          spv::MemoryAccessAlignedMask, alignment);
		return value;
	});
}

uint32_t EmitGetUserData(EmitterState& state, IR::ScalarReg reg) {

	uint32_t dword = 0;
	if (!UserDataDwordIndex(state, reg, dword)) {
		return ConstantU32(state, 0);
	} else {
		return EmitShaderDataDwordLoad(state, dword);
	}
}

uint32_t EmitGetBuiltin(ValueEmitContext& ctx, IR::Value kind, IR::Value index) {
	return EmitBuiltinU32(ctx.state, static_cast<IR::StageInputKind>(kind.U32()), index.U32());
}

uint32_t EmitGetDispatchThreadExtent(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto start = ctx.state.program.bindings.dispatch_thread_dword;
	if (start == IR::PushData::NoStart || !inst.Arg(0).IsImmediate() || inst.Arg(0).U32() >= 3u) {
		ctx.Fail(inst, "invalid dispatch thread extent");
	}
	if (!DispatchDimensionsIndirect(ctx.state)) {
		return EmitShaderDataDwordLoad(ctx.state, start + inst.Arg(0).U32());
	}
	// KYTY_INDIRECT_THREADS_GPU (GT7 fork 633e00962): the counts stay in GPU memory, a copy the
	// renderer made of the guest's indirect arguments; the shader data holds its device address.
	auto&      state = ctx.state;
	const auto u64   = TypeU64(state);
	const auto low   = Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, start));
	const auto high  = Binary(state, spv::OpShiftLeftLogical, u64,
	                          Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, start + 1u)),
	                          state.builder.Constant(spv::OpConstant, u64, 32u, 0u));
	const auto address =
	    Binary(state, spv::OpIAdd, u64, Binary(state, spv::OpBitwiseOr, u64, low, high),
	           state.builder.Constant(spv::OpConstant, u64, inst.Arg(0).U32() * 4u, 0u));
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, address);
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
	                          spv::MemoryAccessAlignedMask, static_cast<uint32_t>(sizeof(uint32_t)));
	return value;
}

uint32_t EmitUndefU1(EmitterState& state, const IR::Inst& inst) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUndef, TypeId(state, inst.GetType()), result);
	return result;
}

uint32_t EmitDppMoveU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state    = ctx.state;
	const auto flags    = inst.Flags<IR::DppMoveFlags>();
	const auto target   = EmitDppTargetLane(state, flags);
	// DPP reads within rows of 16 lanes, which never leave a split wave's half.
	const auto shuffled = state.split_wave ? ReadHostSubgroupLane(state, TypeU32(state),
	                                                              ctx.Arg(inst, 0), target.lane)
	                                       : ctx.Shuffle(inst, 0, target.lane);
	if (flags.fetch_inactive) {
		return shuffled;
	}
	const auto ballot = state.split_wave ? EmitHalfBallot(state, ctx.Arg(inst, 1))
	                                     : ctx.Ballot(inst.Arg(1));
	const auto source_active = EmitBallotLaneActiveBool(state, ballot, target.lane);
	const auto can_fetch     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), can_fetch, target.valid,
	                          source_active);
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, can_fetch, shuffled,
	                                                ConstantU32(state, 0));
}

uint32_t EmitDppUpdateU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto flags = inst.Flags<IR::DppMoveFlags>();
	auto       write = EmitDppWriteCondition(ctx, flags, ctx.Arg(inst, 2));
	if (GetCodegenOptions().dpp_skip_inactive && !flags.bound_control && !flags.fetch_inactive &&
	    !flags.dpp8) {
		// A source lane EXEC disables is invalid like a vacated one, and without bound_ctrl the
		// receiving lane keeps its value (PS5 ISA, DPP options); so does a lane the host subgroup
		// lacks. EmitDppMoveU32 read zero from it (KYTY_DPP_SKIP_INACTIVE, from Senaxx's wolverine
		// branch).
		const auto target        = EmitDppTargetLane(ctx.state, flags);
		const auto ballot        = ctx.state.split_wave ? EmitHalfBallot(ctx.state, ctx.Arg(inst, 2))
		                                                    : ctx.Ballot(inst.Arg(2));
		const auto source_active = EmitBallotLaneActiveBool(ctx.state, ballot, target.lane);
		write = Binary(ctx.state, spv::OpLogicalAnd, TypeBool(ctx.state), write, source_active);
	}
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, write, ctx.Arg(inst, 0),
	                                                ctx.Arg(inst, 1));
}

uint32_t EmitBallot(ValueEmitContext& ctx, IR::Value predicate) {
	return ctx.Ballot(predicate);
}

uint32_t EmitAnyLane(ValueEmitContext& ctx, IR::Value predicate) {
	auto&      state  = ctx.state;
	const auto ballot = ctx.Ballot(predicate);
	const auto low    = state.builder.AllocateId();
	const auto high   = state.builder.AllocateId();
	const auto any    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), any,
	                          EmitBinaryU32(state, spv::OpBitwiseOr, low, high),
	                          ConstantU32(state, 0));
	return any;
}

// KYTY_UNIFORM_LANE_READS (from BryanKAdams/KytyPS5 d514872): after a shuffle by a uniform lane
// every invocation holds the same value, so OpGroupNonUniformBroadcastFirst of it returns it
// unchanged. It tells the host compiler that the value is uniform: AMD compiles the shuffle alone
// to ds_bpermute and treats the result, and everything derived from a waterfall loop's key
// (addresses, scalar loads, loop exits), as per-lane vector work.
static uint32_t MarkLaneReadUniform(ValueEmitContext& ctx, const IR::Inst& inst,
                                    uint32_t shuffled) {
	// KYTY_WAVE32_CLUSTERS: the subgroup's first invocation may belong to the other guest wave.
	if (!GetCodegenOptions().uniform_lane_reads || ctx.state.wave_clusters) {
		return shuffled;
	}
	auto&      state  = ctx.state;
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBroadcastFirst,
	                          TypeId(state, inst.Arg(0).GetType()), result,
	                          ConstantU32(state, spv::ScopeSubgroup), shuffled);
	return result;
}

// A waterfall loop's read: the loop compares each lane's value with the one read, handles the
// lanes that match and repeats for the rest (EmK530/KytyPS5 420fc4ead).
static bool IsWaterfallRead(const IR::Inst& inst) {
	const auto source = inst.Arg(0).Resolve();
	return !source.IsImmediate() && std::ranges::any_of(inst.Uses(), [&](const IR::Use& use) {
		return use.user->GetOpcode() == IR::ValueOpcode::IEqual32 && use.operand < 2 &&
		       use.user->Arg(1 - use.operand).Resolve() == source;
	});
}

uint32_t EmitReadFirstLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (state.split_wave && IsWaterfallRead(inst)) {
		// Each half of a split wave can read its own first active lane: the loop handles the lanes
		// whose value matches and repeats for the rest. With no active lane it reads lane 0.
		const auto word  = EmitHostBallotWord(state, ctx.Arg(inst, 1));
		const auto first = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpExtInst, TypeU32(state), first, GlslStd450(state),
		                          GLSLstd450FindILsb, word);
		const auto lane = Select(state, TypeU32(state),
		                         Binary(state, spv::OpINotEqual, TypeBool(state), word,
		                                ConstantU32(state, 0)),
		                         first, ConstantU32(state, 0));
		return ReadHostSubgroupLane(state, TypeId(state, inst.Arg(0).GetType()), ctx.Arg(inst, 0),
		                            lane);
	}
	const auto ballot = ctx.Ballot(inst.Arg(1));
	const auto first  = ctx.FirstLane(ballot);
	// With EXEC == 0, V_READFIRSTLANE_B32 reads lane 0; FindLSB of an empty ballot is -1.
	uint32_t any_bits = ConstantU32(state, 0);
	for (uint32_t word = 0; word < 4u; word++) {
		const auto bits = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), bits, ballot, word);
		any_bits = EmitBinaryU32(state, spv::OpBitwiseOr, any_bits, bits);
	}
	const auto active = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), active, any_bits,
	                          ConstantU32(state, 0));
	const auto lane = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), lane, active, first,
	                          ConstantU32(state, 0));
	return MarkLaneReadUniform(ctx, inst, ctx.Shuffle(inst, 0, lane));
}

// A split wave's reduction: each host subgroup reduces the range's lanes it holds, and the halves
// combine their results through the exchange.
static uint32_t EmitSplitLaneReduction(ValueEmitContext& ctx, const IR::LaneReduction& reduction) {
	auto&      state = ctx.state;
	const auto guest = EmitLocalInvocationIndex(state);
	const auto in_range =
	    Binary(state, spv::OpULessThan, TypeBool(state),
	           Binary(state, spv::OpISub, TypeU32(state), guest, ConstantU32(state, reduction.first_lane)),
	           ConstantU32(state, reduction.lanes));
	const auto identity     = ConstantU32(state, IR::ReductionIdentity(reduction.operation));
	const auto contribution = Select(state, TypeU32(state), in_range, ctx.Def(reduction.source), identity);
	spv::Op    operation    = spv::OpGroupNonUniformIAdd;
	spv::Op    combine      = spv::OpIAdd;
	uint32_t   glsl         = 0;
	switch (reduction.operation) {
		case IR::ValueOpcode::UMax32: operation = spv::OpGroupNonUniformUMax; glsl = GLSLstd450UMax; break;
		case IR::ValueOpcode::UMin32: operation = spv::OpGroupNonUniformUMin; glsl = GLSLstd450UMin; break;
		case IR::ValueOpcode::SMax32: operation = spv::OpGroupNonUniformSMax; glsl = GLSLstd450SMax; break;
		case IR::ValueOpcode::SMin32: operation = spv::OpGroupNonUniformSMin; glsl = GLSLstd450SMin; break;
		case IR::ValueOpcode::BitwiseOr32:
			operation = spv::OpGroupNonUniformBitwiseOr;
			combine   = spv::OpBitwiseOr;
			break;
		case IR::ValueOpcode::BitwiseAnd32:
			operation = spv::OpGroupNonUniformBitwiseAnd;
			combine   = spv::OpBitwiseAnd;
			break;
		case IR::ValueOpcode::BitwiseXor32:
			operation = spv::OpGroupNonUniformBitwiseXor;
			combine   = spv::OpBitwiseXor;
			break;
		default: break;
	}
	const auto half_result = state.builder.AllocateId();
	state.builder.AddFunction(operation, TypeU32(state), half_result,
	                          ConstantU32(state, spv::ScopeSubgroup), spv::GroupOperationReduce,
	                          contribution);
	// Every lane of a half holds its half's result: read the other half's from the mirror lane.
	const auto other = ExchangeWaveValue(
	    state, half_result, EmitBinaryU32(state, spv::OpBitwiseXor, guest, ConstantU32(state, 32)));
	if (glsl != 0) {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpExtInst, TypeU32(state), result, GlslStd450(state), glsl,
		                          half_result, other);
		return result;
	}
	return EmitBinaryU32(state, combine, half_result, other);
}

// The lanes' reduction, natively over the lanes the host subgroup has (IR::MatchLaneReduction,
// KYTY_LANE_REDUCTIONS; from Senaxx's wolverine branch).
static uint32_t EmitLaneReduction(ValueEmitContext& ctx, const IR::LaneReduction& reduction) {
	auto& state = ctx.state;
	if (state.split_wave) {
		return EmitSplitLaneReduction(ctx, reduction);
	}
	// A wave64 on a 32-wide subgroup keeps lanes 32-63 in the second half.
	const auto host_lanes = state.lane_count == 2 ? 32u : state.program.wave_size;
	auto&      lane = reduction.first_lane / host_lanes == ctx.half ? ctx : *ctx.other_half;
	const auto subid = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), subid,
	                          state.subgroup_local_invocation_id_variable);
	const auto in_range =
	    Binary(state, spv::OpULessThan, TypeBool(state),
	           Binary(state, spv::OpISub, TypeU32(state), subid,
	                  ConstantU32(state, reduction.first_lane % host_lanes)),
	           ConstantU32(state, reduction.lanes));
	const auto contribution =
	    Select(state, TypeU32(state), in_range, lane.Def(reduction.source),
	           ConstantU32(state, IR::ReductionIdentity(reduction.operation)));
	spv::Op operation = spv::OpGroupNonUniformIAdd;
	switch (reduction.operation) {
		case IR::ValueOpcode::UMax32: operation = spv::OpGroupNonUniformUMax; break;
		case IR::ValueOpcode::UMin32: operation = spv::OpGroupNonUniformUMin; break;
		case IR::ValueOpcode::SMax32: operation = spv::OpGroupNonUniformSMax; break;
		case IR::ValueOpcode::SMin32: operation = spv::OpGroupNonUniformSMin; break;
		case IR::ValueOpcode::BitwiseOr32: operation = spv::OpGroupNonUniformBitwiseOr; break;
		case IR::ValueOpcode::BitwiseAnd32: operation = spv::OpGroupNonUniformBitwiseAnd; break;
		case IR::ValueOpcode::BitwiseXor32: operation = spv::OpGroupNonUniformBitwiseXor; break;
		default: break;
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(operation, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), spv::GroupOperationReduce,
	                          contribution);
	return result;
}

// KYTY_WATERFALL_ACTIVE_LEADER (Erlou93/KytyPS5-Yotei WaterfallPendingLow): the waterfall form of
// V_READLANE, whose lane is S_FF1_I32_B64(pending) & 63:
// BitwiseAnd32(SelectU32(INotEqual32(low, 0), FindILsb32(low), ...), 63). Returns the pending mask's
// low word (a graphics wave64 runs in one 32-wide host subgroup, its high word is 0), or empty.
static IR::Value WaterfallPendingLow(const IR::Inst& inst) {
	const auto* masked = inst.Arg(1).Resolve().TryInstruction();
	if (masked == nullptr || masked->GetOpcode() != IR::ValueOpcode::BitwiseAnd32) return {};
	const auto is_63 = [](IR::Value v) {
		v = v.Resolve();
		return v.IsImmediate() && v.GetType() == IR::Type::U32 && v.U32() == 63u;
	};
	const IR::Inst* select = nullptr;
	if (is_63(masked->Arg(1))) select = masked->Arg(0).Resolve().TryInstruction();
	else if (is_63(masked->Arg(0))) select = masked->Arg(1).Resolve().TryInstruction();
	if (select == nullptr || select->GetOpcode() != IR::ValueOpcode::SelectU32) return {};
	const auto* nonzero = select->Arg(0).Resolve().TryInstruction();
	const auto* first   = select->Arg(1).Resolve().TryInstruction();
	if (nonzero == nullptr || first == nullptr || nonzero->GetOpcode() != IR::ValueOpcode::INotEqual32 ||
	    first->GetOpcode() != IR::ValueOpcode::FindILsb32) {
		return {};
	}
	const auto low  = first->Arg(0).Resolve();
	const auto zero = nonzero->Arg(1).Resolve();
	if (!(nonzero->Arg(0).Resolve() == low) || !zero.IsImmediate() || zero.U32() != 0u) return {};
	return low;
}

// KYTY_WATERFALL_ACTIVE_LEADER: a waterfall loop (S_FF1 pending -> V_READLANE -> V_CMP_EQ ->
// S_AND_SAVEEXEC -> S_ANDN2 pending) of a graphics shader needs its leader to be a lane that runs the
// compare in this iteration, but the lowest pending bit can name a host invocation that is not active
// at the read. The leader is the lowest lane that is pending AND active; with only inactive lanes
// pending the read returns a key that no active invocation holds (max + 1, else min - 1).
static uint32_t EmitWaterfallLeaderRead(ValueEmitContext& ctx, const IR::Inst& inst,
                                        const IR::Value& pending_low) {
	auto&      state    = ctx.state;
	const auto u32      = TypeU32(state);
	const auto boolean  = TypeBool(state);
	const auto zero     = ConstantU32(state, 0);
	const auto subgroup = ConstantU32(state, spv::ScopeSubgroup);
	const auto lane     = ctx.Arg(inst, 1);
	const auto native   = EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31));
	const auto ballot   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot, subgroup,
	                          ConstantBool(state, true));
	const auto active = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, u32, active, ballot, 0);
	const auto pending = ctx.Def(pending_low);
	const auto live    = EmitBinaryU32(state, spv::OpBitwiseAnd, pending, active);
	const auto first   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, u32, first, GlslStd450(state), GLSLstd450FindILsb, live);
	const auto has_live = Binary(state, spv::OpINotEqual, boolean, live, zero);
	const auto leader   = Select(state, u32, has_live, first, native);
	const auto value    = ctx.Shuffle(inst, 0, leader);
	state.builder.RequireCapability(spv::CapabilityGroupNonUniformArithmetic);
	const auto source = ctx.Arg(inst, 0);
	const auto high   = state.builder.AllocateId();
	const auto low    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformUMax, u32, high, subgroup, spv::GroupOperationReduce, source);
	state.builder.AddFunction(spv::OpGroupNonUniformUMin, u32, low, subgroup, spv::GroupOperationReduce, source);
	const auto below = Select(state, u32, Binary(state, spv::OpINotEqual, boolean, low, zero),
	                          EmitBinaryU32(state, spv::OpISub, low, ConstantU32(state, 1)), value);
	const auto distinct =
	    Select(state, u32, Binary(state, spv::OpINotEqual, boolean, high, ConstantU32(state, 0xffffffffu)),
	           EmitBinaryU32(state, spv::OpIAdd, high, ConstantU32(state, 1)), below);
	const auto pending_any = Binary(state, spv::OpINotEqual, boolean, pending, zero);
	const auto exhausted   = Binary(state, spv::OpLogicalAnd, boolean,
	                                Unary(state, spv::OpLogicalNot, boolean, has_live), pending_any);
	return Select(state, u32, exhausted, distinct, value);
}

uint32_t EmitReadLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (GetCodegenOptions().waterfall_active_leader && ctx.other_half == nullptr &&
	    ctx.state.lane_count == 1u && !ctx.state.split_wave && !ctx.state.wave_clusters &&
	    ctx.state.program.stage != ShaderType::Compute && inst.Arg(0).GetType() == IR::Type::U32) {
		if (const auto pending_low = WaterfallPendingLow(inst); !pending_low.IsEmpty()) {
			return EmitWaterfallLeaderRead(ctx, inst, pending_low);
		}
	}
	// KYTY_WAVE32_CLUSTERS: a subgroup reduction would cover both guest waves; the scan stays.
	if (GetCodegenOptions().lane_reductions && !ctx.state.wave_clusters) {
		if (const auto reduction = IR::MatchLaneReduction(inst, ctx.state.program.wave_size)) {
			// One reduction per wave: the second half of a wave64 that one invocation runs
			// (lane_count 2) takes the first half's result, which covers the same lanes.
			if (ctx.half == 1) {
				return ctx.other_half->Def(IR::Value(const_cast<IR::Inst*>(&inst)));
			}
			return EmitLaneReduction(ctx, *reduction);
		}
	}
	// V_READLANE's lane is an SGPR, M0 or a constant, so the shuffle's lane is uniform too.
	return MarkLaneReadUniform(ctx, inst, ctx.Shuffle(inst, 0, ctx.Arg(inst, 1)));
}

uint32_t EmitWriteLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto hit   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), hit,
	                          EmitSubgroupLocalInvocationId(state), ctx.Arg(inst, 2));
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, hit, ctx.Arg(inst, 1),
	                                                ctx.Arg(inst, 0));
}

uint32_t EmitPermlane16U32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state    = ctx.state;
	const auto flags    = inst.Flags<IR::PermlaneFlags>();
	const auto subid    = EmitSubgroupLocalInvocationId(state);
	const auto row      = state.builder.AllocateId();
	const auto lane     = state.builder.AllocateId();
	const auto lane8    = state.builder.AllocateId();
	const auto shift    = state.builder.AllocateId();
	const auto upper    = state.builder.AllocateId();
	const auto selected = state.builder.AllocateId();
	const auto shifted  = state.builder.AllocateId();
	const auto index    = state.builder.AllocateId();
	const auto target   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, subid,
	                          ConstantU32(state, 0xfffffff0u));
	const auto row_value =
	    flags.x16 ? Binary(state, spv::OpBitwiseXor, TypeU32(state), row, ConstantU32(state, 16))
		          : row;
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, 15));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane8, lane,
	                          ConstantU32(state, 7));
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), shift, lane8,
	                          ConstantU32(state, 2));
	state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), upper, lane,
	                          ConstantU32(state, 8));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), selected, upper, ctx.Arg(inst, 2),
	                          ctx.Arg(inst, 1));
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), shifted, selected, shift);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), index, shifted,
	                          ConstantU32(state, 15));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, row_value, index);
	// Permlane16 reads within the caller's 32-lane half, a split wave's host subgroup.
	const auto read = [&](size_t argument) {
		return state.split_wave
		           ? ReadHostSubgroupLane(state, TypeId(state, inst.Arg(argument).GetType()),
		                                  ctx.Arg(inst, argument), target)
		           : ctx.Shuffle(inst, argument, target);
	};
	const auto shuffled = read(0);
	if (flags.fetch_inactive) {
		return shuffled;
	}
	// Upstream 67c89e091: an inactive source lane without BOUND_CTRL keeps the old destination.
	const auto source_active =
	    state.split_wave ? read(3) : EmitSourceLaneActive(ctx, ctx.Arg(inst, 3), target);
	return Select(state, TypeU32(state), source_active, shuffled, ctx.Arg(inst, 4));
}
uint32_t EmitGetAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitAttribute(ctx.state, inst.Arg(0).U32(), inst.Arg(1).U32(),
	                     static_cast<IR::InterpolationMode>(inst.Flags<uint32_t>()));
}

uint32_t EmitGetAttributeWithBary(ValueEmitContext& ctx, const IR::Inst& inst) {
	ctx.Fail(inst, "was not resolved to GetAttribute by constant propagation");
}

uint32_t EmitGetAttributeF16(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  attr  = inst.Arg(0).U32();
	const auto* input = InputBindingForParameter(ctx.state, attr);
	if (input == nullptr || input->variable_id == 0) {
		return ConstantF32Value(ctx.state, 0.0f);
	}
	return EmitPixelAttribute(ctx.state, *input, attr, inst.Arg(1).U32(), inst.Arg(2).U32());
}

uint32_t EmitGetInterpolationParameter(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitInterpolationParameter(ctx, inst.Arg(0).U32(), inst.Arg(1).U32(), inst.Arg(2).U32());
}

uint32_t EmitGetShaderBase(ValueEmitContext& ctx) {
	// Guest S_GETPC values stay shader-relative in SPIR-V, matching the runtime ABI. The
	// runtime descriptor evaluator supplies the mapped shader base for host-side planning.
	return ctx.Def(IR::Value(uint64_t {0}));
}

uint32_t EmitReadClockRealtime64(ValueEmitContext& ctx, const IR::Inst& inst) {
	// S_MEMREALTIME is a scalar instruction: one value per wave. The second half of a wave64 that
	// one invocation runs (lane_count 2) takes the first half's value.
	if (ctx.half == 1) {
		return ctx.other_half->Def(IR::Value(const_cast<IR::Inst*>(&inst)));
	}
	auto&      state = ctx.state;
	const auto clock = GetHostShaderClock();
	if (clock.scope == HostClockScope::None) {
		// No shader clock on this device: the placeholder.
		return ctx.Def(IR::Value(UINT64_MAX));
	}
	// Read the clock once as a uvec2 (the halves cannot tear) and take the subgroup's first active
	// invocation's value, so every lane sees the same time, as on the guest.
	const auto read = state.builder.AllocateId();
	state.builder.AddFunction(
	    spv::OpReadClockKHR, TypeU32Vector(state, 2), read,
	    ConstantU32(state, clock.scope == HostClockScope::Device ? spv::ScopeDevice
	                                                             : spv::ScopeSubgroup));
	const auto uniform = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBroadcastFirst, TypeU32Vector(state, 2), uniform,
	                          ConstantU32(state, spv::ScopeSubgroup), read);
	if (clock.divisor > 1u) {
		// KYTY_MEMREALTIME_EXACT: the whole divisor nearest to 100 MHz (10 for a 1 GHz clock).
		const auto value = Unary(state, spv::OpBitcast, TypeU64(state), uniform);
		return Binary(state, spv::OpUDiv, TypeU64(state), value, ConstantU64(state, clock.divisor));
	}
	if (clock.shift == 0) {
		return Unary(state, spv::OpBitcast, TypeU64(state), uniform);
	}
	// Scale toward 100 MHz with a shift (no 64-bit arithmetic): right by `shift` bits for a faster
	// clock (1 GHz: 125 MHz, so guest timeouts expire slightly early rather than 10x late), left
	// for a slower one.
	const auto low  = state.builder.AllocateId();
	const auto high = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, uniform, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, uniform, 1);
	const bool     right  = clock.shift > 0;
	const uint32_t amount = static_cast<uint32_t>(right ? clock.shift : -clock.shift);
	const auto     shift  = right ? spv::OpShiftRightLogical : spv::OpShiftLeftLogical;
	const auto     carry  = right ? spv::OpShiftLeftLogical : spv::OpShiftRightLogical;
	// Right: low' = low >> n | high << (32 - n), high' = high >> n.
	// Left: high' = high << n | low >> (32 - n), low' = low << n.
	const auto receiving = right ? low : high; // gets the bits that cross the word boundary
	const auto giving    = right ? high : low;
	const auto kept      = Binary(state, shift, TypeU32(state), receiving, ConstantU32(state, amount));
	const auto crossing =
	    Binary(state, carry, TypeU32(state), giving, ConstantU32(state, 32u - amount));
	const auto received = Binary(state, spv::OpBitwiseOr, TypeU32(state), kept, crossing);
	const auto given    = Binary(state, shift, TypeU32(state), giving, ConstantU32(state, amount));
	return PackU64(state, right ? received : given, right ? given : received);
}

void EmitUnreachable(ValueEmitContext& ctx, const IR::Inst& inst) {
	ctx.Fail(inst, "must be lowered before SPIR-V emission");
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
