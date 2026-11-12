#include "common/assert.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

uint32_t PixelParameterLocation(const EmitterState& state, uint32_t attr) {
	std::array<uint32_t, 32> active_inputs {};
	uint32_t                 active_count = 0;
	for (const auto& input: state.inputs) {
		if (input.kind == IR::StageInputKind::Parameter) {
			active_inputs[active_count++] = input.location;
		}
	}
	return state.program.stage == ShaderType::Pixel
	           ? ShaderPixelParameterLocation(*state.input_info.pixel,
	                                          {active_inputs.data(), active_count}, attr)
	           : attr;
}

bool PixelParameterIsFlat(const EmitterState& state, uint32_t attr) {
	return state.program.stage == ShaderType::Pixel &&
	       ShaderPixelParameterIsFlat(*state.input_info.pixel, attr);
}

bool PixelParameterIsCustom(const EmitterState& state, uint32_t attr) {
	return state.program.stage == ShaderType::Pixel &&
	       ShaderPixelParameterIsCustom(*state.input_info.pixel, attr);
}

uint32_t MrtOutputType(EmitterState& state, uint32_t index) {
	const auto& pixel = *state.input_info.pixel;
	if ((pixel.target_sint_mask & (1u << index)) != 0u) return TypeI32Vector(state, 4);
	if ((pixel.target_uint_mask & (1u << index)) != 0u) return TypeU32Vector(state, 4);
	return TypeF32Vector(state, 4);
}

uint32_t OutputVariableForExport(const EmitterState& state, const IR::ExportInfo& exp) {
	if (exp.kind == IR::ExportTargetKind::Position && exp.index == 0) {
		return state.per_vertex_variable;
	}
	if (exp.kind == IR::ExportTargetKind::MrtZ) {
		return state.depth_variable;
	}
	for (const auto& binding: state.outputs) {
		const auto expected_kind = exp.kind == IR::ExportTargetKind::Mrt
		                               ? IR::StageOutputKind::Mrt
		                               : IR::StageOutputKind::Parameter;
		if (binding.kind == expected_kind && binding.index == exp.index) {
			return binding.variable_id;
		}
	}
	return 0;
}

uint32_t          ConstantU32(EmitterState& state, uint32_t value);
[[noreturn]] void ExitDescriptorBindingFailure(const EmitterState&       state,
                                               IR::DescriptorBindingKind kind, uint32_t resource,
                                               const char* reason) {
	EXIT("shader binding resolution failed during SPIR-V emit: hash=0x%016" PRIx64
	     " stage=%u resource=%" PRIu32 " binding_kind=%u reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage), resource,
	     static_cast<unsigned>(kind), reason);
	std::abort();
}

uint32_t ResourceForDescriptor(const EmitterState& state, IR::DescriptorBindingKind kind,
                               uint32_t resource) {
	const auto* descriptor = IR::FindBinding(state.program.bindings, kind);
	if (descriptor == nullptr) {
		ExitDescriptorBindingFailure(state, kind, resource, "descriptor group was not allocated");
	}
	const auto found =
	    std::find(descriptor->resources.begin(), descriptor->resources.end(), resource);
	if (found == descriptor->resources.end()) {
		ExitDescriptorBindingFailure(state, kind, resource,
		                             "resource is absent from descriptor group");
	}
	return static_cast<uint32_t>(found - descriptor->resources.begin());
}

uint32_t DescriptorElementPointer(EmitterState& state, uint32_t result_ptr_type,
                                  uint32_t variable_id, uint32_t array_index,
                                  IR::DescriptorBindingKind kind, uint32_t resource,
                                  const char* variable_name) {
	if (variable_id == 0) {
		ExitDescriptorBindingFailure(state, kind, resource, variable_name);
	}
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, result_ptr_type, pointer, variable_id,
	                          array_index);
	return pointer;
}

const ImageDimensionInfo& ImageDimensionInfoFor(ImageDimension dimension) {
	for (const auto& info: ImageDimensions) {
		if (info.dimension == dimension) {
			return info;
		}
	}
	EXIT("SPIR-V image dimension %u is invalid\n", static_cast<uint32_t>(dimension));
	std::abort();
}

uint32_t ImageScalarType(EmitterState& state, Prospero::TextureNumericClass numeric_class) {
	switch (numeric_class) {
		case Prospero::TextureNumericClass::Float: return TypeF32(state);
		case Prospero::TextureNumericClass::Uint: return TypeU32(state);
		case Prospero::TextureNumericClass::Sint: return TypeI32(state);
		case Prospero::TextureNumericClass::Unsupported: break;
	}
	EXIT("invalid image numeric class");
}

uint32_t ImageVectorType(EmitterState& state, Prospero::TextureNumericClass numeric_class,
                         uint32_t components) {
	return state.builder.Type(spv::OpTypeVector, ImageScalarType(state, numeric_class), components);
}

uint32_t ImageType(EmitterState& state, const IR::ImageResource& image) {
	uint32_t sampled = 0;
	uint32_t format  = spv::ImageFormatUnknown;
	if (image.resource_class == IR::ImageResourceClass::Sampled) {
		EXIT_IF(image.atomic);
		sampled = 1;
	} else if (image.resource_class == IR::ImageResourceClass::Storage) {
		EXIT_IF(image.numeric_class == Prospero::TextureNumericClass::Sint ||
		        image.numeric_class == Prospero::TextureNumericClass::Unsupported);
		sampled = 2;
		if (image.atomic) {
			EXIT_IF(image.numeric_class != Prospero::TextureNumericClass::Uint);
			format = image.atomic64 ? spv::ImageFormatR64ui : spv::ImageFormatR32ui;
			if (image.atomic64) {
				state.builder.RequireExtension("SPV_EXT_shader_image_int64");
				state.builder.RequireCapability(spv::CapabilityInt64);
				state.builder.RequireCapability(spv::CapabilityInt64Atomics);
				state.builder.RequireCapability(spv::CapabilityInt64ImageEXT);
			}
		}
	} else {
		EXIT("invalid image resource class");
	}
	const auto& info = ImageDimensionInfoFor(image.dimension);
	if (sampled == 2 && info.multisampled != 0u) {
		state.builder.RequireCapability(spv::CapabilityStorageImageMultisample);
	}
	const auto  scalar_type =
	    image.atomic64 ? TypeU64(state) : ImageScalarType(state, image.numeric_class);
	return state.builder.Type(spv::OpTypeImage, scalar_type,
	                          info.spirv_dimension, image.depth_compare ? 1u : 0u, info.arrayed,
	                          info.multisampled, sampled, format);
}

uint32_t ImageViewSizeType(EmitterState& state, ImageDimension dimension) {
	const auto& info = ImageDimensionInfoFor(dimension);
	return TypeU32Vector(state, info.coordinate_components + (info.one_dimensional ? 1u : 0u));
}

uint32_t LoadImageDescriptor(EmitterState& state, uint32_t resource, uint32_t mip,
                             uint32_t array_index) {
	const auto& image_resource = ImageResourceAt(state, resource);
	if (image_resource.bindless) {
		EXIT_IF(image_resource.resource_class != IR::ImageResourceClass::Sampled);
		const auto type     = ImageType(state, image_resource);
		const auto variable = state.bindless_image_variables.find(type);
		EXIT_IF(variable == state.bindless_image_variables.end() || state.bindless_slot == 0);
		const auto pointer_type =
		    state.builder.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, type);
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, pointer_type, pointer, variable->second,
		                          state.bindless_slot);
		state.builder.AddAnnotation(spv::OpDecorate, pointer, spv::DecorationNonUniform);
		const auto image = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, type, image, pointer);
		state.builder.AddAnnotation(spv::OpDecorate, image, spv::DecorationNonUniform);
		return image;
	}
	const auto pointer = ImageDescriptorPointer(state, resource, mip, array_index);
	const auto image = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, ImageType(state, ImageResourceAt(state, resource)),
	                          image, pointer);
	return image;
}

uint32_t LoadSamplerDescriptor(EmitterState& state, uint32_t sampler) {
	const auto sampler_type = state.builder.Type(spv::OpTypeSampler);
	const auto pointer_type =
	    state.builder.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, sampler_type);
	if (state.program.info.samplers.at(sampler).bindless) {
		// KYTY_BINDLESS_SAMPLERS: the slot EmitImage looked up for this instruction.
		EXIT_IF(state.bindless_sampler_variable == 0 || state.bindless_sampler_slot == 0);
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, pointer_type, pointer,
		                          state.bindless_sampler_variable, state.bindless_sampler_slot);
		state.builder.AddAnnotation(spv::OpDecorate, pointer, spv::DecorationNonUniform);
		const auto sampler_id = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, sampler_type, sampler_id, pointer);
		state.builder.AddAnnotation(spv::OpDecorate, sampler_id, spv::DecorationNonUniform);
		return sampler_id;
	}
	const auto array_index =
	    ResourceForDescriptor(state, IR::DescriptorBindingKind::Samplers, sampler);
	const auto pointer = DescriptorElementPointer(
	    state, pointer_type, state.sampler_variable, ConstantU32(state, array_index),
	    IR::DescriptorBindingKind::Samplers, sampler, "sampler descriptor array was not emitted");
	const auto sampler_id = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, sampler_type, sampler_id, pointer);
	return sampler_id;
}

uint32_t MakeSampledImage(EmitterState& state, uint32_t resource, uint32_t sampler_id, uint32_t mip,
                          uint32_t array_index) {
	const auto& image_resource = ImageResourceAt(state, resource);
	EXIT_IF(image_resource.resource_class != IR::ImageResourceClass::Sampled);
	const auto  image          = LoadImageDescriptor(state, resource, mip, array_index);
	const auto  sampled_image = state.builder.AllocateId();
	const auto  sampled_type =
	    state.builder.Type(spv::OpTypeSampledImage, ImageType(state, image_resource));
	state.builder.AddFunction(spv::OpSampledImage, sampled_type, sampled_image, image, sampler_id);
	if (array_index != 0u) {
		state.builder.RequireExtension("SPV_EXT_descriptor_indexing");
		state.builder.RequireCapability(spv::CapabilityShaderNonUniform);
		state.builder.RequireCapability(spv::CapabilitySampledImageArrayNonUniformIndexing);
		state.builder.AddAnnotation(spv::OpDecorate, sampled_image, spv::DecorationNonUniform);
	} else if (image_resource.bindless || state.bindless_sampler_slot != 0u) {
		state.builder.AddAnnotation(spv::OpDecorate, sampled_image, spv::DecorationNonUniform);
	}
	return sampled_image;
}

bool ImageAccessCoherent(const EmitterState& state, const IR::MemoryInfo& mem) {
	return (GetCodegenOptions().image_glc_coherent && mem.coherent) || PopsInterlockActive(state);
}

uint32_t ImageDescriptorPointer(EmitterState& state, uint32_t resource, uint32_t mip,
                                uint32_t array_index) {
	const auto& image = ImageResourceAt(state, resource);
	EXIT_IF(mip >= image.mip_count);
	const auto kind = IR::DescriptorBindingForImage(image);
	EXIT_IF(!kind.has_value());
	if (array_index == 0u) {
		array_index = ConstantU32(state, ResourceForDescriptor(state, *kind, resource) + mip);
	}
	const auto pointer_type = state.builder.Type(
	    spv::OpTypePointer, spv::StorageClassUniformConstant, ImageType(state, image));
	auto variable = state.image_variables[IR::ImageBindingIndex(*kind)];
	if (state.coherent_image_access && image.resource_class == IR::ImageResourceClass::Storage &&
	    state.image_coherent_variables[IR::ImageBindingIndex(*kind)] != 0u) {
		variable = state.image_coherent_variables[IR::ImageBindingIndex(*kind)];
	}
	return DescriptorElementPointer(state, pointer_type, variable, array_index, *kind, resource,
	                                "image descriptor array was not emitted");
}

void EmitStorageImageWrite(EmitterState& state, uint32_t resource, uint32_t mip_lod, uint32_t coord,
                           uint32_t texel) {
	const auto& image = ImageResourceAt(state, resource);
	EXIT_IF(image.resource_class != IR::ImageResourceClass::Storage);
	if (!image.atomic) {
		state.builder.RequireCapability(spv::CapabilityStorageImageWriteWithoutFormat);
	}
	const auto EmitWrite = [&](uint32_t mip) {
		state.builder.AddFunction(spv::OpImageWrite, LoadImageDescriptor(state, resource, mip),
		                          coord, texel);
	};
	if (image.mip_mode != IR::ImageMipMode::Dynamic) {
		EmitWrite(0);
		return;
	}
	EmitIndexSwitch(state, mip_lod, image.mip_count, 0, EmitWrite);
}

spv::ExecutionModel ExecutionModelForStage(ShaderType stage) {
	switch (stage) {
		case ShaderType::Local:
		case ShaderType::Vertex: return spv::ExecutionModelVertex;
		case ShaderType::TessellationControl: return spv::ExecutionModelTessellationControl;
		case ShaderType::TessellationEvaluation: return spv::ExecutionModelTessellationEvaluation;
		case ShaderType::Mesh: return spv::ExecutionModelMeshEXT; // MeshEXT
		case ShaderType::Pixel: return spv::ExecutionModelFragment;
		default: return spv::ExecutionModelGLCompute;
	}
}

namespace {

// The immediate post-dominator of each block, with blocks.size() standing for the exit, or an
// empty result when some block never reaches the exit (from EmK530/KytyPS5 420fc4ead).
std::vector<size_t> SplitWavePostDominators(
    const IR::BlockList& blocks, const std::unordered_map<const IR::Block*, size_t>& index) {
	const auto     count = blocks.size();
	const auto     exit  = count;
	constexpr auto none  = SIZE_MAX;
	const auto successors = [&](size_t block) {
		std::vector<size_t> result;
		if (block == exit) {
			return result;
		}
		for (const auto* successor: blocks[block]->ImmSuccessors()) {
			result.push_back(index.at(successor));
		}
		if (result.empty()) {
			result.push_back(exit);
		}
		return result;
	};
	std::vector<std::vector<size_t>> reverse(count + 1);
	for (size_t block = 0; block < count; block++) {
		for (const auto successor: successors(block)) {
			reverse[successor].push_back(block);
		}
	}
	std::vector<size_t>                    order(count + 1, none);
	std::vector<size_t>                    postorder;
	std::vector<std::pair<size_t, size_t>> stack {{exit, 0}};
	order[exit] = 0;
	while (!stack.empty()) {
		auto& [node, next] = stack.back();
		if (next < reverse[node].size()) {
			const auto child = reverse[node][next++];
			if (order[child] == none) {
				order[child] = 0;
				stack.emplace_back(child, 0);
			}
			continue;
		}
		order[node] = postorder.size();
		postorder.push_back(node);
		stack.pop_back();
	}
	if (postorder.size() != count + 1) {
		return {};
	}
	std::vector<size_t> idom(count + 1, none);
	idom[exit]           = exit;
	const auto intersect = [&](size_t lhs, size_t rhs) {
		while (lhs != rhs) {
			while (order[lhs] < order[rhs]) lhs = idom[lhs];
			while (order[rhs] < order[lhs]) rhs = idom[rhs];
		}
		return lhs;
	};
	for (bool changed = true; changed;) {
		changed = false;
		for (auto node = postorder.rbegin() + 1; node != postorder.rend(); ++node) {
			auto dominator = none;
			for (const auto successor: successors(*node)) {
				if (idom[successor] != none) {
					dominator = dominator == none ? successor : intersect(successor, dominator);
				}
			}
			if (idom[*node] != dominator) {
				idom[*node] = dominator;
				changed     = true;
			}
		}
	}
	return idom;
}

bool IsWaveBranch(const CFG::Terminator& terminator) {
	if (terminator.kind != CFG::TerminatorKind::ConditionalBranch) {
		return false;
	}
	switch (terminator.condition) {
		case CFG::BranchCondition::SccZero:
		case CFG::BranchCondition::SccNonZero:
		case CFG::BranchCondition::VccZero:
		case CFG::BranchCondition::VccNonZero:
		case CFG::BranchCondition::ExecZero:
		case CFG::BranchCondition::ExecNonZero: return true;
		default: return false;
	}
}

} // namespace

std::unordered_set<const IR::Block*> SplitWaveLocalBranches(
    const IR::Program& program, const std::unordered_set<const IR::Block*>& sync_blocks) {
	// Each half of a split wave may branch on its own lanes where the guest branched on all of
	// them: a half whose lanes are all inactive then skips code the guest ran with them masked.
	// The halves must still meet at every barrier together, so a branch stays whole-wave when a
	// block that depends on it (transitively, through the branches controlling it) synchronizes
	// the halves or ends the program.
	std::unordered_set<const IR::Block*> result;
	const auto&                          blocks = program.blocks;
	const auto                           count  = blocks.size();
	if (count == 0 || program.block_info.size() != count || program.dispatcher_fallback) {
		return result;
	}
	std::unordered_map<const IR::Block*, size_t> index;
	for (size_t block = 0; block < count; block++) {
		index.emplace(blocks[block], block);
	}
	const auto idom = SplitWavePostDominators(blocks, index);
	if (idom.empty()) {
		return result;
	}
	// A block depends on a branch when one of the branch's successors leads to it but the
	// branch's block does not always reach it.
	std::vector<std::vector<size_t>> controllers(count);
	for (size_t block = 0; block < count; block++) {
		const auto successors = blocks[block]->ImmSuccessors();
		if (successors.size() < 2) {
			continue;
		}
		for (const auto* successor: successors) {
			for (auto runner = index.at(successor); runner != idom[block] && runner != count;
			     runner = idom[runner]) {
				if (std::ranges::find(controllers[runner], block) == controllers[runner].end()) {
					controllers[runner].push_back(block);
				}
			}
		}
	}
	// Every branch a synchronizing block depends on, transitively, stays whole-wave.
	std::vector<bool>   whole(count, false);
	std::vector<size_t> work;
	for (size_t block = 0; block < count; block++) {
		if (sync_blocks.contains(blocks[block]) || blocks[block]->ImmSuccessors().empty()) {
			work.push_back(block);
		}
	}
	std::vector<bool> visited(count, false);
	while (!work.empty()) {
		const auto block = work.back();
		work.pop_back();
		if (visited[block]) {
			continue;
		}
		visited[block] = true;
		for (const auto controller: controllers[block]) {
			whole[controller] = true;
			work.push_back(controller);
		}
	}
	for (size_t block = 0; block < count; block++) {
		if (!whole[block] && IsWaveBranch(program.block_info[block].terminator)) {
			result.insert(blocks[block]);
		}
	}
	return result;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
