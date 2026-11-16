#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"

#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/logging/log.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

namespace {

// A compute shader with derivatives (IMAGE_GET_LOD) on a device without a compute-derivatives
// extension keeps the KHR declaration, which the driver may reject; each such shader is named once.
void NoteMissingComputeDerivatives(uint64_t shader_hash) {
	static std::mutex         mutex;
	static std::set<uint64_t> logged;
	std::scoped_lock          lock(mutex);
	if (logged.size() >= 64 || !logged.insert(shader_hash).second) {
		return;
	}
	std::fprintf(stderr,
	             "Warning: compute shader 0x%016" PRIx64 " uses derivatives (IMAGE_GET_LOD), but the "
	             "Vulkan device has neither VK_KHR_compute_shader_derivatives nor "
	             "VK_NV_compute_shader_derivatives; the driver may reject its pipeline\n",
	             shader_hash);
}

// KYTY_COHERENT_GLC_LOG=1: one line per emitted program with coherent buffers (why it keeps
// whole-program coherence, or that only its GLC accesses are Coherent), with running totals; also
// one line per program with KYTY_NATIVE_F16 operations.
std::atomic<uint64_t> g_coherent_programs {0};
std::atomic<uint64_t> g_coherent_per_access {0};

} // namespace

void CountCoherentProgram(const EmitterState& state, bool per_access) {
	if (!state.requirements.coherent_buffers) {
		return;
	}
	const auto total = g_coherent_programs.fetch_add(1, std::memory_order_relaxed) + 1;
	const auto glc   = g_coherent_per_access.fetch_add(per_access ? 1 : 0, std::memory_order_relaxed) +
	                 (per_access ? 1 : 0);
	static const bool log = [] {
		const auto* value = std::getenv("KYTY_COHERENT_GLC_LOG");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	if (!log) {
		return;
	}
	const auto& r = state.requirements;
	char        text[320];
	std::snprintf(text, sizeof(text),
	              "Kyty coherent program 0x%016" PRIx64 " stage %u: %s (buffer writes %d, narrow %d, "
	              "image writes %d, u64 %d, dcache inv %d); totals %" PRIu64 " coherent, %" PRIu64
	              " GLC-only\n",
	              state.program.shader_hash, static_cast<unsigned>(state.program.stage),
	              per_access ? "GLC accesses only" : "whole program", r.buffer_writes ? 1 : 0,
	              r.coherent_narrow ? 1 : 0, r.image_writes ? 1 : 0, r.buffer_int64_atomics ? 1 : 0,
	              r.coherent_dcache ? 1 : 0, total, glc);
	Log::WriteToConsoleAndLog(text);
}

bool DispatchDimensionsIndirect(const EmitterState& state) {
	return state.program.stage == ShaderType::Compute && state.input_info.compute != nullptr &&
	       state.input_info.compute->dispatch_dimensions_indirect &&
	       state.program.bindings.dispatch_thread_dword != IR::PushData::NoStart;
}

bool UsesPhysicalAddresses(const EmitterState& state) {
	// Guest address accesses (DMA), or a mesh program that may read its draw dwords from a native
	// indirect draw's parameter block (CodegenOptions::mesh_indirect_params), or a compute program
	// that reads its thread counts through a device address (KYTY_INDIRECT_THREADS_GPU).
	return state.program.info.uses_dma || DispatchDimensionsIndirect(state) ||
	       state.program.bindings.call_state_dword != IR::PushData::NoStart ||
	       (state.program.stage == ShaderType::Mesh && GetCodegenOptions().mesh_indirect_params);
}

uint32_t TypeVoid(EmitterState& state) {
	if (state.void_type == 0) {
		state.void_type = state.builder.Type(spv::OpTypeVoid);
	}
	return state.void_type;
}

uint32_t TypeBool(EmitterState& state) {
	if (state.bool_type == 0) {
		state.bool_type = state.builder.Type(spv::OpTypeBool);
	}
	return state.bool_type;
}

uint32_t TypeBoolVector(EmitterState& state, uint32_t components) {
	auto& type = state.bool_vector_types[components - 2u];
	if (type == 0) {
		type = state.builder.Type(spv::OpTypeVector, TypeBool(state), components);
	}
	return type;
}

uint32_t TypeU32(EmitterState& state) {
	if (state.u32_type == 0) {
		state.u32_type = state.builder.Type(spv::OpTypeInt, 32, 0);
	}
	return state.u32_type;
}

uint32_t TypeU64(EmitterState& state) {
	if (state.u64_type == 0) {
		state.builder.RequireCapability(spv::CapabilityInt64);
		state.u64_type = state.builder.Type(spv::OpTypeInt, 64, 0);
	}
	return state.u64_type;
}

uint32_t TypeU32Pair(EmitterState& state) {
	if (state.u32_pair_type == 0) {
		const auto element = TypeU32(state);
		state.u32_pair_type = state.builder.Type(spv::OpTypeStruct, element, element);
	}
	return state.u32_pair_type;
}

uint32_t TypeI32(EmitterState& state) {
	if (state.i32_type == 0) {
		state.i32_type = state.builder.Type(spv::OpTypeInt, 32, 1);
	}
	return state.i32_type;
}

uint32_t TypeI32Pair(EmitterState& state) {
	if (state.i32_pair_type == 0) {
		const auto element = TypeI32(state);
		state.i32_pair_type = state.builder.Type(spv::OpTypeStruct, element, element);
	}
	return state.i32_pair_type;
}

uint32_t TypeF32(EmitterState& state) {
	if (state.f32_type == 0) {
		state.f32_type = state.builder.Type(spv::OpTypeFloat, 32);
	}
	return state.f32_type;
}

uint32_t TypeF64(EmitterState& state) {
	if (state.f64_type == 0) {
		state.f64_type = state.builder.Type(spv::OpTypeFloat, 64);
	}
	return state.f64_type;
}

uint32_t TypeU32Vector(EmitterState& state, uint32_t components) {
	auto& type = state.u32_vector_types[components - 2u];
	if (type == 0) {
		type = state.builder.Type(spv::OpTypeVector, TypeU32(state), components);
	}
	return type;
}

uint32_t TypeU32Composite(EmitterState& state, uint32_t components) {
	EXIT_IF(components < 1u || components > 4u);
	if (components == 1u) return TypeU32(state);
	return components == 2u ? TypeU32Pair(state) : TypeU32Vector(state, components);
}

uint32_t TypeI32Vector(EmitterState& state, uint32_t components) {
	auto& type = state.i32_vector_types[components - 2u];
	if (type == 0) {
		type = state.builder.Type(spv::OpTypeVector, TypeI32(state), components);
	}
	return type;
}

uint32_t TypeF32Vector(EmitterState& state, uint32_t components) {
	auto& type = state.f32_vector_types[components - 2u];
	if (type == 0) {
		type = state.builder.Type(spv::OpTypeVector, TypeF32(state), components);
	}
	return type;
}

uint32_t TypePointer(EmitterState& state, spv::StorageClass storage_class, uint32_t pointee) {
	return state.builder.Type(spv::OpTypePointer, storage_class, pointee);
}

uint32_t TypeFunction(EmitterState& state) {
	if (state.function_type == 0) {
		state.function_type = state.builder.Type(spv::OpTypeFunction, TypeVoid(state));
	}
	return state.function_type;
}

uint32_t TypeStorageBufferElement(EmitterState& state, uint32_t bits) {
	return bits == 32u   ? TypeU32(state)
	       : bits == 64u ? TypeU64(state)
	                     : state.builder.Type(spv::OpTypeInt, bits, 0);
}

uint32_t StorageRuntimeArrayType(EmitterState& state, uint32_t bits) {
	return state.builder.DecoratedType(
	    spv::OpTypeRuntimeArray,
	    {{spv::OpDecorate, {spv::DecorationArrayStride, bits / 8u}}},
	    TypeStorageBufferElement(state, bits));
}

uint32_t StorageBufferType(EmitterState& state, uint32_t bits = 32) {
	return state.builder.DecoratedType(spv::OpTypeStruct,
	                                   {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	                                    {spv::OpDecorate, {spv::DecorationBlock}}},
	                                   StorageRuntimeArrayType(state, bits));
}

uint32_t TypeStorageBufferPointer(EmitterState& state, uint32_t bits) {
	return TypePointer(state, spv::StorageClassStorageBuffer, StorageBufferType(state, bits));
}

uint32_t TypeStorageBufferElementPointer(EmitterState& state, uint32_t bits) {
	return TypePointer(state, spv::StorageClassStorageBuffer, TypeStorageBufferElement(state, bits));
}

// KYTY_SRT_UNIFORM: a dword array of the largest uniform range (std430-style,
// uniformBufferStandardLayout) in a uniform block; the host binds at most that much.
static uint32_t UniformDwordBlockType(EmitterState& state) {
	const auto array = state.builder.DecoratedType(
	    spv::OpTypeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 4u}}}, TypeU32(state),
	    ConstantU32(state, IR::UniformBufferBytes / 4u));
	return state.builder.DecoratedType(spv::OpTypeStruct,
	                                   {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	                                    {spv::OpDecorate, {spv::DecorationBlock}}},
	                                   array);
}

uint32_t FlattenedSrtElementPointerType(EmitterState& state) {
	return state.flattened_srt_uniform ? TypePointer(state, spv::StorageClassUniform, TypeU32(state))
	                                   : TypeStorageBufferElementPointer(state);
}

uint32_t TypePhysicalU32Pointer(EmitterState& state) {
	return TypePointer(state, spv::StorageClassPhysicalStorageBuffer, TypeU32(state));
}

uint32_t TypePhysicalU64Pointer(EmitterState& state) {
	return TypePointer(state, spv::StorageClassPhysicalStorageBuffer, TypeU64(state));
}

uint32_t TypePushConstantElementPointer(EmitterState& state) {
	return TypePointer(state, spv::StorageClassPushConstant, TypeU32(state));
}

uint32_t TypeU32ArrayPointer(EmitterState& state, spv::StorageClass storage_class,
                             uint32_t dwords) {
	const auto count = ConstantU32(state, std::max(dwords, 1u));
	const auto array = state.builder.Type(spv::OpTypeArray, TypeU32(state), count);
	return TypePointer(state, storage_class, array);
}

uint32_t TypeU32ElementPointer(EmitterState& state, spv::StorageClass storage_class) {
	return TypePointer(state, storage_class, TypeU32(state));
}

namespace {

uint32_t PushConstantArrayType(EmitterState& state) {
	const auto count = ConstantU32(state, IR::PushData::DwordCount);
	return state.builder.DecoratedType(
	    spv::OpTypeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, sizeof(uint32_t)}}},
	    TypeU32(state), count);
}

uint32_t PushConstantBlockType(EmitterState& state) {
	return state.builder.DecoratedType(spv::OpTypeStruct,
	                                   {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	                                    {spv::OpDecorate, {spv::DecorationBlock}}},
	                                   PushConstantArrayType(state));
}

uint32_t PerVertexType(EmitterState& state) {
	// MadMode::Exact/Position: the position is computed without contraction; Invariant also asks
	// the host compiler to evaluate it identically in every shader that computes it the same way.
	if (GetCodegenOptions().mad_mode != MadMode::Fused) {
		return state.builder.DecoratedType(
		    spv::OpTypeStruct,
		    {{spv::OpMemberDecorate, {0, spv::DecorationBuiltIn, spv::BuiltInPosition}},
		     {spv::OpMemberDecorate, {0, spv::DecorationInvariant}},
		     {spv::OpDecorate, {spv::DecorationBlock}}},
		    TypeF32Vector(state, 4));
	}
	return state.builder.DecoratedType(
	    spv::OpTypeStruct,
	    {{spv::OpMemberDecorate, {0, spv::DecorationBuiltIn, spv::BuiltInPosition}},
	     {spv::OpDecorate, {spv::DecorationBlock}}},
	    TypeF32Vector(state, 4));
}

uint32_t SampleMaskArrayType(EmitterState& state) {
	return state.builder.Type(spv::OpTypeArray, TypeI32(state), ConstantU32(state, 1));
}

uint32_t F32ArrayType(EmitterState& state, uint32_t count) {
	return state.builder.Type(spv::OpTypeArray, TypeF32(state), ConstantU32(state, count));
}

// Set 1 (BindlessBindings.h): one runtime array per image type, aliasing the binding of its
// dimension, and the translation and feedback buffers. Declared before the entry point collects
// globals.
void DefineBindlessImages(EmitterState& state) {
	std::vector<IR::ImageResource> images;
	for (const auto& image: state.program.info.images) {
		if (!image.bindless) {
			continue;
		}
		images.push_back(image);
		// KYTY_RUNTIME_INT_IMAGES: a runtime T# may name an integer texture; its loads read the slot
		// through an alias of the binding with the texture's component type (EmitImage).
		if (image.runtime && GetCodegenOptions().runtime_int_images) {
			for (const auto numeric_class:
			     {Prospero::TextureNumericClass::Uint, Prospero::TextureNumericClass::Sint}) {
				auto alias          = image;
				alias.numeric_class = numeric_class;
				alias.depth_compare = false;
				images.push_back(alias);
			}
		}
		// KYTY_BINDLESS_3D_AS_ARRAY: a 3D image instruction may get a 2D or 2D-array T#
		// (BindlessSlotArrayFlag); it reads the 2D-array array too.
		if (BindlessVolumeMayBeArray(image)) {
			auto as_array      = image;
			as_array.dimension = ImageDimension::Dim2DArray;
			images.push_back(as_array);
		}
	}
	for (const auto& image: images) {
		const auto type = ImageType(state, image);
		if (state.bindless_image_variables.contains(type)) {
			continue;
		}
		uint32_t binding = IR::BindlessImages2D;
		if (image.cube) {
			binding = IR::BindlessImagesCube;
		} else if (image.dimension == ImageDimension::Dim3D) {
			binding = IR::BindlessImages3D;
		} else if (image.dimension == ImageDimension::Dim2DArray ||
		           image.dimension == ImageDimension::Dim2DMsaaArray) {
			binding = IR::BindlessImages2DArray;
		}
		const auto array    = state.builder.Type(spv::OpTypeRuntimeArray, type);
		const auto variable = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassUniformConstant, array),
		    spv::StorageClassUniformConstant);
		state.builder.AddName(variable, "bindless_images");
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet,
		                            IR::BindlessDescriptorSet);
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding, binding);
		state.bindless_image_variables[type] = variable;
		state.builder.RequireCapability(spv::CapabilityRuntimeDescriptorArray);
		state.builder.RequireCapability(spv::CapabilitySampledImageArrayNonUniformIndexing);
		state.builder.RequireCapability(spv::CapabilityShaderNonUniform);
		state.builder.RequireVersion(0x00010500u);
	}
	// KYTY_BINDLESS_SAMPLERS: the samplers the host mirrors from guest sampler heaps.
	if (state.bindless_sampler_variable == 0 &&
	    std::ranges::any_of(state.program.info.samplers, &IR::SamplerResource::bindless)) {
		const auto array    = state.builder.Type(spv::OpTypeRuntimeArray,
		                                         state.builder.Type(spv::OpTypeSampler));
		const auto variable = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassUniformConstant, array),
		    spv::StorageClassUniformConstant);
		state.builder.AddName(variable, "bindless_samplers");
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet,
		                            IR::BindlessDescriptorSet);
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding,
		                            IR::BindlessSamplers);
		state.bindless_sampler_variable = variable;
		state.builder.RequireCapability(spv::CapabilityRuntimeDescriptorArray);
		state.builder.RequireCapability(spv::CapabilitySampledImageArrayNonUniformIndexing);
		state.builder.RequireCapability(spv::CapabilityShaderNonUniform);
		state.builder.RequireVersion(0x00010500u);
	}
	if (!state.bindless_image_variables.empty() && state.bindless_translation_variable == 0) {
		const auto variable = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassStorageBuffer, StorageBufferType(state)),
		    spv::StorageClassStorageBuffer);
		state.builder.AddName(variable, "bindless_translation");
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet,
		                            IR::BindlessDescriptorSet);
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding,
		                            IR::BindlessTranslation);
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationNonWritable);
		state.bindless_translation_variable = variable;
		const auto feedback = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassStorageBuffer, StorageBufferType(state)),
		    spv::StorageClassStorageBuffer);
		state.builder.AddName(feedback, "bindless_feedback");
		state.builder.AddAnnotation(spv::OpDecorate, feedback, spv::DecorationDescriptorSet,
		                            IR::BindlessDescriptorSet);
		state.builder.AddAnnotation(spv::OpDecorate, feedback, spv::DecorationBinding,
		                            IR::BindlessFeedback);
		state.bindless_feedback_variable = feedback;
	}
	// KYTY_RUNTIME_IMAGES: the T# -> translation entry table (the host writes it) and the buffer
	// where shaders request T#s the table lacks.
	if (state.runtime_image_table_variable == 0 &&
	    std::ranges::any_of(state.program.info.images, &IR::ImageResource::runtime)) {
		const auto table = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassStorageBuffer, StorageBufferType(state)),
		    spv::StorageClassStorageBuffer);
		state.builder.AddName(table, "runtime_image_table");
		state.builder.AddAnnotation(spv::OpDecorate, table, spv::DecorationDescriptorSet,
		                            IR::BindlessDescriptorSet);
		state.builder.AddAnnotation(spv::OpDecorate, table, spv::DecorationBinding,
		                            IR::BindlessRuntimeTable);
		state.builder.AddAnnotation(spv::OpDecorate, table, spv::DecorationNonWritable);
		state.runtime_image_table_variable = table;
		const auto requests                = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassStorageBuffer, StorageBufferType(state)),
		    spv::StorageClassStorageBuffer);
		state.builder.AddName(requests, "runtime_image_requests");
		state.builder.AddAnnotation(spv::OpDecorate, requests, spv::DecorationDescriptorSet,
		                            IR::BindlessDescriptorSet);
		state.builder.AddAnnotation(spv::OpDecorate, requests, spv::DecorationBinding,
		                            IR::BindlessRuntimeRequests);
		state.runtime_image_requests_variable = requests;
	}
}

// Whether a storage image access to one of `resources` is coherent (ImageAccessCoherent).
template <typename Resources>
bool StorageBindingHasCoherentAccess(const EmitterState& state, const Resources& resources) {
	if (!GetCodegenOptions().image_glc_coherent && !PopsInterlockActive(state)) {
		return false;
	}
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (IR::ImageOpcodeInfoOf(inst.GetOpcode()).resource_class !=
			    IR::ImageResourceClass::Storage) {
				continue;
			}
			const auto index = inst.Flags<IR::MemoryFlags>().index;
			if (index >= state.program.memory_info.size()) continue;
			const auto& mem = state.program.memory_info[index];
			if (ImageAccessCoherent(state, mem) &&
			    std::find(resources.begin(), resources.end(), mem.resource) != resources.end()) {
				return true;
			}
		}
	}
	return false;
}

void DefineDescriptors(EmitterState& state) {
	DefineBindlessImages(state);
	const bool user_clip_planes = state.program.stage != ShaderType::Pixel &&
	                              state.program.stage != ShaderType::Compute &&
	                              state.input_info.vertex != nullptr &&
	                              state.input_info.vertex->user_clip_planes != 0;
	// KYTY_OBJPRIM_ID: the per-draw VGT_OBJECT_ID is a push-constant dword even without push data.
	const bool objprim_push = state.program.stage == ShaderType::Pixel &&
	                          state.input_info.pixel != nullptr &&
	                          state.input_info.pixel->HasObjprimOutput();
	if (state.program.bindings.UsesPushData() || state.program.stage == ShaderType::Mesh ||
	    user_clip_planes || objprim_push) {
		const auto type              = PushConstantBlockType(state);
		state.push_constant_variable = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassPushConstant, type), spv::StorageClassPushConstant);
		state.builder.AddName(type, "BufferResource");
		state.builder.AddName(state.push_constant_variable, "vsharp");
	}
	for (const auto& binding: state.program.bindings.descriptors) {
		const auto Define = [&](uint32_t type, const char* name,
		                        spv::StorageClass storage = spv::StorageClassStorageBuffer) {
			const auto variable =
			    state.builder.DefineGlobalVariable(TypePointer(state, storage, type), storage);
			state.builder.AddName(variable, name);
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet, 0);
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding,
			                            IR::NativeBinding(state.program.stage, binding.kind));
			return variable;
		};
		const auto ArrayType = [&](uint32_t type) {
			return state.builder.Type(
			    spv::OpTypeArray, type,
			    ConstantU32(state, static_cast<uint32_t>(binding.resources.size())));
		};
		switch (binding.kind) {
			case IR::DescriptorBindingKind::Buffers: {
				state.storage_buffer_variable =
				    Define(ArrayType(StorageBufferType(state)), "buffers");
				if (state.requirements.buffer_u8) {
					state.storage_buffer_u8_variable =
					    Define(ArrayType(StorageBufferType(state, 8)), "buffers_u8");
				}
				if (state.requirements.buffer_u16) {
					state.storage_buffer_u16_variable =
					    Define(ArrayType(StorageBufferType(state, 16)), "buffers_u16");
				}
				if (state.requirements.buffer_int64_atomics) {
					state.storage_buffer_u64_variable =
					    Define(ArrayType(StorageBufferType(state, 64)), "buffers_u64");
				}
				// A program that only reads its buffers says so (serbru20066666/brunoKytyPs5
				// def478029). The AMD driver then loads a uniform address once per wave into
				// scalar registers, as the guest's S_BUFFER_LOAD does, instead of once per lane
				// with the address in a vector register. Coherent (polling) loads keep the
				// buffers writable-declared so no load can be assumed loop-invariant.
				// KYTY_COHERENT_GLC_ONLY (Erlou93/KytyPS5-Yotei: a grass vertex shader with one GLC read
				// among 107 per vertex 7.2 -> <0.15 ms per frame on their GPU): in a program that writes
				// no buffer or image memory, only the GLC/DLC accesses use a Coherent alias. Nothing the
				// command itself does can change what a plain read sees (no invocation of it writes
				// buffer memory); writes of earlier commands are made visible by the emulator's barriers;
				// and the guest's own non-GLC loads go through its non-coherent GL0/GL1 caches too.
				const auto glc_only_mode = GetCodegenOptions().coherent_glc_only;
				const bool per_access_coherence =
				    glc_only_mode != 0u && state.requirements.coherent_buffers &&
				    !state.requirements.buffer_writes && !state.requirements.coherent_narrow &&
				    !state.requirements.image_writes && !state.requirements.buffer_int64_atomics &&
				    (glc_only_mode == 2u || !state.requirements.coherent_dcache);
				CountCoherentProgram(state, per_access_coherence);
				if (per_access_coherence) {
					state.storage_buffer_coherent_variable =
					    Define(ArrayType(StorageBufferType(state)), "buffers_coherent");
					state.builder.AddAnnotation(spv::OpDecorate, state.storage_buffer_coherent_variable,
					                            spv::DecorationAliased);
					state.builder.AddAnnotation(spv::OpDecorate, state.storage_buffer_coherent_variable,
					                            spv::DecorationCoherent);
				}
				const bool read_only = GetCodegenOptions().readonly_buffers &&
				                       !state.requirements.buffer_writes &&
				                       (!state.requirements.coherent_buffers || per_access_coherence);
				for (const auto variable: {state.storage_buffer_variable, state.storage_buffer_u8_variable,
				                           state.storage_buffer_u16_variable, state.storage_buffer_u64_variable}) {
					if (variable == 0) continue;
					if (state.storage_buffer_u8_variable != 0 || state.storage_buffer_u16_variable != 0 ||
					    state.storage_buffer_u64_variable != 0 || per_access_coherence) {
						state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationAliased);
					}
					if (read_only) {
						state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationNonWritable);
					}
					// RDNA2 stores publish to L2 even without GLC; every alias of the buffer
					// must participate in visibility for cache-bypassing polling loads.
					if (state.requirements.coherent_buffers && !per_access_coherence) {
						state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationCoherent);
					}
				}
				break;
			}
			case IR::DescriptorBindingKind::UniformBuffers: {
				// std430-style dword array (uniformBufferStandardLayout) of the whole binding range.
				const auto array = state.builder.DecoratedType(
				    spv::OpTypeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 4u}}},
				    TypeU32(state), ConstantU32(state, IR::UniformBufferBytes / 4u));
				const auto block = state.builder.DecoratedType(
				    spv::OpTypeStruct, {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
				                        {spv::OpDecorate, {spv::DecorationBlock}}},
				    array);
				state.uniform_buffer_variable =
				    Define(ArrayType(block), "uniform_buffers", spv::StorageClassUniform);
				break;
			}
			case IR::DescriptorBindingKind::BdaPagetable:
				state.bda_pagetable_variable = Define(StorageBufferType(state, 64), "bda_pagetable");
				break;
			case IR::DescriptorBindingKind::FaultBuffer:
				state.fault_buffer_variable = Define(StorageBufferType(state), "fault_buffer");
				break;
			case IR::DescriptorBindingKind::ShaderData:
				state.shader_data_storage_variable =
				    SrtUniformEnabled()
				        ? Define(UniformDwordBlockType(state), "shader_data", spv::StorageClassUniform)
				        : Define(StorageBufferType(state), "shader_data");
				break;
			case IR::DescriptorBindingKind::MipStats:
				state.mip_stats_variable = Define(StorageBufferType(state), "mip_stats");
				break;
			case IR::DescriptorBindingKind::FlattenedSrt:
				if (SrtUniformEnabled()) {
					state.flattened_srt_variable =
					    Define(UniformDwordBlockType(state), "flattened_srt", spv::StorageClassUniform);
					state.flattened_srt_uniform = true;
				} else {
					state.flattened_srt_variable = Define(StorageBufferType(state), "flattened_srt");
				}
				break;
			case IR::DescriptorBindingKind::Samplers:
				state.sampler_variable = Define(ArrayType(state.builder.Type(spv::OpTypeSampler)),
				                                "samplers", spv::StorageClassUniformConstant);
				break;
			case IR::DescriptorBindingKind::Gds:
				state.gds_variable = Define(StorageBufferType(state), "gds");
				break;
			case IR::DescriptorBindingKind::SharedMemory:
				state.lds_variable = Define(StorageBufferType(state), "lds_dwords");
				state.builder.AddAnnotation(spv::OpDecorate, state.lds_variable, spv::DecorationCoherent);
				if (state.requirements.shared_int64_atomics) {
					state.lds_u64_variable = Define(StorageBufferType(state, 64), "lds_qwords");
					state.builder.AddAnnotation(spv::OpDecorate, state.lds_u64_variable, spv::DecorationCoherent);
					for (const auto variable: {state.lds_variable, state.lds_u64_variable}) {
						state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationAliased);
					}
				}
				break;
			default: {
				EXIT_IF(IR::ImageBindingResourceClass(binding.kind) ==
				        IR::ImageResourceClass::None);
				const auto& image = state.program.info.images.at(binding.resources.front());
				const auto  name  = "image_" + std::to_string(static_cast<uint32_t>(binding.kind));
				state.image_variables[IR::ImageBindingIndex(binding.kind)] =
				    Define(ArrayType(ImageType(state, image)), name.c_str(),
				           spv::StorageClassUniformConstant);
				if (image.resource_class == IR::ImageResourceClass::Storage &&
				    StorageBindingHasCoherentAccess(state, binding.resources)) {
					// KYTY_IMAGE_GLC_COHERENT / KYTY_POPS_INTERLOCK: a Coherent alias of the same
					// binding for the coherent accesses only (ImageDescriptorPointer).
					const auto alias = Define(ArrayType(ImageType(state, image)),
					                          (name + "_coherent").c_str(),
					                          spv::StorageClassUniformConstant);
					state.builder.AddAnnotation(spv::OpDecorate, alias, spv::DecorationCoherent);
					state.image_coherent_variables[IR::ImageBindingIndex(binding.kind)] = alias;
				}
				// 1D images are host 2D images one texel high (upstream e6cc1ce30): no Sampled1D /
				// Image1D capability.
				break;
			}
		}
	}
}

} // namespace

bool ComputeImplicitLod(const EmitterState& state) {
	if (!state.requirements.implicit_compute_lod || state.program.stage != ShaderType::Compute ||
	    state.lane_count != 1 || state.split_wave || state.mesh_passes > 1u ||
	    !GetHostComputeDerivativesLinear() ||
	    GetHostImageFeatures().compute_derivatives == HostComputeDerivatives::None) {
		return false;
	}
	const auto* cs = ShaderWorkgroupInput(state.program.stage, state.input_info);
	if (cs == nullptr) return false;
	const uint32_t threads = std::max(cs->threads_num[0], 1u) * std::max(cs->threads_num[1], 1u) *
	                         std::max(cs->threads_num[2], 1u);
	return threads % 4u == 0u;
}

uint32_t ConstantU32(EmitterState& state, uint32_t value) {
	return state.builder.Constant(spv::OpConstant, TypeU32(state), value);
}

uint32_t ConstantI32(EmitterState& state, int32_t value) {
	return state.builder.Constant(spv::OpConstant, TypeI32(state), static_cast<uint32_t>(value));
}

uint32_t ConstantF32(EmitterState& state, uint32_t bits) {
	return state.builder.Constant(spv::OpConstant, TypeF32(state), bits);
}

uint32_t ConstantF32Value(EmitterState& state, float value) {
	return ConstantF32(state, std::bit_cast<uint32_t>(value));
}

uint32_t ConstantBool(EmitterState& state, bool value) {
	return state.builder.Constant(value ? spv::OpConstantTrue : spv::OpConstantFalse,
	                              TypeBool(state));
}

uint32_t ConstantU64(EmitterState& state, uint64_t value) {
	return state.builder.Constant(spv::OpConstant, TypeU64(state), static_cast<uint32_t>(value),
	                              static_cast<uint32_t>(value >> 32u));
}

uint32_t ConstantU32CompositeZero(EmitterState& state, uint32_t components) {
	EXIT_IF(components < 1u || components > 4u);
	const auto            zero = ConstantU32(state, 0);
	if (components == 1u) return zero;
	std::vector<uint32_t> values(components, zero);
	return state.builder.Constant(spv::OpConstantComposite, TypeU32Composite(state, components),
	                              values);
}

uint32_t GlslStd450(EmitterState& state) {
	return state.builder.Import("GLSL.std.450");
}

Format::ComponentType VertexParameterScalarKind(const EmitterState& state, uint32_t location) {
	if ((state.program.stage != ShaderType::Vertex && state.program.stage != ShaderType::Local) ||
	    location >= ShaderVertexInputInfo::RES_MAX ||
	    location >= static_cast<uint32_t>(state.input_info.vertex->resources_num)) {
		return Format::ComponentType::Float;
	}
	return Format::GetFormatComponentType(state.input_info.vertex->resources[location].Format());
}

uint32_t VertexParameterComponentCount(const InputBinding& input) {
	return std::clamp(input.component_count, 1u, 4u);
}

uint32_t VertexParameterScalarType(EmitterState& state, Format::ComponentType kind) {
	switch (kind) {
		case Format::ComponentType::Sint: return TypeI32(state);
		case Format::ComponentType::Uint: return TypeU32(state);
		default: return TypeF32(state);
	}
}

uint32_t DefineInterfaceVariable(EmitterState& state, uint32_t type, spv::StorageClass storage,
                                 const char* name) {
	const auto variable =
	    state.builder.DefineGlobalVariable(TypePointer(state, storage, type), storage);
	state.interface_variables.push_back(variable);
	state.builder.AddName(variable, name);
	return variable;
}

namespace {

uint32_t BuiltInForInput(IR::StageInputKind kind) {
	switch (kind) {
		case IR::StageInputKind::VertexIndex: return spv::BuiltInVertexIndex;
		case IR::StageInputKind::InvocationId: return spv::BuiltInInvocationId;
		case IR::StageInputKind::PrimitiveId: return spv::BuiltInPrimitiveId;
		case IR::StageInputKind::TessCoord: return spv::BuiltInTessCoord;
		case IR::StageInputKind::InstanceIndex: return spv::BuiltInInstanceIndex;
		case IR::StageInputKind::BaseInstance: return spv::BuiltInBaseInstance;
		case IR::StageInputKind::BaseVertex: return spv::BuiltInBaseVertex;
		case IR::StageInputKind::FragCoord: return spv::BuiltInFragCoord;
		case IR::StageInputKind::FrontFacing: return spv::BuiltInFrontFacing;
		case IR::StageInputKind::Layer: return spv::BuiltInLayer;
		case IR::StageInputKind::SampleId: return spv::BuiltInSampleId;
		case IR::StageInputKind::BaryCoordSmooth: return spv::BuiltInBaryCoordKHR;
		case IR::StageInputKind::BaryCoordNoPerspective: return spv::BuiltInBaryCoordNoPerspKHR;
		case IR::StageInputKind::WorkgroupId: return spv::BuiltInWorkgroupId;
		case IR::StageInputKind::NumWorkgroups: return spv::BuiltInNumWorkgroups;
		case IR::StageInputKind::LocalInvocationId: return spv::BuiltInLocalInvocationId;
		case IR::StageInputKind::LocalInvocationIndex: return spv::BuiltInLocalInvocationIndex;
		case IR::StageInputKind::GlobalInvocationId: return spv::BuiltInGlobalInvocationId;
		default: return UINT32_MAX;
	}
}

bool IsLinearMode(IR::InterpolationMode mode) {
	return mode == IR::InterpolationMode::LinearCenter ||
	       mode == IR::InterpolationMode::LinearCentroid ||
	       mode == IR::InterpolationMode::LinearSample;
}

bool IsCentroidMode(IR::InterpolationMode mode) {
	return mode == IR::InterpolationMode::PerspectiveCentroid ||
	       mode == IR::InterpolationMode::LinearCentroid;
}

bool IsSampleMode(IR::InterpolationMode mode) {
	return mode == IR::InterpolationMode::PerspectiveSample ||
	       mode == IR::InterpolationMode::LinearSample;
}

// KYTY_INTERP_MODES: the interpolation of a pixel input variable from the I/J pairs its
// V_INTERP_P2 reads used (all aliases of the location together). Returns Unknown when the
// shader-wide interpolation must stay: a read with a computed J, reads mixing perspective and
// linear pairs, or sample reads mixed with other locations. Otherwise the result's perspective
// is shared by all reads and its location is centroid or sample when every read used it, center
// otherwise (centroid reads of a center-decorated input then use InterpolateAtCentroid).
IR::InterpolationMode InputInterpolation(const EmitterState& state, uint32_t location) {
	if (!GetCodegenOptions().interp_modes) {
		return IR::InterpolationMode::Unknown;
	}
	bool any = false, linear = false, perspective = false;
	bool center = false, centroid = false, sample = false;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != IR::ValueOpcode::GetAttribute ||
			    PixelParameterLocation(state, inst.Arg(0).U32()) != location) {
				continue;
			}
			const auto mode = static_cast<IR::InterpolationMode>(inst.Flags<uint32_t>());
			if (mode == IR::InterpolationMode::Unknown) {
				return IR::InterpolationMode::Unknown;
			}
			any = true;
			(IsLinearMode(mode) ? linear : perspective) = true;
			(IsCentroidMode(mode) ? centroid : IsSampleMode(mode) ? sample : center) = true;
		}
	}
	if (!any || (linear && perspective) || (sample && (center || centroid))) {
		return IR::InterpolationMode::Unknown;
	}
	if (sample) {
		return linear ? IR::InterpolationMode::LinearSample : IR::InterpolationMode::PerspectiveSample;
	}
	if (centroid && !center) {
		return linear ? IR::InterpolationMode::LinearCentroid
		              : IR::InterpolationMode::PerspectiveCentroid;
	}
	return linear ? IR::InterpolationMode::LinearCenter : IR::InterpolationMode::PerspectiveCenter;
}

void DefineInputs(EmitterState& state) {
	state.inputs.reserve(state.program.info.inputs.size());
	for (const auto& input: state.program.info.inputs) {
		state.inputs.push_back({input});
	}
	const auto add_builtin = [&](IR::StageInputKind kind, uint32_t components,
	                             const char* name) {
		if (std::ranges::none_of(state.inputs, [kind](const InputBinding& input) {
			    return input.kind == kind;
		    })) {
			state.inputs.push_back({{kind, 0, components, name}});
		}
	};
	if (IR::FindBinding(state.program.bindings, IR::DescriptorBindingKind::SharedMemory) != nullptr) {
		add_builtin(IR::StageInputKind::WorkgroupId, 3, "gl_WorkGroupID");
		add_builtin(IR::StageInputKind::NumWorkgroups, 3, "gl_NumWorkGroups");
	}
	if (state.lane_count == 2) {
		add_builtin(IR::StageInputKind::LocalInvocationIndex, 1, "gl_LocalInvocationIndex");
		if (std::ranges::any_of(state.inputs, [](const InputBinding& input) {
			    return input.kind == IR::StageInputKind::GlobalInvocationId;
		    })) {
			add_builtin(IR::StageInputKind::WorkgroupId, 3, "gl_WorkGroupID");
		}
	} else if (state.program.lane_local || state.split_wave) {
		// KYTY_LANE_LOCAL: the guest lane of ADD_TID buffers (BufferLane). KYTY_WAVE64_WARPS: the
		// guest lane (EmitSubgroupLocalInvocationId) and the exchange slot.
		add_builtin(IR::StageInputKind::LocalInvocationIndex, 1, "gl_LocalInvocationIndex");
	}
	for (auto& input: state.inputs) {
		if (state.program.stage == ShaderType::Pixel &&
		    input.kind == IR::StageInputKind::Parameter) {
			const auto location = PixelParameterLocation(state, input.location);
			const auto alias = std::ranges::find_if(state.inputs, [&](const InputBinding& other) {
				return other.kind == IR::StageInputKind::Parameter && other.variable_id != 0 &&
				       PixelParameterLocation(state, other.location) == location;
			});
			if (alias != state.inputs.end()) {
				EXIT_IF(alias->per_vertex != input.per_vertex);
				input.variable_id = alias->variable_id;
				continue;
			}
		}
		uint32_t type = TypeU32(state);
		switch (input.kind) {
			case IR::StageInputKind::VertexIndex:
			case IR::StageInputKind::InvocationId:
			case IR::StageInputKind::PrimitiveId:
			case IR::StageInputKind::InstanceIndex:
			case IR::StageInputKind::BaseInstance:
			case IR::StageInputKind::BaseVertex:
			case IR::StageInputKind::Layer:
			case IR::StageInputKind::SampleId: type = TypeI32(state); break;
			case IR::StageInputKind::WorkgroupId:
			case IR::StageInputKind::NumWorkgroups:
			case IR::StageInputKind::LocalInvocationId:
			case IR::StageInputKind::GlobalInvocationId: type = TypeU32Vector(state, 3); break;
			case IR::StageInputKind::FragCoord: type = TypeF32Vector(state, 4); break;
			case IR::StageInputKind::TessCoord:
			case IR::StageInputKind::BaryCoordSmooth:
			case IR::StageInputKind::BaryCoordNoPerspective: type = TypeF32Vector(state, 3); break;
			case IR::StageInputKind::FrontFacing: type = TypeBool(state); break;
			case IR::StageInputKind::Parameter:
				if (state.program.stage == ShaderType::Vertex ||
				    state.program.stage == ShaderType::Local) {
					type = VertexParameterScalarType(
					    state, VertexParameterScalarKind(state, input.location));
					const auto components = VertexParameterComponentCount(input);
					if (components > 1u) {
						type = state.builder.Type(spv::OpTypeVector, type, components);
					}
				} else if (input.per_vertex) {
					type = state.builder.Type(spv::OpTypeArray, TypeF32Vector(state, 4),
					                          ConstantU32(state, 3));
				} else {
					type = TypeF32Vector(state, 4);
				}
				break;
			default: break;
		}
		input.variable_id =
		    DefineInterfaceVariable(state, type, spv::StorageClassInput, input.debug_name.c_str());
		if (input.kind == IR::StageInputKind::Layer || input.kind == IR::StageInputKind::SampleId) {
			state.builder.AddAnnotation(spv::OpDecorate, input.variable_id, spv::DecorationFlat);
		}
		// KYTY_OBJPRIM_ID: the fragment primitive index. SPIR-V enables PrimitiveId in a fragment
		// shader through the Tessellation capability as well as Geometry (based on
		// Erlou93/KytyPS5-Yotei).
		if (input.kind == IR::StageInputKind::PrimitiveId && state.program.stage == ShaderType::Pixel) {
			state.builder.AddAnnotation(spv::OpDecorate, input.variable_id, spv::DecorationFlat);
			state.builder.RequireCapability(spv::CapabilityTessellation);
		}
		if (input.kind == IR::StageInputKind::Parameter) {
			const auto flat = PixelParameterIsFlat(state, input.location);
			if (input.per_vertex) {
				state.builder.AddAnnotation(spv::OpDecorate, input.variable_id,
				                            spv::DecorationPerVertexKHR);
			} else if (flat) {
				state.builder.AddAnnotation(spv::OpDecorate, input.variable_id,
				                            spv::DecorationFlat);
			}
			const auto interpolation =
			    state.program.stage == ShaderType::Pixel && !flat && !input.per_vertex
			        ? InputInterpolation(state, PixelParameterLocation(state, input.location))
			        : IR::InterpolationMode::Unknown;
			if (interpolation != IR::InterpolationMode::Unknown) {
				state.input_interpolation.emplace(input.variable_id, interpolation);
				if (IsLinearMode(interpolation)) {
					state.builder.AddAnnotation(spv::OpDecorate, input.variable_id,
					                            spv::DecorationNoPerspective);
				}
				if (IsCentroidMode(interpolation)) {
					state.builder.AddAnnotation(spv::OpDecorate, input.variable_id,
					                            spv::DecorationCentroid);
				} else if (IsSampleMode(interpolation)) {
					state.builder.RequireCapability(spv::CapabilitySampleRateShading);
					state.builder.AddAnnotation(spv::OpDecorate, input.variable_id,
					                            spv::DecorationSample);
				}
			} else if (state.program.stage == ShaderType::Pixel &&
			           state.input_info.pixel->ps_no_perspective && !flat && !input.per_vertex) {
				state.builder.AddAnnotation(spv::OpDecorate, input.variable_id,
				                            spv::DecorationNoPerspective);
			}
			state.builder.AddAnnotation(spv::OpDecorate, input.variable_id, spv::DecorationLocation,
			                            PixelParameterLocation(state, input.location));
		} else if (const auto builtin = BuiltInForInput(input.kind); builtin != UINT32_MAX) {
			state.builder.AddAnnotation(spv::OpDecorate, input.variable_id, spv::DecorationBuiltIn,
			                            builtin);
		}
	}
	if (state.requirements.subgroup_local_invocation_id || state.split_wave) {
		const auto variable = DefineInterfaceVariable(state, TypeU32(state), spv::StorageClassInput,
		                                              "gl_SubgroupInvocationID");
		state.subgroup_local_invocation_id_variable = variable;
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn,
		                            spv::BuiltInSubgroupLocalInvocationId);
		if (state.program.stage == ShaderType::Pixel) {
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationFlat);
		}
	}
	if (state.requirements.helper_invocation) {
		const auto variable = DefineInterfaceVariable(state, TypeBool(state),
		                                              spv::StorageClassInput, "gl_HelperInvocation");
		state.helper_invocation_variable = variable;
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn,
		                            spv::BuiltInHelperInvocation);
	}
}

void DefineOutputs(EmitterState& state) {
	state.outputs.reserve(state.program.info.outputs.size());
	uint32_t clip_distance_count = 0;
	uint32_t cull_distance_count = 0;
	for (const auto& output: state.program.info.outputs) {
		state.outputs.push_back({output});
		if (output.kind == IR::StageOutputKind::ClipDistance) {
			clip_distance_count = std::max(clip_distance_count, output.index + 1);
		} else if (output.kind == IR::StageOutputKind::CullDistance) {
			cull_distance_count = std::max(cull_distance_count, output.index + 1);
		}
	}
	const bool has_position = std::ranges::any_of(state.outputs, [](const OutputBinding& output) {
		return output.kind == IR::StageOutputKind::Position;
	});
	if (has_position && state.program.stage != ShaderType::Pixel &&
	    state.input_info.vertex != nullptr && state.input_info.vertex->user_clip_planes != 0) {
		// Upstream feed075a9: PipelineCache rejects UCP_ENA with shader clip or cull distances.
		EXIT_IF(clip_distance_count + cull_distance_count != 0);
		state.user_clip_planes = state.input_info.vertex->user_clip_planes;
		for (uint32_t plane = 0; plane < 32u; plane++) {
			if ((state.user_clip_planes & (1u << plane)) != 0) {
				state.outputs.push_back(
				    {{IR::StageOutputKind::ClipDistance, clip_distance_count++, 0, "gl_ClipDistance"}});
			}
		}
	}
	if (state.program.stage == ShaderType::Mesh) {
		DefineMeshOutputs(state, clip_distance_count, cull_distance_count);
		return;
	}
	// KYTY_CLIP_GUARD=0: no plane reserved (an AMD test 3 diagnostic).
	if (state.program.stage == ShaderType::Vertex && GetCodegenOptions().clip_guard &&
	    clip_distance_count + cull_distance_count < 8u &&
	    std::ranges::any_of(state.outputs, [](const OutputBinding& output) {
		    return output.kind == IR::StageOutputKind::Position;
	    })) {
		// Reserve one plane for the enabled PA_CL_CLIP_CNTL clipping-error cull.
		state.invalid_position_clip_distance = clip_distance_count++;
		state.outputs.push_back({{IR::StageOutputKind::ClipDistance,
		                          state.invalid_position_clip_distance, 0, "gl_ClipDistance"}});
	}
	const auto BuiltIn = [&](uint32_t& variable, uint32_t type, const char* name,
	                         spv::BuiltIn builtin) {
		if (variable == 0) {
			variable = DefineInterfaceVariable(state, type, spv::StorageClassOutput, name);
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn, builtin);
		}
		return variable;
	};
	for (auto& binding: state.outputs) {
		switch (binding.kind) {
			case IR::StageOutputKind::Position:
				if (state.per_vertex_variable == 0) {
					const auto type = PerVertexType(state);
					state.builder.AddName(type, "gl_PerVertex");
					state.per_vertex_variable = DefineInterfaceVariable(
					    state, type, spv::StorageClassOutput, "outPerVertex");
				}
				binding.variable_id = state.per_vertex_variable;
				break;
			case IR::StageOutputKind::PointSize:
				binding.variable_id = BuiltIn(state.point_size_variable, TypeF32(state),
				                              "gl_PointSize", spv::BuiltInPointSize);
				break;
			case IR::StageOutputKind::ClipDistance:
				binding.variable_id =
				    BuiltIn(state.clip_distance_variable, F32ArrayType(state, clip_distance_count),
				            "gl_ClipDistance", spv::BuiltInClipDistance);
				break;
			case IR::StageOutputKind::CullDistance:
				binding.variable_id =
				    BuiltIn(state.cull_distance_variable, F32ArrayType(state, cull_distance_count),
				            "gl_CullDistance", spv::BuiltInCullDistance);
				break;
			case IR::StageOutputKind::Layer:
				binding.variable_id =
				    BuiltIn(state.layer_variable, TypeU32(state), "gl_Layer", spv::BuiltInLayer);
				break;
			case IR::StageOutputKind::ViewportIndex:
				binding.variable_id = BuiltIn(state.viewport_index_variable, TypeU32(state),
				                              "gl_ViewportIndex", spv::BuiltInViewportIndex);
				break;
			case IR::StageOutputKind::Depth:
				binding.variable_id = BuiltIn(state.depth_variable, TypeF32(state), "gl_FragDepth",
				                              spv::BuiltInFragDepth);
				break;
			case IR::StageOutputKind::SampleMask:
				binding.variable_id =
				    BuiltIn(state.sample_mask_variable, SampleMaskArrayType(state), "gl_SampleMask",
				            spv::BuiltInSampleMask);
				break;
			case IR::StageOutputKind::ObjPrimId:
				// KYTY_OBJPRIM_ID: one 32-bit unsigned value at the colour slot (location) the
				// hardware writes.
				binding.variable_id = DefineInterfaceVariable(state, TypeU32(state), spv::StorageClassOutput,
				                                              binding.debug_name.c_str());
				state.builder.AddAnnotation(spv::OpDecorate, binding.variable_id,
				                            spv::DecorationLocation, binding.location);
				state.objprim_output_variable = binding.variable_id;
				break;
			case IR::StageOutputKind::Parameter:
			case IR::StageOutputKind::Mrt: {
				auto type = TypeF32Vector(state, 4);
				if (binding.kind == IR::StageOutputKind::Mrt &&
				    state.program.stage == ShaderType::Pixel &&
				    binding.index < std::size(state.input_info.pixel->target_output_mode)) {
					// KYTY_INT_MRT_OUTPUTS=1 (upstream 26ca60923): UINT and SINT targets take integer
					// outputs of their signedness (ExportVector's type, spirvEmitterFlow.cpp).
					const auto mode = state.input_info.pixel->target_output_mode[binding.index];
					if (GetCodegenOptions().int_mrt_outputs) {
						type = mode != 0u ? MrtOutputType(state, binding.index) : type;
					} else if (mode == 7u) {
						type = TypeU32Vector(state, 4);
					}
				}
				binding.variable_id = DefineInterfaceVariable(state, type, spv::StorageClassOutput,
				                                              binding.debug_name.c_str());
				const bool dual_source = binding.kind == IR::StageOutputKind::Mrt &&
				                         state.program.stage == ShaderType::Pixel &&
				                         state.input_info.pixel->dual_source_blending;
				EXIT_NOT_IMPLEMENTED(dual_source && binding.index > 1);
				state.builder.AddAnnotation(spv::OpDecorate, binding.variable_id,
				                            spv::DecorationLocation,
				                            dual_source ? 0u : binding.location);
				if (dual_source) {
					state.builder.AddAnnotation(spv::OpDecorate, binding.variable_id,
					                            spv::DecorationIndex, binding.index);
				}
				break;
			}
		}
	}
}

} // namespace

// The float-control declarations of every module (FLOAT_MODE 0xC0 on this host): also the
// header of the KYTY_F32_FLUSH_HOST start-up self-test module (spirvEmitterFlushSelfTest.cpp).
void DeclareFloatControls(EmitterState& state) {
	state.builder.RequireCapability(spv::CapabilitySignedZeroInfNanPreserve);
	state.builder.RequireExtension("SPV_KHR_float_controls");
	// GCN/RDNA arithmetic preserves 32-bit signed zero, infinity, and NaN. Declaring that
	// contract prevents host compilers from treating synthesized IEEE values as finite.
	state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeSignedZeroInfNanPreserve,
	                               32u);
	// KYTY_NATIVE_F16: f16vec2 arithmetic rounds to nearest even and keeps signed zeros, infinities
	// and NaNs, as the guest's packed f16 ops do (f16 denormals: DenormPreserve 16 below, when the
	// device takes it, as before).
	if (state.requirements.native_half) {
		static std::atomic<uint64_t> native_half_programs {0};
		const auto count = native_half_programs.fetch_add(1, std::memory_order_relaxed) + 1;
		static const bool log = [] {
			const auto* value = std::getenv("KYTY_COHERENT_GLC_LOG");
			return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
		}();
		if (log) {
			char text[160];
			std::snprintf(text, sizeof(text),
			              "Kyty native f16 program 0x%016" PRIx64 " stage %u; total %" PRIu64 "\n",
			              state.program.shader_hash, static_cast<unsigned>(state.program.stage), count);
			Log::WriteToConsoleAndLog(text);
		}
		state.builder.RequireCapability(spv::CapabilityFloat16);
		state.builder.RequireCapability(spv::CapabilityRoundingModeRTE);
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeRoundingModeRTE, 16u);
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeSignedZeroInfNanPreserve,
		                               16u);
	}
	// FLOAT_MODE 0xC0 flushes f32 denormals (inputs and outputs) and keeps f16/f64 ones.
	// Declare only what the device supports; otherwise host defaults stay in effect.
	const auto float_controls = GetHostFloatControls();
	const auto denorm_mode = [&](spv::ExecutionMode mode, spv::Capability capability,
	                             uint32_t width) {
		state.builder.RequireCapability(capability);
		state.builder.AddExecutionMode(state.main_func, mode, width);
	};
	if (float_controls.denorm_flush_f32) {
		denorm_mode(spv::ExecutionModeDenormFlushToZero, spv::CapabilityDenormFlushToZero, 32u);
	}
	if (float_controls.denorm_preserve_f16) {
		denorm_mode(spv::ExecutionModeDenormPreserve, spv::CapabilityDenormPreserve, 16u);
	}
	if (float_controls.denorm_preserve_f64) {
		denorm_mode(spv::ExecutionModeDenormPreserve, spv::CapabilityDenormPreserve, 64u);
	}
	if (state.requirements.float64) {
		EXIT_NOT_IMPLEMENTED(state.program.stage == ShaderType::Compute &&
		                     state.input_info.compute->float_mode != 0xc0);
		// Use native rounding for MODE=0xc0, consistent with ordinary FP32 arithmetic (declaring
		// RoundingModeRTE for 32-bit floats here made the NVIDIA compiler never return from
		// vkCreateComputePipelines for a large FP64 module, upstream PR 1034).
		state.builder.RequireCapability(spv::CapabilityFloat64);
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeSignedZeroInfNanPreserve,
		                               64u);
		// FP64 denormal preservation is temporarily disabled.
		// state.builder.RequireCapability(spv::CapabilityDenormPreserve);
		// state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeDenormPreserve, 64u);
	}
}

void DefineModule(EmitterState& state) {
	state.interface_variables.reserve(state.program.info.inputs.size() +
	                                  state.program.info.outputs.size());
	DefineInputs(state);
	DefineOutputs(state);
	DefineTessellationInterfaces(state);
	DefineDescriptors(state);
	if (state.split_wave) {
		// Two alternating halves of one dword per guest lane (ExchangeWaveValue).
		state.wave_exchange_variable = state.builder.DefineGlobalVariable(
		    TypeU32ArrayPointer(state, spv::StorageClassWorkgroup, 128), spv::StorageClassWorkgroup);
		state.builder.AddName(state.wave_exchange_variable, "wave_exchange");
		state.wave_exchange_parity = state.builder.AllocateId();
	}
	if (state.requirements.function_lds) {
		if (state.lds_slot_plan) {
			for (const auto key: state.requirements.lds_slot_keys) {
				state.lds_slot_variables[key] = state.builder.AllocateId();
			}
		} else {
			state.lds_variable = state.builder.AllocateId();
		}
	}
	if (state.requirements.function_scratch) {
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.scratch_variable[half] = state.builder.AllocateId();
		}
	}
	state.main_func = state.builder.AllocateId();
	if (state.program.stage == ShaderType::Mesh) {
		state.mesh_guest_func = state.builder.AllocateId();
		state.builder.RequireCapability(spv::CapabilityMeshShadingEXT); // MeshShadingEXT
		state.builder.RequireExtension("SPV_EXT_mesh_shader");
		state.builder.AddExecutionMode(state.main_func,
		                               spv::ExecutionModeOutputTrianglesEXT); // OutputTrianglesEXT
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeOutputVertices,
		                               state.input_info.vertex->mesh.max_vertices);
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeOutputPrimitivesEXT,
		                               state.input_info.vertex->mesh.max_primitives);
		if (state.mesh_passes > 1u) {
			state.mesh_pass_variable = state.builder.DefineGlobalVariable(
			    TypePointer(state, spv::StorageClassPrivate, TypeU32(state)),
			    spv::StorageClassPrivate);
			state.builder.AddName(state.mesh_pass_variable, "mesh_pass");
		}
	}
	if (state.program.stage == ShaderType::TessellationControl ||
	    state.program.stage == ShaderType::TessellationEvaluation) {
		DefineTessellationExecutionModes(state);
	}
	state.entry_label = state.builder.AllocateId();

	state.builder.RequireCapability(spv::CapabilityShader);
	state.builder.RequireCapability(spv::CapabilitySignedZeroInfNanPreserve);
	if (state.requirements.buffer_u8) {
		state.builder.RequireExtension("SPV_KHR_8bit_storage");
		state.builder.RequireCapability(spv::CapabilityStorageBuffer8BitAccess);
	}
	if (state.requirements.buffer_u16) {
		state.builder.RequireCapability(spv::CapabilityStorageBuffer16BitAccess);
	}
	if (UsesPhysicalAddresses(state)) {
		state.builder.RequireCapability(spv::CapabilityInt64);
		state.builder.RequireCapability(spv::CapabilityPhysicalStorageBufferAddresses);
		state.builder.RequireExtension("SPV_KHR_physical_storage_buffer");
	}
	if (state.requirements.buffer_int64_atomics || state.requirements.shared_int64_atomics) {
		state.builder.RequireCapability(spv::CapabilityInt64);
		state.builder.RequireCapability(spv::CapabilityInt64Atomics);
	}
	if (state.requirements.shared_int64_atomics &&
	    state.lds_storage_class == spv::StorageClassWorkgroup) {
		state.builder.RequireVersion(0x00010400u);
		state.builder.RequireExtension("SPV_KHR_workgroup_memory_explicit_layout");
		state.builder.RequireCapability(spv::CapabilityWorkgroupMemoryExplicitLayoutKHR);
	}
	if (state.clip_distance_variable != 0) {
		state.builder.RequireCapability(spv::CapabilityClipDistance);
	}
	if (state.cull_distance_variable != 0) {
		state.builder.RequireCapability(spv::CapabilityCullDistance);
	}
	if (state.layer_variable != 0 || InputVariableForKind(state, IR::StageInputKind::Layer) != 0) {
		state.builder.RequireVersion(0x00010500u);
		state.builder.RequireCapability(spv::CapabilityShaderLayer);
	}
	if (state.viewport_index_variable != 0) {
		state.builder.RequireVersion(0x00010500u);
		state.builder.RequireCapability(spv::CapabilityShaderViewportIndex);
	}
	if (InputVariableForKind(state, IR::StageInputKind::SampleId) != 0) {
		state.builder.RequireCapability(spv::CapabilitySampleRateShading);
	}
	if (InputVariableForKind(state, IR::StageInputKind::BaseInstance) != 0 ||
	    InputVariableForKind(state, IR::StageInputKind::BaseVertex) != 0) {
		state.builder.RequireCapability(spv::CapabilityDrawParameters);
	}
	if (state.requirements.image_gather_extended) {
		state.builder.RequireCapability(spv::CapabilityImageGatherExtended);
	}
	if (state.lane_count == 2 || state.split_wave || state.requirements.subgroup_barrier ||
	    state.requirements.subgroup_ballot || state.requirements.subgroup_shuffle ||
	    state.requirements.subgroup_local_invocation_id) {
		state.builder.RequireCapability(spv::CapabilityGroupNonUniform);
	}
	if (state.lane_count == 2 || state.split_wave || state.requirements.subgroup_ballot) {
		state.builder.RequireCapability(spv::CapabilityGroupNonUniformBallot);
	}
	if (state.requirements.subgroup_shuffle || state.split_wave) {
		state.builder.RequireCapability(spv::CapabilityGroupNonUniformShuffle);
	}
	if (state.requirements.subgroup_arithmetic) {
		state.builder.RequireCapability(spv::CapabilityGroupNonUniformArithmetic);
	}
	if (state.requirements.shader_clock) {
		state.builder.RequireCapability(spv::CapabilityShaderClockKHR);
		state.builder.RequireExtension("SPV_KHR_shader_clock");
	}
	if (ComputeImplicitLod(state)) {
		// KYTY_IMPLICIT_LOD_STAGES: implicit-LOD samples (and any IMAGE_GET_LOD) take derivatives
		// over four consecutive invocations, the RDNA2 lane quad.
		const auto derivatives = GetHostImageFeatures().compute_derivatives;
		state.builder.RequireCapability(spv::CapabilityComputeDerivativeGroupLinearKHR);
		state.builder.RequireExtension(derivatives == HostComputeDerivatives::Nv
		                                   ? "SPV_NV_compute_shader_derivatives"
		                                   : "SPV_KHR_compute_shader_derivatives");
	} else if (state.requirements.compute_derivatives &&
	           state.program.stage == ShaderType::Compute) {
		// The NV extension declares the same capability (ComputeDerivativeGroupQuadsNV).
		const auto derivatives = GetHostImageFeatures().compute_derivatives;
		state.builder.RequireCapability(spv::CapabilityComputeDerivativeGroupQuadsKHR);
		state.builder.RequireExtension(derivatives == HostComputeDerivatives::Nv
		                                   ? "SPV_NV_compute_shader_derivatives"
		                                   : "SPV_KHR_compute_shader_derivatives");
		if (derivatives == HostComputeDerivatives::None) {
			NoteMissingComputeDerivatives(state.program.shader_hash);
		}
	}
	const bool fragment_barycentric =
	    state.program.stage == ShaderType::Pixel &&
	    std::any_of(state.inputs.begin(), state.inputs.end(), [](const InputBinding& input) {
		    return input.per_vertex || input.kind == IR::StageInputKind::BaryCoordSmooth ||
		           input.kind == IR::StageInputKind::BaryCoordNoPerspective;
	    });
	if (fragment_barycentric) {
		state.builder.RequireCapability(spv::CapabilityFragmentBarycentricKHR);
		state.builder.RequireExtension("SPV_KHR_fragment_shader_barycentric");
	}
	state.builder.AddMemoryModel(UsesPhysicalAddresses(state)
	                                 ? spv::AddressingModelPhysicalStorageBuffer64
	                                 : spv::AddressingModelLogical,
	                             spv::MemoryModelGLSL450);
	DeclareFloatControls(state);
	if (const auto* cs = ShaderWorkgroupInput(state.program.stage, state.input_info)) {
		uint32_t    local_x = state.requirements.compute_derivatives ? 2u : 1u;
		uint32_t    local_y = state.requirements.compute_derivatives ? 2u : 1u;
		uint32_t    local_z = 1u;
		local_x             = cs->threads_num[0] != 0u ? cs->threads_num[0] : local_x;
		local_y             = cs->threads_num[1] != 0u ? cs->threads_num[1] : local_y;
		local_z             = cs->threads_num[2] != 0u ? cs->threads_num[2] : local_z;
		if (state.lane_count == 2) {
			local_x = ((local_x * local_y * local_z + 63u) / 64u) * 32u;
			local_y = local_z = 1u;
			if (state.requirements.compute_derivatives) {
				local_y = local_x / 2u;
				local_x = 2u;
			}
		}
		// A pass runs 1/mesh_passes of the waves (mesh threads are one-dimensional).
		local_x /= state.mesh_passes;
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeLocalSize, local_x,
		                               local_y, local_z);
	}
	if (state.program.stage == ShaderType::Pixel) {
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeOriginUpperLeft);
		if (state.depth_variable != 0) {
			state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeDepthReplacing);
		}
		if (state.input_info.pixel->ps_early_z && !state.input_info.pixel->ps_pixel_kill_enable &&
		    !state.input_info.pixel->ps_depth_export_enable &&
		    !state.input_info.pixel->ps_sample_mask_export_enable) {
			state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeEarlyFragmentTests);
		}
		if (PopsInterlockActive(state)) {
			// KYTY_POPS_INTERLOCK: the body runs between OpBegin/OpEndInvocationInterlockEXT
			// (EmitProgram, EmitReturn). Pixel ordered covers the guest's per-pixel and per-sample
			// POPS modes (sample order is the weaker guarantee).
			state.builder.RequireExtension("SPV_EXT_fragment_shader_interlock");
			state.builder.RequireCapability(spv::CapabilityFragmentShaderPixelInterlockEXT);
			state.builder.AddExecutionMode(state.main_func, spv::ExecutionModePixelInterlockOrderedEXT);
			static std::mutex         ordered_mutex;
			static std::set<uint64_t> ordered_logged;
			std::scoped_lock          ordered_lock(ordered_mutex);
			if (ordered_logged.size() < 64 && ordered_logged.insert(state.program.shader_hash).second) {
				LOGF("KYTY_POPS_INTERLOCK: pixel shader 0x%016" PRIx64
				     " runs as a pixel-ordered interlock section\n",
				     state.program.shader_hash);
			}
		} else if (state.program.info.pops && GetCodegenOptions().pops_interlock) {
			static std::mutex         mutex;
			static std::set<uint64_t> logged;
			std::scoped_lock          lock(mutex);
			if (logged.size() < 64 && logged.insert(state.program.shader_hash).second) {
				LOGF("KYTY_POPS_INTERLOCK: pixel shader 0x%016" PRIx64
				     " has a POPS section but the device has no fragmentShaderPixelInterlock; "
				     "it runs unordered\n",
				     state.program.shader_hash);
			}
		}
	}
	if (ComputeImplicitLod(state)) {
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeDerivativeGroupLinearKHR);
	} else if (state.requirements.compute_derivatives &&
	           state.program.stage == ShaderType::Compute) {
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeDerivativeGroupQuadsKHR);
	}
	state.builder.AddName(state.main_func, "main");
	if (state.requirements.function_lds) {
		if (state.lds_slot_plan) {
			for (const auto& [key, variable]: state.lds_slot_variables) {
				state.builder.AddName(variable, ("lds_slot_" + std::to_string(key)).c_str());
			}
		} else {
			state.builder.AddName(state.lds_variable, "lds_dwords");
		}
	}
	if (state.requirements.function_scratch) {
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.builder.AddName(state.scratch_variable[half], "scratch_dwords");
		}
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
