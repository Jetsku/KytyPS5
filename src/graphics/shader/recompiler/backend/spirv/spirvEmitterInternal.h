#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_INTERNAL_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_INTERNAL_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvBuilder.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <optional>
#include <spirv/unified1/GLSL.std.450.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

struct InputBinding : IR::StageInput {
	uint32_t variable_id = 0;
};

struct OutputBinding : IR::StageOutput {
	uint32_t variable_id        = 0;
	uint32_t mesh_data_variable = 0;
};

using ImageDimension = Decoder::ImageDimension;

// Component counts are the guest's; 1D images are host 2D images one texel high.
struct ImageDimensionInfo {
	ImageDimension dimension;
	uint32_t       spirv_dimension;
	uint32_t       coordinate_components;
	uint32_t       spatial_components;
	uint32_t       arrayed;
	uint32_t       multisampled;
	bool           one_dimensional;
};

constexpr std::array<ImageDimensionInfo, 7> ImageDimensions {{
    {ImageDimension::Dim1D, spv::Dim2D, 1, 1, 0, 0, true},
    {ImageDimension::Dim1DArray, spv::Dim2D, 2, 1, 1, 0, true},
    {ImageDimension::Dim2D, spv::Dim2D, 2, 2, 0, 0, false},
    {ImageDimension::Dim2DArray, spv::Dim2D, 3, 2, 1, 0, false},
    {ImageDimension::Dim3D, spv::Dim3D, 3, 3, 0, 0, false},
    {ImageDimension::Dim2DMsaa, spv::Dim2D, 2, 2, 0, 1, false},
    {ImageDimension::Dim2DMsaaArray, spv::Dim2D, 3, 2, 1, 1, false},
}};

const ImageDimensionInfo& ImageDimensionInfoFor(ImageDimension dimension);

struct SpirvRequirements {
	bool subgroup_ballot              = false;
	bool subgroup_barrier             = false;
	bool subgroup_shuffle             = false;
	bool subgroup_local_invocation_id = false;
	// Native subgroup reductions of DPP row scans (IR::MatchLaneReduction, KYTY_LANE_REDUCTIONS).
	bool subgroup_arithmetic          = false;
	bool compute_derivatives          = false;
	// KYTY_IMPLICIT_LOD_STAGES: a compute program with implicit-LOD samples (ComputeImplicitLod).
	bool implicit_compute_lod         = false;
	bool image_gather_extended        = false;
	bool function_lds                 = false;
	// KYTY_LDS_SLOTS (CodegenOptions::lds_slots): the element offsets k of a non-compute stage's
	// function-scope LDS accesses when each is a dword access of element LaneId + k
	// (StaticLdsSlot), and whether some access is not (then the array stays).
	std::vector<uint32_t> lds_slot_keys;
	bool                  lds_slots_blocked = false;
	bool function_scratch             = false;
	bool pixel_valid_mask             = false;
	// A pixel shader's DS_APPEND/DS_CONSUME elect a non-helper lane (KYTY_PS_APPEND_LIVE_ELECTION).
	bool helper_invocation            = false;
	bool buffer_int64_atomics         = false;
	bool buffer_u8                    = false;
	bool buffer_u16                   = false;
	bool shared_int64_atomics         = false;
	bool coherent_buffers             = false;
	// KYTY_COHERENT_GLC_ONLY: a coherent buffer access narrower than 32 bits (u8/u16 alias), a
	// BUFFER_GL0/GL1_INV, and an image store or atomic; each keeps whole-program coherence.
	bool coherent_narrow              = false;
	bool coherent_dcache              = false;
	bool image_writes                 = false;
	// A buffer store or atomic anywhere in the program; without one (and without coherent loads)
	// the storage buffers are declared NonWritable (CodegenOptions::readonly_buffers).
	bool buffer_writes                = false;
	bool float64                      = false;
	// KYTY_NATIVE_F16: HalfPk* operations (Float16 and the 16-bit float controls).
	bool native_half                  = false;
	// S_MEMREALTIME reads the host shader clock (OpReadClockKHR, VK_KHR_shader_clock).
	bool shader_clock                 = false;
};

SpirvRequirements AnalyzeProgramRequirements(const IR::Program& program);
// KYTY_BUFFER_PAIR_ATOMIC_HASHES=<hex>[,<hex>...]: limits KYTY_BUFFER_PAIR_ATOMIC to those programs
// (the large Miles RT kernels hit "Shader Instruction Error" device losses with it); unset = all.
[[nodiscard]] bool BufferPairAtomicProgram(uint64_t shader_hash);

// KYTY_LANE2_SHARE (spirvEmitterLaneShare.cpp): with lane_count 2 every IR instruction is emitted once
// per wave half; a value proven the same in both halves is emitted by the low half only.
struct LaneShareState {
	bool active = false;
	// Instructions the high half does not emit (uniform minus the whole-wave ones it already reuses).
	std::unordered_set<const IR::Inst*> shared;
	// Every instruction proven to have the same value in both halves.
	std::unordered_set<const IR::Inst*> uniform;
	size_t   instructions = 0;
	uint64_t reused       = 0;
};

struct EmitterState;
// KYTY_COHERENT_GLC_ONLY bookkeeping (KYTY_COHERENT_GLC_LOG=1 logs each coherent program).
void CountCoherentProgram(const EmitterState& state, bool per_access);

struct EmitterState {
	EmitterState(const IR::Program& program_, ShaderStageInputInfo input_info_)
	    : builder(program_.stage == ShaderType::Mesh ? 0x00010400u : 0x00010300u),
	      program(program_), input_info(input_info_),
	      requirements(AnalyzeProgramRequirements(program_)) {
		if (ShaderWorkgroupInput(program.stage, input_info) != nullptr) {
			lds_storage_class = spv::StorageClassWorkgroup;
		}
		if (IR::FindBinding(program.bindings, IR::DescriptorBindingKind::SharedMemory) != nullptr) {
			lds_storage_class = spv::StorageClassStorageBuffer;
		}
	}

	Builder                                          builder;
	const IR::Program&                               program;
	ShaderStageInputInfo                             input_info;
	// PA_CL_UCP planes the last pre-rasterization stage writes as clip distances (upstream feed075a9).
	uint32_t                                        user_clip_planes = 0;
	uint32_t                                        void_type = 0;
	uint32_t                                        bool_type = 0;
	uint32_t                                        u32_type = 0;
	uint32_t                                        u64_type = 0;
	uint32_t                                        i32_type = 0;
	uint32_t                                        f32_type = 0;
	uint32_t                                        f64_type = 0;
	// KYTY_NATIVE_F16: float16 and f16vec2 (EmitHalfPk).
	uint32_t                                        f16_type = 0;
	uint32_t                                        f16_pair_type = 0;
	uint32_t                                        u32_pair_type = 0;
	uint32_t                                        i32_pair_type = 0;
	uint32_t                                        function_type = 0;
	std::array<uint32_t, 3>                          bool_vector_types {};
	std::array<uint32_t, 3>                          u32_vector_types {};
	std::array<uint32_t, 3>                          i32_vector_types {};
	std::array<uint32_t, 3>                          f32_vector_types {};
	std::array<uint32_t, IR::NumTessellationAttributes> tess_variables {};
	uint32_t                                         tess_patch_base     = 0;
	uint32_t                                         tess_inner_variable = 0;

	const SpirvRequirements                          requirements;
	uint32_t                                         lane_count              = 1;
	uint32_t                                         lane_half               = 0;
	LaneShareState                                   lane_share;
	// KYTY_WAVE64_WARPS (split wave): a wave64 compute workgroup of one wave runs as two 32-wide
	// host subgroups, one guest lane per invocation (lane_count 1). Cross-lane operations
	// exchange the halves through workgroup memory (ExchangeWaveValue), and DS accesses
	// synchronize the halves the guest ran in lockstep.
	bool                                             split_wave              = false;
	uint32_t                                         wave_exchange_variable  = 0;
	uint32_t                                         wave_exchange_parity    = 0;
	// Blocks whose conditional branch each half decides on its own lanes (SplitWaveLocalBranches).
	std::unordered_set<const IR::Block*>             wave_local_branches;
	// The probe emission: every branch is half-local, and wave_sync_blocks collects the blocks
	// that synchronize the halves.
	bool                                             wave_probe              = false;
	std::unordered_set<const IR::Block*>*            wave_sync_blocks        = nullptr;
	// DS instructions write the LDS, so DS reads must wait for the other half too.
	bool                                             ds_writes_lds           = false;
	// KYTY_SPLIT_LDS_WRITE_SYNC: no LDS read since the last split-wave barrier in this block.
	bool                                             lds_split_synced        = false;
	// KYTY_WAVE32_CLUSTERS (Spirv::WaveClustersActive): a wave32 program on a host subgroup that may
	// hold two guest waves keeps each to its own 32 lanes (EmitSubgroupLocalInvocationId,
	// EmitHostLane, EmitWaveBallot).
	bool                                             wave_clusters           = false;
	uint32_t                                         storage_buffer_variable = 0;
	// KYTY_UNIFORM_BUFFERS: the constant-only buffers (DescriptorBindingKind::UniformBuffers).
	uint32_t                                         uniform_buffer_variable = 0;
	uint32_t                                         storage_buffer_u8_variable = 0;
	uint32_t                                         storage_buffer_u16_variable = 0;
	uint32_t                                         storage_buffer_u64_variable = 0;
	// KYTY_COHERENT_GLC_ONLY: the Coherent alias the GLC/DLC 32-bit buffer accesses use (0 = none:
	// every alias carries the program's coherence).
	uint32_t                                         storage_buffer_coherent_variable = 0;
	std::array<uint32_t, IR::ShaderInfo::MaxBuffers> memory_byte_offsets {};
	uint32_t                                         bda_pagetable_variable  = 0;
	uint32_t                                         fault_buffer_variable   = 0;
	uint32_t                                         bda_pointer_function    = 0;
	// KYTY_RT_FUNCTION: the module's one software BVH node test function (DefineBvhNodeTestFunction).
	uint32_t                                         bvh_node_test_function  = 0;
	// Native ray tracing (spirvEmitterRayQuery.cpp): function variables per site and emulated half.
	struct RayQuerySiteVariables {
		std::array<uint32_t, 2> query {}, state {}, instance {}, node {}, word {}, saved_t {}, saved_node {},
		    saved_inst {}, nodes {}, slots {}, keep {};
		uint32_t selected = 0;
		uint32_t rounds   = 0;
	};
	std::array<RayQuerySiteVariables, 4>             ray_query {};
	bool                                             ray_query_keep = false;
	// PS5 Pro IMAGE_BVH8_INTERSECT_RAY: its result struct and module function (spirvEmitterBvh8.cpp).
	uint32_t                                         bvh8_result_type        = 0;
	uint32_t                                         bvh8_intersect_function = 0;
	uint32_t                                         gds_variable            = 0;
	uint32_t                                         gds_length              = 0;
	uint32_t                                         push_constant_variable  = 0;
	uint32_t                                         shader_data_storage_variable = 0;
	uint32_t                                         mip_stats_variable           = 0;
	// False for the plain variant of an instrumented pixel shader (KYTY_LOD_STATS_PLAIN_VARIANT):
	// the mip_stats binding stays declared, but no sample records GET_LOD_STATS feedback.
	bool                                             mip_stats_records            = true;
	uint32_t                                         flattened_srt_variable  = 0;
	// KYTY_SRT_UNIFORM: flattened_srt_variable is a uniform block (FlattenedSrtElementPointerType).
	bool                                             flattened_srt_uniform   = false;
	// Bindless images: set-1 runtime arrays by image type, the key -> slot translation buffer,
	// the feedback buffer, and the slot of the image instruction being emitted.
	std::map<uint32_t, uint32_t>                     bindless_image_variables;
	uint32_t                                         bindless_translation_variable = 0;
	uint32_t                                         bindless_feedback_variable    = 0;
	uint32_t                                         bindless_slot                 = 0;
	// KYTY_BINDLESS_SAMPLERS: the sampler array of set 1, and the slot of the current instruction's
	// bindless sampler (0: its sampler is not bindless).
	uint32_t                                         bindless_sampler_variable     = 0;
	uint32_t                                         bindless_sampler_slot         = 0;
	// KYTY_BINDLESS_3D_AS_ARRAY: the raw translation entry behind bindless_slot
	// (BindlessSlotArrayFlag marks a 2D or 2D-array texture a 3D image instruction samples).
	uint32_t                                         bindless_translation_raw      = 0;
	// KYTY_RUNTIME_IMAGES: the runtime image table and request buffer (set 1, bindings 7 and 8).
	uint32_t                                         runtime_image_table_variable    = 0;
	uint32_t                                         runtime_image_requests_variable = 0;
	// While image_override_index names a resource, ImageResourceAt() returns image_override for
	// it: the same instruction emitted against another view type of its texture.
	uint32_t                                         image_override_index          = UINT32_MAX;
	IR::ImageResource                                image_override {};
	// The module's decoder of formatted loads through a computed V# (DefineIndirectFormatFunction).
	uint32_t                                         indirect_format_function      = 0;
	// KYTY_INDIRECT_RAW_OUTLINE: the module's raw-load functions through a computed V#, by shape
	// (IndirectRawKey / IndirectRawElementKey in spirvEmitterMemory.cpp).
	std::map<uint32_t, uint32_t>                     indirect_raw_functions;
	// KYTY_INDIRECT_FORMAT_COMPACT: the compact decoder when its function control is the default one
	// (no KYTY_INDIRECT_FORMAT_COMPACT_CONTROL), for KYTY_HUGE_HELPER_OUTLINE (SpirvEmitter.cpp).
	uint32_t                                         indirect_format_compact_function = 0;
	spv::StorageClass                                lds_storage_class = spv::StorageClassFunction;
	// StorageBuffer LDS (KYTY_LDS_DEVICE_BUFFER), set at function entry: the workgroup's first dword
	// in the buffer, and the bound of its accesses (LdsDwordCount, or 0 when the bound range holds
	// no region for it).
	uint32_t                                         lds_base_dwords         = 0;
	uint32_t                                         lds_length              = 0;
	uint32_t                                         lds_variable            = 0;
	// With lds_slot_plan, one Function variable per element offset (lds_slot_keys) in place of the
	// function-scope LDS array (KYTY_LDS_SLOTS).
	bool                                             lds_slot_plan = false;
	std::map<uint32_t, uint32_t>                     lds_slot_variables;
	uint32_t                                         lds_u64_variable        = 0;
	std::array<uint32_t, 2>                          scratch_variable {};
	std::array<uint32_t, IR::ImageBindingCount>      image_variables {};
	// KYTY_IMAGE_GLC_COHERENT / KYTY_POPS_INTERLOCK: Coherent aliases of storage image bindings with
	// coherent accesses (0: none), and whether the access being emitted uses them
	// (CoherentImageScope, ImageDescriptorPointer).
	std::array<uint32_t, IR::ImageBindingCount>      image_coherent_variables {};
	bool                                             coherent_image_access = false;
	uint32_t                   sampler_variable                      = 0;
	uint32_t                   main_func                             = 0;
	uint32_t                   mesh_guest_func                       = 0;
	uint32_t                   mesh_allocation                       = 0;
	uint32_t                   mesh_primitive_data                   = 0;
	uint32_t                   mesh_primitives                       = 0;
	uint32_t                   mesh_cull                             = 0;
	uint32_t                   mesh_zero_position_mask               = 0;
	uint32_t                   mesh_zero_position_words              = 0;
	// A subgroup larger than the host workgroup runs in mesh_passes passes (ShaderMeshInputInfo):
	// mesh_pass_variable holds the current pass, and the guest program is cut at its barriers
	// into mesh_segment_funcs, each called once per pass.
	uint32_t                   mesh_passes                           = 1;
	uint32_t                   mesh_pass_variable                    = 0;
	std::vector<uint32_t>      mesh_segment_funcs;
	uint32_t                   entry_label                           = 0;
	uint32_t                   current_label                         = 0;
	const IR::Block*           current_block                         = nullptr;
	uint32_t                   pixel_valid_mask_variable             = 0;
	uint32_t                   subgroup_local_invocation_id_variable = 0;
	uint32_t                   helper_invocation_variable            = 0;
	// KYTY_LOOP_GUARD: the invocation's loop-iteration count (a Function variable).
	uint32_t                   loop_guard_variable                   = 0;
	// KYTY_CALL_LOOP_CAP: rounds of the bracketed callee bodies' loops this invocation ran
	// (CFG::Terminator::call_loop; a Function variable).
	uint32_t                   call_loop_variable                    = 0;
	// KYTY_RT_NODE_BUDGET / KYTY_RT_NODE_STATS: BVH instruction executions of this invocation
	// (Function u32; both halves of a two-lane invocation add to it).
	uint32_t                   bvh_node_count_variable               = 0;
	uint32_t                   per_vertex_variable                   = 0;
	uint32_t                   point_size_variable                   = 0;
	uint32_t                   clip_distance_variable                = 0;
	uint32_t                   invalid_position_clip_distance        = UINT32_MAX;
	uint32_t                   cull_distance_variable                = 0;
	uint32_t                   layer_variable                        = 0;
	uint32_t                   layer_limit_constant                  = 0; // KYTY_LAYER_CLAMP
	uint32_t                   viewport_index_variable               = 0;
	uint32_t                   depth_variable                        = 0;
	uint32_t                   sample_mask_variable                  = 0;
	// KYTY_OBJPRIM_ID: the pixel shader's ObjPrimID output (StageOutputKind::ObjPrimId).
	uint32_t                   objprim_output_variable               = 0;
	std::vector<InputBinding>  inputs;
	std::vector<OutputBinding> outputs;
	std::vector<uint32_t>      interface_variables;
	std::unordered_map<const IR::Block*, uint32_t> labels;
	// Instructions whose values reach a position export (MadMode::Position), built on first use.
	bool                                  position_slice_ready = false;
	std::unordered_set<const IR::Inst*>   position_slice;
	// Pixel inputs whose V_INTERP_P2 reads were all resolved to hardware I/J pairs of one
	// perspective mode (KYTY_INTERP_MODES): variable -> sampling location its decoration selects.
	std::unordered_map<uint32_t, IR::InterpolationMode> input_interpolation;
	// KYTY_MOVREL_SWITCH: long select(index == constant, ...) chains (V_MOVRELS reads) become one
	// OpSwitch at their outermost select; their inner selects emit nothing (EmitIndexedSelect),
	// found on first use.
	bool                                indexed_selects_found = false;
	std::unordered_set<const IR::Inst*> indexed_select_heads;
	std::unordered_set<const IR::Inst*> indexed_select_members;
	std::unordered_set<const IR::Inst*> indexed_select_compares;
};

uint32_t TypeVoid(EmitterState& state);
uint32_t TypeBool(EmitterState& state);
uint32_t TypeBoolVector(EmitterState& state, uint32_t components);
uint32_t TypeU32(EmitterState& state);
uint32_t TypeU64(EmitterState& state);
uint32_t TypeU32Pair(EmitterState& state);
uint32_t TypeBvh8Result(EmitterState& state); // spirvEmitterBvh8.cpp
uint32_t TypeI32(EmitterState& state);
uint32_t TypeI32Pair(EmitterState& state);
uint32_t TypeF32(EmitterState& state);
uint32_t TypeF64(EmitterState& state);
uint32_t TypeU32Vector(EmitterState& state, uint32_t components);

uint32_t TypeU32Composite(EmitterState& state, uint32_t components);
uint32_t TypeI32Vector(EmitterState& state, uint32_t components);
uint32_t TypeF32Vector(EmitterState& state, uint32_t components);
uint32_t TypePointer(EmitterState& state, spv::StorageClass storage_class, uint32_t pointee);
uint32_t TypeFunction(EmitterState& state);
uint32_t TypeStorageBufferElement(EmitterState& state, uint32_t bits);
uint32_t TypeStorageBufferPointer(EmitterState& state, uint32_t bits = 32);
uint32_t TypeStorageBufferElementPointer(EmitterState& state, uint32_t bits = 32);
// The pointer type of one dword of flattened_srt_variable (storage buffer, or KYTY_SRT_UNIFORM's
// uniform block).
uint32_t FlattenedSrtElementPointerType(EmitterState& state);
uint32_t TypePhysicalU32Pointer(EmitterState& state);
// The module declares 64-bit integers and physical storage buffer addresses.
bool UsesPhysicalAddresses(const EmitterState& state);
// KYTY_INDIRECT_THREADS_GPU: a thread-dimension compute dispatch whose counts the shader reads
// from GPU memory (ShaderComputeInputInfo::dispatch_dimensions_indirect).
[[nodiscard]] bool DispatchDimensionsIndirect(const EmitterState& state);
uint32_t TypePhysicalU64Pointer(EmitterState& state);
uint32_t TypePushConstantElementPointer(EmitterState& state);
uint32_t TypeU32ArrayPointer(EmitterState& state, spv::StorageClass storage_class, uint32_t dwords);
uint32_t TypeU32ElementPointer(EmitterState& state, spv::StorageClass storage_class);

// KYTY_BINDLESS_3D_AS_ARRAY: a bindless sampled 3D image whose key may name a 2D or 2D-array
// texture instead (adapted from KytyPS5-Frank fed216ce8).
inline bool BindlessVolumeMayBeArray(const IR::ImageResource& image) {
	return GetCodegenOptions().bindless_3d_as_array && image.bindless && !image.cube &&
	       image.resource_class == IR::ImageResourceClass::Sampled &&
	       image.dimension == Decoder::ImageDimension::Dim3D;
}

inline const IR::ImageResource& ImageResourceAt(const EmitterState& state, uint32_t index) {
	return index == state.image_override_index ? state.image_override
	                                           : state.program.info.images.at(index);
}

inline void EmitLabel(EmitterState& state, uint32_t label) {
	state.current_label = label;
	state.builder.AddFunction(spv::OpLabel, label);
}

uint32_t TypeId(EmitterState& state, IR::Type type);

// Shared instruction construction; typed aliases add no forwarding functions.
template <spv::Op opcode, IR::Type type, typename... Args>
uint32_t EmitNative(EmitterState& state, Args... args) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(opcode, TypeId(state, type), result, args...);
	if constexpr (type == IR::Type::F64 &&
	              (opcode == spv::OpFMul || opcode == spv::OpFDiv || opcode == spv::OpExtInst)) {
		state.builder.AddAnnotation(spv::OpDecorate, result, spv::DecorationNoContraction);
	}
	return result;
}

uint32_t GlslStd450(EmitterState& state);

template <GLSLstd450 opcode, IR::Type type, typename... Args>
uint32_t EmitGlsl(EmitterState& state, Args... args) {
	return EmitNative<spv::OpExtInst, type>(state, GlslStd450(state), opcode, args...);
}

inline uint32_t Unary(EmitterState& state, spv::Op opcode, uint32_t type, uint32_t value) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(opcode, type, result, value);
	return result;
}

inline uint32_t PackU64(EmitterState& state, uint32_t low, uint32_t high) {
	const auto pair = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 2), pair, low, high);
	return Unary(state, spv::OpBitcast, TypeU64(state), pair);
}

inline uint32_t Binary(EmitterState& state, spv::Op opcode, uint32_t type, uint32_t lhs,
                       uint32_t rhs) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(opcode, type, result, lhs, rhs);
	return result;
}

inline uint32_t Select(EmitterState& state, uint32_t type, uint32_t condition, uint32_t true_value,
                       uint32_t false_value) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, type, result, condition, true_value, false_value);
	return result;
}

struct ValueEmitContext {
	explicit ValueEmitContext(EmitterState& state_): state(state_) {}

	uint32_t              Def(IR::Value value);
	uint32_t              Arg(const IR::Inst& inst, size_t index);
	uint32_t              HalfArg(const IR::Inst& inst, size_t index, uint32_t half);
	uint32_t              Ballot(IR::Value predicate);
	uint32_t              FirstLane(uint32_t ballot);
	uint32_t              Shuffle(const IR::Inst& inst, size_t index, uint32_t lane);
	uint32_t              Result(const IR::Inst& inst);
	uint32_t              Define(const IR::Inst& inst, uint32_t value);
	uint32_t              ResourceIndex(IR::Value value, IR::ValueOpcode opcode);
	const IR::Inst*       ImageAddress(IR::Value value);
	const IR::MemoryInfo& Memory(const IR::Inst& inst) const;
	const IR::ExportInfo& Export(const IR::Inst& inst) const;
	uint32_t              Label(const IR::Block* block) const;
	[[noreturn]] void     Fail(const char* reason) const;
	[[noreturn]] void     Fail(const IR::Inst& inst, const char* reason) const;

	EmitterState&                                                      state;
	std::unordered_map<const IR::Inst*, uint32_t>                      definitions;
	const std::unordered_map<const IR::Inst*, uint32_t>*               dispatcher_spills = nullptr;
	std::unordered_map<const IR::Inst*, std::pair<uint32_t, uint32_t>> dispatcher_block_loads;
	uint32_t                                                           scratch_u32_variable = 0;
	ValueEmitContext*                                                  other_half = nullptr;
	uint32_t                                                           half       = 0;
};

constexpr uint32_t NoImageComponent = 0xffffffffu;

struct DppTargetLane {
	uint32_t lane  = 0;
	uint32_t valid = 0;
};

struct ImageSampleLayout {
	uint32_t offset = NoImageComponent;
	uint32_t dref   = NoImageComponent;
	uint32_t bias   = NoImageComponent;
	uint32_t coord  = 0;
	uint32_t lod    = NoImageComponent;
	uint32_t grad_x = NoImageComponent;
	uint32_t grad_y = NoImageComponent;
	uint32_t clamp  = NoImageComponent; // IMAGE_SAMPLE*_CL minimum LOD, the last component
};

struct F32Class {
	uint32_t bits = 0;
	uint32_t nan  = 0;
	uint32_t zero = 0;
};

uint32_t PixelParameterLocation(const EmitterState& state, uint32_t attr);

bool PixelParameterIsFlat(const EmitterState& state, uint32_t attr);

bool PixelParameterIsCustom(const EmitterState& state, uint32_t attr);

Format::ComponentType VertexParameterScalarKind(const EmitterState& state, uint32_t location);

uint32_t VertexParameterComponentCount(const InputBinding& input);

uint32_t VertexParameterScalarType(EmitterState& state, Format::ComponentType kind);

uint32_t OutputVariableForExport(const EmitterState& state, const IR::ExportInfo& exp);
uint32_t MrtOutputType(EmitterState& state, uint32_t index);

uint32_t ConstantU32(EmitterState& state, uint32_t value);

uint32_t EmitSubgroupLocalInvocationId(EmitterState& state);

[[noreturn]] void ExitDescriptorBindingFailure(const EmitterState&       state,
                                               IR::DescriptorBindingKind kind, uint32_t resource,
                                               const char* reason);

uint32_t ResourceForDescriptor(const EmitterState& state, IR::DescriptorBindingKind kind,
                               uint32_t resource);

uint32_t DescriptorElementPointer(EmitterState& state, uint32_t result_ptr_type,
                                  uint32_t variable_id, uint32_t array_index,
                                  IR::DescriptorBindingKind kind, uint32_t resource,
                                  const char* variable_name);

uint32_t ImageScalarType(EmitterState& state, Prospero::TextureNumericClass numeric_class);

uint32_t ImageVectorType(EmitterState& state, Prospero::TextureNumericClass numeric_class,
                         uint32_t components);

uint32_t ImageType(EmitterState& state, const IR::ImageResource& image);

uint32_t ImageViewSizeType(EmitterState& state, ImageDimension dimension);

uint32_t ImageDescriptorPointer(EmitterState& state, uint32_t resource, uint32_t mip = 0,
                                uint32_t array_index = 0);

void DefineIndirectFormatFunction(EmitterState& state);
uint32_t LoadImageDescriptor(EmitterState& state, uint32_t resource, uint32_t mip = 0,
                             uint32_t array_index = 0);

uint32_t LoadSamplerDescriptor(EmitterState& state, uint32_t sampler);

uint32_t MakeSampledImage(EmitterState& state, uint32_t resource, uint32_t sampler_id,
                          uint32_t mip = 0, uint32_t array_index = 0);

// sample is 0 (no id) for single-sampled images.
void EmitStorageImageWrite(EmitterState& state, uint32_t resource, uint32_t mip_lod, uint32_t coord,
                           uint32_t texel);

spv::ExecutionModel ExecutionModelForStage(ShaderType stage);

uint32_t ConstantI32(EmitterState& state, int32_t value);

uint32_t ConstantF32(EmitterState& state, uint32_t bits);

uint32_t ConstantF32Value(EmitterState& state, float value);

uint32_t ConstantBool(EmitterState& state, bool value);

uint32_t ConstantU64(EmitterState& state, uint64_t value);

uint32_t ConstantU32CompositeZero(EmitterState& state, uint32_t components);

uint32_t DefineInterfaceVariable(EmitterState& state, uint32_t type, spv::StorageClass storage,
                                 const char* name);
void     DefineModule(EmitterState& state);
// The float-control capabilities, extension and execution modes of state.main_func that every
// module declares (part of DefineModule; also the KYTY_F32_FLUSH_HOST self-test module's header).
void     DeclareFloatControls(EmitterState& state);
void     DefineTessellationInterfaces(EmitterState& state);
void     DefineTessellationExecutionModes(EmitterState& state);
void     DefineMeshOutputs(EmitterState& state, uint32_t clip_distance_count,
                           uint32_t cull_distance_count);
void     EmitMeshEntryPoint(EmitterState& state);
void     EmitGeometryOutputDefaults(EmitterState& state);
void     EmitMeshAllocate(ValueEmitContext& ctx, const IR::Inst& inst);
uint32_t MeshOutputPointer(EmitterState& state, IR::StageOutputKind kind, uint32_t index = 0);
uint32_t MeshPrimitivePointer(EmitterState& state);
uint32_t MeshLaneSlot(EmitterState& state);

DppTargetLane EmitDppTargetLane(EmitterState& state, const IR::DppMoveFlags& flags);

// The invocation's guest lane: SubgroupLocalInvocationId (+32 in the second half of a two-lane
// invocation; its low 5 bits with KYTY_WAVE32_CLUSTERS).
uint32_t EmitSubgroupLocalInvocationId(EmitterState& state);

// KYTY_WAVE32_CLUSTERS: the host subgroup lane of guest lane `lane` of this invocation's wave (the
// wave's first host lane plus lane & 31); `lane` unchanged without clusters. Every
// OpGroupNonUniformShuffle index goes through it.
uint32_t EmitHostLane(EmitterState& state, uint32_t lane);

// KYTY_WAVE32_CLUSTERS: a subgroup ballot reduced to this invocation's wave, its 32 bits in the
// first word and zero in the others; `ballot` unchanged without clusters.
uint32_t EmitWaveBallot(EmitterState& state, uint32_t ballot);

// Loads gl_HelperInvocation (declared when SpirvRequirements::helper_invocation is set).
uint32_t EmitIsHelperInvocation(EmitterState& state);

uint32_t InputVariableForKind(const EmitterState& state, IR::StageInputKind kind);

const InputBinding* InputBindingForParameter(const EmitterState& state, uint32_t location);

uint32_t EmitVertexParameterComponentU32(EmitterState& state, const InputBinding& input,
                                         uint32_t component);

uint32_t EmitInputComponentU32(EmitterState& state, IR::StageInputKind kind, uint32_t component);

uint32_t EmitLocalInvocationIndex(EmitterState& state);

uint32_t EmitBallotLaneActiveBool(EmitterState& state, uint32_t ballot, uint32_t lane);

uint32_t EmitSubgroupLaneActiveBool(EmitterState& state, uint32_t lane);

// Split waves (EmitterState::split_wave). Whether lane operations confined to 32-lane halves use
// the host subgroup's lane (lane & 31): two lanes per invocation or a split wave.
[[nodiscard]] inline bool HalfWaveHost(const EmitterState& state) {
	return state.lane_count == 2 || state.split_wave;
}
// KYTY_IMAGE_GLC_COHERENT: a GLC/DLC storage image access; KYTY_POPS_INTERLOCK: any storage image
// access of a POPS program (its ordered read-modify-write must see the previous fragment's store).
[[nodiscard]] bool ImageAccessCoherent(const EmitterState& state, const IR::MemoryInfo& mem);
// Sets EmitterState::coherent_image_access for the descriptor loads of one image access.
class CoherentImageScope {
public:
	CoherentImageScope(EmitterState& state, bool coherent)
	    : m_state(state), m_saved(state.coherent_image_access) {
		state.coherent_image_access = coherent;
	}
	~CoherentImageScope() { m_state.coherent_image_access = m_saved; }
	CoherentImageScope(const CoherentImageScope&)            = delete;
	CoherentImageScope& operator=(const CoherentImageScope&) = delete;
private:
	EmitterState& m_state;
	bool          m_saved;
};
// KYTY_IMPLICIT_LOD_STAGES: this compute program's implicit-LOD samples keep implicit LOD, with
// derivatives over groups of four consecutive invocations (DerivativeGroupLinear: the RDNA2 lane
// quad). Needs one guest lane per invocation (no wave64 split), a workgroup of a multiple of 4
// invocations and computeDerivativeGroupLinear.
[[nodiscard]] bool ComputeImplicitLod(const EmitterState& state);
// KYTY_POPS_INTERLOCK: this pixel program runs its body as a fragment shader interlock section.
[[nodiscard]] inline bool PopsInterlockActive(const EmitterState& state) {
	return GetCodegenOptions().pops_interlock && state.program.stage == ShaderType::Pixel &&
	       state.program.info.pops && GetHostFragmentInterlock();
}
// The invocation's lane in its host subgroup (0-31 for a split wave).
uint32_t EmitHostSubgroupLane(EmitterState& state);
// Records that the block being emitted synchronizes a split wave's halves.
void     NoteWaveSync(EmitterState& state);
// Synchronizes a split wave's halves and makes their workgroup memory writes visible.
void     EmitWaveBarrier(EmitterState& state);
// The value (u32) of guest lane `lane` (0-63) of a split wave, through workgroup memory.
uint32_t ExchangeWaveValue(EmitterState& state, uint32_t value, uint32_t lane);
// The 64-lane ballot of `predicate` in a split wave, as {low lanes, high lanes, 0, 0}.
uint32_t ExchangeWaveBallot(EmitterState& state, uint32_t predicate);
// The ballot word of `predicate` over the invocation's own host subgroup.
uint32_t EmitHostBallotWord(EmitterState& state, uint32_t predicate);
// The invocation's host-subgroup ballot placed as a wave ballot whose lane bits are right for
// lanes of the invocation's own half: {word, word, 0, 0}.
uint32_t EmitHalfBallot(EmitterState& state, uint32_t predicate);
// Reads `value` (any 32-bit or bool type) from guest lane `lane` of the invocation's own half.
uint32_t ReadHostSubgroupLane(EmitterState& state, uint32_t type, uint32_t value, uint32_t lane);
// The blocks whose conditional branch a split wave's halves may decide on their own: no block
// the branch controls synchronizes the halves (sync_blocks), directly or through another branch
// that stays whole-wave, and no such block ends the program.
std::unordered_set<const IR::Block*> SplitWaveLocalBranches(
    const IR::Program& program, const std::unordered_set<const IR::Block*>& sync_blocks);

inline constexpr auto EmitAddU32 = EmitNative<spv::OpIAdd, IR::Type::U32, uint32_t, uint32_t>;

uint32_t EmitBinaryU32(EmitterState& state, spv::Op opcode, uint32_t lhs, uint32_t rhs);

uint32_t EmitShaderDataDwordLoad(EmitterState& state, uint32_t dword_index);

uint32_t StorageBufferPackedStride(const EmitterState& state, const IR::MemoryInfo& mem);

Prospero::BufferFormat StorageBufferFormat(const EmitterState& state, const IR::MemoryInfo& mem);
uint32_t StorageBufferElementBits(const IR::Program& program, const IR::MemoryInfo& mem);

void EmitMemoryOffsets(EmitterState& state);

uint32_t LdsDwordCount(const EmitterState& state);
void EnsureLdsStorage(EmitterState& state);

// LDS kept in a device buffer (b0bbaef1e, compute LDS past the device limit) is uniform memory.
inline spv::MemorySemanticsMask LdsMemorySemantics(const EmitterState& state) {
	return state.lds_storage_class == spv::StorageClassStorageBuffer
	           ? spv::MemorySemanticsUniformMemoryMask
	           : spv::MemorySemanticsWorkgroupMemoryMask;
}

struct MemoryResourceAccess {
	IR::ResourceKind      kind             = IR::ResourceKind::None;
	uint32_t              object_pointer   = 0;
	uint32_t              descriptor_index = 0;
	uint32_t              length           = 0;
	uint32_t              byte_offset      = 0;
	uint32_t              element_bits     = 32;
	spv::MemoryAccessMask memory_access    = spv::MemoryAccessMaskNone;
	// KYTY_STORAGE_LENGTH_ZERO: a length of 0 means unbounded (NVIDIA's OpArrayLength of a binding
	// of 2 GiB or more).
	bool                  zero_length_unbounded = false;
};

uint32_t EmitIndirectResourceIndex(EmitterState& state, uint32_t key, uint32_t mapping_offset,
                                   uint32_t search_iterations, uint32_t default_resource);
MemoryResourceAccess PrepareMemoryResourceAccess(EmitterState& state, const IR::MemoryInfo& mem);

MemoryResourceAccess PrepareStorageBufferResourceAccess(EmitterState&         state,
                                                        const IR::MemoryInfo& mem,
                                                        uint32_t variable, uint32_t pointer_type);

uint32_t EmitMemoryElementInBounds(EmitterState& state, const MemoryResourceAccess& access,
                                   uint32_t index);
// EmitMemoryElementInBounds, or true for storage buffers the device bounds-checks
// (KYTY_ROBUST_BUFFER_STORES); only plain stores use it (loads: DeviceChecksDwordLoad).
uint32_t EmitStorageElementInBounds(EmitterState& state, const MemoryResourceAccess& access,
                                    uint32_t index);

uint32_t EmitMemoryElementPointer(EmitterState& state, const MemoryResourceAccess& access,
                                  uint32_t index);

uint32_t EmitStorageBufferElementPointer(EmitterState& state, const MemoryResourceAccess& access,
                                         uint32_t index, uint32_t pointer_type);

inline constexpr auto EmitTBufferSelectF32 =
    EmitNative<spv::OpSelect, IR::Type::F32, uint32_t, uint32_t, uint32_t>;

bool IsSignedFormatComponent(Format::ComponentType type);

uint32_t NormalizeFormatComponent(EmitterState& state, const Format::BufferFormatInfo& info,
                                  uint32_t component, uint32_t raw);

uint32_t PackFormatComponent(EmitterState& state, const Format::BufferFormatInfo& info,
                             uint32_t component, uint32_t raw);

void EmitDeviceAtomicMemoryBarrier(EmitterState& state);

uint32_t EmitFloatAtomicReplacement(EmitterState& state, uint32_t old, uint32_t source,
                                    bool max_value);

uint32_t EmitDsFloatAtomicReplacement(EmitterState& state, uint32_t old, uint32_t source,
                                      bool max_value);

uint32_t EmitDsSwizzleTargetLane(EmitterState& state, uint32_t subid, uint32_t control);

inline constexpr auto EmitSelectValueU32 =
    EmitNative<spv::OpSelect, IR::Type::U32, uint32_t, uint32_t, uint32_t>;

uint32_t EmitAndConstant(EmitterState& state, uint32_t value, uint32_t mask);

// KYTY_FLOAT_STORE_CLAMP: the bits of a 32-bit float (u32) bound for a 16-bit float format, the
// guest GPU's way: a finite value beyond the half range becomes the largest half of its sign
// (65504); infinity and NaN pass unchanged.
uint32_t EmitClampToHalfRangeBits(EmitterState& state, uint32_t f32_bits);

uint32_t EmitShiftRightConstant(EmitterState& state, uint32_t value, uint32_t shift);

inline constexpr auto EmitOrU32 = EmitNative<spv::OpBitwiseOr, IR::Type::U32, uint32_t, uint32_t>;

uint32_t EmitCompareU32Constant(EmitterState& state, spv::Op opcode, uint32_t value,
                                uint32_t constant);

uint32_t EmitSubConstantMinusU32(EmitterState& state, uint32_t constant, uint32_t value);

uint32_t EmitF32ToF16RtzBits(EmitterState& state, uint32_t f32);

uint32_t EmitMinMaxU32Value(EmitterState& state, uint32_t lhs, uint32_t rhs, bool max_value);

uint32_t EmitMinMaxI32Value(EmitterState& state, uint32_t lhs, uint32_t rhs, bool max_value);

inline constexpr auto EmitBitcastF32ToU32 = EmitNative<spv::OpBitcast, IR::Type::U32, uint32_t>;

inline constexpr auto EmitBitcastU32ToF32 = EmitNative<spv::OpBitcast, IR::Type::F32, uint32_t>;

inline constexpr auto EmitAndU32 = EmitNative<spv::OpBitwiseAnd, IR::Type::U32, uint32_t, uint32_t>;

inline constexpr auto EmitLogicalAndBool =
    EmitNative<spv::OpLogicalAnd, IR::Type::U1, uint32_t, uint32_t>;

inline constexpr auto EmitLogicalOrBool =
    EmitNative<spv::OpLogicalOr, IR::Type::U1, uint32_t, uint32_t>;

inline constexpr auto EmitLogicalNotBool = EmitNative<spv::OpLogicalNot, IR::Type::U1, uint32_t>;

// Whether float arithmetic computing this instruction must not be contracted (MadMode).
bool NoContraction(EmitterState& state, const IR::Inst& inst);

F32Class EmitClassifyF32Bits(EmitterState& state, uint32_t bits);

F32Class EmitClassifyF32(EmitterState& state, uint32_t value);

uint32_t EmitClassMaskF32(EmitterState& state, uint32_t value, uint32_t mask);
uint32_t EmitClassMaskF16(EmitterState& state, uint32_t bits, uint32_t mask);

uint32_t EmitMinMaxF32Value(EmitterState& state, uint32_t lhs, uint32_t rhs, bool max_value);

inline constexpr auto EmitTruncF32Value = EmitGlsl<GLSLstd450Trunc, IR::Type::F32, uint32_t>;

uint32_t EmitFlushF32DenormToSignedZero(EmitterState& state, uint32_t value);
// KYTY_PS_FLUSH / KYTY_CS_FLUSH / KYTY_VS_FLUSH=off: this program's stage skips the flushes
// (EmitFlushF32DenormToSignedZero then returns its operand).
bool     DenormFlushSkipped(const EmitterState& state);
// KYTY_F32_FLUSH_HOST in a program without f64: the host's default f32 mode flushes denormal
// inputs of compares and transcendentals, for the op classes (F32FlushClass*) it is on for.
bool     F32FlushHost(const EmitterState& state, uint8_t op_class);
// KYTY_NATIVE_F16: two f16 lanes packed in a u32, one f16vec2 operation.
enum class HalfPkOp : uint32_t { Add, Mul, Min, Max, Fma };
uint32_t EmitHalfPk(EmitterState& state, HalfPkOp op, uint32_t a, uint32_t b, uint32_t c = 0);
// KYTY_LDS_SLOTS: the element offset k of a dword LDS access whose dword index is LaneId + k
// (address (LaneId << 2) + c, or LaneId * 4 + c, plus the DS offset; k = c >> 2), or none.
std::optional<uint32_t> StaticLdsSlot(const IR::Program& program, const IR::Inst& inst);
uint64_t DenormFlushCounter(uint32_t stage_index);

uint32_t EmitTrigCycleF32(EmitterState& state, uint32_t src, bool preserve_signed_zero);

inline constexpr auto EmitFNegateValue = EmitNative<spv::OpFNegate, IR::Type::F32, uint32_t>;

inline constexpr auto EmitFAbsValue = EmitGlsl<GLSLstd450FAbs, IR::Type::F32, uint32_t>;

uint32_t EmitF16BitsToF32(EmitterState& state, uint32_t bits);
uint32_t EmitF32BitsToF16RtzBits(EmitterState& state, uint32_t bits);

void EmitProgram(EmitterState& state);

void LaneShareClassify(EmitterState& state);
// Defines `inst` in the high half with the low half's id when the classification shares it.
bool LaneShareReuse(ValueEmitContext& low, ValueEmitContext& high, const IR::Inst& inst);
void LaneShareReport(EmitterState& state);

void DefineGetBdaPointer(EmitterState& state);

// 64-bit (OpTypeInt 64) guest/device address constant.
uint32_t EmitDeviceAddressConstant(EmitterState& state, uint64_t value);
// Guest address (64-bit int) -> device address of the same byte through the BDA page table, or 0
// (and a recorded fault) when the page is not mapped. Requires program.info.uses_dma.
uint32_t EmitBdaDeviceAddress(ValueEmitContext& ctx, uint32_t guest_address);

// These templates accept local lambdas from several emitter translation units.
template <typename Fn>
auto EmitIndexSwitch(EmitterState& state, uint32_t index, uint32_t count,
                        uint32_t result_type, Fn&& emit) {
	constexpr bool has_result = !std::is_void_v<std::invoke_result_t<Fn, uint32_t>>;
	EXIT_IF(count == 0u);
	const auto merge_label = state.builder.AllocateId();
	std::vector<uint32_t> labels(count);
	std::vector<uint32_t> words {spv::OpSwitch, index, merge_label};
	for (uint32_t item = 0; item < count; item++) {
		labels[item] = state.builder.AllocateId();
		words.push_back(item);
		words.push_back(labels[item]);
	}
	if constexpr (has_result) {
		words[2] = labels.front();
	}
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(words);
	std::vector<uint32_t> phi;
	if constexpr (has_result) {
		phi = {spv::OpPhi, result_type, state.builder.AllocateId()};
	}
	for (uint32_t item = 0; item < count; item++) {
		EmitLabel(state, labels[item]);
		if constexpr (has_result) {
			phi.push_back(emit(item));
			phi.push_back(state.current_label);
		} else {
			emit(item);
		}
		state.builder.AddFunction(spv::OpBranch, merge_label);
	}
	EmitLabel(state, merge_label);
	if constexpr (has_result) {
		state.builder.AddFunction(phi);
		return phi[2];
	}
}

template <typename Fn>
void EmitIfCondition(EmitterState& state, uint32_t condition, Fn&& fn) {
	if (condition == ConstantBool(state, true)) {
		fn();
		return;
	}
	const auto then_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, condition, then_label, merge_label);
	EmitLabel(state, then_label);
	fn();
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
}

template <typename Fn>
uint32_t EmitValueOrDefaultIfCondition(EmitterState& state, uint32_t condition, uint32_t type,
                                       uint32_t default_value, Fn&& fn) {
	if (condition == ConstantBool(state, true)) return fn();
	const auto then_label  = state.builder.AllocateId();
	const auto header     = state.current_label;
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, condition, then_label, merge_label);
	EmitLabel(state, then_label);
	const auto then_value = fn();
	const auto then_exit  = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, type, value, then_value, then_exit, default_value,
	                          header);
	return value;
}

template <typename Fn>
uint32_t EmitValueOrZeroIfCondition(EmitterState& state, uint32_t condition, Fn&& fn) {
	return EmitValueOrDefaultIfCondition(state, condition, TypeU32(state), ConstantU32(state, 0),
	                                     std::forward<Fn>(fn));
}

// DS/BUFFER/IMAGE_ATOMIC_INC and _DEC replacement values.
inline uint32_t AtomicIncrement(EmitterState& state, uint32_t old, uint32_t limit) {
	// old >= limit ? 0 : old + 1 (unsigned).
	const auto wrap = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), old, limit);
	const auto next = Binary(state, spv::OpIAdd, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, ConstantU32(state, 0), next);
}

inline uint32_t AtomicDecrement(EmitterState& state, uint32_t old, uint32_t limit) {
	// old == 0 || old > limit ? limit : old - 1 (unsigned).
	const auto zero  = Binary(state, spv::OpIEqual, TypeBool(state), old, ConstantU32(state, 0));
	const auto above = Binary(state, spv::OpUGreaterThan, TypeBool(state), old, limit);
	const auto wrap  = Binary(state, spv::OpLogicalOr, TypeBool(state), zero, above);
	const auto next  = Binary(state, spv::OpISub, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, limit, next);
}

template <typename Fn>
uint32_t AtomicUpdateLoop(EmitterState& state, uint32_t pointer, IR::ResourceKind kind, Fn&& desired) {
	const auto scope  = kind == IR::ResourceKind::Lds ? spv::ScopeWorkgroup : spv::ScopeDevice;
	const auto memory = [&] {
		switch (kind) {
			case IR::ResourceKind::Lds: return LdsMemorySemantics(state);
			case IR::ResourceKind::Image: return spv::MemorySemanticsImageMemoryMask;
			default: return spv::MemorySemanticsUniformMemoryMask;
		}
	}();
	const auto preheader = state.builder.AllocateId();
	const auto header    = state.builder.AllocateId();
	const auto cont      = state.builder.AllocateId();
	const auto merge     = state.builder.AllocateId();
	const auto initial   = state.builder.AllocateId();
	const auto observed  = state.builder.AllocateId();
	const auto exchanged = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBranch, preheader);
	EmitLabel(state, preheader);
	state.builder.AddFunction(spv::OpAtomicLoad, TypeU32(state), initial, pointer,
	                          ConstantU32(state, scope),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone));
	state.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(state, header);
	state.builder.AddFunction(spv::OpPhi, TypeU32(state), observed, initial, preheader, exchanged,
	                          cont);
	const auto next = desired(observed);
	state.builder.AddFunction(spv::OpAtomicCompareExchange, TypeU32(state), exchanged, pointer,
	                          ConstantU32(state, scope),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone), next, observed);
	const auto success = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), success, exchanged, observed);
	state.builder.AddFunction(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, success, merge, cont);
	EmitLabel(state, cont);
	state.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(state, merge);
	state.builder.AddFunction(spv::OpMemoryBarrier, ConstantU32(state, scope),
	                          ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | memory));
	return observed;
}

// KYTY_PS_HELPER_ATOMICS_SKIP (CodegenOptions::ps_helper_atomics_skip): whether a pixel shader's CAS
// loops leave out helper invocations (gl_HelperInvocation declared, SpirvRequirements).
[[nodiscard]] bool PixelHelperAtomicsSkipped(const EmitterState& state);

// A read-modify-write of one dword as a compare-exchange loop (sub-dword stores, float min/max,
// INC/DEC). Vulkan does not perform a helper invocation's atomics and leaves their results
// undefined, so in a helper lane the loop's exit test (exchanged == observed) depends on whatever the
// driver returns and need never become true: the wave then spins until a device reset (an AMD-only
// hang where helper lanes stay in the wave for derivatives). Helper lanes skip the loop and read
// zero, which is what their discarded atomic may return anyway.
template <typename Fn>
uint32_t AtomicUpdate(EmitterState& state, uint32_t pointer, IR::ResourceKind kind, Fn&& desired) {
	if (PixelHelperAtomicsSkipped(state)) {
		const auto live = Unary(state, spv::OpLogicalNot, TypeBool(state), EmitIsHelperInvocation(state));
		return EmitValueOrZeroIfCondition(state, live, [&]() {
			return AtomicUpdateLoop(state, pointer, kind, std::forward<Fn>(desired));
		});
	}
	return AtomicUpdateLoop(state, pointer, kind, std::forward<Fn>(desired));
}
} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_INTERNAL_H_ */
