#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

enum HostFloatControlBits : uint32_t {
	DenormFlushF32    = 1u << 0u,
	DenormPreserveF16 = 1u << 1u,
	DenormPreserveF64 = 1u << 2u,
};

std::atomic_uint32_t g_host_float_controls {0};
std::atomic_bool     g_storage_dword_loads_return_zero {false};
std::atomic_bool     g_image_min_lod {false};
std::atomic_uint8_t  g_compute_derivatives {static_cast<uint8_t>(HostComputeDerivatives::Khr)};
std::atomic_uint8_t  g_shader_clock_scope {0};
std::atomic_int32_t  g_shader_clock_shift {0};
std::atomic_bool     g_guard_vertex_outputs {false};
std::atomic_bool     g_guard_mesh_outputs {false};
std::atomic_uint8_t  g_wave_clusters {0}; // HostWaveClusters: bit 0 vertex, 1 pixel, 2 mesh

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

void ValidateNativeProgram(const IR::Program& program, bool lds_storage) {
	using Kind                                             = IR::DescriptorBindingKind;
	constexpr auto                               KindCount = static_cast<size_t>(Kind::Count);
	std::array<std::vector<uint32_t>, KindCount> expected;
	std::array<bool, KindCount>                  present {};
	const auto                                   Dense = [](size_t size) {
		std::vector<uint32_t> values(size);
		for (uint32_t i = 0; i < values.size(); i++) {
			values[i] = i;
		}
		return values;
	};
	auto Expect = [&](Kind kind, std::vector<uint32_t> resources = {}) {
		const auto index = static_cast<size_t>(kind);
		present[index]   = true;
		expected[index]  = std::move(resources);
	};
	if (!program.info.buffers.empty()) {
		Expect(Kind::Buffers, Dense(program.info.buffers.size()));
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto kind = IR::DescriptorBindingForImage(program.info.images[i]);
		if (!kind.has_value()) {
			Fail(program, "native shader plan has an invalid image class");
		}
		present[static_cast<size_t>(*kind)] = true;
		const auto dynamic = program.info.images[i].mip_mode == IR::ImageMipMode::Dynamic;
		const auto count   = dynamic ? program.info.images[i].mip_count : 1u;
		if (count == 0u || (!dynamic && program.info.images[i].mip_count != 1u)) {
			Fail(program, "native shader plan has an invalid image mip descriptor count");
		}
		expected[static_cast<size_t>(*kind)].insert(expected[static_cast<size_t>(*kind)].end(),
		                                            count, i);
	}
	if (!program.info.samplers.empty()) {
		Expect(Kind::Samplers, Dense(program.info.samplers.size()));
	}
	// AllocateBindings' rule: CollectSharedMemory also rejects invalid shared-memory metadata.
	const auto shared = IR::CollectSharedMemory(program);
	if (shared.gds || LoopGuardApplies(program.shader_hash) || IR::UsesBvhNodeCount(program)) {
		Expect(Kind::Gds);
	}
	if (shared.lds && lds_storage) {
		Expect(Kind::SharedMemory);
	}
	if (program.info.uses_dma) {
		Expect(Kind::BdaPagetable);
		Expect(Kind::FaultBuffer);
	}
	const bool uses_flattened_runtime =
	    !program.srt_reads.empty() ||
	     std::ranges::any_of(program.info.images, [](const IR::ImageResource& image) {
		     return image.indirect_search_iterations != 0u;
	     });
	if (uses_flattened_runtime) {
		Expect(Kind::FlattenedSrt);
	}
	if (program.bindings.ShaderDataDwords() != 0 && !program.bindings.UsesPushData()) {
		Expect(Kind::ShaderData);
	}
	const bool mip_stats = IR::UsesMipStats(program);
	if (mip_stats) {
		Expect(Kind::MipStats);
	}

	std::array<bool, KindCount> seen {};
	for (const auto& binding: program.bindings.descriptors) {
		const auto kind = static_cast<size_t>(binding.kind);
		if (kind >= KindCount || seen[kind] || !present[kind] ||
		    binding.resources != expected[kind]) {
			Fail(program, "native descriptor groups do not match shader topology");
		}
		seen[kind] = true;
	}
	for (size_t i = 0; i < KindCount; i++) {
		if (present[i] != seen[i]) {
			Fail(program, "native shader plan is missing a required descriptor group");
		}
	}
	const auto has_shader_data_storage = present[static_cast<size_t>(Kind::ShaderData)];
	const auto shader_data_dwords = program.bindings.ShaderDataDwords();
	const auto user_data_dwords = program.bindings.user_data_registers.size();
	const bool has_dispatch_threads = program.bindings.dispatch_thread_dword != IR::PushData::NoStart;
	if ((program.bindings.UsesPushData() &&
	     !IR::PushData::CanFit(program.bindings.push_data_start_dword, shader_data_dwords)) ||
	    (has_dispatch_threads && (program.stage != ShaderType::Compute ||
	                              program.bindings.dispatch_thread_dword != user_data_dwords)) ||
	    program.bindings.memory_offset_dword != user_data_dwords + (has_dispatch_threads ? 3u : 0u) ||
	    program.bindings.memory_offset_count != program.info.buffers.size() ||
	    program.bindings.mip_stats_count != (mip_stats ? program.info.images.size() : 0u) ||
	    has_shader_data_storage != (shader_data_dwords != 0 && !program.bindings.UsesPushData()) ||
	    !std::is_sorted(program.bindings.user_data_registers.begin(),
	                    program.bindings.user_data_registers.end()) ||
	    std::adjacent_find(program.bindings.user_data_registers.begin(),
	                       program.bindings.user_data_registers.end()) !=
	        program.bindings.user_data_registers.end()) {
		Fail(program, "native shader-data layout is inconsistent");
	}

	const auto planning_only_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       const auto op = use.user->GetOpcode();
			       if (op != IR::ValueOpcode::LoadAddressU32 &&
			           op != IR::ValueOpcode::ReadConstBuffer) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].planning_only;
		       });
	};
	const auto indirect_buffer_handle = [&](const IR::Inst& handle) {
		return program.info.uses_dma && handle.NumArgs() == 4u && !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       const auto access = IR::BufferAccessOf(use.user->GetOpcode());
			       // KYTY_BDA_WRITES: stores and atomics through the V# as well.
			       if (access != IR::BufferAccess::Read &&
			           !(program.info.bda_writes && access != IR::BufferAccess::None)) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].kind == IR::ResourceKind::IndirectBuffer;
		       });
	};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto dense = inst.Flags<uint32_t>();
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::GetBufferResource:
					if (planning_only_handle(inst) || indirect_buffer_handle(inst)) {
						break;
					}
					if (dense >= program.info.buffers.size()) {
						Fail(program, "typed buffer handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetAddressResource:
					if (planning_only_handle(inst)) {
						break;
					}
					if (inst.NumArgs() != 2 || !program.info.uses_dma) {
						Fail(program, "typed address handle has invalid DMA metadata");
					}
					break;
				case IR::ValueOpcode::GetScratchResource:
					if (inst.NumArgs() != 0 || program.scratch_dwords == 0) {
						Fail(program, "typed scratch handle has invalid shader metadata");
					}
					break;
				case IR::ValueOpcode::GetImageResource:
					if (dense >= program.info.images.size()) {
						Fail(program, "typed image handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetSamplerResource:
					if (dense >= program.info.samplers.size()) {
						Fail(program, "typed sampler handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::ReadConst: {
					const auto slot = inst.Arg(1).Resolve();
					if (!slot.IsImmediate() || slot.GetType() != IR::Type::U32 ||
					    slot.U32() >= program.srt_reads.size()) {
						Fail(program, "flattened SRT read has an invalid dense slot");
					}
					break;
				}
				default: break;
			}
		}
	}
}

} // namespace

Emitter::SpirvRequirements Emitter::AnalyzeProgramRequirements(const IR::Program& program) {
	SpirvRequirements requirements {};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			requirements.float64 |= inst.GetType() == IR::Type::F64;
			if (IR::BufferAccessOf(inst.GetOpcode()) == IR::BufferAccess::Atomic &&
			    inst.GetType() == IR::Type::U64) {
				requirements.buffer_int64_atomics = true;
			}
			// KYTY_PS_HELPER_ATOMICS_SKIP: emulated read-modify-writes (AtomicUpdate) of pixel shaders
			// test gl_HelperInvocation.
			if (program.stage == ShaderType::Pixel && GetCodegenOptions().ps_helper_atomics_skip) {
				const auto buffer = IR::BufferAccessOf(inst.GetOpcode());
				requirements.helper_invocation |=
				    buffer == IR::BufferAccess::Write || buffer == IR::BufferAccess::Atomic ||
				    IR::ImageOpcodeInfoOf(inst.GetOpcode()).access == IR::ImageAccess::Atomic;
			}
			const auto address_access = IR::AddressOpcodeInfoOf(inst.GetOpcode()).access;
			if (address_access != IR::AddressAccess::None) {
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					Fail(program, "address operation has invalid memory metadata");
				}
				const auto& memory = program.memory_info[memory_index];
				if (memory.kind == IR::ResourceKind::Scratch) {
					if (program.scratch_dwords == 0) {
						Fail(program, "scratch operation has no per-thread storage");
					}
					requirements.function_scratch = true;
				} else if (address_access == IR::AddressAccess::Write) {
					Fail(program, "writable FLAT/GLOBAL addresses require GPU ownership tracking");
				} else {
					// Descriptor stores can alias coherent physical-address loads.
					requirements.coherent_buffers |= memory.coherent;
				}
			}
			if (IR::BufferAccessOf(inst.GetOpcode()) != IR::BufferAccess::None) {
				requirements.buffer_writes |=
				    IR::BufferAccessOf(inst.GetOpcode()) != IR::BufferAccess::Read;
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					Fail(program, "buffer operation has invalid memory metadata");
				}
				const auto& memory = program.memory_info[memory_index];
				if (memory.kind == IR::ResourceKind::IndirectBuffer &&
				    inst.GetOpcode() != IR::ValueOpcode::ReadConstBuffer) {
					requirements.subgroup_local_invocation_id = true;
				}
				if (memory.kind == IR::ResourceKind::Buffer) {
					requirements.coherent_buffers |= memory.coherent;
					if (memory.resource >= program.info.buffers.size()) {
						Fail(program, "buffer operation has invalid resource metadata");
					}
					const auto bits = StorageBufferElementBits(program, memory);
					requirements.buffer_u8 |= bits == 8u;
					requirements.buffer_u16 |= bits == 16u;
					if ((program.info.buffers[memory.resource].packed_stride & (1u << 20u)) != 0u) {
						if (program.stage != ShaderType::Compute) {
							Fail(program, "buffer ADD_TID is only valid for compute shaders");
						}
						requirements.subgroup_local_invocation_id = true;
					}
				}
			}
			const auto shared_access = IR::SharedAccessOf(inst.GetOpcode());
			if (shared_access != IR::SharedAccess::None) {
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index >= program.memory_info.size()) {
					Fail(program, "shared operation has invalid memory metadata");
				}
				const auto kind = program.memory_info[index].kind;
				if (kind != IR::ResourceKind::Lds && kind != IR::ResourceKind::Gds) {
					Fail(program, "shared operation has invalid resource kind");
				}
				if (shared_access == IR::SharedAccess::Atomic &&
				    IR::SharedComponentCount(inst.GetOpcode()) == 2u) {
					if (kind != IR::ResourceKind::Lds || program.stage != ShaderType::Compute) {
						Fail(program, "64-bit shared atomics require compute LDS");
					}
					requirements.shared_int64_atomics = true;
				}
				if (program.stage != ShaderType::Compute && program.stage != ShaderType::Mesh &&
				    kind == IR::ResourceKind::Lds) {
					requirements.function_lds = true;
				}
				if (shared_access == IR::SharedAccess::Append ||
				    shared_access == IR::SharedAccess::Consume) {
					requirements.subgroup_ballot              = true;
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
					requirements.helper_invocation |=
					    program.stage == ShaderType::Pixel &&
					    GetCodegenOptions().ps_append_live_election;
				}
			}
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::StoreCompletion: requirements.subgroup_barrier = true; break;
				case IR::ValueOpcode::Ballot:
				case IR::ValueOpcode::AnyLane: requirements.subgroup_ballot = true; break;
				case IR::ValueOpcode::IsHelperInvocation:
					if (program.stage != ShaderType::Pixel) {
						Fail(program, "helper-invocation query outside a pixel shader");
					}
					requirements.helper_invocation = true;
					break;
				case IR::ValueOpcode::DppMoveU32:
				case IR::ValueOpcode::ReadFirstLane:
				case IR::ValueOpcode::ReadLane: {
					requirements.subgroup_ballot  = true;
					requirements.subgroup_shuffle = true;
					if (inst.GetOpcode() == IR::ValueOpcode::DppMoveU32) {
						requirements.subgroup_local_invocation_id = true;
					}
					if (inst.GetOpcode() == IR::ValueOpcode::ReadLane &&
					    GetCodegenOptions().lane_reductions && !WaveClustersActive(program) &&
					    IR::MatchLaneReduction(inst, program.wave_size)) {
						requirements.subgroup_arithmetic          = true;
						requirements.subgroup_local_invocation_id = true;
					}
					break;
				}
				case IR::ValueOpcode::DppUpdateU32:
				case IR::ValueOpcode::WriteLane: {
					requirements.subgroup_ballot              = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::Permlane16U32: {
					requirements.subgroup_ballot              = true;
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::SwizzleU32:
				case IR::ValueOpcode::PermuteU32:
				case IR::ValueOpcode::BpermuteU32: {
					requirements.subgroup_ballot              = true;
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::LaneId:
					requirements.subgroup_local_invocation_id |=
					    program.stage != ShaderType::TessellationControl;
					break;
				case IR::ValueOpcode::ReadClockRealtime64:
					if (GetHostShaderClock().scope != HostClockScope::None) {
						requirements.shader_clock = true;
						// The read is made wave-uniform (OpGroupNonUniformBroadcastFirst).
						requirements.subgroup_ballot = true;
					}
					break;
				case IR::ValueOpcode::ImageQueryLod: requirements.compute_derivatives = true; break;
				case IR::ValueOpcode::ImageGatherRaw:
					requirements.image_gather_extended = true;
					break;
				case IR::ValueOpcode::SetAttribute: {
					const auto index = inst.Flags<IR::ExportFlags>().index;
					if (index >= program.export_info.size()) {
						Fail(program, "attribute export has invalid metadata");
					}
					if (program.stage == ShaderType::Pixel &&
					    program.export_info[index].vm) {
						requirements.pixel_valid_mask = true;
					}
					break;
				}
				default: break;
			}
		}
	}
	// KYTY_WAVE32_CLUSTERS: every ballot and shuffle selects the wave's own 32 lanes by the
	// invocation's subgroup lane.
	if (WaveClustersActive(program) &&
	    (requirements.subgroup_ballot || requirements.subgroup_shuffle)) {
		requirements.subgroup_local_invocation_id = true;
	}
	return requirements;
}

void SetHostFloatControls(const HostFloatControls& controls) {
	uint32_t bits = 0;
	bits |= controls.denorm_flush_f32 ? DenormFlushF32 : 0u;
	bits |= controls.denorm_preserve_f16 ? DenormPreserveF16 : 0u;
	bits |= controls.denorm_preserve_f64 ? DenormPreserveF64 : 0u;
	g_host_float_controls.store(bits, std::memory_order_relaxed);
}

HostFloatControls GetHostFloatControls() {
	const auto bits = g_host_float_controls.load(std::memory_order_relaxed);
	return {
	    .denorm_flush_f32    = (bits & DenormFlushF32) != 0u,
	    .denorm_preserve_f16 = (bits & DenormPreserveF16) != 0u,
	    .denorm_preserve_f64 = (bits & DenormPreserveF64) != 0u,
	};
}

void SetHostBufferRobustness(const HostBufferRobustness& robustness) {
	g_storage_dword_loads_return_zero.store(robustness.storage_dword_loads_return_zero,
	                                        std::memory_order_relaxed);
}

HostBufferRobustness GetHostBufferRobustness() {
	return {.storage_dword_loads_return_zero =
	            g_storage_dword_loads_return_zero.load(std::memory_order_relaxed)};
}

void SetHostImageFeatures(const HostImageFeatures& features) {
	g_image_min_lod.store(features.min_lod, std::memory_order_relaxed);
	g_compute_derivatives.store(static_cast<uint8_t>(features.compute_derivatives),
	                            std::memory_order_relaxed);
}

HostImageFeatures GetHostImageFeatures() {
	return {.min_lod             = g_image_min_lod.load(std::memory_order_relaxed),
	        .compute_derivatives = static_cast<HostComputeDerivatives>(
	            g_compute_derivatives.load(std::memory_order_relaxed))};
}

void SetHostGeometryGuard(const HostGeometryGuard& guard) {
	g_guard_vertex_outputs.store(guard.vertex_outputs, std::memory_order_relaxed);
	g_guard_mesh_outputs.store(guard.mesh_outputs, std::memory_order_relaxed);
}

HostGeometryGuard GetHostGeometryGuard() {
	return {.vertex_outputs = g_guard_vertex_outputs.load(std::memory_order_relaxed),
	        .mesh_outputs   = g_guard_mesh_outputs.load(std::memory_order_relaxed)};
}

void SetHostShaderClock(const HostShaderClock& clock) {
	g_shader_clock_scope.store(static_cast<uint8_t>(clock.scope), std::memory_order_relaxed);
	g_shader_clock_shift.store(std::clamp(clock.shift, -8, 8), std::memory_order_relaxed);
}

HostShaderClock GetHostShaderClock() {
	return {.scope = static_cast<HostClockScope>(g_shader_clock_scope.load(std::memory_order_relaxed)),
	        .shift = g_shader_clock_shift.load(std::memory_order_relaxed)};
}

void SetHostWaveClusters(const HostWaveClusters& clusters) {
	g_wave_clusters.store(static_cast<uint8_t>((clusters.vertex ? 1u : 0u) | (clusters.pixel ? 2u : 0u) |
	                                           (clusters.mesh ? 4u : 0u)),
	                      std::memory_order_relaxed);
}

HostWaveClusters GetHostWaveClusters() {
	const auto bits = g_wave_clusters.load(std::memory_order_relaxed);
	return {.vertex = (bits & 1u) != 0u, .pixel = (bits & 2u) != 0u, .mesh = (bits & 4u) != 0u};
}

bool WaveClustersActive(const IR::Program& program) {
	const auto mode = GetCodegenOptions().wave32_clusters;
	if (program.wave_size != 32u || mode == Wave32Clusters::Off) {
		return false;
	}
	const auto host  = GetHostWaveClusters();
	const bool force = mode == Wave32Clusters::Force;
	switch (program.stage) {
		case ShaderType::Vertex:
		case ShaderType::Local:
		case ShaderType::TessellationEvaluation: return force || host.vertex;
		case ShaderType::Pixel: return force || host.pixel;
		case ShaderType::Mesh: return force || host.mesh;
		// Compute pipelines require the wave size (DeviceCompat::ComputeSubgroupSize), so only a test
		// forces clusters there (it runs the wave32 compute cases through the cluster code); hull
		// lanes are invocation ids, not subgroup lanes.
		case ShaderType::Compute: return force;
		default: return false;
	}
}

int32_t RealtimeClockShift(double timestamp_period_ns) {
	const double     rate = timestamp_period_ns > 0.0 ? 1e9 / timestamp_period_ns : 1e9;
	constexpr double Low  = 100e6 / 1.5;
	int32_t          shift = 0;
	while (shift < 8 && rate / std::ldexp(1.0, shift + 1) >= Low) {
		shift++;
	}
	while (shift > -8 && rate * std::ldexp(1.0, -shift) < Low) {
		shift--;
	}
	return shift;
}

std::vector<uint32_t> EmitProgram(const IR::Program& program,
                                  ShaderStageInputInfo input_info, bool mip_stats_records) {
	using namespace Emitter;

	if (program.stage != ShaderType::Compute && program.stage != ShaderType::Vertex &&
	    program.stage != ShaderType::Pixel && program.stage != ShaderType::Mesh &&
	    program.stage != ShaderType::Local && program.stage != ShaderType::TessellationControl &&
	    program.stage != ShaderType::TessellationEvaluation) {
		Fail(program, "binary SPIR-V emitter received an unsupported shader stage");
	}
	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete || !program.binding_layout_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	ValidateNativeProgram(program, program.stage == ShaderType::Compute &&
	                                   input_info.compute != nullptr && input_info.compute->lds_storage);
	if (IR::ValidationEnabled()) {
		IR::ValidateProgram(program, true);
	}
	EmitterState state(program, input_info);
	state.mip_stats_records = mip_stats_records;
	state.lane_count = ShaderLanesPerInvocation(program.stage, program.wave_size, input_info);
	state.wave_clusters = state.lane_count == 1 && WaveClustersActive(program);
	DefineModule(state);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
