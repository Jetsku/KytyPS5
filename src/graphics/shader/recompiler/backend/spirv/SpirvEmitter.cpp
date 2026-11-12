#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"

#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

enum HostFloatControlBits : uint32_t {
	DenormFlushF32    = 1u << 0u,
	DenormPreserveF16 = 1u << 1u,
	DenormPreserveF64 = 1u << 2u,
};

std::atomic_uint32_t g_host_float_controls {0};
std::atomic_bool     g_storage_dword_loads_return_zero {false};
std::atomic_bool     g_uniform_dword_loads_return_zero {false};
std::atomic_bool     g_buffer_pair_atomics {false};
std::atomic_bool     g_image_min_lod {false};
std::atomic_uint8_t  g_compute_derivatives {static_cast<uint8_t>(HostComputeDerivatives::Khr)};
std::atomic_bool     g_fragment_interlock {false};
std::atomic_bool     g_compute_derivatives_linear {false};
std::atomic_uint8_t  g_shader_clock_scope {0};
std::atomic_int32_t  g_shader_clock_shift {0};
std::atomic_uint32_t g_shader_clock_divisor {1};
std::atomic_bool     g_guard_vertex_outputs {false};
std::atomic_bool     g_guard_mesh_outputs {false};
std::atomic_uint8_t  g_wave_clusters {0}; // HostWaveClusters: bit 0 vertex, 1 pixel, 2 mesh
// KYTY_LDS_SLOTS verdicts of programs with function-scope LDS: 0 slots, 1 array kept.
std::array<std::atomic<uint64_t>, 2> g_lds_slot_verdicts {};

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
	{
		auto uniform = IR::UniformBufferResources(program);
		std::vector<uint32_t> storage;
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			if (!std::ranges::binary_search(uniform, i)) {
				storage.push_back(i);
			}
		}
		if (!storage.empty()) {
			Expect(Kind::Buffers, std::move(storage));
		}
		if (!uniform.empty()) {
			Expect(Kind::UniformBuffers, std::move(uniform));
		}
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
		     return image.indirect_search_iterations != 0u || image.bindless;
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
	const bool has_call_state       = program.bindings.call_state_dword != IR::PushData::NoStart;
	const auto call_state_start     = user_data_dwords + (has_dispatch_threads ? 3u : 0u);
	if ((program.bindings.UsesPushData() &&
	     !IR::PushData::CanFit(program.bindings.push_data_start_dword, shader_data_dwords)) ||
	    (has_dispatch_threads && (program.stage != ShaderType::Compute ||
	                              program.bindings.dispatch_thread_dword != user_data_dwords)) ||
	    (has_call_state && (program.stage != ShaderType::Compute ||
	                        program.bindings.call_state_dword != call_state_start)) ||
	    program.bindings.memory_offset_dword !=
	        call_state_start + (has_call_state ? IR::BindingLayout::CallStateDwords : 0u) ||
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

// KYTY_WAVE32_CLUSTERS: one log line per wave32 graphics program that shares a wider host subgroup
// (AMD fragment and vertex waves are 64 wide), with its cross-lane operations, so a player's log
// names the shaders whose wave-level code runs on clusters (Playroom cloud blocks, AMD test 2).
// Once per graphics program whose buffer descriptors set ADD_TID (compute uses it routinely).
void NoteGraphicsAddTid(const IR::Program& program, uint32_t lane_count, bool wave_clusters) {
	uint32_t buffers = 0, swizzled = 0, stride = 0;
	for (const auto& buffer: program.info.buffers) {
		if ((buffer.packed_stride & (1u << 20u)) != 0u) {
			buffers++;
			swizzled += (buffer.packed_stride & (1u << 14u)) != 0u ? 1u : 0u;
			stride = std::max(stride, buffer.packed_stride & 0x3fffu);
		}
	}
	if (buffers == 0) {
		return;
	}
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> noted;
	{
		const std::lock_guard lock(mutex);
		if (!noted.insert(program.shader_hash ^ (static_cast<uint64_t>(program.stage) << 60u)).second) {
			return;
		}
	}
	char text[256];
	std::snprintf(text, sizeof(text),
	              "Kyty buffer ADD_TID: stage %u shader 0x%016" PRIx64
	              " wave%u, %u lane(s) per invocation%s: %u buffer(s), %u swizzled, max stride %u\n",
	              static_cast<unsigned>(program.stage), program.shader_hash, program.wave_size,
	              lane_count, wave_clusters ? ", wave32 clusters" : "", buffers, swizzled, stride);
	Log::WriteToConsoleAndLog(text);
}

void NoteClusteredProgram(const IR::Program& program) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> noted;
	{
		const std::lock_guard lock(mutex);
		if (!noted.insert(program.shader_hash ^ (static_cast<uint64_t>(program.stage) << 60u)).second) {
			return;
		}
	}
	uint32_t first_lane = 0, lane_rw = 0, ballot = 0, dpp = 0, swizzle = 0, permute = 0, lane_id = 0;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::ReadFirstLane: first_lane++; break;
				case IR::ValueOpcode::ReadLane:
				case IR::ValueOpcode::WriteLane: lane_rw++; break;
				case IR::ValueOpcode::Ballot:
				case IR::ValueOpcode::AnyLane: ballot++; break;
				case IR::ValueOpcode::DppMoveU32:
				case IR::ValueOpcode::DppUpdateU32: dpp++; break;
				case IR::ValueOpcode::SwizzleU32: swizzle++; break;
				case IR::ValueOpcode::PermuteU32:
				case IR::ValueOpcode::BpermuteU32:
				case IR::ValueOpcode::Permlane16U32: permute++; break;
				case IR::ValueOpcode::LaneId: lane_id++; break;
				default: break;
			}
		}
	}
	char text[320];
	std::snprintf(text, sizeof(text),
	              "Kyty wave32 clusters: %s shader 0x%016" PRIx64
	              " runs on a wider host subgroup (KYTY_WAVE32_CLUSTERS); cross-lane ops: "
	              "readfirstlane %u, readlane/writelane %u, ballot/any %u, dpp %u, swizzle %u, "
	              "permute %u, lane id %u\n",
	              program.stage == ShaderType::Pixel ? "pixel"
	              : program.stage == ShaderType::Mesh ? "mesh"
	                                                  : "vertex",
	              program.shader_hash, first_lane, lane_rw, ballot, dpp, swizzle, permute, lane_id);
	Log::WriteToConsoleAndLog(text);
}

} // namespace

uint64_t LdsSlotVerdicts(uint32_t verdict) {
	return verdict < g_lds_slot_verdicts.size()
	           ? g_lds_slot_verdicts[verdict].load(std::memory_order_relaxed)
	           : 0u;
}

Emitter::SpirvRequirements Emitter::AnalyzeProgramRequirements(const IR::Program& program) {
	SpirvRequirements requirements {};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			requirements.float64 |= inst.GetType() == IR::Type::F64;
			// KYTY_DCACHE_INV: an invalidation makes every buffer the program reads coherent.
			requirements.coherent_buffers |= inst.GetOpcode() == IR::ValueOpcode::DataCacheInvalidate;
			requirements.coherent_dcache |= inst.GetOpcode() == IR::ValueOpcode::DataCacheInvalidate;
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::HalfPkAdd:
				case IR::ValueOpcode::HalfPkMul:
				case IR::ValueOpcode::HalfPkMin:
				case IR::ValueOpcode::HalfPkMax:
				case IR::ValueOpcode::HalfPkFma: requirements.native_half = true; break;
				default: break;
			}
			{
				const auto image_access = IR::ImageOpcodeInfoOf(inst.GetOpcode()).access;
				requirements.image_writes |=
				    image_access == IR::ImageAccess::Write || image_access == IR::ImageAccess::Atomic;
			}
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
				// KYTY_BUFFER_PAIR_ATOMIC: a DWORDX2 pair through a bound V# reads or writes the u64
				// view of the buffer atomically.
				if (memory.kind == IR::ResourceKind::Buffer &&
				    GetCodegenOptions().buffer_pair_atomic && GetHostBufferPairAtomics() &&
				    BufferPairAtomicProgram(program.shader_hash) &&
				    memory.IsCoherentBufferPair(inst.GetOpcode())) {
					requirements.buffer_int64_atomics = true;
				}
				if (memory.kind == IR::ResourceKind::Buffer) {
					requirements.coherent_buffers |= memory.coherent;
					if (memory.resource >= program.info.buffers.size()) {
						Fail(program, "buffer operation has invalid resource metadata");
					}
					const auto bits = StorageBufferElementBits(program, memory);
					requirements.buffer_u8 |= bits == 8u;
					requirements.buffer_u16 |= bits == 16u;
					requirements.coherent_narrow |= memory.coherent && bits != 32u;
					// ADD_TID adds the lane's id within its wave to the index. BufferLane takes it from
					// EmitSubgroupLocalInvocationId, the lane id every cross-lane op of the stage uses
					// (wave64 halves, wave32 clusters). A wave64 vertex or pixel shader on a 32-wide
					// host subgroup runs each 32 lanes as a wave of their own, so its lane is 0-31,
					// the lane its LaneId (v_mbcnt) and shuffles see too. A hull shader's lane is its
					// control-point invocation id (EmitLaneId), not a lane of the guest wave that packs several
					// patches, so the guest address cannot be formed there.
					if ((program.info.buffers[memory.resource].packed_stride & (1u << 20u)) != 0u) {
						if (program.stage == ShaderType::TessellationControl) {
							Fail(program, "buffer ADD_TID in a hull shader has no guest wave lane");
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
					if (GetCodegenOptions().lds_slots && !requirements.lds_slots_blocked) {
						if (const auto key = StaticLdsSlot(program, inst); key.has_value()) {
							auto&      keys = requirements.lds_slot_keys;
							const auto at   = std::lower_bound(keys.begin(), keys.end(), *key);
							if (at == keys.end() || *at != *key) {
								keys.insert(at, *key);
							}
						} else {
							requirements.lds_slots_blocked = true;
							static const bool log = [] {
								const auto* value = std::getenv("KYTY_LDS_SLOTS_LOG");
								return value != nullptr && value[0] == '1';
							}();
							if (log) {
								// The first access that is not an own-lane dword: its opcode and
								// its address' defining opcode (two levels).
								const auto  address = inst.Arg(0).Resolve();
								const auto* def     = address.TryInstruction();
								std::string chain   = def == nullptr ? "immediate/arg" : "";
								for (int depth = 0; def != nullptr && depth < 3; depth++) {
									chain += std::string(depth ? " <- " : "") +
									         std::string(IR::ValueOpcodeName(def->GetOpcode()));
									const auto next = def->NumArgs() > 0 ? def->Arg(0).Resolve() : IR::Value {};
									def = next.IsEmpty() ? nullptr : next.TryInstruction();
								}
								std::printf("LDS slots: 0x%016" PRIx64 " blocked by %s at %s\n",
								            program.shader_hash,
								            std::string(IR::ValueOpcodeName(inst.GetOpcode())).c_str(),
								            chain.c_str());
							}
						}
					}
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
				case IR::ValueOpcode::ImageSampleRaw: {
					// KYTY_IMPLICIT_LOD_STAGES: an implicit-LOD sample of a compute shader takes its
					// LOD from lane-quad derivatives (ComputeImplicitLod decides per module).
					if (!GetCodegenOptions().implicit_lod_stages ||
					    program.stage != ShaderType::Compute) {
						break;
					}
					const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
					if (memory_index >= program.memory_info.size()) break;
					const auto flags = program.memory_info[memory_index].image_sample_flags;
					requirements.implicit_compute_lod |=
					    (flags & (Decoder::ImageSampleFlagDerivative | Decoder::ImageSampleFlagLod |
					              Decoder::ImageSampleFlagLevelZero)) == 0u;
					break;
				}
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
	g_uniform_dword_loads_return_zero.store(robustness.uniform_dword_loads_return_zero,
	                                        std::memory_order_relaxed);
	SetHostUniformBufferRobust(robustness.uniform_dword_loads_return_zero);
}

HostBufferRobustness GetHostBufferRobustness() {
	return {.storage_dword_loads_return_zero =
	            g_storage_dword_loads_return_zero.load(std::memory_order_relaxed),
	        .uniform_dword_loads_return_zero =
	            g_uniform_dword_loads_return_zero.load(std::memory_order_relaxed)};
}

bool SrtUniformEnabled() {
	return GetCodegenOptions().srt_uniform &&
	       g_uniform_dword_loads_return_zero.load(std::memory_order_relaxed);
}

void SetHostBufferPairAtomics(bool supported) {
	g_buffer_pair_atomics.store(supported, std::memory_order_relaxed);
}

bool GetHostBufferPairAtomics() {
	return g_buffer_pair_atomics.load(std::memory_order_relaxed);
}

static std::atomic<bool> g_mesh_layer_drop {false};
void SetHostMeshLayerDrop(bool drop) {
	g_mesh_layer_drop.store(drop, std::memory_order_relaxed);
}
bool GetHostMeshLayerDrop() {
	return g_mesh_layer_drop.load(std::memory_order_relaxed);
}

static std::atomic<bool> g_coherent_load_vendor_barrier {false};
void SetCoherentLoadVendorDefault(bool barrier) {
	g_coherent_load_vendor_barrier.store(barrier, std::memory_order_relaxed);
}

CoherentLoadMode GetCoherentLoadMode() {
	static const int forced = [] {
		const char* value = std::getenv("KYTY_COHERENT_LOAD_MODE");
		if (value == nullptr) return -1;
		const std::string_view mode(value);
		if (mode == "barrier") return static_cast<int>(CoherentLoadMode::Barrier);
		if (mode == "acquire") return static_cast<int>(CoherentLoadMode::Acquire);
		if (mode == "relaxed") return static_cast<int>(CoherentLoadMode::Relaxed);
		if (mode == "plain" || mode == "off") return static_cast<int>(CoherentLoadMode::Plain);
		return -1;
	}();
	if (forced >= 0) {
		return static_cast<CoherentLoadMode>(forced);
	}
	static const int acquire = [] {
		const char* value = std::getenv("KYTY_COHERENT_LOAD_ACQUIRE");
		if (value == nullptr) return 0;
		const std::string_view setting(value);
		return setting == "1" ? 1 : setting == "auto" ? 2 : 0;
	}();
	const bool barrier =
	    acquire == 1 || (acquire == 2 && g_coherent_load_vendor_barrier.load(std::memory_order_relaxed));
	return barrier ? CoherentLoadMode::Barrier : CoherentLoadMode::Plain;
}

const char* CoherentLoadModeName(CoherentLoadMode mode) {
	switch (mode) {
		case CoherentLoadMode::Plain: return "plain";
		case CoherentLoadMode::Barrier: return "barrier";
		case CoherentLoadMode::Acquire: return "acquire";
		case CoherentLoadMode::Relaxed: return "relaxed";
	}
	return "?";
}

static std::atomic<uint64_t> g_indirect_format_count_address {0};
void SetIndirectFormatCountAddress(uint64_t address) {
	g_indirect_format_count_address.store(address, std::memory_order_release);
}
uint64_t IndirectFormatCountAddress() {
	return g_indirect_format_count_address.load(std::memory_order_acquire);
}

bool Emitter::BufferPairAtomicProgram(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> list;
		const char*           text = std::getenv("KYTY_BUFFER_PAIR_ATOMIC_HASHES");
		while (text != nullptr && *text != '\0') {
			char*      end   = nullptr;
			const auto value = std::strtoull(text, &end, 16);
			if (end == text) break;
			list.push_back(value);
			if (*end != ',') break;
			text = end + 1;
		}
		return list;
	}();
	return hashes.empty() || std::find(hashes.begin(), hashes.end(), shader_hash) != hashes.end();
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

void SetHostFragmentInterlock(bool supported) {
	g_fragment_interlock.store(supported, std::memory_order_relaxed);
}

bool GetHostFragmentInterlock() {
	return g_fragment_interlock.load(std::memory_order_relaxed);
}

void SetHostComputeDerivativesLinear(bool supported) {
	g_compute_derivatives_linear.store(supported, std::memory_order_relaxed);
}

bool GetHostComputeDerivativesLinear() {
	return g_compute_derivatives_linear.load(std::memory_order_relaxed);
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
	g_shader_clock_divisor.store(std::clamp(clock.divisor, 1u, 1024u), std::memory_order_relaxed);
}

HostShaderClock GetHostShaderClock() {
	return {.scope = static_cast<HostClockScope>(g_shader_clock_scope.load(std::memory_order_relaxed)),
	        .shift   = g_shader_clock_shift.load(std::memory_order_relaxed),
	        .divisor = g_shader_clock_divisor.load(std::memory_order_relaxed)};
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

// KYTY_HUGE_HELPER_OUTLINE (CodegenOptions::huge_helper_outline): in a module of at least
// HugeHelperWords words that already calls DontInline helpers, each of `helpers` with at least
// HugeHelperCalls call sites gets the DontInline function control (the NVIDIA compiler spent 10-35 s
// inlining them at every site). Only the function control word changes.
void OutlineHugeModuleHelpers(std::vector<uint32_t>& words, std::span<const uint32_t> helpers) {
	constexpr size_t   HugeHelperWords = 100'000;
	constexpr uint32_t HugeHelperCalls = 16;
	if (words.size() < HugeHelperWords) {
		return;
	}
	std::array<uint32_t, 4> calls {};
	EXIT_IF(helpers.size() > calls.size());
	for (size_t i = 5; i < words.size();) {
		const uint32_t op    = words[i] & 0xffffu;
		const uint32_t count = words[i] >> 16u;
		if (count == 0u || i + count > words.size()) {
			return; // malformed: leave the module as it is
		}
		if (op == spv::OpFunctionCall && count >= 4u) {
			for (size_t h = 0; h < helpers.size(); h++) {
				calls[h] += helpers[h] != 0u && words[i + 3] == helpers[h] ? 1u : 0u;
			}
		}
		i += count;
	}
	for (size_t i = 5; i < words.size();) {
		const uint32_t op    = words[i] & 0xffffu;
		const uint32_t count = words[i] >> 16u;
		if (op == spv::OpFunction && count >= 5u) {
			for (size_t h = 0; h < helpers.size(); h++) {
				if (helpers[h] != 0u && words[i + 2] == helpers[h] && calls[h] >= HugeHelperCalls) {
					words[i + 3] = (words[i + 3] & ~static_cast<uint32_t>(spv::FunctionControlInlineMask)) |
					               spv::FunctionControlDontInlineMask;
				}
			}
		}
		i += count;
	}
}

HostShaderClock RealtimeClockScale(double timestamp_period_ns, bool exact) {
	HostShaderClock clock {};
	clock.shift = RealtimeClockShift(timestamp_period_ns);
	if (!exact || clock.shift <= 0 || timestamp_period_ns <= 0.0) {
		return clock;
	}
	const auto period_ps = static_cast<uint32_t>(std::lround(timestamp_period_ns * 1000.0));
	const auto divisor   = RealtimeClockDivisor(period_ps);
	if (std::has_single_bit(divisor)) {
		clock.shift = std::countr_zero(divisor);
	} else {
		clock.shift   = 0;
		clock.divisor = divisor;
	}
	return clock;
}

std::vector<uint32_t> EmitProgram(const IR::Program& program,
                                  ShaderStageInputInfo input_info, bool mip_stats_records) {
	using namespace Emitter;

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
	// KYTY_LANE_LOCAL: a proven lane-local wave64 program runs one guest lane per invocation.
	state.lane_count =
	    program.lane_local ? 1u : ShaderLanesPerInvocation(program.stage, program.wave_size, input_info);
	if (state.lane_count == 2 && GetCodegenOptions().wave64_warps &&
	    program.stage == ShaderType::Compute && !program.dispatcher_fallback &&
	    // BVH node counts (KYTY_RT_NODE_BUDGET/STATS) report per two-lane invocation.
	    !IR::UsesBvhNodeCount(program) &&
	    !state.requirements.compute_derivatives && !state.requirements.shared_int64_atomics &&
	    state.lds_storage_class == spv::StorageClassWorkgroup) {
		// KYTY_WAVE64_WARPS: the workgroup must be exactly the one wave (its barriers then
		// synchronize the wave, and every lane exists).
		const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
		if (workgroup != nullptr && std::max(workgroup->threads_num[0], 1u) *
		                                    std::max(workgroup->threads_num[1], 1u) *
		                                    std::max(workgroup->threads_num[2], 1u) ==
		                                64u) {
			state.split_wave = true;
			state.lane_count = 1;
			// A probe emission with every branch half-local finds the blocks that synchronize
			// the halves; branches that control none of them stay half-local.
			std::unordered_set<const IR::Block*> sync_blocks;
			{
				EmitterState probe(program, input_info);
				probe.mip_stats_records = mip_stats_records;
				probe.split_wave        = true;
				probe.lane_count        = 1;
				probe.wave_probe        = true;
				probe.wave_sync_blocks  = &sync_blocks;
				DefineModule(probe);
				EmitProgram(probe);
			}
			state.wave_local_branches = SplitWaveLocalBranches(program, sync_blocks);
			static const bool log = std::getenv("KYTY_WAVE64_WARPS_LOG") != nullptr;
			if (log) {
				size_t branches = 0;
				for (const auto& info: program.block_info) {
					branches += info.terminator.kind == CFG::TerminatorKind::ConditionalBranch ? 1u : 0u;
				}
				std::printf("SplitWave CS %016" PRIx64 ": %zu of %zu branches half-local, %zu sync blocks\n",
				            program.shader_hash, state.wave_local_branches.size(), branches,
				            sync_blocks.size());
			}
		}
	}
	// KYTY_LDS_SLOTS: one guest lane per invocation, every LDS access an own-lane dword (each key
	// k reaches element LaneId + k of this invocation's array, so distinct keys never alias), and
	// every lane's element in bounds (LaneId < wave_size), as the array's bounds check would allow.
	if (state.requirements.function_lds && GetCodegenOptions().lds_slots) {
		const auto& keys = state.requirements.lds_slot_keys; // sorted (AnalyzeProgramRequirements)
		const uint64_t lanes = std::max<uint32_t>(program.wave_size, 1u);
		state.lds_slot_plan  = state.lane_count == 1 && !state.requirements.lds_slots_blocked &&
		                      !keys.empty() && keys.size() <= 64u &&
		                      static_cast<uint64_t>(keys.back()) + lanes <= LdsDwordCount(state);
		g_lds_slot_verdicts[state.lds_slot_plan ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
		static const bool log = [] {
			const auto* value = std::getenv("KYTY_LDS_SLOTS_LOG");
			return value != nullptr && value[0] == '1';
		}();
		if (log) {
			std::printf("LDS slots: stage %u 0x%016" PRIx64 ": %s (%u-dword function array, %zu "
			            "offset(s)%s, lanes/invocation %u)\n",
			            static_cast<unsigned>(program.stage), program.shader_hash,
			            state.lds_slot_plan ? "slots in variables" : "array kept",
			            LdsDwordCount(state), keys.size(),
			            state.requirements.lds_slots_blocked ? ", an access is not an own-lane dword"
			                                                 : "",
			            state.lane_count);
		}
	}
	state.wave_clusters = state.lane_count == 1 && WaveClustersActive(program);
	if (state.wave_clusters && program.stage != ShaderType::Compute) {
		NoteClusteredProgram(program);
	}
	if (program.stage != ShaderType::Compute) {
		NoteGraphicsAddTid(program, state.lane_count, state.wave_clusters);
	}
	if (program.stage == ShaderType::Mesh) {
		state.mesh_passes = std::max(input_info.vertex->mesh.passes, 1u);
	}
	DefineModule(state);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	auto words = state.builder.Build();
	if (GetCodegenOptions().huge_helper_outline && !state.indirect_raw_functions.empty()) {
		const std::array<uint32_t, 2> helpers {state.indirect_format_compact_function,
		                                       state.bda_pointer_function};
		OutlineHugeModuleHelpers(words, helpers);
	}
	return words;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
