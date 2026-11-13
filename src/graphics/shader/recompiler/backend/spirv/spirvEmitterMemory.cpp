#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <bit>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t AndCondition(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return Binary(state, spv::OpLogicalAnd, TypeBool(state), lhs, rhs);
}

uint32_t EmitDsMaskedLaneRead(EmitterState& state, uint32_t source, uint32_t target,
                              uint32_t exec) {
	// These reads stay within 32-lane halves of the wave: a split wave's host subgroups.
	if (HalfWaveHost(state)) {
		target = Binary(state, spv::OpBitwiseAnd, TypeU32(state), target, ConstantU32(state, 31));
	}
	const auto host_target = EmitHostLane(state, target);
	const auto shuffled    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), shuffled,
	                          ConstantU32(state, spv::ScopeSubgroup), source, host_target);
	const auto source_exec = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeBool(state), source_exec,
	                          ConstantU32(state, spv::ScopeSubgroup), exec, host_target);
	const auto source_active =
	    AndCondition(state, source_exec, EmitSubgroupLaneActiveBool(state, target));
	return Select(state, TypeU32(state), source_active, shuffled, ConstantU32(state, 0));
}

struct BufferAddress {
	uint32_t offset;
	uint32_t byte;
};

BufferAddress CalculateBufferAddress(EmitterState& state, uint32_t index, uint32_t offset,
                                     uint32_t soffset, uint32_t immediate, uint32_t stride,
                                     uint32_t swizzle, uint32_t index_stride) {
	const auto zero = ConstantU32(state, 0);
	const auto one  = ConstantU32(state, 1);
	const auto add = [&](uint32_t lhs, uint32_t rhs) {
		return lhs == zero ? rhs : rhs == zero ? lhs
		                                      : Binary(state, spv::OpIAdd, TypeU32(state), lhs, rhs);
	};
	const auto mul = [&](uint32_t lhs, uint32_t rhs) {
		return lhs == zero || rhs == zero ? zero
		       : lhs == one              ? rhs
		       : rhs == one              ? lhs
		                                 : Binary(state, spv::OpIMul, TypeU32(state), lhs, rhs);
	};
	if (immediate != 0u) {
		offset = add(offset, ConstantU32(state, immediate));
	}
	auto address = add(mul(index, stride), offset);
	if (swizzle != 0u) {
		const auto index_shift =
		    Binary(state, spv::OpIAdd, TypeU32(state), index_stride, ConstantU32(state, 3));
		const auto indices = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		                            ConstantU32(state, 1), index_shift);
		const auto index_msb =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), index, index_shift);
		const auto index_lsb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), index,
		           Binary(state, spv::OpISub, TypeU32(state), indices, ConstantU32(state, 1)));
		const auto offset_msb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, ~3u));
		const auto offset_lsb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, 3u));
		const auto msb = mul(add(mul(index_msb, stride), offset_msb), indices);
		const auto lsb = add(Binary(state, spv::OpShiftLeftLogical, TypeU32(state), index_lsb,
		                            ConstantU32(state, 2u)), offset_lsb);
		address = Select(state, TypeU32(state), swizzle, add(msb, lsb), address);
	}
	return {offset, add(address, soffset)};
}

uint32_t BufferLane(EmitterState& state) {
	// ADD_TID: the guest lane. A lane-local wave64 program's subgroup is half a wave (KYTY_LANE_LOCAL):
	// the lane is the invocation's index in its group modulo 64.
	const auto id = state.lane_count == 1 && state.program.lane_local ? EmitLocalInvocationIndex(state)
	                                                                 : EmitSubgroupLocalInvocationId(state);
	return Binary(state, spv::OpBitwiseAnd, TypeU32(state), id, ConstantU32(state, 63));
}

uint32_t BufferByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto&      state  = ctx.state;
	const auto packed = StorageBufferPackedStride(state, mem);
	const auto stride = packed & 0x3fffu;
	auto       index  = ctx.Arg(inst, 1);
	if ((packed & (1u << 20u)) != 0u) {
		index = Binary(state, spv::OpIAdd, TypeU32(state), index, BufferLane(state));
	}
	const bool swizzle = stride != 0u && (packed & (1u << 14u)) != 0u;
	return CalculateBufferAddress(state, index, ctx.Arg(inst, 2), ctx.Arg(inst, 3), mem.offset,
	                              ConstantU32(state, stride),
	                              swizzle ? ConstantBool(state, true) : 0u,
	                              ConstantU32(state, (packed >> 16u) & 3u))
	    .byte;
}

uint32_t AddU64Low(EmitterState& state, uint32_t low, uint32_t high, uint32_t add_low,
                   uint32_t add_high, uint32_t& out_high) {
	const auto result = Binary(state, spv::OpIAdd, TypeU32(state), low, add_low);
	const auto carry  = Binary(state, spv::OpULessThan, TypeBool(state), result, low);
	out_high =
	    Binary(state, spv::OpIAdd, TypeU32(state),
	           Binary(state, spv::OpIAdd, TypeU32(state), high, add_high),
	           Select(state, TypeU32(state), carry, ConstantU32(state, 1), ConstantU32(state, 0)));
	return result;
}

uint32_t ScratchByteAddress(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t low,
                            uint32_t high) {
	auto& state     = ctx.state;
	auto  immediate = static_cast<int32_t>(mem.offset);
	const auto immediate_low  = ConstantU32(state, static_cast<uint32_t>(immediate));
	const auto immediate_high = ConstantU32(state, immediate < 0 ? UINT32_MAX : 0u);
	low                      = AddU64Low(state, low, high, immediate_low, immediate_high, high);
	const auto valid = Binary(state, spv::OpIEqual, TypeBool(state), high, ConstantU32(state, 0));
	return Select(state, TypeU32(state), valid, low, ConstantU32(state, UINT32_MAX));
}

uint32_t ConstantDeviceAddress(EmitterState& state, uint64_t value) {
	return ConstantU64(state, value);
}

uint32_t DeviceAddressFromWords(EmitterState& state, uint32_t low, uint32_t high) {
	return PackU64(state, low, high);
}

uint32_t GuestAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto& state = ctx.state;
	auto  low   = ctx.Arg(inst, 1);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, ~3u));
	}
	uint32_t address = 0;
	if (mem.address_is_full) {
		address = PackU64(state, low, ctx.Arg(inst, 2));
	} else {
		const auto* handle = inst.Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != IR::ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) {
			ctx.Fail(inst, "has no address base pair");
			return ConstantU64(state, 0);
		}
		auto base_low = ctx.Arg(*handle, 0);
		if (mem.kind == IR::ResourceKind::ScalarAddress) {
			base_low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), base_low,
			                  ConstantU32(state, ~3u));
		}
		const auto base = PackU64(state, base_low, ctx.Arg(*handle, 1));
		address         = Binary(state, spv::OpIAdd, TypeU64(state), base,
		                         Unary(state, spv::OpUConvert, TypeU64(state), low));
	}
	auto immediate = static_cast<int32_t>(mem.offset);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		immediate = static_cast<int32_t>(static_cast<uint32_t>(immediate) & ~3u);
	}
	return immediate == 0
	           ? address
	           : Binary(state, spv::OpIAdd, TypeU64(state), address,
	                    ConstantU64(state, static_cast<uint64_t>(static_cast<int64_t>(immediate))));
}

uint32_t FaultElementPointer(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.fault_buffer_variable, ConstantU32(state, 0), index);
	return pointer;
}

void RecordBdaFault(EmitterState& state, uint32_t page) {
	const auto word =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), page, ConstantU32(state, 5));
	const auto bit =
	    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), page, ConstantU32(state, 31)));
	// Atomic: lanes and waves faulting on pages that share a bitmap word must not lose bits
	// (a load/or/store sequence races and drops faults).
	const auto pointer = FaultElementPointer(state, word);
	const auto result  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), result, pointer,
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone), bit);
}

// KYTY_BDA_TAGS_GPU_COUNT: one count in word BDA_TAG_COUNTER_WORD + index of the fault buffer.
void CountBdaTagEvent(EmitterState& state, uint32_t index) {
	if (!GetCodegenOptions().bda_tags_gpu_count) {
		return;
	}
	const auto pointer = FaultElementPointer(
	    state, ConstantU32(state, static_cast<uint32_t>(BufferCache::BDA_TAG_COUNTER_WORD + index)));
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), result, pointer,
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone), ConstantU32(state, 1));
}

uint32_t GetBdaPointer(ValueEmitContext& ctx, uint32_t address) {
	auto&      state  = ctx.state;
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, TypeU64(state), result,
	                          state.bda_pointer_function, address);
	return result;
}

uint32_t LoadBdaDword(ValueEmitContext& ctx, uint32_t address, bool coherent = false) {
	auto&      state   = ctx.state;
	const auto bda     = GetBdaPointer(ctx, address);
	const auto present =
	    Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
	// A page that no buffer backs yet reads as zero, except in BVH traversal shaders: there it
	// reads as all ones, the absent-node link bvh_intersect returns, because guest list walks end
	// on negative links and would advance over zero entries forever.
	const auto missing = ConstantU32(state, state.program.info.uses_bvh ? ~0u : 0u);
	return EmitValueOrDefaultIfCondition(state, present, TypeU32(state), missing, [&]() {
		auto pointer = state.builder.AllocateId();
		if (coherent) {
			const auto block = state.builder.DecoratedType(
			    spv::OpTypeStruct, {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
			                        {spv::OpMemberDecorate, {0, spv::DecorationCoherent}},
			                        {spv::OpDecorate, {spv::DecorationBlock}}}, TypeU32(state));
			state.builder.AddFunction(
			    spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, block),
			    pointer, bda);
			const auto base = pointer;
			pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, TypePhysicalU32Pointer(state), pointer,
			                          base, ConstantU32(state, 0));
		} else {
			state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
			                          bda);
		}
		const auto         value     = state.builder.AllocateId();
		constexpr uint32_t alignment = sizeof(uint32_t);
		// KYTY_VOLATILE_LOADS: FLAT/GLOBAL and indirect-V# reads too; DLC/GLC (coherent) loads always.
		const auto access = GetCodegenOptions().volatile_loads || coherent
		                        ? spv::MemoryAccessAlignedMask | spv::MemoryAccessVolatileMask
		                        : spv::MemoryAccessAlignedMask;
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer, access, alignment);
		return value;
	});
}

uint32_t LoadBda(ValueEmitContext& ctx, uint32_t address, uint32_t active, uint32_t bits,
                 bool coherent = false) {
	auto& state = ctx.state;
	return EmitValueOrZeroIfCondition(state, active, [&]() {
		const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), address,
		                            ConstantU64(state, ~uint64_t {3}));
		const auto first   = LoadBdaDword(ctx, aligned, coherent);
		const auto byte =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		           Unary(state, spv::OpUConvert, TypeU32(state), address), ConstantU32(state, 3));
		const auto crosses =
		    bits == 8u ? ConstantBool(state, false)
		               : Binary(state, bits == 16u ? spv::OpUGreaterThan : spv::OpINotEqual,
		                        TypeBool(state), byte, ConstantU32(state, bits == 16u ? 2u : 0u));
		const auto second = EmitValueOrZeroIfCondition(state, crosses, [&]() {
			return LoadBdaDword(ctx, Binary(state, spv::OpIAdd, TypeU64(state), aligned,
			                                ConstantU64(state, sizeof(uint32_t))), coherent);
		});
		const auto shift =
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), byte, ConstantU32(state, 3));
		const auto upper_shift =
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		           Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                  Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 4), byte),
		                  ConstantU32(state, 3)),
		           ConstantU32(state, 3));
		const auto merged =
		    Binary(state, spv::OpBitwiseOr, TypeU32(state),
		           Binary(state, spv::OpShiftRightLogical, TypeU32(state), first, shift),
		           Binary(state, spv::OpShiftLeftLogical, TypeU32(state), second, upper_shift));
		return bits == 32u ? merged
		                   : Binary(state, spv::OpBitwiseAnd, TypeU32(state), merged,
		                            ConstantU32(state, bits == 8u ? 0xffu : 0xffffu));
	});
}

uint32_t ByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	if (mem.kind == IR::ResourceKind::Buffer) {
		(void)ResourceForDescriptor(ctx.state, IR::DescriptorBindingKind::Buffers, mem.resource);
		// Shader-data byte offsets are per tracked buffer (memory_offset_count == buffers).
		const auto prefix = ctx.state.memory_byte_offsets[mem.resource];
		const auto address = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state),
		                            BufferByteAddress(ctx, inst, mem), prefix);
		// The host binding prefix must not wrap an out-of-range guest address into the buffer.
		auto in_range =
		    Binary(ctx.state, spv::OpUGreaterThanEqual, TypeBool(ctx.state), address, prefix);
		// KYTY_BUFFER_OOB_MODES: OOB_SELECT 0 with a stride: an offset (VGPR offset + instruction
		// offset, per dword of a DWORDX2-4) at or past the stride is out of range (Table 35), not
		// the next record.
		if (const auto packed = StorageBufferPackedStride(ctx.state, mem);
		    (packed & IR::BufferOobOffsetCheck) != 0u && !mem.typed) {
			auto offset = ctx.Arg(inst, 2);
			if (mem.offset != 0u) {
				offset = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), offset,
				                ConstantU32(ctx.state, mem.offset));
			}
			in_range = AndCondition(ctx.state, in_range,
			                        Binary(ctx.state, spv::OpULessThan, TypeBool(ctx.state), offset,
			                               ConstantU32(ctx.state, packed & 0x3fffu)));
		}
		return Select(ctx.state, TypeU32(ctx.state), in_range, address,
		              ConstantU32(ctx.state, UINT32_MAX));
	}
	if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
		if (mem.offset == 0u) {
			return ctx.Arg(inst, 0);
		}
		return Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), ctx.Arg(inst, 0),
		              ConstantU32(ctx.state, mem.offset));
	}
	if (mem.kind != IR::ResourceKind::Scratch) {
		EXIT("physical address memory must use the BDA emitter\n");
	}
	return ScratchByteAddress(ctx, mem, ctx.Arg(inst, 1), ctx.Arg(inst, 2));
}

uint32_t DwordIndex(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	// DS reads and writes keep the full region address (out-of-range words fail the bounds check);
	// only DS atomics truncate it (EmitAtomicAccess).
	return Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state),
	              ByteAddress(ctx, inst, mem), ConstantU32(ctx.state, 2));
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index);

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits);

// Plain dword loads from storage-buffer descriptors can rely on the device's bounds check: with
// robustBufferAccess2 and a 1-byte robustness alignment (HostBufferRobustness) a dword load with any
// byte past the descriptor range returns 0, which is exactly "index < OpArrayLength" failing (the
// range is "size + adjustment" bytes, OpArrayLength its whole dwords). Every guest OOB_SELECT mode
// is modelled today by that same descriptor-range check, so nothing changes for any mode. LDS,
// GDS, scratch, formatted (all-or-nothing) accesses, stores and atomics keep their explicit checks.
// KYTY_ROBUST_BUFFER_LOADS=0 keeps the explicit check everywhere.
bool DeviceChecksDwordLoad(const MemoryResourceAccess& resource) {
	return (resource.kind == IR::ResourceKind::Buffer ||
	        resource.kind == IR::ResourceKind::ScalarBuffer) &&
	       GetCodegenOptions().robust_buffer_loads &&
	       GetHostBufferRobustness().storage_dword_loads_return_zero;
}

uint32_t LoadWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource) {
	const auto index = DwordIndex(ctx, inst, mem);
	if (DeviceChecksDwordLoad(resource)) {
		return LoadWordInBounds(ctx, resource, index);
	}
	return EmitValueOrZeroIfCondition(
	    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index),
	    [&]() { return LoadWordInBounds(ctx, resource, index); });
}

// KYTY_BRANCHLESS_LOADS (CodegenOptions::branchless_loads): how a raw load of mem may run in every
// lane. Device: a storage buffer the device bounds-checks (the address stays, an inactive lane's
// out-of-range address reads zero). SafeIndex: workgroup or invocation LDS and scratch, where a lane
// that EXEC or the bounds rule out reads element 0 instead. None: keep the branches.
enum class BranchlessLoad : uint8_t { None, Device, SafeIndex };

BranchlessLoad BranchlessLoadKind(EmitterState& state, const IR::MemoryInfo& mem) {
	if (!GetCodegenOptions().branchless_loads) {
		return BranchlessLoad::None;
	}
	switch (mem.kind) {
		case IR::ResourceKind::Buffer:
		case IR::ResourceKind::ScalarBuffer:
			return GetCodegenOptions().robust_buffer_loads &&
			               GetHostBufferRobustness().storage_dword_loads_return_zero
			           ? BranchlessLoad::Device
			           : BranchlessLoad::None;
		case IR::ResourceKind::Lds:
			return GetCodegenOptions().branchless_lds &&
			               state.lds_storage_class != spv::StorageClassStorageBuffer &&
			               LdsDwordCount(state) != 0u
			           ? BranchlessLoad::SafeIndex
			           : BranchlessLoad::None;
		case IR::ResourceKind::Scratch:
			return GetCodegenOptions().branchless_lds && state.program.scratch_dwords != 0u
			           ? BranchlessLoad::SafeIndex
			           : BranchlessLoad::None;
		default: return BranchlessLoad::None;
	}
}

// The element index a SafeIndex load reads, and whether its lane may use the value (usable).
uint32_t SafeElementIndex(EmitterState& state, const MemoryResourceAccess& resource, uint32_t index,
                          uint32_t active, uint32_t& usable) {
	usable = AndCondition(state, active, EmitMemoryElementInBounds(state, resource, index));
	return Select(state, TypeU32(state), usable, index, ConstantU32(state, 0));
}

uint32_t ZeroUnless(EmitterState& state, uint32_t condition, uint32_t value) {
	return Select(state, TypeU32(state), condition, value, ConstantU32(state, 0));
}

// A raw dword load of mem without branches, or 0 when it keeps them.
uint32_t LoadWordBranchless(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                            uint32_t active) {
	auto&      state = ctx.state;
	const auto kind  = BranchlessLoadKind(state, mem);
	if (kind == BranchlessLoad::None) {
		return 0;
	}
	const auto resource = PrepareMemoryResourceAccess(state, mem);
	const auto index    = DwordIndex(ctx, inst, mem);
	if (kind == BranchlessLoad::Device) {
		return ZeroUnless(state, active, LoadWordInBounds(ctx, resource, index));
	}
	uint32_t   usable = 0;
	const auto safe   = SafeElementIndex(state, resource, index, active, usable);
	return ZeroUnless(state, usable, LoadWordInBounds(ctx, resource, safe));
}

uint32_t LoadWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	if (const auto value = LoadWordBranchless(ctx, inst, mem, ctx.Arg(inst, inst.NumArgs() - 1));
	    value != 0) {
		return value;
	}
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadWordPrepared(ctx, inst, mem, resource);
	});
}

uint32_t LoadSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                             const MemoryResourceAccess& resource, uint32_t bits) {
	const auto address = ByteAddress(ctx, inst, mem);
	const auto index = Binary(
	    ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), address,
	    ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	if (DeviceChecksDwordLoad(resource)) {
		return LoadSubwordInBounds(ctx, resource, address, index, bits);
	}
	return EmitValueOrZeroIfCondition(
	    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index),
	    [&]() { return LoadSubwordInBounds(ctx, resource, address, index, bits); });
}

// A coherent (GLC/DLC) load of a 32-bit word as GetCoherentLoadMode asks (SpirvEmitter.h).
static uint32_t EmitCoherentWordLoad(EmitterState& state, uint32_t pointer) {
	const auto mode  = GetCoherentLoadMode();
	const auto value = state.builder.AllocateId();
	if (mode == CoherentLoadMode::Acquire || mode == CoherentLoadMode::Relaxed) {
		const uint32_t semantics =
		    mode == CoherentLoadMode::Acquire
		        ? static_cast<uint32_t>(spv::MemorySemanticsAcquireMask |
		                                spv::MemorySemanticsUniformMemoryMask)
		        : static_cast<uint32_t>(spv::MemorySemanticsMaskNone);
		state.builder.AddFunction(spv::OpAtomicLoad, TypeU32(state), value, pointer,
		                          ConstantU32(state, spv::ScopeDevice), ConstantU32(state, semantics));
		return value;
	}
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
	                          spv::MemoryAccessVolatileMask);
	if (mode == CoherentLoadMode::Barrier) {
		EmitDeviceAtomicMemoryBarrier(state);
	}
	return value;
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index) {
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	if (resource.memory_access != spv::MemoryAccessMaskNone &&
	    GetCoherentLoadMode() != CoherentLoadMode::Plain) {
		return EmitCoherentWordLoad(ctx.state, pointer);
	}
	const auto value = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), value, pointer,
	                              resource.memory_access);
	return value;
}

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits) {
	uint32_t value;
	if (resource.element_bits < 32u) {
		const auto loaded = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpLoad, TypeStorageBufferElement(ctx.state, bits), loaded,
		                              EmitMemoryElementPointer(ctx.state, resource, index),
		                              resource.memory_access);
		value = Unary(ctx.state, spv::OpUConvert, TypeU32(ctx.state), loaded);
	} else {
		const auto word = LoadWordInBounds(ctx, resource, index);
		const auto byte = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
		                         ConstantU32(ctx.state, 3));
		const auto shift = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state), byte,
		                          ConstantU32(ctx.state, 3));
		value = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state),
		               Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), word, shift),
		               ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu));
	}
	return value;
}

uint32_t LoadSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem,
                     uint32_t bits) {
	// KYTY_BRANCHLESS_LOADS: as LoadWordBranchless.
	if (const auto kind = BranchlessLoadKind(ctx.state, mem); kind != BranchlessLoad::None) {
		auto&      state    = ctx.state;
		const auto active   = ctx.Arg(inst, inst.NumArgs() - 1);
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto address  = ByteAddress(ctx, inst, mem);
		const auto index =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
		           ConstantU32(state, std::countr_zero(resource.element_bits / 8u)));
		if (kind == BranchlessLoad::Device) {
			return ZeroUnless(state, active,
			                  LoadSubwordInBounds(ctx, resource, address, index, bits));
		}
		uint32_t   usable = 0;
		const auto safe   = SafeElementIndex(state, resource, index, active, usable);
		return ZeroUnless(state, usable,
		                  LoadSubwordInBounds(ctx, resource, address, safe, bits));
	}
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadSubwordPrepared(ctx, inst, mem, resource, bits);
	});
}

Prospero::BufferFormat BufferFormat(const ValueEmitContext& ctx, const IR::MemoryInfo& mem) {
	return mem.typed ? Format::DecodeTBufferFormat(mem.data_format, mem.number_format)
	                 : StorageBufferFormat(ctx.state, mem);
}

IR::MemoryInfo RebaseRawComponent(IR::MemoryInfo mem, uint32_t component) {
	mem.offset += component * 4u;
	mem.data_dwords     = 1u;
	mem.component_index = component;
	return mem;
}

using Format::FormattedSource;
using Format::FormattedSourceKind;

FormattedSource ResolveFormattedSource(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                       const Format::BufferFormatInfo& info,
                                       uint32_t                        output_component) {
	if (mem.typed) {
		return output_component < info.component_count
		           ? FormattedSource {FormattedSourceKind::Memory, output_component}
		           : FormattedSource {};
	}
	const auto swizzle  = ctx.state.program.info.buffers[mem.resource].descriptor_swizzle;
	const auto selector = GetDstSel(swizzle, output_component);
	if (Format::FormattedSelectorMissing(info, selector)) {
		NoteFormattedSelectorMissing(ctx.state.program.shader_hash,
		                             static_cast<uint32_t>(info.format), swizzle, "bound V#");
	}
	const auto source = Format::ResolveFormattedSource(
	    info, selector, GetCodegenOptions().format_missing_canonical);
	if (source.kind == FormattedSourceKind::Invalid) {
		ExitDescriptorBindingFailure(ctx.state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "buffer descriptor has reserved dst_sel");
	}
	return source;
}

uint32_t FormattedConstant(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                           FormattedSourceKind kind) {
	return ConstantU32(ctx.state, Format::FormattedConstantBits(info, kind));
}

template <typename LoadWordFn, typename LoadSubwordFn>
uint32_t LoadFormattedComponent(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                                FormattedSource source, LoadWordFn&& load_word,
                                LoadSubwordFn&& load_subword) {
	if (source.kind != FormattedSourceKind::Memory) {
		return FormattedConstant(ctx, info, source.kind);
	}
	const auto component = source.component;
	const auto bits      = info.component_bits[component];
	auto raw = info.packed_bitfield || bits == 32u ? load_word(component)
	                                                : load_subword(component, bits);
	if (info.packed_bitfield || (bits < 32u && IsSignedFormatComponent(info.type))) {
		const auto type =
		    IsSignedFormatComponent(info.type) ? TypeI32(ctx.state) : TypeU32(ctx.state);
		const auto source_value =
		    type == TypeI32(ctx.state) ? Unary(ctx.state, spv::OpBitcast, type, raw) : raw;
		const auto extracted = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(IsSignedFormatComponent(info.type) ? spv::OpBitFieldSExtract
		                                                                 : spv::OpBitFieldUExtract,
		                              type, extracted, source_value,
		                              ConstantU32(ctx.state, info.packed_bitfield
		                                                         ? info.component_bit_offset[component]
		                                                         : 0u),
		                              ConstantU32(ctx.state, bits));
		raw = type == TypeI32(ctx.state)
		          ? Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), extracted)
		          : extracted;
	}
	return NormalizeFormatComponent(ctx.state, info, component, raw);
}

void StoreSubwordInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t address, uint32_t index,
                          uint32_t bits, uint32_t data) {
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	if (resource.element_bits < 32u) {
		ctx.state.builder.AddFunction(
		    spv::OpStore, pointer,
		    Unary(ctx.state, spv::OpUConvert, TypeStorageBufferElement(ctx.state, bits), data),
		    resource.memory_access);
		return;
	}
	const auto shift   = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
	                                   ConstantU32(ctx.state, 3)),
	                            ConstantU32(ctx.state, 3));
	const auto mask    = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu), shift);
	const auto value   = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), data,
	                                   ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu)),
	                            shift);
	const auto merge   = [&](uint32_t old) {
		return Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state),
		              Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), old,
		                     Unary(ctx.state, spv::OpNot, TypeU32(ctx.state), mask)),
		              value);
	};
	if (mem.kind == IR::ResourceKind::Scratch) {
		const auto old = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), old, pointer);
		ctx.state.builder.AddFunction(spv::OpStore, pointer, merge(old));
	} else {
		AtomicUpdate(ctx.state, pointer, mem.kind, merge);
	}
}

void StoreSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t bits, uint32_t data) {
	const auto address = ByteAddress(ctx, inst, mem);
	const auto index = Binary(
	    ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), address,
	    ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	// KYTY_ROBUST_BUFFER_STORES (upstream ee034be4b): a subword store through an 8/16-bit view may
	// leave its bounds to the device; 32-bit views merge subwords in a compare-exchange loop, and
	// out-of-range atomics return undefined values, so they keep the explicit check.
	const auto in_bounds = resource.element_bits < 32u
	                           ? EmitStorageElementInBounds(ctx.state, resource, index)
	                           : EmitMemoryElementInBounds(ctx.state, resource, index);
	EmitIfCondition(ctx.state, in_bounds, [&]() {
		StoreSubwordInBounds(ctx, mem, resource, address, index, bits, data);
	});
}

void StoreSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem, uint32_t bits) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		StoreSubwordPrepared(ctx, inst, mem, resource, bits, ctx.Arg(inst, inst.NumArgs() - 2));
	});
}

void StoreWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource, uint32_t index,
                       uint32_t data) {
	ctx.state.builder.AddFunction(spv::OpStore,
	                              EmitMemoryElementPointer(ctx.state, resource, index), data,
	                              resource.memory_access);
}

void StoreWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                       const MemoryResourceAccess& resource, uint32_t data) {
	const auto index = DwordIndex(ctx, inst, mem);
	// KYTY_ROBUST_BUFFER_STORES (upstream ee034be4b): the device drops plain dword stores past the
	// descriptor range.
	EmitIfCondition(ctx.state, EmitStorageElementInBounds(ctx.state, resource, index), [&]() {
		StoreWordInBounds(ctx, resource, index, data);
	});
}

void StoreWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto data = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto store = [&](uint32_t resource) {
			if (mem.kind == IR::ResourceKind::Buffer) {
				const auto& buffer = state.program.info.buffers[resource];
				if (buffer.packed_stride == 0u &&
				    buffer.source < state.program.descriptor_sources.size() &&
				    state.program.descriptor_sources[buffer.source].indirect_descriptor) return;
			}
			mem.resource = resource;
			const auto access = PrepareMemoryResourceAccess(state, mem);
			StoreWordPrepared(ctx, inst, mem, access, data);
		};
		if (mem.kind != IR::ResourceKind::Buffer ||
		    state.program.info.buffers[mem.resource].indirect_root != mem.resource) {
			store(mem.resource);
			return;
		}
		const auto& buffer = state.program.info.buffers[mem.resource];
		const auto* handle = inst.Arg(0).ResolveInstruction();
		if (handle == nullptr || buffer.indirect_resources.empty() ||
		    state.flattened_srt_variable == 0 || buffer.indirect_search_iterations == 0u) {
			ctx.Fail(inst, "has no indirect buffer runtime mapping");
			return;
		}
		const auto selected = EmitIndirectResourceIndex(
		    state, ctx.Arg(*handle, 0), buffer.indirect_mapping_offset,
		    buffer.indirect_search_iterations, UINT32_MAX);
		const auto merge = state.builder.AllocateId();
		std::vector<uint32_t> labels(buffer.indirect_resources.size());
		std::vector<uint32_t> branches {spv::OpSwitch, selected, merge};
		for (uint32_t i = 0; i < labels.size(); ++i) {
			labels[i] = state.builder.AllocateId();
			branches.push_back(i);
			branches.push_back(labels[i]);
		}
		state.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
		state.builder.AddFunction(branches);
		for (uint32_t i = 0; i < labels.size(); ++i) {
			EmitLabel(state, labels[i]);
			store(buffer.indirect_resources[i]);
			state.builder.AddFunction(spv::OpBranch, merge);
		}
		EmitLabel(state, merge);
	});
}

spv::Op SpirvAtomicOpcode(IR::ValueOpcode opcode) {
	switch (opcode) {
		case IR::ValueOpcode::SharedAtomicCmpSwap64:
		case IR::ValueOpcode::SharedAtomicCmpSwap32:
		case IR::ValueOpcode::BufferAtomicCmpSwap32: return spv::OpAtomicCompareExchange;
		case IR::ValueOpcode::BufferAtomicSwap32:
		case IR::ValueOpcode::BufferAtomicSwap64:
		case IR::ValueOpcode::SharedAtomicSwap32: return spv::OpAtomicExchange;
		case IR::ValueOpcode::BufferAtomicIAdd32:
		case IR::ValueOpcode::BufferAtomicIAdd64:
		case IR::ValueOpcode::SharedAtomicIAdd64:
		case IR::ValueOpcode::SharedAtomicIAdd32: return spv::OpAtomicIAdd;
		case IR::ValueOpcode::BufferAtomicISub32:
		case IR::ValueOpcode::BufferAtomicISub64:
		case IR::ValueOpcode::SharedAtomicISub32: return spv::OpAtomicISub;
		case IR::ValueOpcode::BufferAtomicSMin32:
		case IR::ValueOpcode::BufferAtomicSMin64:
		case IR::ValueOpcode::SharedAtomicSMin32: return spv::OpAtomicSMin;
		case IR::ValueOpcode::BufferAtomicUMin32:
		case IR::ValueOpcode::BufferAtomicUMin64:
		case IR::ValueOpcode::SharedAtomicUMin32: return spv::OpAtomicUMin;
		case IR::ValueOpcode::BufferAtomicSMax32:
		case IR::ValueOpcode::BufferAtomicSMax64:
		case IR::ValueOpcode::SharedAtomicSMax32: return spv::OpAtomicSMax;
		case IR::ValueOpcode::BufferAtomicUMax32:
		case IR::ValueOpcode::BufferAtomicUMax64:
		case IR::ValueOpcode::SharedAtomicUMax32:
		case IR::ValueOpcode::SharedAtomicUMax64: return spv::OpAtomicUMax;
		case IR::ValueOpcode::BufferAtomicAnd32:
		case IR::ValueOpcode::BufferAtomicAnd64:
		case IR::ValueOpcode::SharedAtomicAnd32: return spv::OpAtomicAnd;
		case IR::ValueOpcode::BufferAtomicOr32:
		case IR::ValueOpcode::BufferAtomicOr64:
		case IR::ValueOpcode::SharedAtomicOr64:
		case IR::ValueOpcode::SharedAtomicOr32: return spv::OpAtomicOr;
		case IR::ValueOpcode::BufferAtomicXor32:
		case IR::ValueOpcode::BufferAtomicXor64:
		case IR::ValueOpcode::SharedAtomicXor32: return spv::OpAtomicXor;
		default: return spv::OpNop;
	}
}

uint32_t EmitAtomicOperation(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t pointer,
                             uint32_t scope) {
	const auto old = ctx.state.builder.AllocateId();
	if (inst.GetOpcode() == IR::ValueOpcode::BufferAtomicCmpSwap32 ||
	    inst.GetOpcode() == IR::ValueOpcode::SharedAtomicCmpSwap32) {
		auto desired    = ctx.Arg(inst, inst.NumArgs() - 3);
		auto comparator = ctx.Arg(inst, inst.NumArgs() - 2);
		if (inst.GetOpcode() == IR::ValueOpcode::SharedAtomicCmpSwap32) {
			// RDNA2 DS compares DATA and stores DATA2, opposite to BUFFER atomics (upstream e93b4b037).
			std::swap(desired, comparator);
		}
		ctx.state.builder.AddFunction(
		    spv::OpAtomicCompareExchange, TypeU32(ctx.state), old, pointer,
		    ConstantU32(ctx.state, scope), ConstantU32(ctx.state, spv::MemorySemanticsMaskNone),
		    ConstantU32(ctx.state, spv::MemorySemanticsMaskNone), desired, comparator);
	} else {
		const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
		ctx.state.builder.AddFunction(SpirvAtomicOpcode(inst.GetOpcode()), TypeU32(ctx.state), old,
		                              pointer, ConstantU32(ctx.state, scope),
		                              ConstantU32(ctx.state, spv::MemorySemanticsMaskNone), value);
	}
	return old;
}

template <typename Fn>
uint32_t EmitIndirectAtomic(ValueEmitContext& ctx, const IR::Inst& inst, bool wide,
                            Fn&& operation);

template <typename Fn>
uint32_t EmitAtomicAccess(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& operation) {
	if (mem.kind == IR::ResourceKind::IndirectBuffer) {
		// KYTY_BDA_WRITES: through a V# the shader computed.
		return EmitIndirectAtomic(ctx, inst, false, std::forward<Fn>(operation));
	}
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		auto index = DwordIndex(ctx, inst, mem);
		if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
			// Keep region-address truncation local to DS atomics.
			index = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), index,
			               ConstantU32(ctx.state, 0x3fffu));
		}
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return EmitValueOrZeroIfCondition(
		    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
			    return operation(EmitMemoryElementPointer(ctx.state, resource, index));
		    });
	});
}

template <typename Fn>
uint32_t EmitAtomicUpdate(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& replacement) {
	const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
	return EmitAtomicAccess(ctx, inst, mem, [&](uint32_t pointer) {
		return AtomicUpdate(ctx.state, pointer, mem.kind, [&](uint32_t old) {
			return replacement(ctx.state, old, value);
		});
	});
}

struct PreparedFormattedMemory {
	Format::BufferFormatInfo info;
	MemoryResourceAccess     resource;
	uint32_t                 base = 0;
	uint32_t                 in_bounds = 0;
};

PreparedFormattedMemory PrepareFormattedMemory(ValueEmitContext& ctx, const IR::Inst& inst,
                                               const IR::MemoryInfo& mem,
                                               const MemoryResourceAccess& resource,
                                               const Format::BufferFormatInfo& info) {
	PreparedFormattedMemory plan;
	plan.info      = info;
	plan.resource  = resource;
	const auto start = ByteAddress(ctx, inst, mem);
	// Check the whole, unaligned format span before aligning the actual memory address.
	const auto end = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), start,
	                        ConstantU32(ctx.state, info.byte_size - 1u));
	const auto last_index = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), end,
	                               ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	plan.in_bounds = AndCondition(
	    ctx.state, Binary(ctx.state, spv::OpUGreaterThanEqual, TypeBool(ctx.state), end, start),
	    EmitMemoryElementInBounds(ctx.state, plan.resource, last_index));
	plan.base = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), start,
	                   ConstantU32(ctx.state, ~(std::min(4u, info.byte_size) - 1u)));
	return plan;
}

std::pair<uint32_t, uint32_t> FormattedComponentAddress(EmitterState& state,
                                                       const PreparedFormattedMemory& plan,
                                                       uint32_t component) {
	// Components advance sequentially from one address; do not reapply buffer swizzling.
	const auto offset = Format::GetFormatComponentByteOffset(plan.info, component);
	const auto address = offset == 0u ? plan.base : Binary(
	    state, spv::OpIAdd, TypeU32(state), plan.base, ConstantU32(state, offset));
	return {address, Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
	                        ConstantU32(state, std::countr_zero(plan.resource.element_bits / 8u)))};
}

uint32_t LoadFormattedInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                               const PreparedFormattedMemory& plan, uint32_t output_component) {
	return LoadFormattedComponent(
	    ctx, plan.info, ResolveFormattedSource(ctx, mem, plan.info, output_component),
	    [&](uint32_t component) {
		    return LoadWordInBounds(ctx, plan.resource,
		                            FormattedComponentAddress(ctx.state, plan, component).second);
	    },
	    [&](uint32_t component, uint32_t bits) {
		    const auto [address, index] = FormattedComponentAddress(ctx.state, plan, component);
		    return LoadSubwordInBounds(ctx, plan.resource, address, index, bits);
	    });
}

uint32_t ConstructU32Composite(EmitterState& state, uint32_t components,
                               const std::array<uint32_t, 4>& values) {
	if (components == 1u) return values[0];
	const auto            result = state.builder.AllocateId();
	std::vector<uint32_t> words {spv::OpCompositeConstruct, TypeU32Composite(state, components),
	                             result};
	words.insert(words.end(), values.begin(), values.begin() + components);
	state.builder.AddFunction(words);
	return result;
}

uint32_t FormattedOutOfBoundsValue(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                   const PreparedFormattedMemory& plan, uint32_t components) {
	std::array<uint32_t, 4> values {};
	// KYTY_NULL_BUFFER_SHARE: a slot bound as the null buffer (marker offset) reads zero in every
	// channel, as a null V# (DST_SEL 0) does, whatever DST_SELs its shared specialization has.
	uint32_t is_null = 0;
	if (GetCodegenOptions().null_buffer_share && mem.kind == IR::ResourceKind::Buffer &&
	    mem.resource < ctx.state.memory_byte_offsets.size()) {
		is_null = Binary(ctx.state, spv::OpIEqual, TypeBool(ctx.state),
		                 ctx.state.memory_byte_offsets[mem.resource],
		                 ConstantU32(ctx.state, NullBufferMarkerOffset));
	}
	for (uint32_t component = 0; component < components; component++) {
		const auto source = ResolveFormattedSource(ctx, mem, plan.info, component);
		values[component] = FormattedConstant(ctx, plan.info, source.kind);
		if (is_null != 0 && source.kind == FormattedSourceKind::One) {
			values[component] = Select(ctx.state, TypeU32(ctx.state), is_null,
			                           ConstantU32(ctx.state, 0), values[component]);
		}
	}
	return ConstructU32Composite(ctx.state, components, values);
}

uint32_t FormattedLoad(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		const auto info = Format::GetFormatInfo(BufferFormat(ctx, mem));
		if (info.type == Format::ComponentType::Unknown) {
			return LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, 0u), resource);
		}
		const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info);
		const auto value = EmitValueOrDefaultIfCondition(
		    ctx.state, plan.in_bounds, TypeU32(ctx.state),
		    FormattedOutOfBoundsValue(ctx, mem, plan, 1u),
		    [&]() { return LoadFormattedInBounds(ctx, mem, plan, 0u); });
		if (mem.data_bits != 16u) {
			return value;
		}
		if (info.type == Format::ComponentType::Uint) {
			return EmitUMin32(ctx.state, value, ConstantU32(ctx.state, 0xffffu));
		}
		if (info.type == Format::ComponentType::Sint) {
			const auto lower = EmitSMax32(ctx.state, value, ConstantU32(ctx.state, 0xffff8000u));
			return EmitSMin32(ctx.state, lower, ConstantU32(ctx.state, 0x7fffu));
		}
		return EmitConvertF16F32(ctx.state, EmitBitCastF32U32(ctx.state, value));
	});
}

// Formatted vector loads; D16 (BUFFER_LOAD_FORMAT_D16_XYZW) packs component pairs into dwords.
uint32_t LoadFormattedPrepared(ValueEmitContext& ctx, const IR::Inst& inst,
                               const IR::MemoryInfo& mem, const MemoryResourceAccess& resource,
                               const Format::BufferFormatInfo& info, uint32_t components) {
	auto&      state = ctx.state;
	const auto count = mem.data_bits == 16u ? mem.component_count : components;
	const auto plan  = PrepareFormattedMemory(ctx, inst, mem, resource, info);
	const auto value = EmitValueOrDefaultIfCondition(
	    state, plan.in_bounds, TypeU32Composite(state, count),
	    FormattedOutOfBoundsValue(ctx, mem, plan, count), [&]() {
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < count; component++) {
			    values[component] = LoadFormattedInBounds(ctx, mem, plan, component);
		    }
		    return ConstructU32Composite(state, count, values);
	    });
	if (mem.data_bits != 16u) return value;
	const bool integer = info.type == Format::ComponentType::Uint ||
	                     info.type == Format::ComponentType::Sint;
	const auto scalar = [&](uint32_t component) {
		if (component >= count) return integer ? ConstantU32(state, 0u) : ConstantF32(state, 0);
		auto result = value;
		if (count != 1u) {
			result = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), result, value,
			                          component);
		}
		if (info.type == Format::ComponentType::Uint) {
			return EmitUMin32(state, result, ConstantU32(state, 0xffffu));
		}
		if (info.type == Format::ComponentType::Sint) {
			const auto lower = EmitSMax32(state, result, ConstantU32(state, 0xffff8000u));
			return EmitSMin32(state, lower, ConstantU32(state, 0x7fffu));
		}
		return EmitBitCastF32U32(state, result);
	};
	std::array<uint32_t, 4> packed {};
	for (uint32_t word = 0; word < components; word++) {
		const auto low  = scalar(word * 2u);
		const auto high = scalar(word * 2u + 1u);
		if (integer) {
			packed[word] = Binary(
			    state, spv::OpBitwiseOr, TypeU32(state),
			    Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, 0xffffu)),
			    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), high, ConstantU32(state, 16u)));
		} else {
			const auto pair = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair,
			                          low, high);
			packed[word] = EmitGlslPackHalf2x16(state, pair);
		}
	}
	return ConstructU32Composite(state, components, packed);
}

uint32_t EncodeFormattedStoreComponent(ValueEmitContext& ctx,
                                       const Format::BufferFormatInfo& info,
                                       uint32_t component, uint32_t data, uint32_t data_bits) {
	auto& state = ctx.state;
	const auto bits = info.component_bits[component];
	if (data_bits == 16u) {
		if (info.type == Format::ComponentType::Uint) {
			data = EmitBitFieldUExtract(state, data, ConstantU32(state, 0), ConstantU32(state, 16));
		} else if (info.type == Format::ComponentType::Sint) {
			data = EmitBitFieldSExtract(state, data, ConstantU32(state, 0), ConstantU32(state, 16));
		} else {
			data = EmitBitCastU32F32(state, EmitF16BitsToF32(state, data));
		}
	}
	const bool normalized = info.type == Format::ComponentType::Unorm ||
	                        info.type == Format::ComponentType::Snorm;
	const bool scaled = info.type == Format::ComponentType::Uscaled ||
	                    info.type == Format::ComponentType::Sscaled;
	if (normalized || scaled) {
		// Two-bit packed channels are unsigned even in signed formats.
		const bool is_signed = bits != 2u && (info.type == Format::ComponentType::Snorm ||
		                                      info.type == Format::ComponentType::Sscaled);
		const auto max_value = static_cast<float>((1u << (bits - (is_signed ? 1u : 0u))) - 1u);
		const auto lower = normalized ? (is_signed ? -1.0f : 0.0f)
		                              : (is_signed ? -max_value - 1.0f : 0.0f);
		const auto upper = normalized ? 1.0f : max_value;
		auto value = Select(state, TypeF32(state), EmitClassifyF32Bits(state, data).nan,
		                    ConstantF32Value(state, 0.0f), EmitBitCastF32U32(state, data));
		value = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(
		    state, value, ConstantF32Value(state, lower), ConstantF32Value(state, upper));
		if (normalized) {
			value = EmitFPRoundEven32(state, Binary(state, spv::OpFMul, TypeF32(state), value,
			                                        ConstantF32Value(state, max_value)));
		}
		// Float-to-integer conversion truncates scaled values toward zero.
		value = Unary(state, is_signed ? spv::OpConvertFToS : spv::OpConvertFToU,
		              is_signed ? TypeI32(state) : TypeU32(state), value);
		return is_signed ? Unary(state, spv::OpBitcast, TypeU32(state), value) : value;
	}
	// UINT/SINT saturate; 16-bit and 11/10-bit floats encode (PackFormatComponent, 259bb8b83).
	return PackFormatComponent(state, info, component, data);
}

void StoreFormattedPrepared(ValueEmitContext& ctx, const IR::Inst& inst,
                             const IR::MemoryInfo& mem, const MemoryResourceAccess& resource,
                             const Format::BufferFormatInfo& info, uint32_t data,
                             uint32_t components) {
	// RDNA2 formatted stores transfer the entire format, zero-filling missing VGPRs.
	const auto count = mem.data_bits == 16u ? mem.component_count : components;
	const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info);
	// KYTY_FORMAT_STORE_DST_SEL: memory channel c takes the data component whose V# dst_sel names
	// it (Table 31), the first such component; none: 0.
	const bool dst_sel = GetCodegenOptions().format_store_dst_sel && !mem.typed &&
	                     mem.resource < ctx.state.program.info.buffers.size();
	const auto source_of = [&](uint32_t channel) {
		if (!dst_sel) return channel;
		const auto swizzle = ctx.state.program.info.buffers[mem.resource].descriptor_swizzle;
		for (uint32_t source = 0; source < 4u; source++) {
			if (GetDstSel(swizzle, source) == 4u + channel) return source;
		}
		return UINT32_MAX;
	};
	EmitIfCondition(ctx.state, plan.in_bounds, [&]() {
		auto packed = ConstantU32(ctx.state, 0);
		for (uint32_t channel = 0; channel < info.component_count; channel++) {
			const auto component = channel;
			const auto source    = source_of(channel);
			auto value = ConstantU32(ctx.state, 0);
			if (source < count) {
				const auto word = mem.data_bits == 16u ? source / 2u : source;
				value = data;
				if (components != 1u) {
					value = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state),
					                              value, data, word);
				}
				if (mem.data_bits == 16u && (source & 1u) != 0u)
					value = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), value,
					               ConstantU32(ctx.state, 16u));
			}
			value = EncodeFormattedStoreComponent(ctx, info, component, value, mem.data_bits);
			const auto bits = info.component_bits[component];
			if (info.packed_bitfield) {
				packed = EmitNative<spv::OpBitFieldInsert, IR::Type::U32>(
				    ctx.state, packed, value,
				    ConstantU32(ctx.state, info.component_bit_offset[component]),
				    ConstantU32(ctx.state, bits));
			} else {
				const auto [address, index] = FormattedComponentAddress(ctx.state, plan, component);
				if (bits == 8u || bits == 16u) {
					StoreSubwordInBounds(ctx, mem, plan.resource, address, index, bits, value);
				} else {
					StoreWordInBounds(ctx, plan.resource, index, value);
				}
			}
		}
		if (info.packed_bitfield) {
			StoreWordInBounds(ctx, plan.resource,
			                  FormattedComponentAddress(ctx.state, plan, 0u).second, packed);
		}
	});
}

void FormattedStore(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		const auto data     = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto info     = Format::GetFormatInfo(BufferFormat(ctx, mem));
		if (info.type == Format::ComponentType::Unknown) {
			StoreWordPrepared(ctx, inst, RebaseRawComponent(mem, 0u), resource, data);
			return;
		}
		StoreFormattedPrepared(ctx, inst, mem, resource, info, data, 1u);
	});
}

struct IndirectBufferAccess {
	uint32_t address;
	uint32_t offset;
	uint32_t stride;
	uint32_t swizzle;
	uint32_t format;
	uint32_t selector;
	uint32_t mode;
	uint32_t records;
	uint32_t index_in_bounds;
	uint32_t scalar_in_bounds;
	uint32_t raw_records;
	uint32_t raw_index_in_bounds;
};

// The ADD_TID-adjusted element index of an access through a computed V# (word 3 bit 23).
uint32_t IndirectBufferIndex(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t word3) {
	auto&      state = ctx.state;
	const auto add_tid =
	    Binary(state, spv::OpINotEqual, TypeBool(state),
	           EmitBitFieldUExtract(state, word3, ConstantU32(state, 23), ConstantU32(state, 1)),
	           ConstantU32(state, 0));
	return Binary(state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 1),
	              Select(state, TypeU32(state), add_tid, BufferLane(state), ConstantU32(state, 0)));
}

// PrepareIndirectBuffer from the V# words and the instruction's operands; `index` already holds the
// ADD_TID lane. Also the body of the outlined raw loads (KYTY_INDIRECT_RAW_OUTLINE).
IndirectBufferAccess IndirectBufferFromWords(EmitterState& state, uint32_t word0, uint32_t word1,
                                             uint32_t records, uint32_t word3, uint32_t index,
                                             uint32_t voffset, uint32_t soffset, uint32_t immediate,
                                             bool formatted) {
	const auto field = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
	};
	const auto nonzero = [&](uint32_t value) {
		return Binary(state, spv::OpINotEqual, TypeBool(state), value, ConstantU32(state, 0));
	};
	const auto stride  = field(word1, 16, 14);
	const auto swizzle = AndCondition(state, nonzero(stride), nonzero(field(word1, 31, 1)));
	const auto address = CalculateBufferAddress(state, index, voffset, soffset, immediate, stride,
	                                            swizzle, field(word3, 21, 2));
	const auto base        = PackU64(state, word0, field(word1, 0, 16));
	const auto raw_records = Binary(state, spv::OpISub, TypeU32(state), records, soffset);
	return {
	    .address          = Binary(state, spv::OpIAdd, TypeU64(state), base,
	                               Unary(state, spv::OpUConvert, TypeU64(state), address.byte)),
	    .offset           = address.offset,
	    .stride           = stride,
	    .swizzle          = swizzle,
	    .format           = field(word3, 12, 7),
	    .selector         = formatted ? field(word3, 0, 3) : 0u,
	    .mode             = field(word3, 28, 2),
	    .records          = records,
	    .index_in_bounds  = Binary(state, spv::OpULessThan, TypeBool(state), index, records),
	    .scalar_in_bounds = Binary(state, spv::OpULessThanEqual, TypeBool(state), soffset, records),
	    .raw_records      = raw_records,
	    .raw_index_in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, raw_records),
	};
}

IndirectBufferAccess PrepareIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto  word3  = ctx.Arg(handle, 3);
	return IndirectBufferFromWords(ctx.state, ctx.Arg(handle, 0), ctx.Arg(handle, 1),
	                               ctx.Arg(handle, 2), word3, IndirectBufferIndex(ctx, inst, word3),
	                               ctx.Arg(inst, 2), ctx.Arg(inst, 3), ctx.Memory(inst).offset,
	                               ctx.Memory(inst).formatted);
}

// IndirectBufferInBounds with the payload size as a U32 id (KYTY_INDIRECT_FORMAT_COMPACT: the
// element size of a format known only at run time).
uint32_t IndirectBufferInBoundsBytes(EmitterState& state, const IndirectBufferAccess& buffer,
                                     uint32_t displacement, uint32_t bytes_id, bool formatted) {
	const auto offset = displacement == 0u ? buffer.offset : Binary(
	    state, spv::OpIAdd, TypeU32(state), buffer.offset, ConstantU32(state, displacement));
	const auto has_payload = [&](uint32_t size) {
		return AndCondition(
		    state, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), size, bytes_id),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state), offset,
		           Binary(state, spv::OpISub, TypeU32(state), size, bytes_id)));
	};
	const auto structured = AndCondition(
	    state, buffer.index_in_bounds,
	    formatted ? has_payload(buffer.stride)
	              : Binary(state, spv::OpULessThan, TypeBool(state), offset, buffer.stride));
	// OOB_SELECT=3 reduces NUM_RECORDS by SOFFSET before its offset/index checks.
	const auto raw = AndCondition(
	    state, buffer.scalar_in_bounds,
	    Select(state, TypeBool(state), buffer.swizzle,
	           AndCondition(state, buffer.raw_index_in_bounds, has_payload(buffer.stride)),
	           has_payload(buffer.raw_records)));
	auto in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 0)),
	    structured, buffer.index_in_bounds);
	in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpULessThan, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	    in_bounds,
	    Select(state, TypeBool(state),
	           Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	           Binary(state, spv::OpINotEqual, TypeBool(state), buffer.records, ConstantU32(state, 0)),
	           raw));
	return in_bounds;
}

uint32_t IndirectBufferInBounds(EmitterState& state, const IndirectBufferAccess& buffer,
                                uint32_t displacement, uint32_t bytes, bool formatted) {
	const auto offset = displacement == 0u ? buffer.offset : Binary(
	    state, spv::OpIAdd, TypeU32(state), buffer.offset, ConstantU32(state, displacement));
	const auto has_payload = [&](uint32_t size) {
		return AndCondition(
		    state, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), size,
		                  ConstantU32(state, bytes)),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state), offset,
		           Binary(state, spv::OpISub, TypeU32(state), size, ConstantU32(state, bytes))));
	};
	const auto structured = AndCondition(
	    state, buffer.index_in_bounds,
	    formatted ? has_payload(buffer.stride)
	              : Binary(state, spv::OpULessThan, TypeBool(state), offset, buffer.stride));
	// OOB_SELECT=3 reduces NUM_RECORDS by SOFFSET before its offset/index checks.
	const auto raw = AndCondition(
	    state, buffer.scalar_in_bounds,
	    Select(state, TypeBool(state), buffer.swizzle,
	           AndCondition(state, buffer.raw_index_in_bounds, has_payload(buffer.stride)),
	           has_payload(buffer.raw_records)));
	auto in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 0)),
	    structured, buffer.index_in_bounds);
	in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpULessThan, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	    in_bounds,
	    Select(state, TypeBool(state),
	           Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	           Binary(state, spv::OpINotEqual, TypeBool(state), buffer.records, ConstantU32(state, 0)),
	           raw));
	return in_bounds;
}

// KYTY_INDIRECT_COHERENT (CodegenOptions::indirect_coherent): a GLC/DLC load through a computed V#
// is Coherent and Volatile, as through a bound V#.
bool IndirectLoadCoherent(ValueEmitContext& ctx, const IR::Inst& inst) {
	return GetCodegenOptions().indirect_coherent && ctx.Memory(inst).coherent;
}

// The dwords of a raw load through a computed V# (each in range and FORMAT valid, else 0).
uint32_t LoadIndirectBufferAccess(ValueEmitContext& ctx, const IndirectBufferAccess& buffer,
                                  uint32_t components, bool coherent) {
	auto&      state        = ctx.state;
	const auto valid_format = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
	                                  ConstantU32(state, 0));
	const auto base         = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
	                                 ConstantU64(state, ~uint64_t {3}));
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; ++component) {
		const auto address = component == 0u ? base
		                                     : Binary(state, spv::OpIAdd, TypeU64(state), base,
		                                              ConstantU64(state, component * 4u));
		values[component] = EmitValueOrZeroIfCondition(
		    state, AndCondition(state, valid_format,
		                        IndirectBufferInBounds(state, buffer, component * 4u, 4u, false)),
		    [&]() { return LoadBdaDword(ctx, address, coherent); });
	}
	return ConstructU32Composite(state, components, values);
}

// KYTY_INDIRECT_RAW_OUTLINE: the module's function for a raw load of this shape through a computed
// V# (DefineIndirectRawFunctions), else 0. Key: the dword count (1-4) of an aligned-dword load
// (LoadIndirectBuffer), 0x100 | bits (8/16/32) of a single element (LoadIndirectBufferElement);
// | 0x1000 coherent.
uint32_t IndirectRawFunction(const EmitterState& state, uint32_t key) {
	const auto found = state.indirect_raw_functions.find(key);
	return found == state.indirect_raw_functions.end() ? 0u : found->second;
}

uint32_t IndirectRawKey(uint32_t components, uint32_t /*bits*/, bool coherent) {
	return components | (coherent ? 0x1000u : 0u);
}

uint32_t IndirectRawElementKey(uint32_t bits, bool coherent) {
	return 0x100u | bits | (coherent ? 0x1000u : 0u);
}

// A call of an outlined raw load: (word0, word1, records, word3, index with ADD_TID, voffset +
// immediate, soffset).
uint32_t CallIndirectRawFunction(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t function,
                                 uint32_t result_type) {
	auto&       state  = ctx.state;
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto  word3  = ctx.Arg(handle, 3);
	auto        voffset = ctx.Arg(inst, 2);
	if (const auto immediate = ctx.Memory(inst).offset; immediate != 0u) {
		voffset = Binary(state, spv::OpIAdd, TypeU32(state), voffset, ConstantU32(state, immediate));
	}
	const std::array<uint32_t, 7> arguments {ctx.Arg(handle, 0), ctx.Arg(handle, 1),
	                                         ctx.Arg(handle, 2), word3,
	                                         IndirectBufferIndex(ctx, inst, word3), voffset,
	                                         ctx.Arg(inst, 3)};
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, result_type, result, function,
	                          std::span<const uint32_t>(arguments));
	return result;
}

uint32_t LoadIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	if (const auto function =
	        IndirectRawFunction(state, IndirectRawKey(components, 32u, IndirectLoadCoherent(ctx, inst)));
	    function != 0u) {
		return CallIndirectRawFunction(ctx, inst, function, TypeU32Composite(state, components));
	}
	return LoadIndirectBufferAccess(ctx, PrepareIndirectBuffer(ctx, inst), components,
	                                IndirectLoadCoherent(ctx, inst));
}

// One element of a raw access through a V# the shader computed (ResourceKind::IndirectBuffer):
// its guest byte address and whether it is in range. Addressing and the OOB_SELECT range checks
// are LoadIndirectBuffer's for one DWORD (RDNA2 ISA 8.1.5), with `payload` bytes at
// `component_offset` bytes past the instruction offset: OOB_SELECT 3 checks offset + payload
// against the records the SOFFSET leaves. An invalid FORMAT (0: unbound resource) is out of range.
struct IndirectBufferElement {
	uint32_t guest     = 0; // U64 guest byte address
	uint32_t in_bounds = 0; // bool
};

IndirectBufferElement IndirectBufferElementFromWords(EmitterState& state, uint32_t word0,
                                                     uint32_t word1, uint32_t records,
                                                     uint32_t word3, uint32_t index,
                                                     uint32_t voffset, uint32_t soffset,
                                                     uint32_t immediate, uint32_t payload) {
	const auto  field   = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first),
		                            ConstantU32(state, count));
	};
	const auto nonzero = [&](uint32_t value) {
		return Binary(state, spv::OpINotEqual, TypeBool(state), value, ConstantU32(state, 0));
	};
	const auto bytes    = ConstantU32(state, payload);
	const auto has_data = [&](uint32_t offset, uint32_t size) {
		return AndCondition(
		    state, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), size, bytes),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state), offset,
		           Binary(state, spv::OpISub, TypeU32(state), size, bytes)));
	};
	const auto stride       = field(word1, 16, 14);
	const auto swizzle      = AndCondition(state, nonzero(stride), nonzero(field(word1, 31, 1)));
	const auto index_stride = field(word3, 21, 2);
	const auto base            = DeviceAddressFromWords(state, word0, field(word1, 0, 16));
	const auto valid_format    = nonzero(field(word3, 12, 7));
	const auto mode            = field(word3, 28, 2);
	const auto index_in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, records);
	const auto scalar_in_bounds =
	    Binary(state, spv::OpULessThanEqual, TypeBool(state), soffset, records);
	const auto raw_records = Binary(state, spv::OpISub, TypeU32(state), records, soffset);
	const auto raw_index_in_bounds =
	    Binary(state, spv::OpULessThan, TypeBool(state), index, raw_records);
	const auto address = CalculateBufferAddress(state, index, voffset, soffset, immediate, stride,
	                                            swizzle, index_stride);
	const auto structured_bounds =
	    AndCondition(state, index_in_bounds,
	                 Binary(state, spv::OpULessThan, TypeBool(state), address.offset, stride));
	const auto raw_bounds = AndCondition(
	    state, scalar_in_bounds,
	    Select(state, TypeBool(state), swizzle,
	           AndCondition(state, raw_index_in_bounds, has_data(address.offset, stride)),
	           has_data(address.offset, raw_records)));
	auto in_bounds = Select(state, TypeBool(state),
	                        Binary(state, spv::OpIEqual, TypeBool(state), mode, ConstantU32(state, 0)),
	                        structured_bounds, index_in_bounds);
	in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpULessThan, TypeBool(state), mode, ConstantU32(state, 2)), in_bounds,
	    Select(state, TypeBool(state),
	           Binary(state, spv::OpIEqual, TypeBool(state), mode, ConstantU32(state, 2)),
	           nonzero(records), raw_bounds));
	return {
	    .guest     = Binary(state, spv::OpIAdd, TypeU64(state), base,
	                        Unary(state, spv::OpUConvert, TypeU64(state), address.byte)),
	    .in_bounds = AndCondition(state, valid_format, in_bounds),
	};
}

IndirectBufferElement IndirectBufferElementAccess(ValueEmitContext& ctx, const IR::Inst& inst,
                                                  uint32_t component_offset, uint32_t payload) {
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto  word3  = ctx.Arg(handle, 3);
	return IndirectBufferElementFromWords(
	    ctx.state, ctx.Arg(handle, 0), ctx.Arg(handle, 1), ctx.Arg(handle, 2), word3,
	    IndirectBufferIndex(ctx, inst, word3), ctx.Arg(inst, 2), ctx.Arg(inst, 3),
	    ctx.Memory(inst).offset + component_offset, payload);
}

// One raw BUFFER_LOAD_DWORD/UBYTE/USHORT through a V# the shader computed (KYTY_SRT_VARIANT_READS,
// ResourceKind::IndirectBuffer), read through BDA; an out-of-range element reads 0.
uint32_t LoadIndirectBufferElement(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t bits) {
	const bool coherent = IndirectLoadCoherent(ctx, inst);
	if (const auto function = IndirectRawFunction(ctx.state, IndirectRawElementKey(bits, coherent));
	    function != 0u) {
		return CallIndirectRawFunction(ctx, inst, function, TypeU32(ctx.state));
	}
	const auto access = IndirectBufferElementAccess(ctx, inst, 0u, bits / 8u);
	return LoadBda(ctx, access.guest, access.in_bounds, bits, IndirectLoadCoherent(ctx, inst));
}

// KYTY_BDA_WRITES: sets the written-page bit of the page a BDA write lands in (the fault buffer's
// second bitmap, BufferCache::BDA_WRITE_BITMAP_WORD); the renderer's settle turns the bits into
// GPU-owned page runs. A set bit is not set again: at most one atomic per page per wave.
void RecordBdaWrite(EmitterState& state, uint32_t guest) {
	const auto page = Unary(state, spv::OpUConvert, TypeU32(state),
	                        Binary(state, spv::OpShiftRightLogical, TypeU64(state), guest,
	                               ConstantDeviceAddress(state, BufferCache::CACHING_PAGEBITS)));
	const auto word =
	    Binary(state, spv::OpIAdd, TypeU32(state),
	           Binary(state, spv::OpShiftRightLogical, TypeU32(state), page, ConstantU32(state, 5)),
	           ConstantU32(state, static_cast<uint32_t>(BufferCache::BDA_WRITE_BITMAP_WORD)));
	const auto bit =
	    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), page, ConstantU32(state, 31)));
	const auto pointer = FaultElementPointer(state, word);
	const auto current = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicLoad, TypeU32(state), current, pointer,
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone));
	const auto missing =
	    Binary(state, spv::OpIEqual, TypeBool(state),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), current, bit), ConstantU32(state, 0));
	EmitIfCondition(state, missing, [&]() {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), result, pointer,
		                          ConstantU32(state, spv::ScopeDevice),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone), bit);
	});
}

// KYTY_BDA_WRITES: a write to a page without a cache buffer cannot land (get_bda_pointer records
// the page fault, so a later dispatch finds a buffer there); it is counted for the settle.
void RecordBdaDroppedWrite(EmitterState& state) {
	const auto pointer = FaultElementPointer(
	    state, ConstantU32(state, static_cast<uint32_t>(BufferCache::BDA_DROPPED_WRITES_WORD)));
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), result, pointer,
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone),
	                          ConstantU32(state, 1));
}

// KYTY_BDA_STORE_LOG: a dword store to a page without a cache buffer. The next store-log entry
// (BufferCache::BDA_STORE_LOG_WORD) takes the dword-aligned guest address, the value and the mask of
// the bits it replaces, and the settle writes it into guest memory; past the log's capacity the store
// is dropped and counted as before. Entries of one invocation keep its program order.
void LogBdaStore(EmitterState& state, uint32_t guest, uint32_t value, uint32_t mask) {
	const auto count = state.builder.AllocateId();
	state.builder.AddFunction(
	    spv::OpAtomicIAdd, TypeU32(state), count,
	    FaultElementPointer(state,
	                        ConstantU32(state, static_cast<uint32_t>(BufferCache::BDA_STORE_LOG_WORD))),
	    ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone),
	    ConstantU32(state, 1));
	const auto fits =
	    Binary(state, spv::OpULessThan, TypeBool(state), count,
	           ConstantU32(state, static_cast<uint32_t>(BufferCache::BDA_STORE_LOG_ENTRIES)));
	EmitIfCondition(state, fits, [&]() {
		const auto first = Binary(
		    state, spv::OpIAdd, TypeU32(state),
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), count, ConstantU32(state, 2)),
		    ConstantU32(state, static_cast<uint32_t>(BufferCache::BDA_STORE_LOG_WORD + 4u)));
		const std::array<uint32_t, 4> words {
		    Unary(state, spv::OpUConvert, TypeU32(state), guest),
		    Unary(state, spv::OpUConvert, TypeU32(state),
		          Binary(state, spv::OpShiftRightLogical, TypeU64(state), guest,
		                 ConstantDeviceAddress(state, 32))),
		    value, mask};
		for (uint32_t i = 0; i < words.size(); i++) {
			const auto index = i == 0 ? first
			                          : Binary(state, spv::OpIAdd, TypeU32(state), first,
			                                   ConstantU32(state, i));
			state.builder.AddFunction(spv::OpStore, FaultElementPointer(state, index), words[i]);
		}
	});
	EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), fits),
	                [&]() { RecordBdaDroppedWrite(state); });
}

// The value and mask a store to a page without a cache buffer logs (KYTY_BDA_STORE_LOG), or none.
struct BdaStoreLogValue {
	uint32_t value = 0;
	uint32_t mask  = 0;
};

// KYTY_BDA_WRITES: `write(host_address)` once `guest` (in range) is mapped to a cache buffer, after
// recording its page; otherwise a store with `log` is logged (KYTY_BDA_STORE_LOG) and any other write
// is dropped and counted. Returns write's value, or `default_value` when out of range or not mapped
// (an atomic then returns 0).
template <typename Fn>
uint32_t WriteBdaIfMapped(ValueEmitContext& ctx, uint32_t guest, uint32_t in_bounds, uint32_t type,
                          uint32_t default_value, Fn&& write,
                          const BdaStoreLogValue* log = nullptr) {
	auto& state = ctx.state;
	return EmitValueOrDefaultIfCondition(state, in_bounds, type, default_value, [&]() {
		const auto host    = GetBdaPointer(ctx, guest);
		const auto present =
		    Binary(state, spv::OpINotEqual, TypeBool(state), host, ConstantDeviceAddress(state, 0));
		EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), present), [&]() {
			if (log != nullptr) {
				LogBdaStore(state, guest, log->value, log->mask);
			} else {
				RecordBdaDroppedWrite(state);
			}
		});
		return EmitValueOrDefaultIfCondition(state, present, type, default_value, [&]() {
			RecordBdaWrite(state, guest);
			return write(host);
		});
	});
}

// KYTY_BDA_WRITES: a raw BUFFER_STORE_BYTE/SHORT/DWORD[X2-X4] through a V# the shader computed.
// Each DWORD component is range-checked on its own (ISA 8.1.5 note 4); DWORD addresses ignore
// their two LSBs (8.1.7), and a byte or short replaces its bits of the containing DWORD atomically,
// as the bound-buffer store does.
void StoreIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components,
                         uint32_t bits) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto data = ctx.Arg(inst, inst.NumArgs() - 2);
		for (uint32_t component = 0; component < components; component++) {
			const auto access = IndirectBufferElementAccess(ctx, inst, component * 4u, bits / 8u);
			auto       value  = data;
			if (components > 1u) {
				value = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, data,
				                          component);
			}
			const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), access.guest,
			                            ConstantDeviceAddress(state, ~uint64_t {3}));
			// A byte or short replaces `mask` of the dword with `shifted`.
			uint32_t mask    = 0;
			uint32_t shifted = 0;
			if (bits != 32u) {
				const auto mask_bits = ConstantU32(state, bits == 8u ? 0xffu : 0xffffu);
				const auto shift =
				    Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
				           Binary(state, spv::OpBitwiseAnd, TypeU32(state),
				                  Unary(state, spv::OpUConvert, TypeU32(state), access.guest),
				                  ConstantU32(state, 3)),
				           ConstantU32(state, 3));
				mask    = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), mask_bits, shift);
				shifted = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
				                 Binary(state, spv::OpBitwiseAnd, TypeU32(state), value, mask_bits),
				                 shift);
			}
			const BdaStoreLogValue log {bits == 32u ? value : shifted,
			                            bits == 32u ? ConstantU32(state, ~0u) : mask};
			(void)WriteBdaIfMapped(
			    ctx, aligned, access.in_bounds, TypeU32(state), ConstantU32(state, 0),
			    [&](uint32_t host) {
				    const auto pointer = state.builder.AllocateId();
				    state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state),
				                              pointer, host);
				    if (bits == 32u && IndirectLoadCoherent(ctx, inst)) {
					    // KYTY_INDIRECT_COHERENT: a GLC/DLC store is Coherent and Volatile, so a
					    // polling workgroup of the same dispatch sees it.
					    const auto block = state.builder.DecoratedType(
					        spv::OpTypeStruct,
					        {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
					         {spv::OpMemberDecorate, {0, spv::DecorationCoherent}},
					         {spv::OpDecorate, {spv::DecorationBlock}}},
					        TypeU32(state));
					    const auto base = state.builder.AllocateId();
					    state.builder.AddFunction(
					        spv::OpConvertUToPtr,
					        TypePointer(state, spv::StorageClassPhysicalStorageBuffer, block), base,
					        host);
					    const auto member = state.builder.AllocateId();
					    state.builder.AddFunction(spv::OpAccessChain, TypePhysicalU32Pointer(state),
					                              member, base, ConstantU32(state, 0));
					    constexpr uint32_t alignment = sizeof(uint32_t);
					    state.builder.AddFunction(spv::OpStore, member, value,
					                              spv::MemoryAccessAlignedMask |
					                                  spv::MemoryAccessVolatileMask,
					                              alignment);
				    } else if (bits == 32u) {
					    constexpr uint32_t alignment = sizeof(uint32_t);
					    state.builder.AddFunction(spv::OpStore, pointer, value,
					                              spv::MemoryAccessAlignedMask, alignment);
				    } else {
					    (void)AtomicUpdate(
					        state, pointer, IR::ResourceKind::IndirectBuffer, [&](uint32_t old) {
						        return Binary(
						            state, spv::OpBitwiseOr, TypeU32(state),
						            Binary(state, spv::OpBitwiseAnd, TypeU32(state), old,
						                   Unary(state, spv::OpNot, TypeU32(state), mask)),
						            shifted);
					        });
				    }
				    return ConstantU32(state, 0);
			    },
			    GetCodegenOptions().bda_store_log ? &log : nullptr);
		}
	});
}

// KYTY_INDIRECT_FORMAT_STORES: one 8, 16 or 32-bit piece of a formatted store through BDA at the
// byte address `guest`; 8 and 16-bit pieces replace their bits of the dword atomically.
static void StoreIndirectFormattedPiece(ValueEmitContext& ctx, uint32_t guest, uint32_t in_bounds,
                                        uint32_t bits, uint32_t value) {
	auto&      state   = ctx.state;
	const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), guest,
	                            ConstantDeviceAddress(state, ~uint64_t {3}));
	uint32_t   mask    = 0;
	uint32_t   shifted = 0;
	if (bits != 32u) {
		const auto mask_bits = ConstantU32(state, bits == 8u ? 0xffu : 0xffffu);
		const auto shift     = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		                              Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                                     Unary(state, spv::OpUConvert, TypeU32(state), guest),
		                                     ConstantU32(state, bits == 8u ? 3u : 2u)),
		                              ConstantU32(state, 3));
		mask    = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), mask_bits, shift);
		shifted = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		                 Binary(state, spv::OpBitwiseAnd, TypeU32(state), value, mask_bits), shift);
	}
	// KYTY_INDIRECT_FORMAT_STORES_LOG=1 (codegen option): a piece for a page without a cache buffer
	// goes to the store log (KYTY_BDA_STORE_LOG) as raw stores do, instead of being dropped.
	const BdaStoreLogValue log {bits == 32u ? value : shifted,
	                            bits == 32u ? ConstantU32(state, ~0u) : mask};
	const bool use_log = GetCodegenOptions().bda_store_log && GetCodegenOptions().indirect_format_stores_log;
	(void)WriteBdaIfMapped(ctx, aligned, in_bounds, TypeU32(state), ConstantU32(state, 0),
	                       [&](uint32_t host) {
		                       const auto pointer = state.builder.AllocateId();
		                       state.builder.AddFunction(spv::OpConvertUToPtr,
		                                                 TypePhysicalU32Pointer(state), pointer, host);
		                       if (bits == 32u) {
			                       state.builder.AddFunction(spv::OpStore, pointer, value,
			                                                 spv::MemoryAccessAlignedMask, 4u);
		                       } else {
			                       (void)AtomicUpdate(
			                           state, pointer, IR::ResourceKind::IndirectBuffer,
			                           [&](uint32_t old) {
				                           return Binary(
				                               state, spv::OpBitwiseOr, TypeU32(state),
				                               Binary(state, spv::OpBitwiseAnd, TypeU32(state), old,
				                                      Unary(state, spv::OpNot, TypeU32(state), mask)),
				                               shifted);
			                           });
		                       }
		                       return ConstantU32(state, 0);
	                       },
	                       use_log ? &log : nullptr);
}

// KYTY_INDIRECT_FORMAT_STORES (adapted from Erlou93/KytyPS5-Yotei's StoreIndirectFormatted, M48):
// a formatted X..XYZW store of 32-bit data through a V# the shader computed. The V#'s run-time
// format selects the encoding; the whole element is written (missing channels zero, as RDNA2
// does) when it is in bounds. Packed formats are one 16/32-bit piece; other formats write each
// 8/16/32-bit channel at its byte offset. Unknown formats, elements over 16 bytes and wider
// channels are dropped.
void StoreIndirectFormatted(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto buffer = PrepareIndirectBuffer(ctx, inst);
		const auto data   = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto value  = [&](uint32_t component) {
			if (component >= components) return ConstantU32(state, 0);
			if (components == 1u) return data;
			const auto extracted = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), extracted, data,
			                          component);
			return extracted;
		};
		(void)EmitIndexSwitch(
		    state, buffer.format,
		    static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) + 1u, TypeU32(state),
		    [&](uint32_t format) {
			    const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
			    if (info.type == Format::ComponentType::Unknown || info.component_count == 0u ||
			        info.byte_size == 0u || info.byte_size > 16u ||
			        (info.packed_bitfield && info.byte_size != 2u && info.byte_size != 4u)) {
				    return ConstantU32(state, 0);
			    }
			    const auto in_bounds = IndirectBufferInBounds(state, buffer, 0u, info.byte_size, true);
			    if (info.packed_bitfield) {
				    auto packed = ConstantU32(state, 0);
				    for (uint32_t component = 0; component < info.component_count; ++component) {
					    packed = EmitNative<spv::OpBitFieldInsert, IR::Type::U32>(
					        state, packed,
					        EncodeFormattedStoreComponent(ctx, info, component, value(component), 32u),
					        ConstantU32(state, info.component_bit_offset[component]),
					        ConstantU32(state, info.component_bits[component]));
				    }
				    StoreIndirectFormattedPiece(ctx, buffer.address, in_bounds, info.byte_size * 8u,
				                                packed);
				    return ConstantU32(state, 0);
			    }
			    for (uint32_t component = 0; component < info.component_count; ++component) {
				    const auto bits = info.component_bits[component];
				    if (bits != 8u && bits != 16u && bits != 32u) continue;
				    const auto offset  = Format::GetFormatComponentByteOffset(info, component);
				    const auto address = offset == 0u
				                             ? buffer.address
				                             : Binary(state, spv::OpIAdd, TypeU64(state),
				                                      buffer.address, ConstantU64(state, offset));
				    StoreIndirectFormattedPiece(
				        ctx, address, in_bounds, bits,
				        EncodeFormattedStoreComponent(ctx, info, component, value(component), 32u));
			    }
			    return ConstantU32(state, 0);
		    });
	});
}

// KYTY_BDA_WRITES: a BUFFER_ATOMIC_* through a V# the shader computed. The whole payload is
// range-checked at once (ISA 8.1.5 note 3); the address ignores the LSBs below the payload size, as
// the bound-buffer atomics index DWORDs (or DWORD pairs). `operation(pointer)` gets a
// PhysicalStorageBuffer pointer to a u32 (or u64 when `wide`) and returns the pre-op value in the
// IR's representation (U32, or U64 as a u32 pair).
template <typename Fn>
uint32_t EmitIndirectAtomic(ValueEmitContext& ctx, const IR::Inst& inst, bool wide,
                            Fn&& operation) {
	auto&      state = ctx.state;
	const auto type  = wide ? TypeU64(state) : TypeU32(state);
	const auto zero  = wide ? ConstantU64(state, 0) : ConstantU32(state, 0);
	return EmitValueOrDefaultIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), type, zero, [&]() {
		const auto access  = IndirectBufferElementAccess(ctx, inst, 0u, wide ? 8u : 4u);
		const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), access.guest,
		                            ConstantDeviceAddress(state, wide ? ~uint64_t {7} : ~uint64_t {3}));
		return WriteBdaIfMapped(ctx, aligned, access.in_bounds, type, zero, [&](uint32_t host) {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(
			    spv::OpConvertUToPtr,
			    wide ? TypePhysicalU64Pointer(state) : TypePhysicalU32Pointer(state), pointer, host);
			return operation(pointer);
		});
	});
}

// The formats a formatted load through a V# the shader computed decodes (the vertex formats of
// 8, 16 and 32 bits per component, packed 10/11-bit ones): every other format reads 0.
static bool IndirectFormatDecoded(Prospero::BufferFormat format) {
	using F = Prospero::BufferFormat;
	// KYTY_INDIRECT_FORMAT_ALL: every buffer format the bound path decodes (ISA 8.1.4 lists no
	// restriction for computed V#s).
	if (GetCodegenOptions().indirect_format_all) {
		const auto info = Format::GetFormatInfo(format);
		return info.type != Format::ComponentType::Unknown && info.component_count != 0u &&
		       info.component_count <= 4u;
	}
	switch (format) {
		case F::k8UNorm: case F::k8SNorm: case F::k8UInt: case F::k8SInt:
		case F::k16UNorm: case F::k16SNorm: case F::k16UInt: case F::k16SInt: case F::k16Float:
		case F::k8_8UNorm: case F::k8_8SNorm: case F::k8_8UInt: case F::k8_8SInt:
		case F::k32UInt: case F::k32SInt: case F::k32Float:
		case F::k16_16UNorm: case F::k16_16SNorm: case F::k16_16UInt: case F::k16_16SInt:
		case F::k16_16Float: case F::k11_11_10Float: case F::k10_11_11Float:
		case F::k2_10_10_10UNorm: case F::k2_10_10_10SNorm: case F::k10_10_10_2UNorm:
		case F::k10_10_10_2SNorm: case F::k8_8_8_8UNorm: case F::k8_8_8_8SNorm:
		case F::k8_8_8_8UInt: case F::k8_8_8_8SInt: case F::k32_32UInt: case F::k32_32SInt:
		case F::k32_32Float: case F::k16_16_16_16UNorm: case F::k16_16_16_16SNorm:
		case F::k16_16_16_16UInt: case F::k16_16_16_16SInt: case F::k16_16_16_16Float:
		case F::k32_32_32UInt: case F::k32_32_32SInt: case F::k32_32_32Float:
		case F::k32_32_32_32UInt: case F::k32_32_32_32SInt: case F::k32_32_32_32Float: return true;
		default: return false;
	}
}

// The module's decoder of formatted loads through a V# the shader computed (Miles' hit callbacks
// read vertex data so: BUFFER_LOAD_FORMAT_X..XYZW through a material's V#, about 35,000 such
// instructions over the callees). The format, destination selects and bounds come from the V# at
// run time (upstream a55a28380's decode, LoadIndirectFormattedX); one function per module keeps
// each load a call instead of a copy of the format switch.
// Parameters: V# dwords 0-3, the element index (ADD_TID applied), the byte offset (immediate
// added), SOFFSET. Result: the four channels after the destination selects.
static std::vector<uint32_t> IndirectFormatFastSet();
static void DefineIndirectFormatFunctionBody(EmitterState& state, uint32_t function_id, const char* name,
                                             uint32_t function_control, uint32_t function_type,
                                             std::span<const uint32_t> fast, uint32_t slow_function);

// KYTY_INDIRECT_FORMAT_INLINE_HASHES: the programs that inline the full decoder.
static bool IndirectFormatInlineHash(uint64_t hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> parsed;
		const char*           value = std::getenv("KYTY_INDIRECT_FORMAT_INLINE_HASHES");
		if (value == nullptr) return parsed;
		std::string text(value);
		size_t      start = 0;
		while (start < text.size()) {
			const auto end = text.find(',', start);
			parsed.push_back(std::strtoull(text.substr(start, end == std::string::npos ? std::string::npos : end - start).c_str(),
			                               nullptr, 16));
			if (end == std::string::npos) break;
			start = end + 1;
		}
		return parsed;
	}();
	return std::ranges::find(hashes, hash) != hashes.end();
}

// KYTY_INDIRECT_FORMAT_COMPACT: the packed descriptor of a format the compact decoder reads:
// component widths in bits 0-5 / 6-11 / 12-17 / 18-23, numeric kind in bits 24-26 (0 uint,
// 1 sint, 2 unorm, 3 snorm, 4 float), component count in bits 27-29; 0 = a format the decoder
// does not decode (reads 0, as IndirectFormatDecoded). Components lie at the prefix sums of the
// widths (true of every decoded format, packed ones included).
uint32_t IndirectFormatCompactDescriptor(uint32_t format) {
	const auto value = static_cast<Prospero::BufferFormat>(format);
	const auto info  = Format::GetFormatInfo(value);
	if (info.type == Format::ComponentType::Unknown || !IndirectFormatDecoded(value) ||
	    info.component_count == 0u || info.component_count > 4u) {
		return 0;
	}
	uint32_t kind = 0;
	switch (info.type) {
		case Format::ComponentType::Uint: kind = 0; break;
		case Format::ComponentType::Sint: kind = 1; break;
		case Format::ComponentType::Unorm: kind = 2; break;
		case Format::ComponentType::Snorm: kind = 3; break;
		case Format::ComponentType::Float: kind = 4; break;
		// KYTY_INDIRECT_FORMAT_ALL (IndirectFormatDecoded only admits them with it).
		case Format::ComponentType::Uscaled: kind = 5; break;
		case Format::ComponentType::Sscaled: kind = 6; break;
		default: return 0;
	}
	uint32_t descriptor = (kind << 24u) | (info.component_count << 27u);
	uint32_t offset     = 0;
	for (uint32_t component = 0; component < info.component_count; ++component) {
		const auto bits = info.component_bits[component];
		if (bits == 0u || bits > 32u || info.component_bit_offset[component] != offset) {
			return 0;
		}
		descriptor |= bits << (6u * component);
		offset += bits;
	}
	if (offset != info.byte_size * 8u) {
		return 0;
	}
	return descriptor;
}

// KYTY_INDIRECT_FORMAT_COMPACT: one decoder for every format, small enough to inline at each of
// the ~35,000 call sites of Miles' hit kernels. Results equal the per-format switch's.
static void DefineIndirectFormatCompactBody(EmitterState& state, uint32_t function_id,
                                            uint32_t function_type) {
	const auto u32    = TypeU32(state);
	const auto result = TypeU32Vector(state, 4);
	// The descriptor table: a Private array with a constant initializer, never written.
	constexpr uint32_t TableSize = 128u;
	std::vector<uint32_t> entries(TableSize);
	for (uint32_t format = 0; format < TableSize; ++format) {
		entries[format] = ConstantU32(state, IndirectFormatCompactDescriptor(format));
	}
	const auto table_type = state.builder.Type(spv::OpTypeArray, u32, ConstantU32(state, TableSize));
	const auto table_init = state.builder.Constant(spv::OpConstantComposite, table_type, entries);
	// Diagnostics (Miles: the compact decoder alone reads zeros in game, with the full decoder in the
	// same module it does not): KYTY_INDIRECT_FORMAT_COMPACT_TABLE=function keeps the table in a
	// Function variable of the decoder instead of a Private global; KYTY_INDIRECT_FORMAT_COMPACT_CONTROL=
	// inline|dontinline sets the decoder's function control.
	const char* table_mode   = std::getenv("KYTY_INDIRECT_FORMAT_COMPACT_TABLE");
	const bool  function_table = table_mode != nullptr && std::strcmp(table_mode, "function") == 0;
	const char* control_mode = std::getenv("KYTY_INDIRECT_FORMAT_COMPACT_CONTROL");
	const auto  control =
	    control_mode == nullptr                       ? spv::FunctionControlMaskNone
	    : std::strcmp(control_mode, "inline") == 0     ? spv::FunctionControlInlineMask
	    : std::strcmp(control_mode, "dontinline") == 0 ? spv::FunctionControlDontInlineMask
	                                                   : spv::FunctionControlMaskNone;
	const auto table_class = function_table ? spv::StorageClassFunction : spv::StorageClassPrivate;
	uint32_t   table       = 0;
	if (!function_table) {
		table = state.builder.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, table_type),
		                                           spv::StorageClassPrivate, table_init);
		state.builder.AddName(table, "indirect_format_table");
	}

	state.builder.AddName(function_id, "indirect_formatted_load_compact");
	if (control_mode == nullptr) {
		state.indirect_format_compact_function = function_id;
	}
	state.builder.AddFunction(spv::OpFunction, result, function_id, control, function_type);
	std::array<uint32_t, 7> parameters {};
	for (auto& id: parameters) {
		id = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionParameter, u32, id);
	}
	EmitLabel(state, state.builder.AllocateId());
	if (function_table) {
		table = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, table_type), table,
		                          spv::StorageClassFunction, table_init);
		state.builder.AddName(table, "indirect_format_table");
	}
	ValueEmitContext ctx(state);
	const auto [word0, word1, records, word3, index, voffset, soffset] = parameters;
	const auto constant = [&](uint32_t value) { return ConstantU32(state, value); };
	const auto field = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, constant(first), constant(count));
	};
	const auto nonzero = [&](uint32_t value) {
		return Binary(state, spv::OpINotEqual, TypeBool(state), value, constant(0));
	};
	const auto equal = [&](uint32_t lhs, uint32_t rhs) {
		return Binary(state, spv::OpIEqual, TypeBool(state), lhs, rhs);
	};
	const auto stride  = field(word1, 16, 14);
	const auto swizzle = AndCondition(state, nonzero(stride), nonzero(field(word1, 31, 1)));
	const auto address = CalculateBufferAddress(state, index, voffset, soffset, 0u, stride, swizzle,
	                                            field(word3, 21, 2));
	const auto base        = PackU64(state, word0, field(word1, 0, 16));
	const auto raw_records = Binary(state, spv::OpISub, u32, records, soffset);
	const IndirectBufferAccess buffer {
	    .address          = Binary(state, spv::OpIAdd, TypeU64(state), base,
	                               Unary(state, spv::OpUConvert, TypeU64(state), address.byte)),
	    .offset           = address.offset,
	    .stride           = stride,
	    .swizzle          = swizzle,
	    .format           = field(word3, 12, 7),
	    .selector         = 0u,
	    .mode             = field(word3, 28, 2),
	    .records          = records,
	    .index_in_bounds  = Binary(state, spv::OpULessThan, TypeBool(state), index, records),
	    .scalar_in_bounds = Binary(state, spv::OpULessThanEqual, TypeBool(state), soffset, records),
	    .raw_records      = raw_records,
	    .raw_index_in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, raw_records),
	};
	// Descriptor of the format.
	const auto entry_pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, table_class, u32),
	                          entry_pointer, table, buffer.format);
	const auto descriptor = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, u32, descriptor, entry_pointer);
	std::array<uint32_t, 4> widths {};
	for (uint32_t component = 0; component < 4u; ++component) {
		widths[component] = field(descriptor, 6u * component, 6u);
	}
	const auto kind  = field(descriptor, 24, 3);
	const auto count = field(descriptor, 27, 3);
	const auto valid = nonzero(descriptor);
	const auto total = Binary(state, spv::OpIAdd, u32,
	                          Binary(state, spv::OpIAdd, u32, widths[0], widths[1]),
	                          Binary(state, spv::OpIAdd, u32, widths[2], widths[3]));
	const auto bytes = Binary(state, spv::OpShiftRightLogical, u32, total, constant(3));
	const auto wide  = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), bytes, constant(4));
	// The element: aligned to min(4, size) bytes, read as `dwords` dwords from `first`.
	const auto align_mask = Select(state, u32, wide, constant(3),
	                               Binary(state, spv::OpISub, u32, bytes, constant(1)));
	const auto element = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
	                            Unary(state, spv::OpNot, TypeU64(state),
	                                  Unary(state, spv::OpUConvert, TypeU64(state), align_mask)));
	const auto first = Binary(state, spv::OpBitwiseAnd, TypeU64(state), element,
	                          ConstantU64(state, ~uint64_t {3}));
	const auto shift = Binary(state, spv::OpShiftLeftLogical, u32,
	                          Binary(state, spv::OpBitwiseAnd, u32,
	                                 Unary(state, spv::OpUConvert, u32, element), constant(3)),
	                          constant(3));
	const auto dwords = Select(state, u32, wide,
	                           Binary(state, spv::OpShiftRightLogical, u32, bytes, constant(2)), constant(1));
	const auto in_bounds = AndCondition(state, valid, IndirectBufferInBoundsBytes(state, buffer, 0u, bytes, true));
	const auto loaded = EmitValueOrDefaultIfCondition(state, in_bounds, result, ConstantU32CompositeZero(state, 4), [&]() {
		// Two page-table lookups (first and last dword): 16 bytes cross at most one page boundary.
		const auto last = Binary(state, spv::OpIAdd, TypeU64(state), first,
		                         Unary(state, spv::OpUConvert, TypeU64(state),
		                               Binary(state, spv::OpShiftLeftLogical, u32,
		                                      Binary(state, spv::OpISub, u32, dwords, constant(1)),
		                                      constant(2))));
		const auto device_first = GetBdaPointer(ctx, first);
		const auto device_last  = GetBdaPointer(ctx, last);
		const auto page_bits    = ConstantDeviceAddress(state, BufferCache::CACHING_PAGEBITS);
		const auto first_page   = Binary(state, spv::OpShiftRightLogical, TypeU64(state), first, page_bits);
		std::array<uint32_t, 4> values {};
		for (uint32_t dword = 0; dword < 4u; ++dword) {
			const auto guest = dword == 0u ? first
			                               : Binary(state, spv::OpIAdd, TypeU64(state), first,
			                                        ConstantU64(state, dword * 4u));
			const auto later_page = Binary(state, spv::OpINotEqual, TypeBool(state),
			                               Binary(state, spv::OpShiftRightLogical, TypeU64(state), guest, page_bits),
			                               first_page);
			const auto page_base = Select(state, TypeU64(state), later_page, device_last, device_first);
			const auto device    = Select(state, TypeU64(state), later_page,
			                              Binary(state, spv::OpIAdd, TypeU64(state), device_last,
			                                     Binary(state, spv::OpISub, TypeU64(state), guest, last)),
			                              dword == 0u ? device_first
			                                          : Binary(state, spv::OpIAdd, TypeU64(state), device_first,
			                                                   ConstantU64(state, dword * 4u)));
			const auto active = AndCondition(
			    state, Binary(state, spv::OpULessThan, TypeBool(state), constant(dword), dwords),
			    Binary(state, spv::OpINotEqual, TypeBool(state), page_base, ConstantU64(state, 0)));
			values[dword] = EmitValueOrZeroIfCondition(state, active, [&]() {
				const auto pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, device);
				const auto value = state.builder.AllocateId();
				const auto access = GetCodegenOptions().volatile_loads
				                        ? spv::MemoryAccessAlignedMask | spv::MemoryAccessVolatileMask
				                        : spv::MemoryAccessAlignedMask;
				state.builder.AddFunction(spv::OpLoad, u32, value, pointer, access, 4u);
				return value;
			});
		}
		return ConstructU32Composite(state, 4, values);
	});
	std::array<uint32_t, 4> words {};
	for (uint32_t dword = 0; dword < 4u; ++dword) {
		words[dword] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, u32, words[dword], loaded, dword);
	}
	// Components: bits [offset, offset + width) of the element, converted by the kind.
	const auto f32       = TypeF32(state);
	const auto i32       = TypeI32(state);
	const auto is_kind   = [&](uint32_t value) { return equal(kind, constant(value)); };
	std::array<uint32_t, 4> components {};
	auto offset = shift;
	for (uint32_t component = 0; component < 4u; ++component) {
		const auto width = widths[component];
		const auto which = Binary(state, spv::OpShiftRightLogical, u32, offset, constant(5));
		const auto bit   = Binary(state, spv::OpBitwiseAnd, u32, offset, constant(31));
		auto word = words[0];
		for (uint32_t dword = 1; dword < 4u; ++dword) {
			word = Select(state, u32, equal(which, constant(dword)), words[dword], word);
		}
		const auto unsigned_value = EmitBitFieldUExtract(state, word, bit, width);
		const auto signed_value   = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitFieldSExtract, i32, signed_value,
		                          Unary(state, spv::OpBitcast, i32, word), bit, width);
		// unorm: x / (2^w - 1); snorm: max(x / (2^(w-1) - 1), -1) (w >= 2 for those kinds).
		const auto small_width = Select(state, u32,
		                                Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), width, constant(31)),
		                                constant(31), Select(state, u32, equal(width, constant(0)), constant(1), width));
		const auto unorm_max = Unary(state, spv::OpConvertUToF, f32,
		                             Binary(state, spv::OpISub, u32,
		                                    Binary(state, spv::OpShiftLeftLogical, u32, constant(1), small_width),
		                                    constant(1)));
		const auto snorm_max = Unary(state, spv::OpConvertUToF, f32,
		                             Binary(state, spv::OpISub, u32,
		                                    Binary(state, spv::OpShiftLeftLogical, u32, constant(1),
		                                           Binary(state, spv::OpISub, u32, small_width, constant(1))),
		                                    constant(1)));
		const auto unorm = Binary(state, spv::OpFDiv, f32, Unary(state, spv::OpConvertUToF, f32, unsigned_value),
		                          unorm_max);
		const auto snorm_div = Binary(state, spv::OpFDiv, f32,
		                              Unary(state, spv::OpConvertSToF, f32, signed_value), snorm_max);
		const auto snorm     = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpExtInst, f32, snorm, GlslStd450(state), GLSLstd450FMax, snorm_div,
		                          ConstantF32Value(state, -1.0f));
		// float: 32 bits raw; 16 / 11 / 10 bits as a half (the 11/10-bit unsigned floats share the
		// half's exponent: their mantissa shifted up by 4 / 5 bits is the same value).
		const auto half_shift = Select(state, u32, equal(width, constant(16)), constant(0),
		                               Binary(state, spv::OpISub, u32, constant(15), small_width));
		const auto half = EmitF16BitsToF32(
		    state, Binary(state, spv::OpShiftLeftLogical, u32, unsigned_value,
		                  Select(state, u32, Binary(state, spv::OpULessThan, TypeBool(state), width, constant(16)),
		                         half_shift, constant(0))));
		const auto float_bits = Select(state, u32, equal(width, constant(32)), unsigned_value,
		                               Unary(state, spv::OpBitcast, u32, half));
		auto value = Select(state, u32, is_kind(1), Unary(state, spv::OpBitcast, u32, signed_value), unsigned_value);
		value      = Select(state, u32, is_kind(2), Unary(state, spv::OpBitcast, u32, unorm), value);
		value      = Select(state, u32, is_kind(3), Unary(state, spv::OpBitcast, u32, snorm), value);
		value      = Select(state, u32, is_kind(4), float_bits, value);
		if (GetCodegenOptions().indirect_format_all) {
			// uscaled / sscaled: the integer as a float.
			value = Select(state, u32, is_kind(5),
			               Unary(state, spv::OpBitcast, u32,
			                     Unary(state, spv::OpConvertUToF, f32, unsigned_value)),
			               value);
			value = Select(state, u32, is_kind(6),
			               Unary(state, spv::OpBitcast, u32,
			                     Unary(state, spv::OpConvertSToF, f32, signed_value)),
			               value);
		}
		components[component] = value;
		offset = Binary(state, spv::OpIAdd, u32, offset, width);
	}
	const auto one = Select(state, u32, Binary(state, spv::OpULessThan, TypeBool(state), kind, constant(2)),
	                        constant(1), constant(0x3f800000u));
	const auto safe_count = Select(state, u32, equal(count, constant(0)), constant(1), count);
	std::array<uint32_t, 4> selects {};
	for (uint32_t channel = 0; channel < 4u; ++channel) {
		selects[channel] = field(word3, 3u * channel, 3u);
	}
	std::array<uint32_t, 4> channels {};
	for (uint32_t channel = 0; channel < 4u; ++channel) {
		const auto sel   = selects[channel];
		const auto which = Binary(state, spv::OpUMod, u32, Binary(state, spv::OpISub, u32, sel, constant(4)),
		                          safe_count);
		auto picked = components[0];
		for (uint32_t component = 1; component < 4u; ++component) {
			picked = Select(state, u32, equal(which, constant(component)), components[component], picked);
		}
		if (GetCodegenOptions().format_missing_canonical) {
			// KYTY_FORMAT_MISSING_CANON: a channel past the format's reads 0, or 1 for W.
			const auto past = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state),
			                         Binary(state, spv::OpISub, u32, sel, constant(4)), safe_count);
			picked = Select(state, u32, past,
			                Select(state, u32, equal(sel, constant(7)), one, constant(0)), picked);
		}
		const auto fixed = Select(state, u32, equal(sel, constant(1)), one, constant(0));
		const auto value = Select(state, u32,
		                          Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), sel, constant(4)),
		                          picked, fixed);
		channels[channel] = Select(state, u32, valid, value, constant(0));
	}
	state.builder.AddFunction(spv::OpReturnValue, ConstructU32Composite(state, 4, channels));
	state.builder.AddFunction(spv::OpFunctionEnd);
}

// KYTY_INDIRECT_FORMAT_COMPARE=1 (diagnostic, with KYTY_INDIRECT_FORMAT_COUNT=1): every formatted load
// through a computed V# calls both the compact decoder and the full one and returns the full one's
// channels; a mismatch adds 1 to the format's counter and the first one stores its arguments and
// both results (BufferCache logs them every 600 flips).
static bool IndirectFormatCompareEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_INDIRECT_FORMAT_COMPARE");
		return value != nullptr && (value[0] == '1' || value[0] == '2');
	}();
	return enabled && Spirv::IndirectFormatCountAddress() != 0;
}

static void DefineIndirectFormatCompareBody(EmitterState& state, uint32_t function_id, uint32_t function_type,
                                            uint32_t compact, uint32_t full) {
	const auto u32    = TypeU32(state);
	const auto result = TypeU32Vector(state, 4);
	state.builder.AddName(function_id, "indirect_formatted_load_compare");
	state.builder.AddFunction(spv::OpFunction, result, function_id, spv::FunctionControlMaskNone, function_type);
	std::array<uint32_t, 7> parameters {};
	for (auto& id: parameters) {
		id = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionParameter, u32, id);
	}
	EmitLabel(state, state.builder.AllocateId());
	const auto call = [&](uint32_t function) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionCall, result, value, function,
		                          std::span<const uint32_t>(parameters));
		return value;
	};
	const auto a     = call(compact);
	const auto b     = call(full);
	const auto equal = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBoolVector(state, 4), equal, a, b);
	const auto all = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAll, TypeBool(state), all, equal);
	const auto mismatch = Unary(state, spv::OpLogicalNot, TypeBool(state), all);
	const auto counters = Spirv::IndirectFormatCountAddress();
	const auto pointer_at = [&](uint32_t byte_offset_id) {
		const auto address = Binary(state, spv::OpIAdd, TypeU64(state), ConstantU64(state, counters),
		                            Unary(state, spv::OpUConvert, TypeU64(state), byte_offset_id));
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, address);
		return pointer;
	};
	(void)EmitValueOrZeroIfCondition(state, mismatch, [&]() {
		const auto format = EmitBitFieldUExtract(state, parameters[3], ConstantU32(state, 12), ConstantU32(state, 7));
		const auto added  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicIAdd, u32, added,
		                          pointer_at(Binary(state, spv::OpShiftLeftLogical, u32, format, ConstantU32(state, 2))),
		                          ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone),
		                          ConstantU32(state, 1));
		const auto old = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicCompareExchange, u32, old, pointer_at(ConstantU32(state, 128u * 4u)),
		                          ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone), ConstantU32(state, 1),
		                          ConstantU32(state, 0));
		(void)EmitValueOrZeroIfCondition(state, Binary(state, spv::OpIEqual, TypeBool(state), old, ConstantU32(state, 0)),
		                                 [&]() {
			std::array<uint32_t, 15> words {};
			for (uint32_t i = 0; i < 7u; ++i) words[i] = parameters[i];
			for (uint32_t i = 0; i < 4u; ++i) {
				words[7u + i] = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, u32, words[7u + i], a, i);
				words[11u + i] = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, u32, words[11u + i], b, i);
			}
			for (uint32_t i = 0; i < 15u; ++i) {
				state.builder.AddFunction(spv::OpStore, pointer_at(ConstantU32(state, (129u + i) * 4u)), words[i],
				                          spv::MemoryAccessAlignedMask, 4u);
			}
			return ConstantU32(state, 0);
		});
		return ConstantU32(state, 0);
	});
	// KYTY_INDIRECT_FORMAT_COMPARE=2: the compact decoder's channels are returned instead.
	const char* mode = std::getenv("KYTY_INDIRECT_FORMAT_COMPARE");
	state.builder.AddFunction(spv::OpReturnValue, mode != nullptr && mode[0] == '2' ? a : b);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

void DefineIndirectFormatFunctionImpl(EmitterState& state) {
	bool used = false;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if ((op == IR::ValueOpcode::LoadBufferU32 || op == IR::ValueOpcode::LoadBufferU32x2 ||
			     op == IR::ValueOpcode::LoadBufferU32x3 || op == IR::ValueOpcode::LoadBufferU32x4)) {
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index < state.program.memory_info.size() &&
				    state.program.memory_info[index].kind == IR::ResourceKind::IndirectBuffer &&
				    state.program.memory_info[index].formatted) {
					used = true;
				}
			}
		}
	}
	if (!used) {
		return;
	}
	const auto u32    = TypeU32(state);
	const auto result = TypeU32Vector(state, 4);
	const std::array<uint32_t, 7> parameter_types {u32, u32, u32, u32, u32, u32, u32};
	const auto function_type =
	    state.builder.Type(spv::OpTypeFunction, result, std::span<const uint32_t>(parameter_types));
	const auto& options = GetCodegenOptions();
	if (IndirectFormatCompareEnabled()) {
		const auto compact = state.builder.AllocateId();
		DefineIndirectFormatCompactBody(state, compact, function_type);
		const auto full = state.builder.AllocateId();
		DefineIndirectFormatFunctionBody(state, full, "indirect_formatted_load", spv::FunctionControlMaskNone,
		                                 function_type, {}, 0u);
		state.indirect_format_function = state.builder.AllocateId();
		DefineIndirectFormatCompareBody(state, state.indirect_format_function, function_type, compact, full);
		return;
	}
	if (options.indirect_format_compact) {
		state.indirect_format_function = state.builder.AllocateId();
		DefineIndirectFormatCompactBody(state, state.indirect_format_function, function_type);
		return;
	}
	// KYTY_INDIRECT_FORMAT_FAST: the full decoder stays a DontInline function for the rare formats;
	// the loads call a small ordinary function (the driver inlines it) that decodes the common
	// formats itself and calls the full one only for the others.
	const auto fast_formats = IndirectFormatFastSet();
	const bool inline_this = options.indirect_format_inline || IndirectFormatInlineHash(state.program.shader_hash);
	if (options.indirect_format_fast && !inline_this && !fast_formats.empty()) {
		const auto slow = state.builder.AllocateId();
		DefineIndirectFormatFunctionBody(state, slow, "indirect_formatted_load",
		                                 spv::FunctionControlDontInlineMask, function_type, {}, 0u);
		state.indirect_format_function = state.builder.AllocateId();
		DefineIndirectFormatFunctionBody(state, state.indirect_format_function, "indirect_formatted_load_fast",
		                                 spv::FunctionControlMaskNone, function_type, fast_formats, slow);
		return;
	}
	state.indirect_format_function = state.builder.AllocateId();
	DefineIndirectFormatFunctionBody(state, state.indirect_format_function, "indirect_formatted_load",
	                                 inline_this ? spv::FunctionControlMaskNone
	                                             : spv::FunctionControlDontInlineMask,
	                                 function_type, {}, 0u);
}

// A "g" item in KYTY_INDIRECT_FORMAT_FAST (e.g. g,20,22,64,71,74,77): the listed formats made of
// 32-bit components (32 / 32_32 / 32_32_32 / 32_32_32_32, uint or float) share one decode block that
// loads as many dwords as the format has, instead of one block each. The inlined decoder of every
// call site gets smaller (Miles' hit kernels: ~35,000 call sites; driver compile memory).
static bool IndirectFormatGroup32() {
	static const bool group = [] {
		const char* value = std::getenv("KYTY_INDIRECT_FORMAT_FAST");
		if (value == nullptr) return false;
		std::string text(value);
		size_t      start = 0;
		while (start <= text.size()) {
			const auto end  = text.find(',', start);
			const auto item = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
			if (item == "g") return true;
			if (end == std::string::npos) break;
			start = end + 1;
		}
		return false;
	}();
	return group;
}

// The 32-bit-component formats IndirectFormatGroup32 groups: component count, float or not.
static bool IndirectFormatRaw32(uint32_t format, uint32_t* count = nullptr, bool* is_float = nullptr) {
	uint32_t n = 0;
	bool     f = false;
	switch (static_cast<Prospero::BufferFormat>(format)) {
		case Prospero::BufferFormat::k32UInt: n = 1; break;
		case Prospero::BufferFormat::k32Float: n = 1; f = true; break;
		case Prospero::BufferFormat::k32_32UInt: n = 2; break;
		case Prospero::BufferFormat::k32_32Float: n = 2; f = true; break;
		case Prospero::BufferFormat::k32_32_32UInt: n = 3; break;
		case Prospero::BufferFormat::k32_32_32Float: n = 3; f = true; break;
		case Prospero::BufferFormat::k32_32_32_32UInt: n = 4; break;
		case Prospero::BufferFormat::k32_32_32_32Float: n = 4; f = true; break;
		default: return false;
	}
	if (count != nullptr) *count = n;
	if (is_float != nullptr) *is_float = f;
	return true;
}

// KYTY_INDIRECT_FORMAT_FAST=1 (default formats) or =<format>[,<format>...] (decimal BufferFormat
// values): the formats indirect_formatted_load_fast decodes inline.
static std::vector<uint32_t> IndirectFormatFastSet() {
	static const std::vector<uint32_t> formats = [] {
		std::vector<uint32_t> parsed;
		const char*           value = std::getenv("KYTY_INDIRECT_FORMAT_FAST");
		if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) return parsed;
		if (std::strcmp(value, "1") == 0) {
			// 32, 32_32, 32_32_32, 32_32_32_32 (uint, float), 16_16 / 16_16_16_16 (float, unorm,
			// snorm, uint), 8_8_8_8 (unorm, snorm, uint), 16 uint.
			parsed = {11, 20, 22, 23, 24, 27, 29, 56, 57, 60, 62, 64, 65, 66, 69, 71, 72, 74, 75, 77};
			return parsed;
		}
		std::string text(value);
		size_t      start = 0;
		while (start < text.size()) {
			const auto end = text.find(',', start);
			const auto item = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
			const auto number = std::strtoul(item.c_str(), nullptr, 10);
			if (number > 0 && number <= static_cast<unsigned long>(Prospero::BufferFormat::k32_32_32_32Float)) {
				parsed.push_back(static_cast<uint32_t>(number));
			}
			if (end == std::string::npos) break;
			start = end + 1;
		}
		std::ranges::sort(parsed);
		parsed.erase(std::unique(parsed.begin(), parsed.end()), parsed.end());
		return parsed;
	}();
	return formats;
}

// One decoder function: all formats (fast empty) or only the fast ones, the others through a call
// of slow_function with the same parameters.
static void DefineIndirectFormatFunctionBody(EmitterState& state, uint32_t function_id, const char* name,
                                      uint32_t function_control, uint32_t function_type,
                                      std::span<const uint32_t> fast, uint32_t slow_function) {
	const auto u32    = TypeU32(state);
	const auto result = TypeU32Vector(state, 4);
	state.builder.AddName(function_id, name);
	state.builder.AddFunction(spv::OpFunction, result, function_id, function_control, function_type);
	std::array<uint32_t, 7> parameters {};
	for (auto& id: parameters) {
		id = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionParameter, u32, id);
	}
	EmitLabel(state, state.builder.AllocateId());
	ValueEmitContext ctx(state);
	const auto [word0, word1, records, word3, index, voffset, soffset] = parameters;
	const auto field = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
	};
	const auto nonzero = [&](uint32_t value) {
		return Binary(state, spv::OpINotEqual, TypeBool(state), value, ConstantU32(state, 0));
	};
	const auto stride  = field(word1, 16, 14);
	const auto swizzle = AndCondition(state, nonzero(stride), nonzero(field(word1, 31, 1)));
	const auto address = CalculateBufferAddress(state, index, voffset, soffset, 0u, stride, swizzle,
	                                            field(word3, 21, 2));
	const auto base        = PackU64(state, word0, field(word1, 0, 16));
	const auto raw_records = Binary(state, spv::OpISub, TypeU32(state), records, soffset);
	const IndirectBufferAccess buffer {
	    .address          = Binary(state, spv::OpIAdd, TypeU64(state), base,
	                               Unary(state, spv::OpUConvert, TypeU64(state), address.byte)),
	    .offset           = address.offset,
	    .stride           = stride,
	    .swizzle          = swizzle,
	    .format           = field(word3, 12, 7),
	    .selector         = 0u,
	    .mode             = field(word3, 28, 2),
	    .records          = records,
	    .index_in_bounds  = Binary(state, spv::OpULessThan, TypeBool(state), index, records),
	    .scalar_in_bounds = Binary(state, spv::OpULessThanEqual, TypeBool(state), soffset, records),
	    .raw_records      = raw_records,
	    .raw_index_in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, raw_records),
	};
	std::array<uint32_t, 4> selects {};
	for (uint32_t channel = 0; channel < 4u; ++channel) {
		selects[channel] = field(word3, 3u * channel, 3u);
	}
	// KYTY_INDIRECT_FORMAT_COUNT: count the calls of the DontInline full decoder per format.
	if (const auto counters = Spirv::IndirectFormatCountAddress();
	    counters != 0 && fast.empty() && function_control == spv::FunctionControlDontInlineMask) {
		const auto offset  = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		                            Binary(state, spv::OpBitwiseAnd, TypeU32(state), buffer.format,
		                                   ConstantU32(state, 127)),
		                            ConstantU32(state, 2));
		const auto address = Binary(state, spv::OpIAdd, TypeU64(state), ConstantU64(state, counters),
		                            Unary(state, spv::OpUConvert, TypeU64(state), offset));
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, address);
		const auto added = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), added, pointer,
		                          ConstantU32(state, spv::ScopeDevice),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone), ConstantU32(state, 1));
	}
	const auto decode_format = [&](uint32_t format) {
		    const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
		    if (info.type == Format::ComponentType::Unknown ||
		        !IndirectFormatDecoded(static_cast<Prospero::BufferFormat>(format))) {
			    return ConstantU32CompositeZero(state, 4);
		    }
		    const auto in_bounds = IndirectBufferInBounds(state, buffer, 0u, info.byte_size, true);
		    const auto element   = Binary(
                state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
                ConstantU64(state, ~(uint64_t(std::min(4u, info.byte_size)) - 1u)));
		    const auto read = [&](uint32_t component, uint32_t bits) {
			    const auto offset  = Format::GetFormatComponentByteOffset(info, component);
			    const auto address = offset == 0u ? element
			                                      : Binary(state, spv::OpIAdd, TypeU64(state), element,
			                                               ConstantU64(state, offset));
			    return LoadBda(ctx, address, in_bounds, bits);
		    };
		    std::array<uint32_t, 4> components {};
		    for (uint32_t component = 0; component < info.component_count; ++component) {
			    components[component] = LoadFormattedComponent(
			        ctx, info, {FormattedSourceKind::Memory, component},
			        [&](uint32_t source) { return read(source, 32u); }, read);
		    }
		    const auto one = FormattedConstant(ctx, info, FormattedSourceKind::One);
		    std::array<uint32_t, 4> channels {};
		    for (uint32_t channel = 0; channel < 4u; ++channel) {
			    // dst_sel: 0 -> 0, 1 -> 1, 4-7 -> X-W (a component past the format's wraps, as
			    // upstream's decode reads it).
			    const auto sel = selects[channel];
			    auto picked    = components[0];
			    if (info.component_count > 1u) {
				    const auto which = Binary(
				        state, spv::OpUMod, TypeU32(state),
				        Binary(state, spv::OpISub, TypeU32(state), sel, ConstantU32(state, 4)),
				        ConstantU32(state, info.component_count));
				    for (uint32_t component = 1; component < info.component_count; ++component) {
					    picked = Select(state, TypeU32(state),
					                    Binary(state, spv::OpIEqual, TypeBool(state), which,
					                           ConstantU32(state, component)),
					                    components[component], picked);
				    }
			    }
			    const auto constant = Select(
			        state, TypeU32(state),
			        Binary(state, spv::OpIEqual, TypeBool(state), sel, ConstantU32(state, 1)), one,
			        ConstantU32(state, 0));
			    if (GetCodegenOptions().format_missing_canonical && info.component_count < 4u) {
				    // KYTY_FORMAT_MISSING_CANON: a channel past the format's reads 0, or 1 for W.
				    const auto missing = Select(
				        state, TypeU32(state),
				        Binary(state, spv::OpIEqual, TypeBool(state), sel, ConstantU32(state, 7)), one,
				        ConstantU32(state, 0));
				    picked = Select(state, TypeU32(state),
				                    Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), sel,
				                           ConstantU32(state, 4u + info.component_count)),
				                    missing, picked);
			    }
			    channels[channel] = Select(
			        state, TypeU32(state),
			        Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), sel, ConstantU32(state, 4)),
			        picked, constant);
		    }
		    return ConstructU32Composite(state, 4, channels);
	    };
	uint32_t value = 0;
	if (fast.empty()) {
		value = EmitIndexSwitch(state, buffer.format,
		                        static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) + 1u, result,
		                        decode_format);
	} else {
		// Sparse switch: the fast formats inline, the default calls the full decoder.
		const auto            merge_label   = state.builder.AllocateId();
		const auto            default_label = state.builder.AllocateId();
		std::vector<uint32_t> labels(fast.size());
		std::vector<uint32_t> words {spv::OpSwitch, buffer.format, default_label};
		// IndirectFormatGroup32: the 32-bit-component formats branch to one shared block.
		const bool group32     = IndirectFormatGroup32();
		uint32_t   group_label = 0;
		std::vector<uint32_t> grouped;
		for (size_t item = 0; item < fast.size(); item++) {
			if (group32 && IndirectFormatRaw32(fast[item])) {
				if (group_label == 0) group_label = state.builder.AllocateId();
				labels[item] = group_label;
				grouped.push_back(fast[item]);
			} else {
				labels[item] = state.builder.AllocateId();
			}
			words.push_back(fast[item]);
			words.push_back(labels[item]);
		}
		state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
		state.builder.AddFunction(words);
		std::vector<uint32_t> phi {spv::OpPhi, result, state.builder.AllocateId()};
		for (size_t item = 0; item < fast.size(); item++) {
			if (labels[item] == group_label && group_label != 0) continue;
			EmitLabel(state, labels[item]);
			phi.push_back(decode_format(fast[item]));
			phi.push_back(state.current_label);
			state.builder.AddFunction(spv::OpBranch, merge_label);
		}
		if (group_label != 0) {
			EmitLabel(state, group_label);
			// Component count and float-ness of the grouped format at run time.
			auto count    = ConstantU32(state, 0);
			auto is_float = ConstantBool(state, false);
			for (const auto format: grouped) {
				uint32_t n = 0;
				bool     f = false;
				IndirectFormatRaw32(format, &n, &f);
				const auto match = Binary(state, spv::OpIEqual, TypeBool(state), buffer.format,
				                          ConstantU32(state, format));
				count    = Select(state, TypeU32(state), match, ConstantU32(state, n), count);
				is_float = Select(state, TypeBool(state), match, ConstantBool(state, f), is_float);
			}
			// The whole element in bounds, by its byte size (as decode_format checks).
			std::array<uint32_t, 4> whole {};
			for (uint32_t n = 1; n <= 4u; ++n) {
				whole[n - 1] = IndirectBufferInBounds(state, buffer, 0u, 4u * n, true);
			}
			auto in_bounds = whole[0];
			for (uint32_t n = 2; n <= 4u; ++n) {
				in_bounds = Select(state, TypeBool(state),
				                   Binary(state, spv::OpIEqual, TypeBool(state), count, ConstantU32(state, n)),
				                   whole[n - 1], in_bounds);
			}
			const auto element = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
			                            ConstantU64(state, ~uint64_t {3}));
			std::array<uint32_t, 4> components {};
			for (uint32_t component = 0; component < 4u; ++component) {
				const auto address = component == 0u ? element
				                                      : Binary(state, spv::OpIAdd, TypeU64(state), element,
				                                               ConstantU64(state, component * 4u));
				const auto active =
				    component == 0u ? in_bounds
				                    : AndCondition(state, in_bounds,
				                                   Binary(state, spv::OpULessThan, TypeBool(state),
				                                          ConstantU32(state, component), count));
				components[component] = LoadBda(ctx, address, active, 32u);
			}
			const auto one = Select(state, TypeU32(state), is_float, ConstantU32(state, 0x3f800000u),
			                        ConstantU32(state, 1u));
			const auto safe_count =
			    Select(state, TypeU32(state),
			           Binary(state, spv::OpIEqual, TypeBool(state), count, ConstantU32(state, 0)),
			           ConstantU32(state, 1), count);
			std::array<uint32_t, 4> channels {};
			for (uint32_t channel = 0; channel < 4u; ++channel) {
				const auto sel   = selects[channel];
				const auto which = Binary(state, spv::OpUMod, TypeU32(state),
				                          Binary(state, spv::OpISub, TypeU32(state), sel, ConstantU32(state, 4)),
				                          safe_count);
				auto picked = components[0];
				for (uint32_t component = 1; component < 4u; ++component) {
					picked = Select(state, TypeU32(state),
					                Binary(state, spv::OpIEqual, TypeBool(state), which,
					                       ConstantU32(state, component)),
					                components[component], picked);
				}
				const auto constant = Select(
				    state, TypeU32(state),
				    Binary(state, spv::OpIEqual, TypeBool(state), sel, ConstantU32(state, 1)), one,
				    ConstantU32(state, 0));
				channels[channel] = Select(
				    state, TypeU32(state),
				    Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), sel, ConstantU32(state, 4)),
				    picked, constant);
			}
			phi.push_back(ConstructU32Composite(state, 4, channels));
			phi.push_back(state.current_label);
			state.builder.AddFunction(spv::OpBranch, merge_label);
		}
		EmitLabel(state, default_label);
		const auto called = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionCall, result, called, slow_function,
		                          std::span<const uint32_t>(parameters));
		phi.push_back(called);
		phi.push_back(state.current_label);
		state.builder.AddFunction(spv::OpBranch, merge_label);
		EmitLabel(state, merge_label);
		state.builder.AddFunction(phi);
		value = phi[2];
	}
	state.builder.AddFunction(spv::OpReturnValue, value);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

// BUFFER_LOAD_FORMAT_X..XYZW through a V# the shader computed: a call of the module's decoder.
static uint32_t CallIndirectFormatFunction(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state  = ctx.state;
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	EXIT_IF(state.indirect_format_function == 0);
	const auto word3 = ctx.Arg(handle, 3);
	const auto add_tid =
	    Binary(state, spv::OpINotEqual, TypeBool(state),
	           EmitBitFieldUExtract(state, word3, ConstantU32(state, 23), ConstantU32(state, 1)),
	           ConstantU32(state, 0));
	const auto index = Binary(state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 1),
	                          Select(state, TypeU32(state), add_tid, BufferLane(state),
	                                 ConstantU32(state, 0)));
	auto voffset = ctx.Arg(inst, 2);
	if (const auto immediate = ctx.Memory(inst).offset; immediate != 0u) {
		voffset = Binary(state, spv::OpIAdd, TypeU32(state), voffset, ConstantU32(state, immediate));
	}
	const std::array<uint32_t, 7> arguments {ctx.Arg(handle, 0), ctx.Arg(handle, 1),
	                                         ctx.Arg(handle, 2), word3, index, voffset,
	                                         ctx.Arg(inst, 3)};
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 4), result,
	                          state.indirect_format_function, std::span<const uint32_t>(arguments));
	return result;
}

uint32_t LoadIndirectFormattedX(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto channels = CallIndirectFormatFunction(ctx, inst);
		const auto x        = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), x, channels, 0u);
		return x;
	});
}

// KYTY_INDIRECT_FORMAT_D16: the decoded channels of a D16 load packed into `components` dwords, two
// per dword: 16-bit floats (float and normalized formats) or 16-bit integers clamped like
// LoadFormattedPrepared's D16 path (UINT/SINT), chosen by the V#'s format (word 3 bits 12-18).
static uint32_t PackIndirectFormattedD16(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t channels,
                                         uint32_t components) {
	auto&       state  = ctx.state;
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto  count  = std::min<uint32_t>(ctx.Memory(inst).component_count, 4u);
	std::array<uint32_t, 4> uint_mask {};
	std::array<uint32_t, 4> sint_mask {};
	for (uint32_t format = 0; format < 128u; ++format) {
		const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
		if (info.type == Format::ComponentType::Uint) uint_mask[format / 32u] |= 1u << (format % 32u);
		if (info.type == Format::ComponentType::Sint) sint_mask[format / 32u] |= 1u << (format % 32u);
	}
	const auto format = EmitBitFieldUExtract(state, ctx.Arg(handle, 3), ConstantU32(state, 12), ConstantU32(state, 7));
	const auto word   = Binary(state, spv::OpShiftRightLogical, TypeU32(state), format, ConstantU32(state, 5));
	const auto bit    = Binary(state, spv::OpBitwiseAnd, TypeU32(state), format, ConstantU32(state, 31));
	const auto test   = [&](const std::array<uint32_t, 4>& mask) {
		auto selected = ConstantU32(state, mask[3]);
		for (uint32_t i = 3; i-- > 0;) {
			selected = Select(state, TypeU32(state),
			                  Binary(state, spv::OpIEqual, TypeBool(state), word, ConstantU32(state, i)),
			                  ConstantU32(state, mask[i]), selected);
		}
		const auto shifted = Binary(state, spv::OpShiftRightLogical, TypeU32(state), selected, bit);
		return Binary(state, spv::OpINotEqual, TypeBool(state),
		              Binary(state, spv::OpBitwiseAnd, TypeU32(state), shifted, ConstantU32(state, 1)),
		              ConstantU32(state, 0));
	};
	const auto is_uint = test(uint_mask);
	const auto is_sint = test(sint_mask);
	const auto channel = [&](uint32_t index) {
		if (index >= count) return ConstantU32(state, 0u);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, channels, index);
		return value;
	};
	std::array<uint32_t, 4> packed {};
	for (uint32_t out = 0; out < components; ++out) {
		const auto low  = channel(out * 2u);
		const auto high = channel(out * 2u + 1u);
		const auto as_uint = [&](uint32_t value) { return EmitUMin32(state, value, ConstantU32(state, 0xffffu)); };
		const auto as_sint = [&](uint32_t value) {
			return EmitSMin32(state, EmitSMax32(state, value, ConstantU32(state, 0xffff8000u)), ConstantU32(state, 0x7fffu));
		};
		const auto pack_int = [&](uint32_t lo, uint32_t hi) {
			return Binary(state, spv::OpBitwiseOr, TypeU32(state),
			              Binary(state, spv::OpBitwiseAnd, TypeU32(state), lo, ConstantU32(state, 0xffffu)),
			              Binary(state, spv::OpShiftLeftLogical, TypeU32(state), hi, ConstantU32(state, 16u)));
		};
		const auto pair = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair,
		                          EmitBitCastF32U32(state, low), EmitBitCastF32U32(state, high));
		const auto as_half = EmitGlslPackHalf2x16(state, pair);
		packed[out] = Select(state, TypeU32(state), is_uint, pack_int(as_uint(low), as_uint(high)),
		                     Select(state, TypeU32(state), is_sint, pack_int(as_sint(low), as_sint(high)), as_half));
	}
	return ConstructU32Composite(state, components, packed);
}

// BUFFER_LOAD_FORMAT_XY/XYZ/XYZW through a V# the shader computed.
uint32_t LoadIndirectFormatted(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	const auto channels = CallIndirectFormatFunction(ctx, inst);
	if (ctx.Memory(inst).data_bits == 16u) {
		return PackIndirectFormattedD16(ctx, inst, channels, components);
	}
	if (components == 4u) {
		return channels;
	}
	std::array<uint32_t, 4> values {};
	for (uint32_t channel = 0; channel < components; ++channel) {
		values[channel] = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), values[channel],
		                              channels, channel);
	}
	return ConstructU32Composite(ctx.state, components, values);
}

uint32_t LoadIndirectFormatted(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components);

uint32_t LoadWideBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	// KYTY_BRANCHLESS_LOADS: raw dwords of a device-checked storage buffer, each selected by EXEC.
	if (const auto& mem = ctx.Memory(inst);
	    mem.kind != IR::ResourceKind::IndirectBuffer && !mem.formatted &&
	    BranchlessLoadKind(state, mem) == BranchlessLoad::Device) {
		const auto              active   = ctx.Arg(inst, inst.NumArgs() - 1);
		const auto              resource = PrepareMemoryResourceAccess(state, mem);
		std::array<uint32_t, 4> values {};
		for (uint32_t component = 0; component < components; component++) {
			const auto index  = DwordIndex(ctx, inst, RebaseRawComponent(mem, component));
			values[component] = ZeroUnless(state, active, LoadWordInBounds(ctx, resource, index));
		}
		return ConstructU32Composite(state, components, values);
	}
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    const auto mem      = ctx.Memory(inst);
		    if (mem.kind == IR::ResourceKind::IndirectBuffer) {
			    return mem.formatted ? LoadIndirectFormatted(ctx, inst, components)
			                         : LoadIndirectBuffer(ctx, inst, components);
		    }
		    const auto resource = PrepareMemoryResourceAccess(state, mem);
		    const auto info = Format::GetFormatInfo(
		        mem.formatted ? BufferFormat(ctx, mem) : Prospero::BufferFormat::kInvalid);
		    if (info.type != Format::ComponentType::Unknown) {
			    return LoadFormattedPrepared(ctx, inst, mem, resource, info, components);
		    }
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < components; component++) {
			    values[component] =
			        LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource);
		    }
		    return ConstructU32Composite(state, components, values);
	    });
}

void StoreWideBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem       = ctx.Memory(inst);
		const auto resource  = PrepareMemoryResourceAccess(state, mem);
		const auto composite = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto info = Format::GetFormatInfo(
		    mem.formatted ? BufferFormat(ctx, mem) : Prospero::BufferFormat::kInvalid);
		if (info.type != Format::ComponentType::Unknown) {
			StoreFormattedPrepared(ctx, inst, mem, resource, info, composite, components);
			return;
		}
		for (uint32_t component = 0; component < components; component++) {
			const auto data = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), data, composite,
			                          component);
			StoreWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource, data);
		}
	});
}

// KYTY_BUFFER_PAIR_ATOMIC (CodegenOptions::buffer_pair_atomic): a raw GLC/DLC DWORDX2 load or
// store whose two dwords are contiguous, in range and one aligned qword is one 64-bit atomic access
// (an aligned DWORDX2 is a single access on the guest GPU), so another workgroup polling the pair
// never sees half of it: Miles Morales' TLAS builder 0x49d3e4dbb4dd1ac4 publishes the
// {status, prefix} pairs of a decoupled look-back scan so. Any other pair takes the dword path.
// Through a bound V# the u64 view of the buffer is used (the device starts bindings 8-byte
// aligned, GetHostBufferPairAtomics); through a V# the shader computed, the host address of the
// BDA page table (checked for 8-byte alignment at run time).
bool BufferPairAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (!GetCodegenOptions().buffer_pair_atomic || !BufferPairAtomicProgram(ctx.state.program.shader_hash)) {
		return false;
	}
	const auto& mem = ctx.Memory(inst);
	if (!mem.IsCoherentBufferPair(inst.GetOpcode())) {
		return false;
	}
	if (mem.kind == IR::ResourceKind::Buffer) {
		return GetHostBufferPairAtomics() && ctx.state.storage_buffer_u64_variable != 0;
	}
	return ctx.state.bda_pointer_function != 0;
}

struct BufferPairAccess {
	uint32_t             usable = 0; // bool: the pair is one aligned, in-range qword
	bool                 bda    = false;
	uint32_t             guest  = 0; // BDA: U64 guest byte address
	uint32_t             host   = 0; // BDA: U64 host address
	uint32_t             index  = 0; // bound V#: qword index into the u64 view
	MemoryResourceAccess resource;   // bound V#: the u64 view
};

// Whether the pair can be one 64-bit access: exactly the dwords the dword path would access, both
// in range by its rules, at an 8-byte aligned host address.
BufferPairAccess PrepareBufferPair(ValueEmitContext& ctx, const IR::Inst& inst, bool store) {
	auto&            state = ctx.state;
	const auto&      mem   = ctx.Memory(inst);
	BufferPairAccess pair;
	const auto equal = [&](uint32_t lhs, uint32_t rhs) {
		return Binary(state, spv::OpIEqual, TypeBool(state), lhs, rhs);
	};
	const auto qword_aligned = [&](uint32_t address) {
		return equal(Binary(state, spv::OpBitwiseAnd, TypeU64(state), address, ConstantU64(state, 7)),
		             ConstantU64(state, 0));
	};
	if (mem.kind == IR::ResourceKind::Buffer) {
		const auto first  = ByteAddress(ctx, inst, RebaseRawComponent(mem, 0u));
		const auto second = ByteAddress(ctx, inst, RebaseRawComponent(mem, 1u));
		pair.resource     = PrepareStorageBufferResourceAccess(
		        state, mem, state.storage_buffer_u64_variable, TypeStorageBufferPointer(state, 64));
		pair.index =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), first, ConstantU32(state, 3));
		const auto contiguous = equal(
		    second,
		    Binary(state, spv::OpIAdd, TypeU32(state), first, ConstantU32(state, 4)));
		const auto aligned = equal(
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), first, ConstantU32(state, 7)),
		    ConstantU32(state, 0));
		pair.usable = AndCondition(state, AndCondition(state, contiguous, aligned),
		                           EmitMemoryElementInBounds(state, pair.resource, pair.index));
		return pair;
	}
	pair.bda          = true;
	uint32_t in_range = 0;
	if (store) {
		// StoreIndirectBuffer's per-component accesses.
		const auto first  = IndirectBufferElementAccess(ctx, inst, 0u, 4u);
		const auto second = IndirectBufferElementAccess(ctx, inst, 4u, 4u);
		pair.guest        = first.guest;
		in_range          = AndCondition(
		             state, AndCondition(state, first.in_bounds, second.in_bounds),
		             equal(second.guest,
		                   Binary(state, spv::OpIAdd, TypeU64(state), first.guest, ConstantU64(state, 4))));
	} else {
		// LoadIndirectBuffer's.
		const auto buffer = PrepareIndirectBuffer(ctx, inst);
		pair.guest        = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
		                           ConstantU64(state, ~uint64_t {3}));
		in_range          = AndCondition(
		             state,
		             AndCondition(state,
		                          Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
		                                 ConstantU32(state, 0)),
		                          IndirectBufferInBounds(state, buffer, 0u, 4u, false)),
		             IndirectBufferInBounds(state, buffer, 4u, 4u, false));
	}
	pair.host = EmitValueOrDefaultIfCondition(
	    state, AndCondition(state, in_range, qword_aligned(pair.guest)), TypeU64(state),
	    ConstantU64(state, 0), [&]() { return GetBdaPointer(ctx, pair.guest); });
	pair.usable = AndCondition(
	    state, Binary(state, spv::OpINotEqual, TypeBool(state), pair.host, ConstantU64(state, 0)),
	    qword_aligned(pair.host));
	return pair;
}

// The pair's u64 pointer, emitted where the pair is usable: an element of the u64 storage-buffer
// view (Coherent with the program's other views), or a Coherent PhysicalStorageBuffer u64.
uint32_t BufferPairPointer(EmitterState& state, const BufferPairAccess& pair) {
	if (!pair.bda) {
		return EmitStorageBufferElementPointer(state, pair.resource, pair.index,
		                                       TypeStorageBufferElementPointer(state, 64));
	}
	state.builder.RequireCapability(spv::CapabilityInt64Atomics);
	const auto block = state.builder.DecoratedType(
	    spv::OpTypeStruct,
	    {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	     {spv::OpMemberDecorate, {0, spv::DecorationCoherent}},
	     {spv::OpDecorate, {spv::DecorationBlock}}},
	    TypeU64(state));
	const auto base = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToPtr,
	                          TypePointer(state, spv::StorageClassPhysicalStorageBuffer, block), base,
	                          pair.host);
	const auto member = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePhysicalU64Pointer(state), member, base,
	                          ConstantU32(state, 0));
	return member;
}

// Device scope; the load acquires and the store releases buffer memory, so what a workgroup wrote
// before publishing its pair is visible to the workgroup that reads the pair.
constexpr uint32_t PairLoadSemantics =
    static_cast<uint32_t>(spv::MemorySemanticsAcquireMask) |
    static_cast<uint32_t>(spv::MemorySemanticsUniformMemoryMask);
constexpr uint32_t PairStoreSemantics =
    static_cast<uint32_t>(spv::MemorySemanticsReleaseMask) |
    static_cast<uint32_t>(spv::MemorySemanticsUniformMemoryMask);

uint32_t LoadBufferPair(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto type  = TypeU32Composite(state, 2);
	const auto zero  = ConstantU32CompositeZero(state, 2);
	return EmitValueOrDefaultIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), type, zero, [&]() {
		const auto pair  = PrepareBufferPair(ctx, inst, false);
		const auto qword = EmitValueOrDefaultIfCondition(
		    state, pair.usable, TypeU64(state), ConstantU64(state, 0), [&]() {
			    const auto value = state.builder.AllocateId();
			    state.builder.AddFunction(spv::OpAtomicLoad, TypeU64(state), value,
			                              BufferPairPointer(state, pair),
			                              ConstantU32(state, spv::ScopeDevice),
			                              ConstantU32(state, PairLoadSemantics));
			    return value;
		    });
		const auto dwords = EmitValueOrDefaultIfCondition(
		    state, Unary(state, spv::OpLogicalNot, TypeBool(state), pair.usable), type, zero,
		    [&]() {
			    return ctx.Memory(inst).kind == IR::ResourceKind::IndirectBuffer
			               ? LoadIndirectBuffer(ctx, inst, 2u)
			               : LoadWideBuffer(ctx, inst, 2u);
		    });
		std::array<uint32_t, 4> values {};
		for (uint32_t component = 0; component < 2u; component++) {
			const auto dword = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), dword, dwords,
			                          component);
			const auto half = Unary(
			    state, spv::OpUConvert, TypeU32(state),
			    component == 0u ? qword
			                    : Binary(state, spv::OpShiftRightLogical, TypeU64(state), qword,
			                             ConstantU64(state, 32)));
			values[component] = Select(state, TypeU32(state), pair.usable, half, dword);
		}
		return ConstructU32Composite(state, 2u, values);
	});
}

void StoreBufferPair(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto pair = PrepareBufferPair(ctx, inst, true);
		EmitIfCondition(state, pair.usable, [&]() {
			const auto data = ctx.Arg(inst, inst.NumArgs() - 2);
			std::array<uint32_t, 2> halves {};
			for (uint32_t component = 0; component < 2u; component++) {
				halves[component] = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state),
				                          halves[component], data, component);
			}
			if (pair.bda) {
				RecordBdaWrite(state, pair.guest);
			}
			state.builder.AddFunction(spv::OpAtomicStore, BufferPairPointer(state, pair),
			                          ConstantU32(state, spv::ScopeDevice),
			                          ConstantU32(state, PairStoreSemantics),
			                          PackU64(state, halves[0], halves[1]));
		});
		EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), pair.usable), [&]() {
			if (pair.bda) {
				StoreIndirectBuffer(ctx, inst, 2u, 32u);
			} else {
				StoreWideBuffer(ctx, inst, 2u);
			}
		});
	});
}

uint32_t LoadWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	// KYTY_BRANCHLESS_LOADS: each dword reads element 0 where EXEC or its bounds rule it out.
	if (const auto& mem = ctx.Memory(inst); BranchlessLoadKind(state, mem) == BranchlessLoad::SafeIndex) {
		const auto              active   = ctx.Arg(inst, inst.NumArgs() - 1);
		const auto              resource = PrepareMemoryResourceAccess(state, mem);
		const auto              base     = ByteAddress(ctx, inst, mem);
		std::array<uint32_t, 4> values {};
		for (uint32_t component = 0; component < components; component++) {
			const auto address = component == 0u ? base
			                                     : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                              ConstantU32(state, component * 4u));
			const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
			                          ConstantU32(state, 2));
			uint32_t   usable = 0;
			const auto safe   = SafeElementIndex(state, resource, index, active, usable);
			values[component] = ZeroUnless(state, usable, LoadWordInBounds(ctx, resource, safe));
		}
		return ConstructU32Composite(state, components, values);
	}
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    const auto              mem      = ctx.Memory(inst);
		    const auto              resource = PrepareMemoryResourceAccess(state, mem);
		    const auto              base     = ByteAddress(ctx, inst, mem);
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < components; component++) {
			    const auto address   = component == 0u
			                               ? base
			                               : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                        ConstantU32(state, component * 4u));
			    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state),
			                                  address, ConstantU32(state, 2));
			    values[component]    = EmitValueOrZeroIfCondition(
			        state, EmitMemoryElementInBounds(state, resource, index),
			        [&]() { return LoadWordInBounds(ctx, resource, index); });
		    }
		    return ConstructU32Composite(state, components, values);
	    });
}

void StoreWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem      = ctx.Memory(inst);
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto base     = ByteAddress(ctx, inst, mem);
		for (uint32_t component = 0; component < components; component++) {
			const auto address = component == 0u ? base
			                                     : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                              ConstantU32(state, component * 4u));
			const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
			                              ConstantU32(state, 2));
			EmitIfCondition(state, EmitMemoryElementInBounds(state, resource, index),
			                [&]() {
				                StoreWordInBounds(ctx, resource, index, ctx.Arg(inst, component + 1u));
			                });
		}
	});
}

} // namespace

uint32_t EmitDeviceAddressConstant(EmitterState& state, uint64_t value) {
	return ConstantDeviceAddress(state, value);
}

uint32_t EmitBdaDeviceAddress(ValueEmitContext& ctx, uint32_t guest_address) {
	return GetBdaPointer(ctx, guest_address);
}

void DefineGetBdaPointer(EmitterState& state) {
	if (!state.program.info.uses_dma) {
		return;
	}
	const auto type            = TypeU64(state);
	const auto function_type   = state.builder.Type(spv::OpTypeFunction, type, type);
	state.bda_pointer_function = state.builder.AllocateId();
	const auto address         = state.builder.AllocateId();
	const auto entry_label     = state.builder.AllocateId();
	state.builder.AddName(state.bda_pointer_function, "get_bda_pointer");
	state.builder.AddFunction(spv::OpFunction, type, state.bda_pointer_function,
	                          spv::FunctionControlMaskNone, function_type);
	state.builder.AddFunction(spv::OpFunctionParameter, type, address);
	EmitLabel(state, entry_label);
	CountBdaTagEvent(state, 0);

	const auto page64        = Binary(state, spv::OpShiftRightLogical, type, address,
	                                  ConstantDeviceAddress(state, BufferCache::CACHING_PAGEBITS));
	// A guest address past the 40-bit space the page table covers (a garbage pointer, a T# base
	// with high bits set) is unmapped: no page-table or fault-bitmap access out of their buffers,
	// whose robustness behaviour differs between drivers (AMD has reported device loss).
	const auto in_table      = Binary(state, spv::OpULessThan, TypeBool(state), page64,
	                                  ConstantDeviceAddress(state, BufferCache::CACHING_NUMPAGES));
	const auto page          = Select(state, TypeU32(state), in_table,
	                                  Unary(state, spv::OpUConvert, TypeU32(state), page64),
	                                  ConstantU32(state, 0));
	const auto entry_pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state, 64),
	                          entry_pointer, state.bda_pagetable_variable, ConstantU32(state, 0),
	                          page);
	const auto loaded = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, type, loaded, entry_pointer);
	const auto base    = Select(state, type, in_table, loaded, ConstantDeviceAddress(state, 0));
	const auto missing =
	    Binary(state, spv::OpIEqual, TypeBool(state), base, ConstantDeviceAddress(state, 0));
	const auto fault_label     = state.builder.AllocateId();
	const auto available_label = state.builder.AllocateId();
	const auto merge_label     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, missing, fault_label, available_label);

	EmitLabel(state, fault_label);
	EmitIfCondition(state, in_table, [&]() { RecordBdaFault(state, page); });
	CountBdaTagEvent(state, 1);
	const auto fault_exit = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);

	EmitLabel(state, available_label);
	const auto offset    = Binary(state, spv::OpBitwiseAnd, type, address,
	                              ConstantU64(state, BufferCache::CACHING_PAGESIZE - 1));
	auto       available = Binary(state, spv::OpIAdd, type, base, offset);
	if (GetCodegenOptions().bda_access_tags) {
		// KYTY_BDA_ACCESS_TAGS (based on upstream KytyPS5 0ec2cd5d5): a backed page without the
		// accessed tag is read as it is on the GPU and reported once through the fault bitmap; the
		// host then tags it and synchronizes it from then on (bufferCacheTags.cpp).
		const auto tag = Binary(state, spv::OpBitwiseAnd, type, base,
		                        ConstantU64(state, BufferCache::BDA_ACCESSED));
		EmitIfCondition(state,
		                Binary(state, spv::OpIEqual, TypeBool(state), tag, ConstantU64(state, 0)),
		                [&]() {
			                RecordBdaFault(state, page);
			                CountBdaTagEvent(state, 2);
		                });
		available = Binary(state, spv::OpISub, type, available, tag);
		BufferCache::s_tag_hops.programs.fetch_add(1, std::memory_order_relaxed);
	}
	const auto available_exit = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);

	EmitLabel(state, merge_label);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, type, result, ConstantDeviceAddress(state, 0),
	                          fault_exit, available, available_exit);
	state.builder.AddFunction(spv::OpReturnValue, result);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

// KYTY_LDS_ATOMIC_ORDER (adapted from Erlou93/KytyPS5-Yotei M110, EmitAtomic32): DS_ADD_RTN_U32 /
// DS_SUB_RTN_U32 return their old values in lane order, as the hardware does, for the lanes that address
// the same dword (Ghost of Yotei's RTGI BVH builder's tree rotations depend on it; with one atomic per
// invocation in the host's order its parent loops and ray traversal spun). A loop over the distinct
// dword indices of the subgroup's invocations in this region: the lowest invocation of an index adds
// (subtracts) the group's total and every invocation gets the old value plus (minus) the exclusive
// prefix of the lower lanes' values. Every invocation runs every iteration (the exit test is a subgroup
// minimum); only the atomic itself is conditional. Both halves of a lane_count 2 program are emitted one
// after the other, so lanes 0..31 come before 32..63. Not for a split wave (its two host subgroups are
// not ordered against each other) or per-lane LDS slots.
static uint32_t EmitLaneOrderedLdsAdd(ValueEmitContext& ctx, const IR::Inst& inst,
                                      const IR::MemoryInfo& mem, uint32_t pointer, uint32_t index) {
	auto&      s        = ctx.state;
	const bool sub      = inst.GetOpcode() == IR::ValueOpcode::SharedAtomicISub32;
	const auto u32      = TypeU32(s);
	const auto boolean  = TypeBool(s);
	const auto subgroup = ConstantU32(s, spv::ScopeSubgroup);
	const auto value    = ctx.Arg(inst, inst.NumArgs() - 2);
	const auto none     = ConstantU32(s, 0xffffffffu); // no LDS dword index reaches it
	s.builder.RequireCapability(spv::CapabilityGroupNonUniformArithmetic);
	const auto entry  = s.current_label;
	const auto header = s.builder.AllocateId();
	const auto body   = s.builder.AllocateId();
	const auto work   = s.builder.AllocateId();
	const auto cont   = s.builder.AllocateId();
	const auto merge  = s.builder.AllocateId();
	const auto pending      = s.builder.AllocateId();
	const auto old          = s.builder.AllocateId();
	const auto next_pending = s.builder.AllocateId();
	const auto next_old     = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(s, header);
	s.builder.AddFunction(spv::OpPhi, boolean, pending, ConstantBool(s, true), entry, next_pending, cont);
	s.builder.AddFunction(spv::OpPhi, u32, old, ConstantU32(s, 0), entry, next_old, cont);
	s.builder.AddFunction(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
	s.builder.AddFunction(spv::OpBranch, body);
	EmitLabel(s, body);
	// The smallest index still pending in the subgroup (the same in every invocation); none left: done.
	const auto chosen = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpGroupNonUniformUMin, u32, chosen, subgroup, spv::GroupOperationReduce,
	                      Select(s, u32, pending, index, none));
	// A break to the loop merge needs no selection merge.
	s.builder.AddFunction(spv::OpBranchConditional, Binary(s, spv::OpIEqual, boolean, chosen, none), merge,
	                      work);
	EmitLabel(s, work);
	const auto mine   = AndCondition(s, pending, Binary(s, spv::OpIEqual, boolean, index, chosen));
	const auto part   = Select(s, u32, mine, value, ConstantU32(s, 0));
	const auto total  = s.builder.AllocateId();
	const auto prefix = s.builder.AllocateId();
	const auto rank   = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpGroupNonUniformIAdd, u32, total, subgroup, spv::GroupOperationReduce, part);
	s.builder.AddFunction(spv::OpGroupNonUniformIAdd, u32, prefix, subgroup,
	                      spv::GroupOperationExclusiveScan, part);
	s.builder.AddFunction(spv::OpGroupNonUniformIAdd, u32, rank, subgroup, spv::GroupOperationExclusiveScan,
	                      Select(s, u32, mine, ConstantU32(s, 1), ConstantU32(s, 0)));
	const auto leader = AndCondition(s, mine, Binary(s, spv::OpIEqual, boolean, rank, ConstantU32(s, 0)));
	const auto leader_old = EmitValueOrZeroIfCondition(s, leader, [&]() {
		const auto result = s.builder.AllocateId();
		s.builder.AddFunction(sub ? spv::OpAtomicISub : spv::OpAtomicIAdd, u32, result, pointer,
		                      ConstantU32(s, spv::ScopeWorkgroup),
		                      ConstantU32(s, spv::MemorySemanticsMaskNone), total);
		return result;
	});
	// Only the leader contributes a non-zero term: the sum is its old value.
	const auto base = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpGroupNonUniformIAdd, u32, base, subgroup, spv::GroupOperationReduce,
	                      leader_old);
	const auto lane_old = Binary(s, sub ? spv::OpISub : spv::OpIAdd, u32, base, prefix);
	s.builder.AddFunction(spv::OpSelect, u32, next_old, mine, lane_old, old);
	s.builder.AddFunction(spv::OpLogicalAnd, boolean, next_pending, pending,
	                      Unary(s, spv::OpLogicalNot, boolean, mine));
	s.builder.AddFunction(spv::OpBranch, cont);
	EmitLabel(s, cont);
	s.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(s, merge);
	(void)mem;
	return old;
}

uint32_t EmitAtomic32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem = ctx.Memory(inst);
	const bool  lane_order =
	    GetCodegenOptions().lds_atomic_order && mem.kind == IR::ResourceKind::Lds &&
	    (inst.GetOpcode() == IR::ValueOpcode::SharedAtomicIAdd32 ||
	     inst.GetOpcode() == IR::ValueOpcode::SharedAtomicISub32) &&
	    inst.HasUses() && !ctx.state.split_wave && !ctx.state.lds_slot_plan &&
	    ctx.state.lds_storage_class == spv::StorageClassWorkgroup;
	return EmitAtomicAccess(ctx, inst, mem, [&](uint32_t pointer) {
		const auto scope =
		    mem.kind == IR::ResourceKind::Lds ? spv::ScopeWorkgroup : spv::ScopeDevice;
		const auto old =
		    lane_order
		        ? EmitLaneOrderedLdsAdd(ctx, inst, mem, pointer,
		                                Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state),
		                                       DwordIndex(ctx, inst, mem), ConstantU32(ctx.state, 0x3fffu)))
		        : EmitAtomicOperation(ctx, inst, pointer, scope);
		if (mem.kind == IR::ResourceKind::Lds) {
			const auto semantics =
			    spv::MemorySemanticsAcquireReleaseMask | LdsMemorySemantics(ctx.state);
			ctx.state.builder.AddFunction(spv::OpMemoryBarrier, ConstantU32(ctx.state, scope),
			                              ConstantU32(ctx.state, semantics));
		} else {
			EmitDeviceAtomicMemoryBarrier(ctx.state);
		}
		return old;
	});
}

uint32_t EmitBufferAtomic64(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem   = ctx.Memory(inst);
	auto&       state = ctx.state;
	if (mem.kind == IR::ResourceKind::IndirectBuffer) {
		// KYTY_BDA_WRITES: through a V# the shader computed; the same operations on a u64
		// PhysicalStorageBuffer pointer.
		return EmitIndirectAtomic(ctx, inst, true, [&](uint32_t pointer) {
			const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
			const auto old   = state.builder.AllocateId();
			if (inst.GetOpcode() == IR::ValueOpcode::BufferAtomicCmpSwap64) {
				const auto desired = ctx.Arg(inst, inst.NumArgs() - 3);
				state.builder.AddFunction(spv::OpAtomicCompareExchange, TypeU64(state), old,
				                          pointer, ConstantU32(state, spv::ScopeDevice),
				                          ConstantU32(state, spv::MemorySemanticsMaskNone),
				                          ConstantU32(state, spv::MemorySemanticsMaskNone), desired,
				                          value);
			} else {
				state.builder.AddFunction(SpirvAtomicOpcode(inst.GetOpcode()), TypeU64(state),
				                          old, pointer, ConstantU32(state, spv::ScopeDevice),
				                          ConstantU32(state, spv::MemorySemanticsMaskNone), value);
			}
			EmitDeviceAtomicMemoryBarrier(state);
			return old;
		});
	}
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU64(state), ConstantU64(state, 0), [&]() {
		    const auto resource = PrepareStorageBufferResourceAccess(
		        state, mem, state.storage_buffer_u64_variable, TypeStorageBufferPointer(state, 64));
		    const auto byte_address = ByteAddress(ctx, inst, mem);
		    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), byte_address,
		                              ConstantU32(state, 3u));
		    return EmitValueOrDefaultIfCondition(
		        state, EmitMemoryElementInBounds(state, resource, index), TypeU64(state),
		        ConstantU64(state, 0), [&]() {
			        const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
			        const auto old   = state.builder.AllocateId();
			        const auto pointer = EmitStorageBufferElementPointer(
			            state, resource, index, TypeStorageBufferElementPointer(state, 64));
			        if (inst.GetOpcode() == IR::ValueOpcode::BufferAtomicCmpSwap64) {
				        // DATA[0:1] is stored when memory equals the comparator in DATA[2:3].
				        const auto desired = ctx.Arg(inst, inst.NumArgs() - 3);
				        state.builder.AddFunction(
				            spv::OpAtomicCompareExchange, TypeU64(state), old, pointer,
				            ConstantU32(state, spv::ScopeDevice),
				            ConstantU32(state, spv::MemorySemanticsMaskNone),
				            ConstantU32(state, spv::MemorySemanticsMaskNone), desired, value);
			        } else {
				        state.builder.AddFunction(SpirvAtomicOpcode(inst.GetOpcode()),
				                                  TypeU64(state), old, pointer,
				                                  ConstantU32(state, spv::ScopeDevice),
				                                  ConstantU32(state, spv::MemorySemanticsMaskNone),
				                                  value);
			        }
			        EmitDeviceAtomicMemoryBarrier(state);
			        return old;
		        });
	    });
}

void EmitSharedAtomic64(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	const auto& mem = ctx.Memory(inst);
	EnsureLdsStorage(state);
	const auto address = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                            ByteAddress(ctx, inst, mem), ConstantU32(state, 0xfff8u));
	const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
	                          ConstantU32(state, 3u));
	// Device-buffer LDS: the region's bound in qwords (regions are whole qwords: LDS sizes are
	// multiples of 512 bytes).
	const auto bound = state.lds_storage_class == spv::StorageClassStorageBuffer
	                       ? EmitBinaryU32(state, spv::OpShiftRightLogical, state.lds_length,
	                                       ConstantU32(state, 1))
	                       : ConstantU32(state, LdsDwordCount(state) / 2u);
	const auto in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, bound);
	const auto active = EmitLogicalAndBool(state, ctx.Arg(inst, inst.NumArgs() - 1), in_bounds);
	const auto emit = [&]() {
		auto native_index = index;
		auto semantics = spv::MemorySemanticsWorkgroupMemoryMask;
		if (state.lds_storage_class == spv::StorageClassStorageBuffer) {
			native_index = EmitAddU32(state, index, EmitBinaryU32(
			    state, spv::OpShiftRightLogical, state.lds_base_dwords, ConstantU32(state, 1)));
			semantics = spv::MemorySemanticsUniformMemoryMask;
		}
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, state.lds_storage_class, TypeU64(state)),
		    pointer, state.lds_u64_variable, ConstantU32(state, 0), native_index);
		const auto old = state.builder.AllocateId();
		const auto opcode = SpirvAtomicOpcode(inst.GetOpcode());
		const auto scope = ConstantU32(state, spv::ScopeWorkgroup);
		const auto acquire_release =
		    ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | semantics);
		if (opcode == spv::OpAtomicCompareExchange) {
			// RDNA2 DS compares DATA and stores DATA2, opposite to BUFFER atomics.
			state.builder.AddFunction(
			    opcode, TypeU64(state), old, pointer, scope, acquire_release,
			    ConstantU32(state, spv::MemorySemanticsAcquireMask | semantics),
			    ctx.Arg(inst, 2), ctx.Arg(inst, 1));
		} else {
			state.builder.AddFunction(opcode, TypeU64(state), old, pointer, scope,
			                          acquire_release, ctx.Arg(inst, 1));
		}
		return old;
	};
	// DS_MAX_RTN_U64 and DS_CMPST_RTN_B64 return the old value (0 for an inactive or
	// out-of-range lane); the no-return forms are Void.
	if (inst.GetType() != IR::Type::Void && inst.HasUses()) {
		ctx.Define(inst, EmitValueOrDefaultIfCondition(state, active, TypeU64(state),
		                                               ConstantU64(state, 0), emit));
	} else {
		EmitIfCondition(state, active, [&]() { emit(); });
	}
}

uint32_t EmitBufferFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem       = ctx.Memory(inst);
	const bool  max_value = inst.GetOpcode() == IR::ValueOpcode::BufferAtomicFMax32;
	return EmitAtomicUpdate(ctx, inst, mem,
	                        [max_value](EmitterState& state, uint32_t old, uint32_t value) {
		                        return EmitFloatAtomicReplacement(state, old, value, max_value);
	                        });
}

void EmitSharedFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem       = ctx.Memory(inst);
	const bool  max_value = inst.GetOpcode() == IR::ValueOpcode::SharedAtomicFMax32;
	EmitAtomicUpdate(ctx, inst, mem,
	                 [max_value](EmitterState& state, uint32_t old, uint32_t value) {
		                 return EmitDsFloatAtomicReplacement(state, old, value, max_value);
	                 });
}

uint32_t EmitAppendConsume(ValueEmitContext& ctx, const IR::Inst& inst) {
	const bool append = inst.GetOpcode() == IR::ValueOpcode::DataAppend;
	auto&      state  = ctx.state;
	if (ctx.half == 1) {
		return ctx.other_half->Def(IR::Value(const_cast<IR::Inst*>(&inst)));
	}
	const auto m0 = ctx.Arg(inst, 0);
	const auto base =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), m0, ConstantU32(state, 16));
	const auto size =
	    Binary(state, spv::OpBitwiseAnd, TypeU32(state), m0, ConstantU32(state, 0xffffu));
	const auto address = Binary(state, spv::OpIAdd, TypeU32(state), base,
	                            ConstantU32(state, ctx.Memory(inst).offset));
	const auto index =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), address, ConstantU32(state, 2));
	const auto mem    = ctx.Memory(inst);
	const auto access = PrepareMemoryResourceAccess(state, mem);
	const auto exec   = ctx.Arg(inst, 1);
	const auto ballot = ctx.Ballot(inst.Arg(1));
	const auto low    = state.builder.AllocateId();
	const auto high   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1);
	const auto count = Binary(state, spv::OpIAdd, TypeU32(state),
	                          Unary(state, spv::OpBitCount, TypeU32(state), low),
	                          Unary(state, spv::OpBitCount, TypeU32(state), high));
	// KYTY_PS_APPEND_LIVE_ELECTION: a helper invocation's atomic has no effect and returns an
	// undefined value, so a pixel shader elects the first EXEC lane that is not a helper (the
	// count above still covers every EXEC lane). With no such lane nobody consumes the result.
	auto elected_ballot = ballot;
	auto lane_guard     = exec;
	if (state.program.stage == ShaderType::Pixel && state.helper_invocation_variable != 0 &&
	    state.lane_count == 1) {
		const auto live =
		    Binary(state, spv::OpLogicalAnd, TypeBool(state), exec,
		           Unary(state, spv::OpLogicalNot, TypeBool(state), EmitIsHelperInvocation(state)));
		const auto live_ballot = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4),
		                          live_ballot, ConstantU32(state, spv::ScopeSubgroup), live);
		elected_ballot = EmitWaveBallot(state, live_ballot);
		const auto elected_low  = state.builder.AllocateId();
		const auto elected_high = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), elected_low,
		                          elected_ballot, 0);
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), elected_high,
		                          elected_ballot, 1);
		const auto any_live =
		    Binary(state, spv::OpINotEqual, TypeBool(state),
		           Binary(state, spv::OpBitwiseOr, TypeU32(state), elected_low, elected_high),
		           ConstantU32(state, 0));
		// FindLSB of an empty ballot is undefined: without a live lane nothing is added.
		lane_guard = AndCondition(state, exec, any_live);
	}
	const auto first = ctx.FirstLane(elected_ballot);
	const auto source_lane =
	    state.lane_count == 2
	        ? Binary(state, spv::OpBitwiseAnd, TypeU32(state), first, ConstantU32(state, 31))
	        : first;
	const auto is_first       = Binary(state, spv::OpIEqual, TypeBool(state),
	                                   EmitSubgroupLocalInvocationId(state), source_lane);
	const auto storage_bounds = EmitMemoryElementInBounds(state, access, index);
	const auto m0_bounds =
	    mem.kind == IR::ResourceKind::Gds
	        ? Binary(state, spv::OpINotEqual, TypeBool(state), size, ConstantU32(state, 0))
	        : Binary(state, spv::OpULessThan, TypeBool(state),
	                 ConstantU32(state, ctx.Memory(inst).offset + 3u), size);
	const auto condition = AndCondition(
	    state, is_first,
	    AndCondition(state,
	                 state.lane_count == 2 || state.split_wave
	                     ? Binary(state, spv::OpINotEqual, TypeBool(state), count,
	                              ConstantU32(state, 0))
	                     : lane_guard,
	                 AndCondition(state, storage_bounds, m0_bounds)));
	const auto atomic = EmitValueOrZeroIfCondition(state, condition, [&]() {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(append ? spv::OpAtomicIAdd : spv::OpAtomicISub, TypeU32(state),
		                          value, EmitMemoryElementPointer(state, access, index),
		                          ConstantU32(state, mem.kind == IR::ResourceKind::Gds
		                                                 ? spv::ScopeDevice
		                                                 : spv::ScopeWorkgroup),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone), count);
		return value;
	});
	// A split wave's first active lane may be in the other host subgroup.
	if (state.split_wave) {
		return ExchangeWaveValue(state, atomic, source_lane);
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), atomic,
	                          EmitHostLane(state, source_lane));
	return result;
}

uint32_t EmitReadConst(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (state.flattened_srt_variable == 0) {
		ctx.Fail(inst, "requires the flattened SRT descriptor");
	}
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, FlattenedSrtElementPointerType(state), pointer,
	                          state.flattened_srt_variable, ConstantU32(state, 0),
	                          ctx.Arg(inst, 1));
	return EmitNative<spv::OpLoad, IR::Type::U32>(state, pointer);
}

// One dword of an S_BUFFER_LOAD through a V# the shader computed at runtime (KYTY_SRT_VARIANT_READS,
// ResourceKind::IndirectBuffer), read through BDA. RDNA2 ISA 7.2.1, "Reads using Buffer Constant":
// only base, stride and num_records are used; addr = (base + OFFSET + SOFFSET) & ~3. The bound is
// the one a bound scalar buffer gets (ShaderBufferResource::GetSize: stride 0 makes num_records a
// byte count, otherwise stride * num_records bytes), and a dword with any byte past it reads 0, as
// the robust storage-buffer load of a bound V# does.
uint32_t LoadIndirectScalarBuffer(ValueEmitContext& ctx, const IR::Inst& inst,
                                  const IR::MemoryInfo& mem) {
	auto&       state   = ctx.state;
	const auto& handle  = *inst.Arg(0).ResolveInstruction();
	const auto  word1   = ctx.Arg(handle, 1);
	const auto  records = ctx.Arg(handle, 2);
	const auto  field   = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first),
		                            ConstantU32(state, count));
	};
	const auto u64 = [&](uint32_t value) {
		return Unary(state, spv::OpUConvert, TypeU64(state), value);
	};
	const auto stride = field(word1, 16, 14);
	const auto base   = DeviceAddressFromWords(state, ctx.Arg(handle, 0), field(word1, 0, 16));
	const auto offset = Binary(state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 1),
	                           ConstantU32(state, mem.offset));
	const auto size   = Select(state, TypeU64(state),
	                           Binary(state, spv::OpIEqual, TypeBool(state), stride,
	                                  ConstantU32(state, 0)),
	                           u64(records),
	                           Binary(state, spv::OpIMul, TypeU64(state), u64(stride),
	                                  u64(records)));
	const auto dword_end =
	    Binary(state, spv::OpIAdd, TypeU64(state),
	           u64(Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, ~3u))),
	           ConstantDeviceAddress(state, sizeof(uint32_t)));
	const auto in_bounds =
	    Binary(state, spv::OpULessThanEqual, TypeBool(state), dword_end, size);
	// The ISA drops the two LSBs of the sum, so the dword never straddles (no unaligned merge).
	const auto guest = Binary(state, spv::OpBitwiseAnd, TypeU64(state),
	                          Binary(state, spv::OpIAdd, TypeU64(state), base, u64(offset)),
	                          ConstantDeviceAddress(state, ~uint64_t {3}));
	return LoadBda(ctx, guest, in_bounds, 32u,
	               GetCodegenOptions().indirect_coherent && mem.coherent);
}

static bool EmitScalarLoadGroup(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem);

void EmitReadConstBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto mem = ctx.Memory(inst);
	if (mem.planning_only) return;
	if (mem.kind == IR::ResourceKind::IndirectBuffer && GetCodegenOptions().scalar_load_groups &&
	    ctx.definitions.contains(&inst)) {
		return; // KYTY_SCALAR_LOAD_GROUPS: a member its group's leader already loaded.
	}
	if (mem.kind == IR::ResourceKind::IndirectBuffer && EmitScalarLoadGroup(ctx, inst, mem)) {
		return;
	}
	if (mem.kind == IR::ResourceKind::IndirectBuffer) {
		ctx.Define(inst, LoadIndirectScalarBuffer(ctx, inst, mem));
		return;
	}
	auto& state        = ctx.state;
	mem.kind           = IR::ResourceKind::ScalarBuffer;
	const auto address = Binary(state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 1),
	                            ConstantU32(state, mem.offset));
	const auto index =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), address, ConstantU32(state, 2));
	if (const auto* uniform =
	        IR::FindBinding(state.program.bindings, IR::DescriptorBindingKind::UniformBuffers);
	    uniform != nullptr && std::ranges::binary_search(uniform->resources, mem.resource)) {
		// KYTY_UNIFORM_BUFFERS: the uniform view shares the storage view's base adjustment, and
		// the device reads zero past its dword-aligned range (robustBufferAccess2).
		const auto array_index =
		    ResourceForDescriptor(state, IR::DescriptorBindingKind::UniformBuffers, mem.resource);
		const auto element = Binary(
		    state, spv::OpIAdd, TypeU32(state), index,
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state),
		           state.memory_byte_offsets[mem.resource], ConstantU32(state, 2)));
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		                          TypePointer(state, spv::StorageClassUniform, TypeU32(state)),
		                          pointer, state.uniform_buffer_variable,
		                          ConstantU32(state, array_index), ConstantU32(state, 0), element);
		ctx.Define(inst, EmitNative<spv::OpLoad, IR::Type::U32>(state, pointer));
		return;
	}
	const auto access    = PrepareMemoryResourceAccess(state, mem);
	// Scalar buffer addressing aligns the base and each offset independently.
	auto element = Binary(
	    state, spv::OpIAdd, TypeU32(state), index,
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), access.byte_offset, ConstantU32(state, 2)));
	// KYTY_COHERENT_LOAD_MODE / _ACQUIRE: a coherent scalar load as the vector ones (not Plain).
	const bool coherent = access.memory_access != spv::MemoryAccessMaskNone &&
	                      GetCoherentLoadMode() != CoherentLoadMode::Plain;
	const auto load = [&](uint32_t pointer) {
		return coherent ? EmitCoherentWordLoad(state, pointer)
		                : EmitNative<spv::OpLoad, IR::Type::U32>(state, pointer);
	};
	if (DeviceChecksDwordLoad(access)) {
		// Upstream ee034be4b: the device computes byte offsets in 32 bits, so elements from 2^30 on
		// would wrap back into the range; the host prefix adds at most 63 bytes.
		constexpr uint32_t last_element = UINT32_MAX >> 2u;
		const auto         offset       = inst.Arg(1).Resolve();
		if (!offset.IsImmediate() ||
		    (offset.U32() >> 2u) + (mem.offset >> 2u) > last_element - 63u) {
			element = EmitGlsl<GLSLstd450UMin, IR::Type::U32>(state, element,
			                                                  ConstantU32(state, last_element));
		}
		ctx.Define(inst, load(EmitMemoryElementPointer(state, access, element)));
		return;
	}
	const auto condition = EmitMemoryElementInBounds(state, access, element);
	ctx.Define(inst, EmitValueOrZeroIfCondition(state, condition, [&]() {
		           return load(EmitMemoryElementPointer(state, access, element));
	           }));
}

namespace {

// The value of a lane-only expression (immediates and LaneId under pure 32-bit integer and
// boolean operations) for one lane, or none for anything else. Booleans are 0 or 1.
std::optional<uint32_t> EvaluateForLane(const IR::Value& value, uint32_t lane, uint32_t depth,
                                        uint32_t& budget) {
	if (budget == 0u) {
		return std::nullopt;
	}
	budget--;
	const auto resolved = value.Resolve();
	if (resolved.IsImmediate()) {
		if (resolved.GetType() == IR::Type::U1) {
			return resolved.U1() ? 1u : 0u;
		}
		if (resolved.GetType() == IR::Type::U32) {
			return resolved.U32();
		}
		return std::nullopt;
	}
	const auto* inst = resolved.TryInstruction();
	if (inst == nullptr || depth > 24u) {
		return std::nullopt;
	}
	using Op = IR::ValueOpcode;
	if (inst->GetOpcode() == Op::LaneId) {
		return lane;
	}
	std::array<uint32_t, 3> a {};
	const auto              count = inst->NumArgs();
	if (count == 0u || count > a.size()) {
		return std::nullopt;
	}
	for (size_t index = 0; index < count; index++) {
		const auto arg = EvaluateForLane(inst->Arg(index), lane, depth + 1u, budget);
		if (!arg.has_value()) {
			return std::nullopt;
		}
		a[index] = *arg;
	}
	switch (inst->GetOpcode()) {
		case Op::IAdd32: return a[0] + a[1];
		case Op::ISub32: return a[0] - a[1];
		case Op::IMul32: return a[0] * a[1];
		case Op::ShiftLeftLogical32: return a[1] < 32u ? a[0] << a[1] : 0u;
		case Op::ShiftRightLogical32: return a[1] < 32u ? a[0] >> a[1] : 0u;
		case Op::BitwiseAnd32: return a[0] & a[1];
		case Op::BitwiseOr32: return a[0] | a[1];
		case Op::BitwiseXor32: return a[0] ^ a[1];
		case Op::BitwiseNot32: return ~a[0];
		case Op::BitCount32: return static_cast<uint32_t>(std::popcount(a[0]));
		case Op::SelectU32:
		case Op::SelectU1: return a[0] != 0u ? a[1] : a[2];
		case Op::ULessThan32: return a[0] < a[1] ? 1u : 0u;
		case Op::ULessThanEqual32: return a[0] <= a[1] ? 1u : 0u;
		case Op::UGreaterThan32: return a[0] > a[1] ? 1u : 0u;
		case Op::UGreaterThanEqual32: return a[0] >= a[1] ? 1u : 0u;
		case Op::IEqual32: return a[0] == a[1] ? 1u : 0u;
		case Op::INotEqual32: return a[0] != a[1] ? 1u : 0u;
		case Op::LogicalAnd: return a[0] & a[1];
		case Op::LogicalOr: return a[0] | a[1];
		case Op::LogicalNot: return a[0] ^ 1u;
		default: return std::nullopt;
	}
}

} // namespace

// KYTY_LDS_SLOTS (CodegenOptions::lds_slots, strxno/KytyPS5-GT7 220abc01c adapted). The guest
// compiler spills VGPRs of non-compute stages to LDS at (LaneId << 2) + c bytes, LaneId built
// with V_MBCNT (folded to LaneId or not): lane l then reads and writes only dword l + (c >> 2) of
// the invocation's own function-scope array, never another lane's, and one variable per distinct
// c >> 2 holds exactly what the array would. The address must depend on nothing but the lane, and
// is evaluated for every lane of the wave: it must be 4 * lane + c (plus the DS offset) for all.
std::optional<uint32_t> StaticLdsSlot(const IR::Program& program, const IR::Inst& inst) {
	if (inst.GetOpcode() != IR::ValueOpcode::LoadSharedU32 &&
	    inst.GetOpcode() != IR::ValueOpcode::WriteSharedU32) {
		return std::nullopt;
	}
	const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
	if (memory_index >= program.memory_info.size() ||
	    program.memory_info[memory_index].kind != IR::ResourceKind::Lds) {
		return std::nullopt;
	}
	const auto lanes  = std::max<uint32_t>(program.wave_size, 1u);
	uint32_t   budget = 1024u; // instruction visits per lane (the walk does not share subtrees)
	const auto base   = EvaluateForLane(inst.Arg(0), 0u, 0u, budget);
	if (!base.has_value()) {
		return std::nullopt;
	}
	for (uint32_t lane = 1; lane < lanes; lane++) {
		budget             = 1024u;
		const auto address = EvaluateForLane(inst.Arg(0), lane, 0u, budget);
		if (!address.has_value() || *address != *base + lane * 4u) {
			return std::nullopt;
		}
	}
	// The emitter adds the DS offset in 32 bits (ByteAddress); keep far from any wrap (the plan
	// also bounds every key by the array size).
	const uint64_t constant = static_cast<uint64_t>(*base) + program.memory_info[memory_index].offset;
	if (constant >= (1ull << 24u)) {
		return std::nullopt;
	}
	return static_cast<uint32_t>(constant >> 2u);
}

static uint32_t LdsSlotVariable(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto key = StaticLdsSlot(ctx.state.program, inst);
	EXIT_IF(!key.has_value());
	const auto found = ctx.state.lds_slot_variables.find(*key);
	EXIT_IF(found == ctx.state.lds_slot_variables.end());
	return found->second;
}

// Read as zero (and written not at all) while the lane is inactive, as the array's accesses are.
static uint32_t LoadLdsSlot(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state    = ctx.state;
	const auto variable = LdsSlotVariable(ctx, inst);
	const auto value    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, variable);
	return Select(state, TypeU32(state), ctx.Arg(inst, inst.NumArgs() - 1), value,
	              ConstantU32(state, 0));
}

static void StoreLdsSlot(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state    = ctx.state;
	const auto variable = LdsSlotVariable(ctx, inst);
	const auto before   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), before, variable);
	state.builder.AddFunction(spv::OpStore, variable,
	                          Select(state, TypeU32(state), ctx.Arg(inst, inst.NumArgs() - 1),
	                                 ctx.Arg(inst, inst.NumArgs() - 2), before));
}

// KYTY_SCALAR_LOAD_GROUPS (adapted from Erlou93/KytyPS5-Yotei FindScalarLoadGroups /
// EmitScalarLoadGroup): the dwords of one S_LOAD_DWORDXn through a scalar address are consecutive
// LoadAddressU32 instructions on the same operands. Their leader loads every member: when the
// highest member's dword lies in the page of the lowest one, one get_bda_pointer of the lowest
// address gives every member's pointer as pointer + delta (a missing page gives 0 for every member
// and records the same fault bit, as each member would); otherwise each member takes the per-dword
// path. The members are defined here; their own emission finds them defined.
static bool EmitScalarLoadGroup(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto& state = ctx.state;
	const auto opcode   = inst.GetOpcode();
	const bool indirect = opcode == IR::ValueOpcode::ReadConstBuffer && mem.kind == IR::ResourceKind::IndirectBuffer;
	if (!GetCodegenOptions().scalar_load_groups || mem.planning_only || state.current_block == nullptr ||
	    ctx.definitions.contains(&inst) ||
	    !(indirect || (opcode == IR::ValueOpcode::LoadAddressU32 && mem.kind == IR::ResourceKind::ScalarAddress))) {
		return false;
	}
	auto it = state.current_block->begin();
	while (it != state.current_block->end() && &*it != &inst) ++it;
	if (it == state.current_block->end()) {
		return false;
	}
	const uint32_t base = indirect ? mem.offset : mem.offset & ~3u;
	std::vector<std::pair<const IR::Inst*, uint32_t>> members; // instruction, byte delta
	uint32_t last_component = mem.component_index;
	for (++it; it != state.current_block->end() && members.size() < 15u; ++it) {
		const auto& next = *it;
		if (next.GetOpcode() != opcode || next.NumArgs() != inst.NumArgs()) break;
		const auto index = next.Flags<IR::MemoryFlags>().index;
		if (index >= state.program.memory_info.size()) break;
		const auto& other = state.program.memory_info[index];
		const uint32_t other_offset = indirect ? other.offset : other.offset & ~3u;
		if (other.kind != mem.kind || other.planning_only ||
		    other.address_is_full != mem.address_is_full || other.coherent != mem.coherent ||
		    other.component_index <= last_component || other_offset <= base || other_offset - base >= 64u ||
		    ((other_offset - base) & 3u) != 0u) {
			break;
		}
		bool same = true;
		for (size_t arg = 0; arg < inst.NumArgs() && same; arg++) {
			same = next.Arg(arg) == inst.Arg(arg);
		}
		if (!same) break;
		members.emplace_back(&next, other_offset - base);
		last_component = other.component_index;
	}
	if (members.empty()) {
		static const bool log = std::getenv("KYTY_SCALAR_LOAD_GROUPS_LOG") != nullptr;
		if (log) {
			auto next = state.current_block->begin();
			while (next != state.current_block->end() && &*next != &inst) ++next;
			if (next != state.current_block->end()) ++next;
			std::printf("ScalarLoadGroup: no member after offset %u component %u (next %.*s)\n", mem.offset,
			            mem.component_index,
			            static_cast<int>(next == state.current_block->end() ? 3 : IR::ValueOpcodeName(next->GetOpcode()).size()),
			            next == state.current_block->end() ? "end" : IR::ValueOpcodeName(next->GetOpcode()).data());
		}
		return false;
	}
	const bool coherent = GetCodegenOptions().indirect_coherent && mem.coherent;
	const auto u64      = TypeU64(state);
	const auto span     = members.back().second + static_cast<uint32_t>(sizeof(uint32_t));
	uint32_t   first    = 0;
	uint32_t   in_range = 0; // indirect: the highest member's dword is in the buffer's range
	if (indirect) {
		// LoadIndirectScalarBuffer's address and range for the leader and the highest member.
		const auto& handle = *inst.Arg(0).ResolveInstruction();
		const auto  word1  = ctx.Arg(handle, 1);
		const auto  records = ctx.Arg(handle, 2);
		const auto  stride =
		    EmitBitFieldUExtract(state, word1, ConstantU32(state, 16), ConstantU32(state, 14));
		const auto widen = [&](uint32_t value) { return Unary(state, spv::OpUConvert, u64, value); };
		const auto buffer_base = DeviceAddressFromWords(
		    state, ctx.Arg(handle, 0),
		    EmitBitFieldUExtract(state, word1, ConstantU32(state, 0), ConstantU32(state, 16)));
		const auto offset = Binary(state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 1), ConstantU32(state, mem.offset));
		const auto size   = Select(state, u64, Binary(state, spv::OpIEqual, TypeBool(state), stride, ConstantU32(state, 0)),
		                           widen(records), Binary(state, spv::OpIMul, u64, widen(stride), widen(records)));
		const auto last   = Binary(state, spv::OpIAdd, TypeU32(state), offset, ConstantU32(state, members.back().second));
		const auto end    = Binary(state, spv::OpIAdd, u64,
		                           widen(Binary(state, spv::OpBitwiseAnd, TypeU32(state), last, ConstantU32(state, ~3u))),
		                           ConstantDeviceAddress(state, sizeof(uint32_t)));
		// (No 32-bit wrap of the members' offsets: the fast path's addresses are first + delta.)
		in_range = AndCondition(state, Binary(state, spv::OpULessThanEqual, TypeBool(state), end, size),
		                        Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), last, offset));
		first    = Binary(state, spv::OpBitwiseAnd, u64, Binary(state, spv::OpIAdd, u64, buffer_base, widen(offset)),
		                  ConstantDeviceAddress(state, ~uint64_t {3}));
	} else {
		first = GuestAddress(ctx, inst, mem);
	}
	auto in_page  = Binary(
	    state, spv::OpULessThanEqual, TypeBool(state),
	    Binary(state, spv::OpIAdd, u64,
	           Binary(state, spv::OpBitwiseAnd, u64, first, ConstantU64(state, BufferCache::CACHING_PAGESIZE - 1)),
	           ConstantU64(state, span)),
	    ConstantU64(state, BufferCache::CACHING_PAGESIZE));
	if (indirect) {
		in_page = AndCondition(state, in_range, in_page);
	}
	const auto count = members.size() + 1u;
	std::vector<uint32_t> fast(count);
	std::vector<uint32_t> slow(count);
	const auto fast_label  = state.builder.AllocateId();
	const auto slow_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, in_page, fast_label, slow_label);
	EmitLabel(state, fast_label);
	{
		const auto bda     = GetBdaPointer(ctx, first);
		const auto present = Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
		const auto header  = state.current_label;
		const auto load_label   = state.builder.AllocateId();
		const auto loaded_label = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelectionMerge, loaded_label, spv::SelectionControlMaskNone);
		state.builder.AddFunction(spv::OpBranchConditional, present, load_label, loaded_label);
		EmitLabel(state, load_label);
		const auto access = GetCodegenOptions().volatile_loads || coherent
		                        ? spv::MemoryAccessAlignedMask | spv::MemoryAccessVolatileMask
		                        : spv::MemoryAccessAlignedMask;
		std::vector<uint32_t> loads(count);
		for (size_t k = 0; k < count; k++) {
			const auto address =
			    k == 0 ? bda : Binary(state, spv::OpIAdd, u64, bda, ConstantU64(state, members[k - 1].second));
			uint32_t pointer = state.builder.AllocateId();
			if (coherent) {
				const auto block = state.builder.DecoratedType(
				    spv::OpTypeStruct, {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
				                        {spv::OpMemberDecorate, {0, spv::DecorationCoherent}},
				                        {spv::OpDecorate, {spv::DecorationBlock}}}, TypeU32(state));
				state.builder.AddFunction(spv::OpConvertUToPtr,
				                          TypePointer(state, spv::StorageClassPhysicalStorageBuffer, block),
				                          pointer, address);
				const auto struct_pointer = pointer;
				pointer                   = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpAccessChain, TypePhysicalU32Pointer(state), pointer,
				                          struct_pointer, ConstantU32(state, 0));
			} else {
				state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
				                          address);
			}
			loads[k] = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeU32(state), loads[k], pointer, access,
			                          static_cast<uint32_t>(sizeof(uint32_t)));
		}
		state.builder.AddFunction(spv::OpBranch, loaded_label);
		EmitLabel(state, loaded_label);
		for (size_t k = 0; k < count; k++) {
			fast[k] = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpPhi, TypeU32(state), fast[k], loads[k], load_label,
			                          ConstantU32(state, 0), header);
		}
	}
	const auto fast_exit = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, slow_label);
	for (size_t k = 0; k < count; k++) {
		if (indirect) {
			const auto& member = k == 0 ? inst : *members[k - 1].first;
			slow[k] = LoadIndirectScalarBuffer(ctx, member, ctx.Memory(member));
		} else {
			slow[k] = LoadBdaDword(
			    ctx, k == 0 ? first : Binary(state, spv::OpIAdd, u64, first, ConstantU64(state, members[k - 1].second)),
			    coherent);
		}
	}
	const auto slow_exit = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
	for (size_t k = 0; k < count; k++) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpPhi, TypeU32(state), value, fast[k], fast_exit, slow[k], slow_exit);
		ctx.Define(k == 0 ? inst : *members[k - 1].first, value);
	}
	return true;
}

// Upstream 6fd43ac65: a dword at a 64-bit device address plus a byte offset, zero where the lane
// is inactive (IR LoadDeviceAddressU32; our frontend does not emit it yet).
uint32_t EmitLoadDeviceAddressU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	return EmitValueOrZeroIfCondition(state, ctx.Arg(inst, 3), [&]() {
		const auto address = Binary(
		    state, spv::OpIAdd, TypeU64(state), PackU64(state, ctx.Arg(inst, 0), ctx.Arg(inst, 1)),
		    Unary(state, spv::OpUConvert, TypeU64(state), ctx.Arg(inst, 2)));
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpLoad, TypeU32(state), value,
		    Unary(state, spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), address),
		    spv::MemoryAccessAlignedMask, 4u);
		return value;
	});
}

void EmitLoadMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (GetCodegenOptions().scalar_load_groups && ctx.definitions.contains(&inst) &&
	    inst.GetOpcode() == IR::ValueOpcode::LoadAddressU32 &&
	    ctx.Memory(inst).kind == IR::ResourceKind::ScalarAddress) {
		return; // A member of a scalar load group its leader already loaded.
	}
	if (EmitScalarLoadGroup(ctx, inst, ctx.Memory(inst))) {
		return;
	}
	if (ctx.state.lds_slot_plan && ctx.Memory(inst).kind == IR::ResourceKind::Lds) {
		ctx.Define(inst, LoadLdsSlot(ctx, inst));
		return;
	}
	const auto  op  = inst.GetOpcode();
	const auto& mem = ctx.Memory(inst);
	if (op == IR::ValueOpcode::LoadAddressU32 && mem.planning_only) return;
	const auto buffer_components = IR::BufferComponentCount(op);
	const auto shared_components = IR::SharedComponentCount(op);
	const auto address_info      = IR::AddressOpcodeInfoOf(op);
	uint32_t   value;
	if (buffer_components == 2u && BufferPairAtomic(ctx, inst))
		value = LoadBufferPair(ctx, inst);
	else if (buffer_components > 1u)
		value = LoadWideBuffer(ctx, inst, buffer_components);
	else if (buffer_components == 1u && mem.kind == IR::ResourceKind::IndirectBuffer &&
	         !mem.formatted)
		value = EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
			return LoadIndirectBufferElement(ctx, inst,
			                                 inst.GetType() == IR::Type::U8    ? 8u
			                                 : inst.GetType() == IR::Type::U16 ? 16u
			                                                                   : 32u);
		});
	else if (shared_components > 1u)
		value = LoadWideShared(ctx, inst, shared_components);
	else if (mem.kind == IR::ResourceKind::ScalarAddress)
		value = LoadBdaDword(ctx, GuestAddress(ctx, inst, mem),
		                     GetCodegenOptions().indirect_coherent && mem.coherent);
	else if (address_info.access == IR::AddressAccess::Read &&
	         mem.kind != IR::ResourceKind::Scratch)
		value = LoadBda(ctx, GuestAddress(ctx, inst, mem), ctx.Arg(inst, inst.NumArgs() - 1),
		                address_info.data_bits, mem.coherent);
	else if (op == IR::ValueOpcode::LoadBufferU32 && mem.formatted)
		value = mem.kind == IR::ResourceKind::IndirectBuffer ? LoadIndirectFormattedX(ctx, inst)
		                                                     : FormattedLoad(ctx, inst, mem);
	else if (inst.GetType() == IR::Type::U8)
		value = LoadSubword(ctx, inst, mem, 8);
	else if (inst.GetType() == IR::Type::U16)
		value = LoadSubword(ctx, inst, mem, 16);
	else
		value = LoadWord(ctx, inst, mem);
	ctx.Define(inst, value);
}

void EmitStoreMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (ctx.state.lds_slot_plan && ctx.Memory(inst).kind == IR::ResourceKind::Lds) {
		StoreLdsSlot(ctx, inst);
		return;
	}
	const auto  op                = inst.GetOpcode();
	const auto& mem               = ctx.Memory(inst);
	const auto  buffer_components = IR::BufferComponentCount(op);
	const auto  shared_components = IR::SharedComponentCount(op);
	const auto  type              = inst.Arg(inst.NumArgs() - 2).GetType();
	if (buffer_components == 2u && BufferPairAtomic(ctx, inst))
		StoreBufferPair(ctx, inst);
	else if (buffer_components != 0u && mem.kind == IR::ResourceKind::IndirectBuffer && mem.formatted)
		StoreIndirectFormatted(ctx, inst, buffer_components);
	else if (buffer_components != 0u && mem.kind == IR::ResourceKind::IndirectBuffer)
		StoreIndirectBuffer(ctx, inst, buffer_components,
		                    type == IR::Type::U8 ? 8u : type == IR::Type::U16 ? 16u : 32u);
	else if (buffer_components > 1u)
		StoreWideBuffer(ctx, inst, buffer_components);
	else if (shared_components > 1u)
		StoreWideShared(ctx, inst, shared_components);
	else if (op == IR::ValueOpcode::StoreBufferU32 && mem.formatted)
		FormattedStore(ctx, inst, mem);
	else if (type == IR::Type::U8)
		StoreSubword(ctx, inst, mem, 8);
	else if (type == IR::Type::U16)
		StoreSubword(ctx, inst, mem, 16);
	else
		StoreWord(ctx, inst, mem);
}

uint32_t EmitAtomicIncDec(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto op          = inst.GetOpcode();
	const bool increment   = op == IR::ValueOpcode::SharedAtomicInc32 ||
	                       op == IR::ValueOpcode::BufferAtomicInc32;
	const auto replacement = increment ? AtomicIncrement : AtomicDecrement;
	return EmitAtomicUpdate(ctx, inst, ctx.Memory(inst), replacement);
}

void EmitSharedAtomicMaskedOr32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto keep = Unary(ctx.state, spv::OpNot, TypeU32(ctx.state), ctx.Arg(inst, 1));
	EmitAtomicUpdate(ctx, inst, ctx.Memory(inst),
	                 [keep](EmitterState& state, uint32_t old, uint32_t value) {
		                 return Binary(state, spv::OpBitwiseOr, TypeU32(state),
		                               Binary(state, spv::OpBitwiseAnd, TypeU32(state), old, keep), value);
	                 });
}

uint32_t EmitSwizzleU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	state.builder.AddFunction(spv::OpStore, ctx.scratch_u32_variable, ctx.Arg(inst, 0));
	const auto source = EmitNative<spv::OpLoad, IR::Type::U32>(state, ctx.scratch_u32_variable);
	const auto target = EmitDsSwizzleTargetLane(state, EmitSubgroupLocalInvocationId(state),
	                                            inst.Arg(1).IsImmediate() ? inst.Arg(1).U32() : 0);
	return EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2));
}

uint32_t EmitPermuteU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state   = ctx.state;
	auto       lane    = EmitSubgroupLocalInvocationId(state);
	if (HalfWaveHost(state)) {
		lane = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane, ConstantU32(state, 31));
	}
	const auto address = ctx.Arg(inst, 1);
	const auto word    = Binary(state, spv::OpShiftRightLogical, TypeU32(state), lane,
	                            ConstantU32(state, 5));
	const auto ballot_word = [&](uint32_t predicate) {
		const auto ballot = state.builder.AllocateId();
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
		                          ConstantU32(state, spv::ScopeSubgroup), predicate);
		state.builder.AddFunction(spv::OpVectorExtractDynamic, TypeU32(state), result,
		                          EmitWaveBallot(state, ballot), word);
		return result;
	};
	// RDNA2 permutes independently within each 32-lane half. Intersect the source
	// address bit ballots to find this destination's enabled writers without LDS.
	auto writers = ballot_word(ctx.Arg(inst, 2));
	for (uint32_t bit = 0; bit < 5; ++bit) {
		const auto address_bit = Binary(state, spv::OpBitwiseAnd, TypeU32(state), address,
		                                ConstantU32(state, 1u << (bit + 2)));
		const auto mask = ballot_word(Binary(state, spv::OpINotEqual, TypeBool(state),
		                                      address_bit, ConstantU32(state, 0)));
		const auto lane_bit = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane,
		                             ConstantU32(state, 1u << bit));
		const auto selected = Select(
		    state, TypeU32(state),
		    Binary(state, spv::OpINotEqual, TypeBool(state), lane_bit, ConstantU32(state, 0)),
		    mask, Unary(state, spv::OpNot, TypeU32(state), mask));
		writers = Binary(state, spv::OpBitwiseAnd, TypeU32(state), writers, selected);
	}
	const auto active = Binary(state, spv::OpINotEqual, TypeBool(state), writers,
	                           ConstantU32(state, 0));
	const auto base = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane,
	                         ConstantU32(state, ~31u));
	const auto source = Select(state, TypeU32(state), active,
	                           Binary(state, spv::OpBitwiseOr, TypeU32(state), base,
	                                  EmitFindUMsb32(state, writers)), lane);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), ctx.Arg(inst, 0),
	                          EmitHostLane(state, source));
	return Select(state, TypeU32(state), active, result, ConstantU32(state, 0));
}

uint32_t EmitBpermuteU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto source = ctx.Arg(inst, 0);
	const auto index  = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                           Binary(state, spv::OpShiftRightLogical, TypeU32(state),
	                                  ctx.Arg(inst, 1), ConstantU32(state, 2)),
	                           ConstantU32(state, 31));
	const auto base   = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                           EmitSubgroupLocalInvocationId(state), ConstantU32(state, ~31u));
	const auto target = Binary(state, spv::OpBitwiseOr, TypeU32(state), base, index);
	return EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2));
}

// KYTY_INDIRECT_RAW_OUTLINE: one DontInline function per raw-load shape through a computed V# the
// program uses (an aligned-dword load of 1-4 dwords, or one 8/16/32-bit element), each the
// inline path's code on its parameters (word0, word1, records, word3, index with ADD_TID, voffset +
// immediate, soffset).
static void DefineIndirectRawFunctions(EmitterState& state) {
	if (!GetCodegenOptions().indirect_raw_outline || state.bda_pointer_function == 0) {
		return;
	}
	struct Shape {
		uint32_t components; // 0: one element of `bits`
		uint32_t bits;
		bool     coherent;
	};
	std::map<uint32_t, Shape> shapes;
	const bool                coherent_option = GetCodegenOptions().indirect_coherent;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			const auto op         = inst.GetOpcode();
			const auto components = IR::BufferComponentCount(op);
			if (components == 0u || IR::BufferAccessOf(op) != IR::BufferAccess::Read) {
				continue;
			}
			const auto index = inst.Flags<IR::MemoryFlags>().index;
			if (index >= state.program.memory_info.size()) {
				continue;
			}
			const auto& mem = state.program.memory_info[index];
			if (mem.kind != IR::ResourceKind::IndirectBuffer || mem.formatted) {
				continue;
			}
			const bool coherent = coherent_option && mem.coherent;
			if (components == 1u) {
				const uint32_t bits = inst.GetType() == IR::Type::U8    ? 8u
				                      : inst.GetType() == IR::Type::U16 ? 16u
				                                                        : 32u;
				shapes.emplace(IndirectRawElementKey(bits, coherent), Shape {0u, bits, coherent});
			}
			if (components > 1u) {
				shapes.emplace(IndirectRawKey(components, 32u, coherent),
				               Shape {components, 32u, coherent});
			}
		}
	}
	const auto u32 = TypeU32(state);
	for (const auto& [key, shape]: shapes) {
		const auto result_type = shape.components == 0u ? u32 : TypeU32Composite(state, shape.components);
		const std::array<uint32_t, 7> parameter_types {u32, u32, u32, u32, u32, u32, u32};
		const auto function_type = state.builder.Type(spv::OpTypeFunction, result_type,
		                                              std::span<const uint32_t>(parameter_types));
		const auto function = state.builder.AllocateId();
		std::string name = shape.components == 0u ? "indirect_raw_u" + std::to_string(shape.bits)
		                                          : "indirect_raw_x" + std::to_string(shape.components);
		if (shape.coherent) {
			name += "_coherent";
		}
		state.builder.AddName(function, name.c_str());
		state.builder.AddFunction(spv::OpFunction, result_type, function,
		                          spv::FunctionControlDontInlineMask, function_type);
		std::array<uint32_t, 7> parameters {};
		for (auto& id: parameters) {
			id = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpFunctionParameter, u32, id);
		}
		EmitLabel(state, state.builder.AllocateId());
		ValueEmitContext ctx(state);
		const auto [word0, word1, records, word3, index, voffset, soffset] = parameters;
		uint32_t value = 0;
		if (shape.components == 0u) {
			const auto access = IndirectBufferElementFromWords(state, word0, word1, records, word3,
			                                                   index, voffset, soffset, 0u,
			                                                   shape.bits / 8u);
			value = LoadBda(ctx, access.guest, access.in_bounds, shape.bits, shape.coherent);
		} else {
			const auto buffer = IndirectBufferFromWords(state, word0, word1, records, word3, index,
			                                            voffset, soffset, 0u, false);
			value = LoadIndirectBufferAccess(ctx, buffer, shape.components, shape.coherent);
		}
		state.builder.AddFunction(spv::OpReturnValue, value);
		state.builder.AddFunction(spv::OpFunctionEnd);
		state.indirect_raw_functions[key] = function;
	}
}

void DefineIndirectFormatFunction(EmitterState& state) {
	DefineIndirectFormatFunctionImpl(state);
	DefineIndirectRawFunctions(state);
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
