#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"

#include <algorithm>
#include <array>

// IMAGE_BVH_INTERSECT_RAY / IMAGE_BVH64_INTERSECT_RAY (MIMG 0xe6/0xe7, RDNA2 ISA 8.2.10).
// Specification: Profiling/analysis/RT-SOFTWARE-DESIGN.md.

namespace Libs::Graphics::ShaderRecompiler::Frontend {

IR::U32 Translator::ReadBvhAddress(const Decoder::Instruction& inst, uint32_t index) {
	// NSA supplies every address VGPR after the first one individually (VADDR is address 0).
	const auto nsa_components =
	    std::min(inst.image_nsa_dwords * 4u, Decoder::MaxImageNsaAddressComponents);
	if (index != 0u && index - 1u < nsa_components) {
		return ir.GetVectorReg(static_cast<IR::VectorReg>(inst.image_nsa_addr[index - 1u]));
	}
	return ReadRawU32(OffsetOperand(PlainOperand(inst.src0), index));
}

void Translator::IMAGE_BVH_INTERSECT_RAY(const Decoder::Instruction& inst) {
	program.info.uses_bvh = true;
	const auto u          = [](uint32_t value) { return IR::U32(IR::Value(value)); };
	if (!GetCodegenOptions().rt_software) {
		// KYTY_RT_STUB: every lane gets the result of a node the ray misses (see the SPIR-V
		// lowering). The value is opaque to IR folding: constant results would let the optimizer
		// delete the game's hit handling and reshape its CFG.
		const auto result =
		    ir.Emit(IR::ValueOpcode::BvhIntersectRayStub, {ReadBvhAddress(inst, 0)});
		for (uint32_t dword = 0; dword < 4u; dword++) {
			WriteOperand(
			    OffsetOperand(inst.dst, dword),
			    ir.Emit(IR::ValueOpcode::CompositeExtractU32x4, {result, IR::Value(dword)}));
		}
		return;
	}

	// KYTY_RT_SOFTWARE: one BvhIntersectRay value (lowered in the SPIR-V backend).
	const bool bvh64 = inst.opcode == Decoder::Opcode::IMAGE_BVH64_INTERSECT_RAY;
	const bool a16   = (inst.image_sample_flags & Decoder::ImageSampleFlagA16) != 0u;
	// The 128-bit T# in four consecutive SGPRs (R128=1 is checked by the decoder).
	std::array<IR::U32, 4> tsharp;
	for (uint32_t dword = 0; dword < 4u; dword++) {
		tsharp[dword] = ReadRawU32(OffsetOperand(PlainOperand(inst.src1), dword));
	}
	// Base address: T# bits 39:0 hold address bits 47:8.
	const auto base_lo = ir.ShiftLeftLogical(tsharp[0], u(8u));
	const auto base_hi =
	    ir.BitwiseOr(ir.ShiftRightLogical(tsharp[0], u(24u)),
	                 ir.ShiftLeftLogical(ir.BitwiseAnd(tsharp[1], u(0xffu)), u(8u)));
	const auto resource   = GetAddressResource(base_lo, base_hi);
	const auto descriptor = ir.Emit(IR::ValueOpcode::CompositeConstructU32x4,
	                                {tsharp[0], tsharp[1], tsharp[2], tsharp[3]});

	uint32_t               cursor  = 0;
	const auto             next    = [&]() { return ReadBvhAddress(inst, cursor++); };
	const auto             node_lo = next();
	const auto             node_hi = bvh64 ? next() : u(0u);
	const auto             extent  = next();
	std::array<IR::U32, 3> origin {next(), next(), next()};
	std::array<IR::U32, 3> direction;
	std::array<IR::U32, 3> inverse;
	if (!a16) {
		for (auto& value: direction)
			value = next();
		for (auto& value: inverse)
			value = next();
	} else {
		// {dir.x, dir.y}, {dir.z, inv.x}, {inv.y, inv.z}, low half first; f16 -> f32 is exact.
		std::array<IR::U32, 6> halves;
		for (uint32_t dword = 0; dword < 3u; dword++) {
			const auto packed      = next();
			halves[dword * 2u]     = ir.BitwiseAnd(packed, u(0xffffu));
			halves[dword * 2u + 1] = ir.ShiftRightLogical(packed, u(16u));
		}
		const auto widen = [&](IR::U32 half) {
			const auto f16 = ir.Emit(IR::ValueOpcode::BitCastF16U16,
			                         {ir.Emit(IR::ValueOpcode::ConvertU16U32, {half})});
			return ir.BitCastU32(IR::F32(ir.Emit(IR::ValueOpcode::ConvertF32F16, {f16})));
		};
		for (uint32_t axis = 0; axis < 3u; axis++) {
			direction[axis] = widen(halves[axis]);
			inverse[axis]   = widen(halves[3u + axis]);
		}
	}

	// An address read for resource tracking (it turns on the BDA page table); the node fetch
	// itself is sized by the backend, so the metadata is that of one dword.
	IR::MemoryInfo memory;
	memory.kind            = IR::ResourceKind::Flat;
	memory.address_is_full = true;
	memory.data_dwords     = 1u;
	memory.data_bits       = 32u;
	memory.component_count = 1u;
	const auto result = ir.Emit(IR::ValueOpcode::BvhIntersectRay,
	                            {resource, descriptor, node_lo, node_hi, extent, origin[0],
	                             origin[1], origin[2], direction[0], direction[1], direction[2],
	                             inverse[0], inverse[1], inverse[2], ir.GetExec()},
	                            AddMemoryInfo(memory, inst.pc));
	for (uint32_t dword = 0; dword < 4u; dword++) {
		WriteOperand(OffsetOperand(inst.dst, dword),
		             ir.Emit(IR::ValueOpcode::CompositeExtractU32x4, {result, IR::Value(dword)}));
	}
	return;
}

// PS5 Pro (Trinity) instructions, ported from upstream 5eab0ec73 (IMAGE_BVH8_INTERSECT_RAY) and
// bfd4c34b8 (DS_BVH_STACK_PUSH8_POP1_RTN_B32). Upstream lowered them on its BvhIntersect design; here
// BVH8 is one address-reading IR value lowered beside our BVH4 node test (spirvEmitterBvh8.cpp), and
// the stack is plain IR over the LDS accesses every other DS instruction uses.

IR::U32 Translator::ReadBvh8Address(const Decoder::Instruction& inst, uint32_t index) {
	// NSA (two dwords): the decoder expanded the ten addresses after VADDR, one per component.
	if (index != 0u && inst.image_nsa_dwords != 0u) {
		return ir.GetVectorReg(static_cast<IR::VectorReg>(inst.image_nsa_addr[index - 1u]));
	}
	return ReadRawU32(OffsetOperand(PlainOperand(inst.src0), index));
}

void Translator::IMAGE_BVH8_INTERSECT_RAY(const Decoder::Instruction& inst) {
	program.info.uses_bvh = true;
	const auto u          = [](uint32_t value) { return IR::U32(IR::Value(value)); };
	// The destination's previous contents: an instance node leaves six of its dwords as they were.
	std::array<IR::U32, 10> previous;
	for (uint32_t i = 0; i < previous.size(); i++) {
		previous[i] = ReadRawU32(OffsetOperand(PlainOperand(inst.dst), i));
	}
	std::array<IR::U32, 4> tsharp;
	for (uint32_t dword = 0; dword < 4u; dword++) {
		tsharp[dword] = ReadRawU32(OffsetOperand(PlainOperand(inst.src1), dword));
	}
	const auto base_lo = ir.ShiftLeftLogical(tsharp[0], u(8u));
	const auto base_hi =
	    ir.BitwiseOr(ir.ShiftRightLogical(tsharp[0], u(24u)),
	                 ir.ShiftLeftLogical(ir.BitwiseAnd(tsharp[1], u(0xffu)), u(8u)));
	const auto resource   = GetAddressResource(base_lo, base_hi);
	const auto descriptor = ir.Emit(IR::ValueOpcode::CompositeConstructU32x4,
	                                {tsharp[0], tsharp[1], tsharp[2], tsharp[3]});
	std::array<IR::U32, 11> address;
	for (uint32_t i = 0; i < address.size(); i++) {
		address[i] = ReadBvh8Address(inst, i);
	}
	IR::MemoryInfo memory;
	memory.kind            = IR::ResourceKind::Flat;
	memory.address_is_full = true;
	memory.data_dwords     = 1u;
	memory.data_bits       = 32u;
	memory.component_count = 1u;
	const auto result      = ir.Emit(IR::ValueOpcode::Bvh8IntersectRay,
	                                 {resource, descriptor, address[0], address[1], address[2],
	                                  address[3], address[4], address[5], address[6], address[7],
	                                  address[8], address[9], address[10], ir.GetExec()},
	                                 AddMemoryInfo(memory, inst.pc));
	const auto get         = [&](uint32_t index) {
        return IR::U32(ir.Emit(IR::ValueOpcode::Bvh8Extract, {result, IR::Value(index)}));
	};
	const auto instance = ir.INotEqual(get(16u), u(0u));
	for (uint32_t i = 0; i < 10u; i++) {
		auto value = get(i);
		if (i == 0u || i == 1u || i == 4u || i == 5u || i == 8u || i == 9u) {
			value = ir.Select(instance, previous[i], value);
		}
		WriteOperand(OffsetOperand(inst.dst, i), value);
	}
	// An instance node writes the ray transformed into its object space over the origin and
	// direction registers. Non-instance results may alias the ray registers: those are read after
	// the destination writes above.
	for (uint32_t i = 0; i < 6u; i++) {
		auto target = PlainOperand(inst.src0);
		target.reg =
		    inst.image_nsa_dwords != 0u ? inst.image_nsa_addr[3u + i] : inst.src0.reg + 4u + i;
		WriteOperand(target, ir.Select(instance, get(10u + i), ReadRawU32(target)));
	}
}

// DS_BVH_STACK_PUSH8_POP1_RTN_B32: ADDR holds the stack control word (bits 0-4 entry count, 5-9
// entries since the instance boundary, 10-14 ring position, 15-28 the lane's stack base in dwords,
// 29 inside an instance, 30 overflowed, 31 returning from an instance); DATA0 the node just visited
// (a walk back resumes after its position among the children), DATA1..+7 its eight children. The
// children are pushed from the eighth down, then one node is popped: ADDR gets the new control
// word, VDST the popped node. Entry k of a lane's ring is LDS dword base + 32k; OFFSET is the
// capacity. Upstream's stack function, unrolled into straight-line IR (eight predicated pushes).
void Translator::DS_BVH_STACK(const Decoder::Instruction& inst) {
	const auto u    = [](uint32_t value) { return IR::U32(IR::Value(value)); };
	const auto bits = [&](IR::U32 value, uint32_t offset, uint32_t count) {
		return IR::U32(ir.Emit(IR::ValueOpcode::BitFieldUExtract, {value, u(offset), u(count)}));
	};
	const auto set = [&](IR::U32 value, IR::U32 field, uint32_t offset, uint32_t count) {
		return IR::U32(
		    ir.Emit(IR::ValueOpcode::BitFieldInsert, {value, field, u(offset), u(count)}));
	};
	const auto umin = [&](IR::U32 a, IR::U32 b) {
		return IR::U32(ir.Emit(IR::ValueOpcode::UMin32, {a, b}));
	};
	const auto select_u1 = [&](IR::U1 condition, IR::U1 a, IR::U1 b) {
		return IR::U1(ir.Emit(IR::ValueOpcode::SelectU1, {condition, a, b}));
	};
	const uint32_t capacity_value = inst.offset;
	const auto     capacity       = u(capacity_value);
	const auto     exec           = ir.GetExec();
	IR::MemoryInfo memory;
	memory.kind            = IR::ResourceKind::Lds;
	memory.data_dwords     = 1u;
	memory.data_bits       = 32u;
	memory.component_count = 1u;
	const auto byte_address = [&](IR::U32 control, IR::U32 ring) {
		const auto index = ir.IAdd(bits(control, 15u, 14u), ir.ShiftLeftLogical(ring, u(5u)));
		return ir.ShiftLeftLogical(index, u(2u));
	};
	const auto is_sentinel = [&](IR::U32 node) {
		return ir.IEqual(bits(node, 3u, 29u), u(0x1fffffffu));
	};

	auto       control = ReadU32(inst.src0);
	const auto last    = ReadU32(inst.src1);
	std::array<IR::U32, 8> children;
	for (uint32_t i = 0; i < children.size(); i++) {
		children[i] = ReadU32(OffsetOperand(inst.src2, i));
	}
	const auto walkback = ir.LogicalNot(is_sentinel(last));
	auto       found    = IR::U1(IR::Value(false));
	for (const auto& child: children) {
		found = ir.LogicalOr(found, ir.IEqual(child, last));
	}
	auto alive     = ir.LogicalOr(ir.LogicalNot(walkback), found);
	auto skip_low  = IR::U1(IR::Value(false));
	auto skip_high = IR::U1(IR::Value(false));
	for (uint32_t step = 0; step < 8u; step++) {
		const uint32_t index    = 7u - step;
		const bool     high     = index >= 4u;
		const auto     node     = children[index];
		const auto     sentinel = is_sentinel(node);
		const auto     marker   = ir.LogicalAnd(sentinel, ir.LogicalNot(high ? skip_high : skip_low));
		const auto     next_low =
		    ir.LogicalOr(skip_low, ir.LogicalAnd(marker, ir.IEqual(bits(node, 1u, 1u), u(0u))));
		const auto next_high =
		    high ? ir.LogicalOr(skip_high,
		                        ir.LogicalAnd(marker, ir.IEqual(bits(node, 2u, 1u), u(0u))))
		         : skip_high;
		const auto next_valid = ir.LogicalNot(ir.LogicalAnd(walkback, ir.IEqual(node, last)));
		const auto push =
		    ir.LogicalAnd(ir.LogicalAnd(alive, next_valid),
		                  ir.LogicalNot(ir.LogicalOr(sentinel, high ? next_high : next_low)));
		// Push: the node at the ring position, then the counters.
		const auto count  = bits(control, 0u, 5u);
		const auto parent = bits(control, 5u, 5u);
		const auto ring   = bits(control, 10u, 5u);
		ir.Emit(IR::ValueOpcode::WriteSharedU32,
		        {byte_address(control, ring), node, ir.LogicalAnd(exec, push)},
		        AddMemoryInfo(memory, inst.pc));
		auto pushed =
		    ir.BitwiseOr(control, ir.Select(ir.IEqual(count, capacity), u(1u << 30u), u(0u)));
		pushed = ir.Select(ir.IEqual(parent, capacity), ir.BitwiseAnd(pushed, u(~(1u << 29u))),
		                   pushed);
		pushed = set(pushed, umin(ir.IAdd(count, u(1u)), capacity), 0u, 5u);
		pushed = set(pushed, umin(ir.IAdd(parent, u(1u)), capacity), 5u, 5u);
		const auto next_ring = ir.IAdd(ring, u(1u));
		pushed  = set(pushed, ir.Select(ir.IEqual(next_ring, capacity), u(0u), next_ring), 10u, 5u);
		control = ir.Select(push, pushed, control);
		// Once the walk-back node is reached nothing more is pushed (upstream's loop exit).
		skip_low  = select_u1(alive, next_low, skip_low);
		skip_high = select_u1(alive, next_high, skip_high);
		alive     = ir.LogicalAnd(alive, next_valid);
	}

	// Pop.
	const auto count      = bits(control, 0u, 5u);
	const auto parent     = bits(control, 5u, 5u);
	const auto ring       = bits(control, 10u, 5u);
	const auto returning  = ir.LogicalAnd(ir.INotEqual(bits(control, 29u, 1u), u(0u)),
	                                      ir.IEqual(parent, u(0u)));
	control               = set(control, ir.Select(returning, u(1u), u(0u)), 31u, 1u);
	control               = ir.Select(returning, ir.BitwiseAnd(control, u(~(1u << 29u))), control);
	const auto overflow   = ir.INotEqual(bits(control, 30u, 1u), u(0u));
	const auto nonempty   = ir.INotEqual(count, u(0u));
	const auto empty_node = ir.Select(overflow, u(0xffffffffu), u(0xfffffffeu));
	auto       popped     = set(control, ir.ISub(count, u(1u)), 0u, 5u);
	popped                = set(popped, ir.ISub(parent, u(1u)), 5u, 5u);
	const auto previous_ring =
	    ir.Select(ir.IEqual(ring, u(0u)), u(capacity_value - 1u), ir.ISub(ring, u(1u)));
	popped            = set(popped, previous_ring, 10u, 5u);
	const auto loaded = IR::U32(ir.Emit(
	    IR::ValueOpcode::LoadSharedU32,
	    {byte_address(popped, previous_ring), ir.LogicalAnd(exec, nonempty)},
	    AddMemoryInfo(memory, inst.pc)));
	// Popping an instance node: the walk continues in the instance's parent BVH. Inside a nested
	// instance the entries pushed since its boundary are dropped (and the stack marked overflowed).
	const auto instance  = ir.IEqual(bits(loaded, 0u, 3u), u(6u));
	const auto nested    = ir.INotEqual(bits(popped, 29u, 1u), u(0u));
	const auto discarded = ir.BitwiseOr(set(popped, bits(popped, 5u, 5u), 0u, 5u), u(1u << 30u));
	const auto entered =
	    ir.BitwiseOr(set(ir.Select(nested, discarded, popped), u(0u), 5u, 5u), u(1u << 29u));
	popped = ir.Select(instance, entered, popped);
	WriteOperand(inst.src0,
	             ir.Select(nonempty, popped, ir.BitwiseAnd(control, u(0x5fffffffu))));
	WriteOperand(inst.dst, ir.Select(nonempty, loaded, empty_node));
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
