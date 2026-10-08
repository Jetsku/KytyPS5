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

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
