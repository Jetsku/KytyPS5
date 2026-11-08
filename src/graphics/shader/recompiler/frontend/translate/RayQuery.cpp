#include "common/assert.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/frontend/decode/RayTracingPatch.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"

#include <array>
#include <bit>

// The pseudo instructions of the native ray tracing patch (see RayTracingPatch.h).

namespace Libs::Graphics::ShaderRecompiler::Frontend {

namespace {

Decoder::Operand VectorOperand(uint32_t reg) {
	Decoder::Operand operand;
	operand.kind = Decoder::OperandKind::Vgpr;
	operand.reg  = reg;
	return operand;
}

Decoder::Operand ExecOperand() {
	Decoder::Operand operand;
	operand.kind = Decoder::OperandKind::ExecLo;
	return operand;
}

} // namespace

void Translator::EmitRayTracing(const Decoder::Instruction& inst) {
	using Op       = IR::ValueOpcode;
	const auto site = inst.offset;
	if (site >= IR::MaxRayQuerySites) {
		EXIT("ray tracing patch site %u out of range at pc 0x%08x", site, inst.pc);
	}
	program.info.ray_query_sites |= 1u << site;
	const auto flags = [site](auto variant) {
		return static_cast<uint64_t>(IR::RayQueryFlags(site, static_cast<uint32_t>(variant)));
	};
	const auto any_lane = [&](IR::U1 lanes) {
		const auto mask = BallotMask(lanes);
		return ir.INotEqual(ir.BitwiseOr(mask[0], mask[1]), IR::U32(IR::Value(0u)));
	};
	const auto vreg = [&](uint32_t reg) {
		return IR::U32(ir.GetVectorReg(static_cast<IR::VectorReg>(reg)));
	};
	const auto set_vreg = [&](uint32_t reg, IR::U1 lanes, IR::U32 value) {
		const auto r = static_cast<IR::VectorReg>(reg);
		ir.SetVectorReg(r, ir.Select(lanes, value, IR::U32(ir.GetVectorReg(r))));
	};
	const auto keep = [&](uint32_t slot, IR::U32 value) {
		ir.Emit(Op::RayQueryKeep, {value}, flags(slot));
	};
	const auto kept = [&](uint32_t slot) {
		return IR::U32(ir.Emit(Op::RayQueryKept, {}, flags(slot)));
	};
	const auto in_traversal = [&]() {
		return IR::U1(ir.Emit(Op::RayQueryFlag, {}, flags(IR::RayQueryFlagKind::InTraversal)));
	};
	const auto results = [&](const Decoder::Instruction& pseudo) {
		return std::array {vreg(pseudo.image_nsa_addr[0]), vreg(pseudo.image_nsa_addr[1]),
		                   vreg(pseudo.image_nsa_addr[2])};
	};

	switch (inst.opcode) {
		case Decoder::Opcode::RT_INIT: {
			std::array<IR::U32, 8> ray;
			for (uint32_t i = 0; i < 8; i++) {
				// A site without a mask register traces against every instance.
				ray[i] = i == 7 && inst.image_nsa_addr[i] == Decoder::NoRegister ? IR::U32(IR::Value(0xffu))
				                                                                   : vreg(inst.image_nsa_addr[i]);
			}
			const auto header = ReadU32Pair(inst.src0);
			const auto native = IR::U1(ir.Emit(Op::RayQueryInit,
			                                   {ray[0], ray[1], ray[2], ray[3], ray[4], ray[5], ray[6], ray[7],
			                                    header[0], header[1], ir.GetExec()},
			                                   flags(0u)));
			// The same in every lane: the host has (or not) the TLAS of the header.
			ir.SetScc(any_lane(native));
			if (GetCodegenOptions().rt_hw_verify) {
				ir.Emit(Op::RayQueryCount, {ir.GetExec()}, flags(IR::RayQueryCounter::EntryLanes));
				ir.Emit(Op::RayQueryCount, {native}, flags(IR::RayQueryCounter::NativeLanes));
			}
			return;
		}
		case Decoder::Opcode::RT_SELECT: {
			// The wave's smallest instance with a candidate: a uniform value.
			const auto next     = ir.Emit(Op::RayQueryNext, {}, flags(0u));
			const auto instance = IR::U32(ir.CompositeExtract(next, 0));
			const auto has      = ir.INotEqual(IR::U32(ir.CompositeExtract(next, 1)), IR::U32(IR::Value(0u)));
			const auto match    = IR::U1(ir.Emit(Op::RayQueryMatch, {}, flags(0u)));
			WriteOperand(inst.dst, instance);
			WriteMask(ExecOperand(), match);
			ir.SetScc(has);
			return;
		}
		case Decoder::Opcode::RT_LEAF: {
			// The lanes of the instance (the setup kept EXEC) test their own candidates.
			const auto lanes = IR::U1(ir.Emit(Op::RayQueryMatch, {}, flags(1u)));
			const auto node  = IR::U32(ir.Emit(Op::RayQueryCandidate, {}, flags(IR::RayQueryCandidateKind::Node)));
			const auto word  = IR::U32(ir.Emit(Op::RayQueryCandidate, {}, flags(IR::RayQueryCandidateKind::Word)));
			set_vreg(inst.dst.reg, lanes, node);
			set_vreg(inst.dst2.reg, lanes, word);
			WriteMask(ExecOperand(), lanes);
			const auto r = results(inst);
			ir.Emit(Op::RayQuerySave, {r[0], r[1], r[2]}, flags(0u));
			return;
		}
		case Decoder::Opcode::RT_RESOLVE: {
			const auto r = results(inst);
			ir.Emit(Op::RayQueryResolve, {r[0], r[1], r[2]}, flags(inst.dmask != 0 ? 1u : 0u));
			return;
		}
		case Decoder::Opcode::RT_EXIT:
			WriteMask(ExecOperand(), in_traversal());
			ir.Emit(Op::RayQueryExit, {}, flags(0u));
			return;
		case Decoder::Opcode::RT_VERIFY_SAVE: {
			if (inst.dmask == 2) {
				// The entry values.
				for (uint32_t i = 0; i < 8; i++) {
					if (inst.image_nsa_addr[i] != Decoder::NoRegister) {
						keep(i, vreg(inst.image_nsa_addr[i]));
					}
				}
				keep(IR::RayQueryKeepFlag, IR::U32(IR::Value(0u)));
				return;
			}
			// The native results; the guest loop runs again from the entry values.
			const auto lanes = in_traversal();
			for (uint32_t i = 0; i < 8; i++) {
				const auto reg = inst.image_nsa_addr[i];
				if (reg != Decoder::NoRegister) {
					keep(8 + i, vreg(reg));
					set_vreg(reg, lanes, kept(i));
				}
			}
			keep(IR::RayQueryKeepFlag, ir.Select(lanes, IR::U32(IR::Value(1u)), IR::U32(IR::Value(0u))));
			WriteOperand(inst.dst, IR::U32(IR::Value(inst.secondary_offset)));
			if (inst.dst2.kind == Decoder::OperandKind::Sgpr) {
				WriteOperand(inst.dst2, IR::U32(IR::Value(inst.data_format)));
			}
			return;
		}
		case Decoder::Opcode::RT_VERIFY_CHECK: {
			const auto lanes = ir.LogicalAnd(ir.INotEqual(kept(IR::RayQueryKeepFlag), IR::U32(IR::Value(0u))),
			                                 ir.GetExec());
			// 0 t, 1 u, 2 v, 3 word, 4 node, 5 instance low, 6 high, 7 alpha counter.
			std::array<IR::U32, 7> guest, native, entry;
			for (uint32_t i = 0; i < 7; i++) {
				const bool has = inst.image_nsa_addr[i] != Decoder::NoRegister;
				guest[i]       = has ? vreg(inst.image_nsa_addr[i]) : IR::U32(IR::Value(0u));
				native[i]      = has ? kept(8 + i) : IR::U32(IR::Value(0u));
				entry[i]       = has ? kept(i) : IR::U32(IR::Value(0u));
			}
			const auto differs = [&](const std::array<IR::U32, 7>& a, const std::array<IR::U32, 7>& b,
			                         std::initializer_list<uint32_t> fields) {
				IR::U1 any = IR::U1(IR::Value(false));
				for (const auto f: fields) {
					any = ir.LogicalOr(any, ir.INotEqual(a[f], b[f]));
				}
				return any;
			};
			const auto guest_hit  = differs(guest, entry, {0, 4, 5, 6});
			const auto native_hit = differs(native, entry, {0, 4, 5, 6});
			const auto both       = ir.LogicalAnd(guest_hit, native_hit);
			const auto same_inst  = ir.LogicalNot(differs(guest, native, {5, 6}));
			const auto same_prim  = ir.LogicalAnd(same_inst, ir.LogicalNot(differs(guest, native, {3, 4})));
			const auto f32        = [&](IR::U32 value) { return IR::F32(ir.Emit(Op::BitCastF32U32, {value})); };
			const auto far_apart  = [&](uint32_t field, float scale_relative, float absolute) {
				const auto a     = f32(guest[field]);
				const auto b     = f32(native[field]);
				const auto delta = IR::F32(ir.Emit(Op::FPAbs32, {IR::F32(ir.Emit(Op::FPSub32, {a, b}))}));
				const auto limit = scale_relative != 0.0f
				                       ? IR::F32(ir.Emit(Op::FPMul32, {IR::F32(ir.Emit(Op::FPAbs32, {a})),
				                                                       IR::F32(IR::Value::F32(scale_relative))}))
				                       : IR::F32(IR::Value::F32(absolute));
				return IR::U1(ir.Emit(Op::FPOrdGreaterThan32, {delta, limit}));
			};
			const auto count = [&](IR::RayQueryCounter counter, IR::U1 value) {
				ir.Emit(Op::RayQueryCount, {ir.LogicalAnd(lanes, value)}, flags(counter));
			};
			const auto both_prim = ir.LogicalAnd(both, same_prim);
			count(IR::RayQueryCounter::Rays, IR::U1(IR::Value(true)));
			count(IR::RayQueryCounter::NativeHits, native_hit);
			count(IR::RayQueryCounter::GuestHits, guest_hit);
			count(IR::RayQueryCounter::HitMismatch, ir.LogicalOr(ir.LogicalAnd(guest_hit, ir.LogicalNot(native_hit)),
			                                                     ir.LogicalAnd(native_hit, ir.LogicalNot(guest_hit))));
			count(IR::RayQueryCounter::InstanceDiff, ir.LogicalAnd(both, ir.LogicalNot(same_inst)));
			count(IR::RayQueryCounter::PrimitiveDiff, ir.LogicalAnd(ir.LogicalAnd(both, same_inst), ir.LogicalNot(same_prim)));
			count(IR::RayQueryCounter::DistanceDiff, ir.LogicalAnd(both_prim, far_apart(0, 1e-4f, 0.0f)));
			if (inst.image_nsa_addr[1] != Decoder::NoRegister && inst.image_nsa_addr[2] != Decoder::NoRegister) {
				count(IR::RayQueryCounter::BaryDiff,
				      ir.LogicalAnd(both_prim, ir.LogicalOr(far_apart(1, 0.0f, 1e-3f), far_apart(2, 0.0f, 1e-3f))));
			}
			keep(IR::RayQueryKeepFlag, IR::U32(IR::Value(0u)));
			return;
		}
		default: FailMissingTranslation(inst);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
