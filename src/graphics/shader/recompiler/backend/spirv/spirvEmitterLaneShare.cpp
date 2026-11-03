// KYTY_LANE2_SHARE (adapted from Erlou93/KytyPS5-Yotei spirvEmitterLaneShare.cpp): a wave64 compute
// program that runs two guest lanes per host invocation (lane_count 2: lanes l and l + 32) emits every
// IR instruction once per half (EmitBlock). A value that is the same in both halves (scalar state,
// loop counters, addresses built from user data) then costs two registers and twice the code, and the
// host compiler cannot merge the two copies of a loop-carried value. This pass proves which values are
// the same in both halves; the high half reuses the low half's id instead of emitting the instruction.
//
// Both halves run in one instruction stream: every branch decides on one condition for both halves
// (BranchCondition combines the halves' ballots) and every phi takes the same incoming edge in both. So
// a value is the same in both halves when the instruction computing it
//   1. is a pure function of its IR operands (it reads no lane id, half index, per-half variable or
//      other lane), and
//   2. has operands that are the same in both halves (constants, user data, workgroup ids, whole-wave
//      results, or results of such instructions; phi cycles are solved optimistically).
// The high half's copy of such an instruction would be the same SPIR-V on the same ids.
//
// Not shared, from reading this tree's emitters:
//   * LaneId, GetBuiltin LocalInvocationId/Index/GlobalInvocationId (EmitSubgroupLocalInvocationId and
//     EmitLocalInvocationIndex add lane_half * 32).
//   * Cross-lane operations (DPP, WriteLane, permutes, swizzles, append/consume).
//   * Buffer loads whose V# adds the lane id (ADD_TID, packed stride bit 20: BufferLane) and every
//     IndirectBuffer load (the V#, and so ADD_TID, is a run-time value).
//   * Scratch (scratch_variable[lane_half]), LDS (lane-slot plans, per-lane contents) and FLAT loads
//     (aperture routing to LDS/scratch), image reads and samples, atomics and every store.
//   * Void instructions, resource handles (they emit nothing) and anything not listed below.
// Ballot, ReadFirstLane, ReadConstBuffer and scalar-address loads already give both halves one result
// (EmitDirectInstruction); they count as uniform operands here.
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

enum class ShareRule : uint8_t {
	Never,    // lane dependent, side effects or unknown
	Pure,     // the same in both halves when every operand is; emitted once
	Source,   // the same in both halves whatever the operands (user data, workgroup id, undef)
	Whole,    // the emitters already give both halves one result
	ReadLane, // the same in both halves when the lane operand is (the value operand may be per lane)
	Phi,      // the same when every incoming value is
};

ShareRule MemoryRule(const EmitterState& state, const IR::Inst& inst) {
	using Op         = IR::ValueOpcode;
	const auto index = inst.Flags<IR::MemoryFlags>().index;
	if (index >= state.program.memory_info.size()) {
		return ShareRule::Never;
	}
	const auto& mem = state.program.memory_info[index];
	switch (inst.GetOpcode()) {
		case Op::ReadConstBuffer: return ShareRule::Whole;
		case Op::LoadAddressU8:
		case Op::LoadAddressU16:
		case Op::LoadAddressU32:
			// ScalarAddress and Global reads go through the BDA path with an address built from the
			// operands only. Flat may route to LDS/scratch by aperture; Scratch is per half.
			if (mem.kind == IR::ResourceKind::ScalarAddress) {
				return inst.GetOpcode() == Op::LoadAddressU32 ? ShareRule::Whole : ShareRule::Pure;
			}
			return mem.kind == IR::ResourceKind::Global ? ShareRule::Pure : ShareRule::Never;
		case Op::LoadBufferU8:
		case Op::LoadBufferU16:
		case Op::LoadBufferU32:
		case Op::LoadBufferU32x2:
		case Op::LoadBufferU32x3:
		case Op::LoadBufferU32x4:
			if (mem.kind != IR::ResourceKind::Buffer || mem.resource >= state.program.info.buffers.size()) {
				return ShareRule::Never;
			}
			return (StorageBufferPackedStride(state, mem) & (1u << 20u)) == 0u ? ShareRule::Pure
			                                                                   : ShareRule::Never;
		default: return ShareRule::Never;
	}
}

ShareRule RuleFor(const EmitterState& state, const IR::Inst& inst) {
	using Op = IR::ValueOpcode;
	switch (inst.GetOpcode()) {
		case Op::Phi: return ShareRule::Phi;
		case Op::Ballot:
		case Op::ReadFirstLane: return ShareRule::Whole;
		case Op::ReadLane: return ShareRule::ReadLane;
		case Op::UndefU1:
		case Op::UndefU8:
		case Op::UndefU16:
		case Op::UndefU32:
		case Op::UndefU64:
		case Op::GetUserData:
		case Op::GetShaderBase:
		case Op::GetDispatchThreadExtent: return ShareRule::Source;
		case Op::GetBuiltin: {
			if (!inst.Arg(0).IsImmediate()) {
				return ShareRule::Never;
			}
			const auto kind = static_cast<IR::StageInputKind>(inst.Arg(0).U32());
			return kind == IR::StageInputKind::WorkgroupId || kind == IR::StageInputKind::NumWorkgroups
			           ? ShareRule::Source
			           : ShareRule::Never;
		}
		case Op::ReadConstBuffer:
		case Op::LoadAddressU8:
		case Op::LoadAddressU16:
		case Op::LoadAddressU32:
		case Op::LoadBufferU8:
		case Op::LoadBufferU16:
		case Op::LoadBufferU32:
		case Op::LoadBufferU32x2:
		case Op::LoadBufferU32x3:
		case Op::LoadBufferU32x4: return MemoryRule(state, inst);
		case Op::Identity:
		case Op::BitCastU16F16:
		case Op::BitCastF16U16:
		case Op::BitCastU32F32:
		case Op::BitCastF32U32:
		case Op::BitCastU64F64:
		case Op::BitCastF64U64:
		case Op::ConvertU16U32:
		case Op::ConvertU32U16:
		case Op::ConvertU8U32:
		case Op::ConvertU32U8:
		case Op::ConvertF32F16:
		case Op::ConvertF16F32:
		case Op::ConvertS32F32:
		case Op::ConvertU32F32:
		case Op::ConvertF32S32:
		case Op::ConvertF64S32:
		case Op::ConvertF32F64:
		case Op::ConvertF64F32:
		case Op::ConvertF32U32:
		case Op::ConvertF64U32:
		case Op::CompositeConstructU64:
		case Op::CompositeConstructU32x2:
		case Op::CompositeConstructU32x3:
		case Op::CompositeConstructF32x2:
		case Op::CompositeConstructU32x4:
		case Op::CompositeExtractU64:
		case Op::CompositeExtractU32x2:
		case Op::CompositeExtractU32x3:
		case Op::CompositeExtractU32x4:
		case Op::PackHalf2x16:
		case Op::PackSnorm2x16:
		case Op::PackUnorm2x16:
		case Op::PackFloat2x16Rtz:
		case Op::FPAbs32:
		case Op::FPNeg32:
		case Op::FPSaturate32:
		case Op::BitFieldInsert:
		case Op::BitFieldUExtract:
		case Op::BitFieldSExtract:
		case Op::WqmU64:
		case Op::SelectU1:
		case Op::SelectF32:
		case Op::SelectU32:
		case Op::IAdd32:
		case Op::IAdd64:
		case Op::ISub32:
		case Op::ISub64:
		case Op::IMul32:
		case Op::IMul64:
		case Op::UDiv32:
		case Op::SMulHi:
		case Op::UMulHi:
		case Op::IAbs32:
		case Op::ShiftLeftLogical32:
		case Op::ShiftLeftLogical64:
		case Op::ShiftRightLogical32:
		case Op::ShiftRightLogical64:
		case Op::ShiftRightArithmetic32:
		case Op::ShiftRightArithmetic64:
		case Op::BitwiseAnd32:
		case Op::BitwiseAnd64:
		case Op::BitwiseOr32:
		case Op::BitwiseXor32:
		case Op::BitwiseNot32:
		case Op::BitReverse32:
		case Op::BitCount32:
		case Op::BitCount64:
		case Op::FindUMsb32:
		case Op::FindUMsb64:
		case Op::FindILsb32:
		case Op::SMin32:
		case Op::UMin32:
		case Op::SMax32:
		case Op::UMax32:
		case Op::SMinTri32:
		case Op::UMinTri32:
		case Op::SMaxTri32:
		case Op::UMaxTri32:
		case Op::SMedTri32:
		case Op::UMedTri32:
		case Op::SLessThan32:
		case Op::SLessThan64:
		case Op::SLessThanEqual64:
		case Op::ULessThanEqual64:
		case Op::UGreaterThanEqual64:
		case Op::ULessThan32:
		case Op::ULessThan64:
		case Op::IEqual32:
		case Op::IEqual64:
		case Op::SLessThanEqual32:
		case Op::ULessThanEqual32:
		case Op::SGreaterThan32:
		case Op::UGreaterThan32:
		case Op::UGreaterThan64:
		case Op::INotEqual32:
		case Op::INotEqual64:
		case Op::SGreaterThanEqual32:
		case Op::UGreaterThanEqual32:
		case Op::LogicalOr:
		case Op::LogicalAnd:
		case Op::LogicalXor:
		case Op::LogicalNot:
		case Op::FPOrdEqual32:
		case Op::FPOrdEqual64:
		case Op::FPOrdLessThanEqual64:
		case Op::FPOrdGreaterThanEqual64:
		case Op::FPOrdNotEqual64:
		case Op::FPOrdLessThan64:
		case Op::FPOrdGreaterThan64:
		case Op::FPUnordEqual64:
		case Op::FPUnordNotEqual64:
		case Op::FPUnordLessThan64:
		case Op::FPUnordGreaterThan64:
		case Op::FPUnordLessThanEqual64:
		case Op::FPUnordGreaterThanEqual64:
		case Op::FPUnordEqual32:
		case Op::FPOrdNotEqual32:
		case Op::FPUnordNotEqual32:
		case Op::FPOrdLessThan32:
		case Op::FPUnordLessThan32:
		case Op::FPOrdGreaterThan32:
		case Op::FPUnordGreaterThan32:
		case Op::FPOrdLessThanEqual32:
		case Op::FPUnordLessThanEqual32:
		case Op::FPOrdGreaterThanEqual32:
		case Op::FPUnordGreaterThanEqual32:
		case Op::FPIsNan32:
		case Op::FPIsNan64:
		case Op::FPCmpClass32:
		case Op::FPCmpClass16:
		case Op::FPAdd32:
		case Op::FPSub32:
		case Op::FPFma32:
		case Op::FPMad32:
		case Op::FPMul32:
		case Op::FPAdd64:
		case Op::FPMul64:
		case Op::FPFma64:
		case Op::FPRecip64:
		case Op::FPMin32:
		case Op::FPMax32:
		case Op::FPMin64:
		case Op::FPMax64:
		case Op::FPMinTri32:
		case Op::FPMaxTri32:
		case Op::FPMedTri32:
		case Op::FPRecip32:
		case Op::FPRecipIFlag32:
		case Op::FPRecipSqrt32:
		case Op::FPSqrt:
		case Op::FPSin:
		case Op::FPCos:
		case Op::FPExp2:
		case Op::FPLog2:
		case Op::FPLdexp:
		case Op::FPRoundEven32:
		case Op::FPFloor32:
		case Op::FPCeil32:
		case Op::FPTrunc32:
		case Op::FPFloor64:
		case Op::FPCeil64:
		case Op::FPTrunc64:
		case Op::FPFract32:
		case Op::FPFract64: return ShareRule::Pure;
		default: return ShareRule::Never;
	}
}

struct Candidate {
	const IR::Inst* inst = nullptr;
	ShareRule       rule = ShareRule::Never;
};

// Whether `value` is the same in both halves under the current assumptions.
bool OperandUniform(const std::unordered_set<const IR::Inst*>& uniform, IR::Value value) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		// Constants are; a register operand is not an IR value at all.
		const auto type = value.GetType();
		return type != IR::Type::ScalarReg && type != IR::Type::VectorReg;
	}
	const auto* operand = value.TryInstruction();
	return operand != nullptr && uniform.contains(operand);
}

bool StillUniform(const std::unordered_set<const IR::Inst*>& uniform, const Candidate& candidate) {
	const auto& inst = *candidate.inst;
	switch (candidate.rule) {
		case ShareRule::Source:
		case ShareRule::Whole: return true;
		case ShareRule::ReadLane: return inst.NumArgs() > 1 && OperandUniform(uniform, inst.Arg(1));
		case ShareRule::Never: return false;
		case ShareRule::Pure:
		case ShareRule::Phi:
			for (size_t index = 0; index < inst.NumArgs(); index++) {
				if (!OperandUniform(uniform, inst.Arg(index))) {
					return false;
				}
			}
			return true;
	}
	return false;
}

std::atomic_uint64_t g_lane_share_programs {0};
std::atomic_uint64_t g_lane_share_reused {0};

} // namespace

void LaneShareClassify(EmitterState& state) {
	auto& share = state.lane_share;
	share       = {};
	if (!GetCodegenOptions().lane2_share || state.lane_count != 2 || state.split_wave ||
	    state.program.dispatcher_fallback || state.program.stage != ShaderType::Compute) {
		return;
	}
	share.active = true;
	std::vector<Candidate> candidates;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			share.instructions++;
			const auto rule = RuleFor(state, inst);
			if (rule == ShareRule::Never || inst.GetType() == IR::Type::Void) {
				continue;
			}
			candidates.push_back({&inst, rule});
		}
	}
	// Optimistic fixed point: every candidate starts out uniform; one with an operand that is not is
	// dropped, until nothing changes (a loop phi fed by itself and uniform values stays uniform).
	auto& uniform = share.uniform;
	uniform.reserve(candidates.size() * 2);
	for (const auto& candidate: candidates) {
		uniform.insert(candidate.inst);
	}
	for (bool changed = true; changed;) {
		changed = false;
		for (const auto& candidate: candidates) {
			if (uniform.contains(candidate.inst) && !StillUniform(uniform, candidate)) {
				uniform.erase(candidate.inst);
				changed = true;
			}
		}
	}
	for (const auto& candidate: candidates) {
		if (uniform.contains(candidate.inst) && candidate.rule != ShareRule::Whole) {
			share.shared.insert(candidate.inst);
		}
	}
}

bool LaneShareReuse(ValueEmitContext& low, ValueEmitContext& high, const IR::Inst& inst) {
	auto& share = low.state.lane_share;
	if (!share.active || !share.shared.contains(&inst)) {
		return false;
	}
	// An instruction the low half emitted no value for (a planning-only load, a member of an
	// indexed select chain) is emitted as before.
	const auto found = low.definitions.find(&inst);
	if (found == low.definitions.end()) {
		return false;
	}
	high.Define(inst, found->second);
	share.reused++;
	return true;
}

void LaneShareReport(EmitterState& state) {
	const auto& share = state.lane_share;
	if (!share.active) {
		return;
	}
	g_lane_share_programs.fetch_add(1, std::memory_order_relaxed);
	g_lane_share_reused.fetch_add(share.reused, std::memory_order_relaxed);
	static const bool log = [] {
		const auto* value = std::getenv("KYTY_LANE2_SHARE_LOG");
		return value != nullptr && value[0] == '1';
	}();
	if (log) {
		std::printf("LaneShare CS %016" PRIx64 ": %zu instructions, %zu uniform, %zu shared, %" PRIu64
		            " reused\n",
		            state.program.shader_hash, share.instructions, share.uniform.size(),
		            share.shared.size(), share.reused);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
