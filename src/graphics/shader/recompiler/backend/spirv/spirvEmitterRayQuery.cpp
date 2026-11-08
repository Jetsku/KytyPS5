#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"
// windows.h defines these.
#undef near
#undef far

// Native ray tracing (KYTY_RT_HW; RayTracingPatch.h, NativeRayTracing), adapted from TheCruZ's
// native-hardware-raytracing branch. Each patched traversal (site) keeps, per lane: a host ray
// query over the TLAS the host built from the guest's BVH, paused at the lane's current candidate
// (a triangle the hardware found; every triangle is a candidate, none is committed by the
// hardware), the candidate's guest instance node, triangle node and leaf entry info word, and a
// state word. The wave processes the smallest instance any lane's candidate is in, with every lane
// whose candidate is in it: the guest instance setup runs once, then each lane's guest leaf code
// tests its own triangle; a lane whose leaf code accepted the hit confirms the candidate (the
// query's t max shrinks to it). The host publishes its acceleration structures in a context whose
// device address is the BDA page table entry after the last page (NativeRayTracing).

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

constexpr uint32_t DoneBit      = 1u << 0u; // The query has no more candidates (or never ran).
constexpr uint32_t TraversalBit = 1u << 1u; // The lane started the native traversal.
constexpr uint32_t PendingBit   = 1u << 2u; // The query is paused at a candidate.
constexpr uint32_t MatchedBit   = 1u << 3u; // The candidate is in the selected instance.
constexpr uint32_t None         = 0xffffffffu;

// The context (rt_tlas_walk.comp ContextEntry), the slot info (rt_common.inc SlotInfo) and the
// verify counters (after the contexts).
constexpr uint32_t ContextHeader        = 0;
constexpr uint32_t ContextTlas          = 8;
constexpr uint32_t ContextInstanceNodes = 16;
constexpr uint32_t ContextSlots         = 24;
constexpr uint32_t ContextValid         = 36;
constexpr uint32_t ContextCounters      = 1024;
constexpr uint32_t SlotStride           = 64;
constexpr uint32_t SlotEntryTable       = 8;
constexpr uint32_t SlotCapacity0        = 16;
constexpr uint32_t SlotCapacity1        = 20;

constexpr uint32_t CandidateIntersection = 0; // RayQueryCandidateIntersectionKHR

// gl_RayFlagsNoOpaqueEXT: every triangle is a candidate the guest leaf code tests.
constexpr uint32_t RayFlagNoOpaque = 0x2;
// A wave that keeps selecting instances stops after this many rounds instead of hanging the GPU
// (its lanes keep the nearest hit accepted so far).
constexpr uint32_t MaxRounds = 8192;

struct Lane {
	uint32_t query      = 0;
	uint32_t state      = 0;
	uint32_t instance   = 0;
	uint32_t node       = 0;
	uint32_t word       = 0;
	uint32_t saved_t    = 0;
	uint32_t saved_node = 0;
	uint32_t saved_inst = 0;
	uint32_t nodes      = 0;
	uint32_t slots      = 0;
	uint32_t keep       = 0;
};

uint32_t SiteOf(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto site = IR::RayQuerySite(inst.Flags<uint32_t>());
	if (site >= IR::MaxRayQuerySites || ctx.state.ray_query[site].query[0] == 0) {
		ctx.Fail(inst, "ray query site without variables");
	}
	return site;
}

Lane LaneOf(EmitterState& s, uint32_t site, uint32_t half) {
	const auto& v = s.ray_query[site];
	return {v.query[half],   v.state[half],      v.instance[half],   v.node[half],
	        v.word[half],    v.saved_t[half],    v.saved_node[half], v.saved_inst[half],
	        v.nodes[half],   v.slots[half],      v.keep[half]};
}

uint32_t Load(EmitterState& s, uint32_t type, uint32_t pointer) {
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpLoad, type, result, pointer);
	return result;
}

void Store(EmitterState& s, uint32_t pointer, uint32_t value) {
	s.builder.AddFunction(spv::OpStore, pointer, value);
}

uint32_t U32(EmitterState& s, uint32_t value) {
	return ConstantU32(s, value);
}

uint32_t And(EmitterState& s, uint32_t value, uint32_t mask) {
	return Binary(s, spv::OpBitwiseAnd, TypeU32(s), value, U32(s, mask));
}

uint32_t Or(EmitterState& s, uint32_t value, uint32_t bits) {
	return Binary(s, spv::OpBitwiseOr, TypeU32(s), value, bits);
}

uint32_t BitSet(EmitterState& s, uint32_t value, uint32_t bit) {
	return Binary(s, spv::OpINotEqual, TypeBool(s), And(s, value, bit), U32(s, 0));
}

uint32_t Equal(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpIEqual, TypeBool(s), a, b);
}

uint32_t NotEqual(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpINotEqual, TypeBool(s), a, b);
}

uint32_t Less(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpULessThan, TypeBool(s), a, b);
}

uint32_t LogicalAnd(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpLogicalAnd, TypeBool(s), a, b);
}

uint32_t LogicalOr(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpLogicalOr, TypeBool(s), a, b);
}

uint32_t LogicalNot(EmitterState& s, uint32_t a) {
	return Unary(s, spv::OpLogicalNot, TypeBool(s), a);
}

uint32_t Add(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpIAdd, TypeU32(s), a, b);
}

// A 4-byte aligned load through a device address.
uint32_t LoadAddressU32(EmitterState& s, uint32_t address) {
	const auto pointer = Unary(s, spv::OpConvertUToPtr, TypePhysicalU32Pointer(s), address);
	const auto result  = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpLoad, TypeU32(s), result, pointer, spv::MemoryAccessAlignedMask, 4u);
	return result;
}

uint32_t LoadAddressU64(EmitterState& s, uint32_t address) {
	const auto low  = LoadAddressU32(s, address);
	const auto high = LoadAddressU32(s, Binary(s, spv::OpIAdd, TypeU64(s), address, ConstantU64(s, 4)));
	return PackU64(s, low, high);
}

uint32_t AddressPlus(EmitterState& s, uint32_t address, uint32_t offset) {
	return Binary(s, spv::OpIAdd, TypeU64(s), address, ConstantU64(s, offset));
}

uint32_t ScaledAddress(EmitterState& s, uint32_t base, uint32_t index, uint32_t scale) {
	const auto wide = TypeU64(s);
	return Binary(s, spv::OpIAdd, wide, base,
	              Binary(s, spv::OpIMul, wide, Unary(s, spv::OpUConvert, wide, index), ConstantU64(s, scale)));
}

uint32_t Finite(EmitterState& s, uint32_t value) {
	const auto b = TypeBool(s);
	return LogicalAnd(s, LogicalNot(s, Unary(s, spv::OpIsNan, b, value)), LogicalNot(s, Unary(s, spv::OpIsInf, b, value)));
}

// The context the host published (the BDA page table entry after the last page), 0 when none.
uint32_t ContextAddress(EmitterState& s) {
	const auto entry = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(s, 64), entry,
	                      s.bda_pagetable_variable, U32(s, 0),
	                      U32(s, static_cast<uint32_t>(BufferCache::NATIVE_RT_CONTEXT_PAGE)));
	return Load(s, TypeU64(s), entry);
}

// A structured loop: `body(merge, cont)` emits the body from the current label and ends with a
// branch to one of them (the continue target goes back to the header).
template <typename Body>
void EmitLoop(EmitterState& s, Body&& body) {
	const auto header = s.builder.AllocateId(), start = s.builder.AllocateId();
	const auto cont = s.builder.AllocateId(), merge = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(s, header);
	s.builder.AddFunction(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
	s.builder.AddFunction(spv::OpBranch, start);
	EmitLabel(s, start);
	body(merge, cont);
	EmitLabel(s, cont);
	s.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(s, merge);
}

uint32_t QueryU32(EmitterState& s, const Lane& lane, spv::Op op, uint32_t intersection) {
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(op, TypeU32(s), result, lane.query, U32(s, intersection));
	return result;
}

// Advances the query of a lane that is not paused at a candidate to its next candidate (a
// triangle whose entry is in its host BLAS table), or to its end.
void Advance(EmitterState& s, const Lane& lane) {
	const auto u     = TypeU32(s), b = TypeBool(s), wide = TypeU64(s);
	const auto state = Load(s, u, lane.state);
	const auto go    = LogicalAnd(s, BitSet(s, state, TraversalBit), LogicalNot(s, BitSet(s, state, DoneBit | PendingBit)));
	EmitIfCondition(s, go, [&] {
		EmitLoop(s, [&](uint32_t merge, uint32_t cont) {
			const auto proceed = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpRayQueryProceedKHR, b, proceed, lane.query);
			EmitIfCondition(s, LogicalNot(s, proceed), [&] {
				Store(s, lane.state, Or(s, Load(s, u, lane.state), U32(s, DoneBit)));
			});
			const auto found = EmitValueOrDefaultIfCondition(s, proceed, b, ConstantBool(s, false), [&] {
				const auto type     = QueryU32(s, lane, spv::OpRayQueryGetIntersectionTypeKHR, CandidateIntersection);
				const auto triangle = Equal(
				    s, type, U32(s, spv::RayQueryCandidateIntersectionTypeRayQueryCandidateIntersectionTriangleKHR));
				return EmitValueOrDefaultIfCondition(s, triangle, b, ConstantBool(s, false), [&] {
					// The guest list entry of the triangle (rt_blas_walk.comp's entry table).
					const auto slot      = QueryU32(s, lane, spv::OpRayQueryGetIntersectionInstanceCustomIndexKHR,
					                                CandidateIntersection);
					const auto geometry  = QueryU32(s, lane, spv::OpRayQueryGetIntersectionGeometryIndexKHR,
					                                CandidateIntersection);
					const auto primitive = QueryU32(s, lane, spv::OpRayQueryGetIntersectionPrimitiveIndexKHR,
					                                CandidateIntersection);
					const auto instance  = QueryU32(s, lane, spv::OpRayQueryGetIntersectionInstanceIdKHR,
					                                CandidateIntersection);
					const auto info      = ScaledAddress(s, Load(s, wide, lane.slots), slot, SlotStride);
					const auto table     = LoadAddressU64(s, AddressPlus(s, info, SlotEntryTable));
					const auto capacity0 = LoadAddressU32(s, AddressPlus(s, info, SlotCapacity0));
					const auto capacity1 = LoadAddressU32(s, AddressPlus(s, info, SlotCapacity1));
					const auto opaque    = Equal(s, geometry, U32(s, 0));
					const auto index     = Select(s, u, opaque, primitive, Add(s, capacity0, primitive));
					const auto valid     = LogicalAnd(s, Binary(s, spv::OpINotEqual, b, table, ConstantU64(s, 0)),
					                                  Less(s, primitive, Select(s, u, opaque, capacity0, capacity1)));
					EmitIfCondition(s, valid, [&] {
						const auto entry = ScaledAddress(s, table, index, 16);
						Store(s, lane.word, LoadAddressU32(s, AddressPlus(s, entry, 4)));
						Store(s, lane.node, LoadAddressU32(s, AddressPlus(s, entry, 8)));
						Store(s, lane.instance, LoadAddressU32(s, ScaledAddress(s, Load(s, wide, lane.nodes), instance, 4)));
						Store(s, lane.state, Or(s, Load(s, u, lane.state), U32(s, PendingBit)));
					});
					return valid;
				});
			});
			// Again until a candidate is found or the query ends.
			s.builder.AddFunction(spv::OpBranchConditional, LogicalAnd(s, proceed, LogicalNot(s, found)), cont, merge);
		});
	});
}

uint32_t GroupMin(EmitterState& s, uint32_t value) {
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpGroupNonUniformUMin, TypeU32(s), result, U32(s, spv::ScopeSubgroup),
	                      spv::GroupOperationReduce, value);
	return result;
}

uint32_t KeepPointer(EmitterState& s, const Lane& lane, uint32_t slot) {
	const auto pointer = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpAccessChain, TypePointer(s, spv::StorageClassFunction, TypeU32(s)), pointer,
	                      lane.keep, U32(s, slot));
	return pointer;
}

bool UsesKeep(const IR::Program& program) {
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == IR::ValueOpcode::RayQueryKeep) {
				return true;
			}
		}
	}
	return false;
}

} // namespace

void DeclareRayQueryVariables(EmitterState& s) {
	if (s.program.info.ray_query_sites == 0) {
		return;
	}
	s.builder.RequireExtension("SPV_KHR_ray_query");
	s.builder.RequireCapability(spv::CapabilityRayQueryKHR);
	s.builder.RequireCapability(spv::CapabilityGroupNonUniform);
	s.builder.RequireCapability(spv::CapabilityGroupNonUniformArithmetic);
	const auto u          = TypeU32(s);
	const auto query_type = s.builder.Type(spv::OpTypeRayQueryKHR);
	s.ray_query_keep      = UsesKeep(s.program);
	const auto keep_type  = s.builder.Type(spv::OpTypeArray, u, ConstantU32(s, IR::RayQueryKeepSlots));
	const auto declare    = [&](uint32_t pointee, uint32_t& id) {
		id = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpVariable, TypePointer(s, spv::StorageClassFunction, pointee), id,
		                      spv::StorageClassFunction);
	};
	for (uint32_t site = 0; site < IR::MaxRayQuerySites; site++) {
		if ((s.program.info.ray_query_sites & (1u << site)) == 0) {
			continue;
		}
		auto& v = s.ray_query[site];
		for (uint32_t half = 0; half < s.lane_count; half++) {
			declare(query_type, v.query[half]);
			declare(u, v.state[half]);
			declare(u, v.instance[half]);
			declare(u, v.node[half]);
			declare(u, v.word[half]);
			declare(u, v.saved_t[half]);
			declare(u, v.saved_node[half]);
			declare(u, v.saved_inst[half]);
			declare(TypeU64(s), v.nodes[half]);
			declare(TypeU64(s), v.slots[half]);
			if (s.ray_query_keep) {
				declare(keep_type, v.keep[half]);
			}
		}
		declare(u, v.selected);
		declare(u, v.rounds);
	}
}

void InitializeRayQueryVariables(EmitterState& s) {
	for (uint32_t site = 0; site < IR::MaxRayQuerySites; site++) {
		auto& v = s.ray_query[site];
		if (v.query[0] == 0) {
			continue;
		}
		for (uint32_t half = 0; half < s.lane_count; half++) {
			Store(s, v.state[half], U32(s, DoneBit));
			if (v.keep[half] != 0) {
				Store(s, KeepPointer(s, LaneOf(s, site, half), IR::RayQueryKeepFlag), U32(s, 0));
			}
		}
		Store(s, v.selected, U32(s, None));
		Store(s, v.rounds, U32(s, 0));
	}
}

uint32_t EmitRayQueryInit(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = LaneOf(s, site, ctx.half);
	const auto u = TypeU32(s), f = TypeF32(s), b = TypeBool(s), wide = TypeU64(s);
	std::array<uint32_t, 8> ray;
	for (uint32_t i = 0; i < 8; i++) {
		ray[i] = i < 7 ? Unary(s, spv::OpBitcast, f, ctx.Arg(inst, i)) : ctx.Arg(inst, i);
	}
	const auto header = PackU64(s, ctx.Arg(inst, 8), ctx.Arg(inst, 9));
	const auto exec   = ctx.Arg(inst, 10);
	if (s.bda_pagetable_variable == 0) {
		// No device addresses in this program: the guest loop runs.
		Store(s, lane.state, U32(s, DoneBit));
		return ConstantBool(s, false);
	}
	const auto context     = ContextAddress(s);
	const auto has_context = Binary(s, spv::OpINotEqual, b, context, ConstantU64(s, 0));
	const auto native      = EmitValueOrDefaultIfCondition(s, has_context, b, ConstantBool(s, false), [&] {
		const auto valid = NotEqual(s, LoadAddressU32(s, AddressPlus(s, context, ContextValid)), U32(s, 0));
		const auto same  = Binary(s, spv::OpIEqual, b, LoadAddressU64(s, AddressPlus(s, context, ContextHeader)), header);
		return LogicalAnd(s, valid, same);
	});
	// Lanes whose ray the hardware cannot take (NaN, infinite or negative extents) find nothing,
	// as the guest's box tests would not.
	const auto traversal = LogicalAnd(s, native, exec);
	auto       lane_ok   = traversal;
	for (uint32_t i = 0; i < 6; i++) {
		lane_ok = LogicalAnd(s, lane_ok, Finite(s, ray[i]));
	}
	lane_ok = LogicalAnd(s, lane_ok, Binary(s, spv::OpFOrdGreaterThanEqual, b, ray[6], ConstantF32Value(s, 0.0f)));
	Store(s, lane.state, Or(s, Select(s, u, traversal, U32(s, TraversalBit), U32(s, 0)),
	                         Select(s, u, lane_ok, U32(s, 0), U32(s, DoneBit))));
	if (ctx.half == 0) {
		Store(s, s.ray_query[site].rounds, U32(s, 0));
	}
	EmitIfCondition(s, lane_ok, [&] {
		const auto tlas = LoadAddressU64(s, AddressPlus(s, context, ContextTlas));
		Store(s, lane.nodes, LoadAddressU64(s, AddressPlus(s, context, ContextInstanceNodes)));
		Store(s, lane.slots, LoadAddressU64(s, AddressPlus(s, context, ContextSlots)));
		const auto structure = Unary(s, spv::OpConvertUToAccelerationStructureKHR,
		                             s.builder.Type(spv::OpTypeAccelerationStructureKHR), tlas);
		const auto origin = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(s, 3), origin, ray[0], ray[1], ray[2]);
		const auto direction = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(s, 3), direction, ray[3], ray[4], ray[5]);
		s.builder.AddFunction(spv::OpRayQueryInitializeKHR, lane.query, structure, U32(s, RayFlagNoOpaque),
		                      And(s, ray[7], 0xffu), origin, ConstantF32Value(s, 0.0f), direction, ray[6]);
	});
	(void)wide;
	return traversal;
}

// Wave operation (emitted once for both emulated halves): advances the lanes without a candidate
// and selects the smallest instance a candidate is in. Result: instance node, 1 when there is one.
uint32_t EmitRayQueryNext(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto u    = TypeU32(s), b = TypeBool(s);
	uint32_t   local = 0;
	for (uint32_t half = 0; half < s.lane_count; half++) {
		const auto lane = LaneOf(s, site, half);
		Advance(s, lane);
		const auto mine = Select(s, u, BitSet(s, Load(s, u, lane.state), PendingBit), Load(s, u, lane.instance), U32(s, None));
		local = half == 0 ? mine : Select(s, u, Less(s, mine, local), mine, local);
	}
	const auto rounds  = Add(s, Load(s, u, s.ray_query[site].rounds), U32(s, 1));
	Store(s, s.ray_query[site].rounds, rounds);
	const auto running  = Less(s, rounds, U32(s, MaxRounds));
	const auto instance = Select(s, u, running, GroupMin(s, local), U32(s, None));
	const auto has      = Binary(s, spv::OpINotEqual, b, instance, U32(s, None));
	Store(s, s.ray_query[site].selected, instance);
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(s, 4), result, instance,
	                      Select(s, u, has, U32(s, 1), U32(s, 0)), U32(s, 0), U32(s, 0));
	return result;
}

// Variant 0: whether the lane's candidate is in the selected instance (marks the lane); 1: the mark.
uint32_t EmitRayQueryMatch(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = LaneOf(s, site, ctx.half);
	const auto u    = TypeU32(s);
	const auto state = Load(s, u, lane.state);
	if (IR::RayQueryVariant(inst.Flags<uint32_t>()) != 0) {
		return BitSet(s, state, MatchedBit);
	}
	const auto matched = LogicalAnd(s, BitSet(s, state, PendingBit),
	                                Equal(s, Load(s, u, lane.instance), Load(s, u, s.ray_query[site].selected)));
	Store(s, lane.state, Or(s, And(s, state, ~MatchedBit), Select(s, u, matched, U32(s, MatchedBit), U32(s, 0))));
	return matched;
}

uint32_t EmitRayQueryCandidate(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = LaneOf(s, site, ctx.half);
	const auto kind = static_cast<IR::RayQueryCandidateKind>(IR::RayQueryVariant(inst.Flags<uint32_t>()));
	return Load(s, TypeU32(s), kind == IR::RayQueryCandidateKind::Word ? lane.word : lane.node);
}

uint32_t EmitRayQueryFlag(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = LaneOf(s, site, ctx.half);
	return BitSet(s, Load(s, TypeU32(s), lane.state), TraversalBit);
}

// The result registers before the leaf code (a change means the leaf code accepted the hit).
void EmitRayQuerySave(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = LaneOf(s, site, ctx.half);
	Store(s, lane.saved_t, ctx.Arg(inst, 0));
	Store(s, lane.saved_node, ctx.Arg(inst, 1));
	Store(s, lane.saved_inst, ctx.Arg(inst, 2));
}

// After the leaf code (variant 0) or a culled instance (1): the lanes of the instance are done
// with their candidates; those whose result registers changed confirm theirs.
void EmitRayQueryResolve(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = LaneOf(s, site, ctx.half);
	const auto u    = TypeU32(s);
	const auto state   = Load(s, u, lane.state);
	const auto matched = BitSet(s, state, MatchedBit);
	if (IR::RayQueryVariant(inst.Flags<uint32_t>()) == 0) {
		const auto changed = LogicalOr(s, LogicalOr(s, NotEqual(s, ctx.Arg(inst, 0), Load(s, u, lane.saved_t)),
		                                             NotEqual(s, ctx.Arg(inst, 1), Load(s, u, lane.saved_node))),
		                               NotEqual(s, ctx.Arg(inst, 2), Load(s, u, lane.saved_inst)));
		EmitIfCondition(s, LogicalAnd(s, matched, changed),
		                [&] { s.builder.AddFunction(spv::OpRayQueryConfirmIntersectionKHR, lane.query); });
	}
	Store(s, lane.state, Select(s, u, matched, And(s, state, ~(PendingBit | MatchedBit)), state));
}

void EmitRayQueryExit(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = LaneOf(s, site, ctx.half);
	// The traversal bit stays: a verifying site reads it after the exit.
	Store(s, lane.state, Or(s, And(s, Load(s, TypeU32(s), lane.state), TraversalBit), U32(s, DoneBit)));
}

void EmitRayQueryKeep(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = LaneOf(s, site, ctx.half);
	const auto slot = IR::RayQueryVariant(inst.Flags<uint32_t>());
	if (lane.keep == 0 || slot >= IR::RayQueryKeepSlots) {
		ctx.Fail(inst, "ray query keep slot without storage");
	}
	Store(s, KeepPointer(s, lane, slot), ctx.Arg(inst, 0));
}

uint32_t EmitRayQueryKept(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = LaneOf(s, site, ctx.half);
	const auto slot = IR::RayQueryVariant(inst.Flags<uint32_t>());
	if (lane.keep == 0 || slot >= IR::RayQueryKeepSlots) {
		ctx.Fail(inst, "ray query keep slot without storage");
	}
	return Load(s, TypeU32(s), KeepPointer(s, lane, slot));
}

// Wave operation (emitted once for both halves): adds the lanes whose argument is true to a
// verify counter of the context.
void EmitRayQueryCount(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s       = ctx.state;
	const auto counter = IR::RayQueryVariant(inst.Flags<uint32_t>());
	if (s.bda_pagetable_variable == 0 || counter >= static_cast<uint32_t>(IR::RayQueryCounter::Count)) {
		return;
	}
	const auto u = TypeU32(s);
	uint32_t   value = U32(s, 0);
	for (uint32_t half = 0; half < s.lane_count; half++) {
		const auto lanes = half == ctx.half ? ctx.Arg(inst, 0) : ctx.HalfArg(inst, 0, half);
		value            = Add(s, value, Select(s, u, lanes, U32(s, 1), U32(s, 0)));
	}
	const auto sum = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpGroupNonUniformIAdd, u, sum, U32(s, spv::ScopeSubgroup), spv::GroupOperationReduce,
	                      value);
	const auto elect = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpGroupNonUniformElect, TypeBool(s), elect, U32(s, spv::ScopeSubgroup));
	const auto context = ContextAddress(s);
	EmitIfCondition(s, LogicalAnd(s, elect, Binary(s, spv::OpINotEqual, TypeBool(s), context, ConstantU64(s, 0))), [&] {
		const auto address = AddressPlus(s, context, ContextCounters + counter * 4u);
		const auto pointer = Unary(s, spv::OpConvertUToPtr, TypePhysicalU32Pointer(s), address);
		const auto old     = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpAtomicIAdd, u, old, pointer, U32(s, spv::ScopeDevice), U32(s, 0), sum);
	});
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
