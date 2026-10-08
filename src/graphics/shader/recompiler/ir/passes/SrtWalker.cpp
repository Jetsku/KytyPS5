#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	if (runtime.read_specialization_memory == nullptr) {
		runtime.try_read_clean_backing = nullptr;
	} else if (runtime.try_read_specialization_backing != nullptr) {
		runtime.try_read_clean_backing = runtime.try_read_specialization_backing;
	}
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;
constexpr uint32_t MaxFlatReadRun = 16;

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Fetch: return "fetch";
		case ShaderType::Compute: return "compute";
		default: return "unknown";
	}
}

std::string Diagnostic(const ResourcePlan& program, uint32_t pc, const std::string& message) {
	return fmt::format("shader SRT: hash=0x{:016x} stage={} pc=0x{:08x} {}", program.shader_hash,
	                   StageName(program.stage), pc, message);
}

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto kind = values.memory_info[index].kind;
	return (op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
	       (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer);
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMin32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type)
	    : m_program(program), m_type(type) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type == RuntimeValueType::Integer &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU8 ||
		    op == ValueOpcode::UndefU16 || op == ValueOpcode::UndefU32 ||
		    op == ValueOpcode::UndefU64 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type == RuntimeValueType::Integer && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    (source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 &&
			     source->GetOpcode() != ValueOpcode::IAddCarry32)) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};

class PlanBuilder {
public:
	PlanBuilder(Program& program, bool variant_reads)
	    : m_program(program), m_variant_reads(variant_reads) {}

	void Run() {
		m_program.srt_reads.clear();
		m_program.dynamic_reads.clear();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
					const auto flags = inst.Flags<MemoryFlags>();
					if (flags.index < m_program.memory_info.size()) {
						const auto kind       = m_program.memory_info[flags.index].kind;
						const bool crosswired = (op == ValueOpcode::LoadAddressU32 &&
						                         kind == ResourceKind::ScalarBuffer) ||
						                        (op == ValueOpcode::ReadConstBuffer &&
						                         kind == ResourceKind::ScalarAddress);
						if (crosswired) {
							Fail(flags.pc,
							     fmt::format("{} has incompatible scalar memory metadata",
							                 ValueOpcodeName(op)));
						}
					}
				}
				if (IsDescriptorHandle(inst.GetOpcode())) {
					for (size_t index = 0; index < inst.NumArgs(); index++) {
						Collect(inst.Arg(index), 0);
					}
				}
			}
		}
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 && IsRawRead(m_program, inst) &&
				    inst.Arg(1).Resolve().IsImmediate() &&
				    ValidateRuntimeValue(m_program, Value(&inst))) {
					Collect(Value(&inst), inst.Flags<MemoryFlags>().pc);
				}
			}
		}
		PatchReads();
	}

private:
	struct Patch {
		Inst*    inst = nullptr;
		uint32_t slot = 0;
		bool     keep = false;
	};

	[[noreturn]] void Fail(uint32_t pc, const std::string& message) const {
		const auto diagnostic = Diagnostic(m_program, pc, message);
		EXIT("shader SRT planning failed: %s", diagnostic.c_str());
		std::abort();
	}

	void Collect(Value value, uint32_t use_pc) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return;
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			Fail(use_pc, "invalid typed planning value");
		}
		const auto cycle = std::ranges::find(m_visiting, inst);
		if (cycle != m_visiting.end()) {
			const auto contains_phi = std::any_of(cycle, m_visiting.end(), [](const Inst* value) {
				return value->GetOpcode() == ValueOpcode::Phi;
			});
			if (contains_phi) {
				return;
			}
			Fail(use_pc, fmt::format("cyclic typed planning value {} without a phi",
			                         ValueOpcodeName(inst->GetOpcode())));
		}
		if (std::ranges::find(m_visited, inst) != m_visited.end()) {
			return;
		}
		m_visiting.push_back(inst);
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			Collect(inst->Arg(index), use_pc);
		}
		m_visiting.pop_back();
		m_visited.push_back(inst);
		if (!IsRawRead(m_program, *inst)) {
			return;
		}
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32 ||
		    (m_variant_reads && !EvaluableBeforeDispatch(inst->Arg(0)))) {
			if (std::ranges::find(m_program.dynamic_reads, value) ==
			    m_program.dynamic_reads.end()) {
				m_program.dynamic_reads.push_back(value);
			}
			return;
		}
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); slot++) {
			if (EquivalentValue(m_program, value, m_program.srt_reads[slot].value)) {
				m_patches.push_back({inst, slot, false});
				return;
			}
		}
		const auto slot = static_cast<uint32_t>(m_program.srt_reads.size());
		m_program.srt_reads.push_back({value, slot});
		m_patches.push_back({inst, slot, true});
	}

	void PatchReads() {
		for (const auto& patch: m_patches) {
			auto* block = patch.inst->Parent();
			auto& list  = block->Instructions();
			auto  where =
			    std::ranges::find_if(list, [&](const Inst& inst) { return &inst == patch.inst; });
			const auto resource =
			    Value(&*block->PrependNewInst(where, ValueOpcode::GetSrtResource));
			const auto flat = Value(&*block->PrependNewInst(where, ValueOpcode::ReadConst,
			                                                {resource, Value(patch.slot)}));
			const auto uses = patch.inst->Uses();
			for (const auto& use: uses) {
				use.user->SetArg(use.operand, flat);
			}
			for (auto& info: m_program.block_info) {
				if (info.condition.Resolve() == Value(patch.inst)) {
					info.condition = flat;
				}
				if (info.indirect_target.Resolve() == Value(patch.inst)) {
					info.indirect_target = flat;
				}
			}
			if (patch.keep) {
				const auto memory = patch.inst->Flags<MemoryFlags>().index;
				if (memory < m_program.memory_info.size()) {
					m_program.memory_info[memory].planning_only = true;
				}
				block->AppendNewInst(ValueOpcode::ReferenceU32, {Value(patch.inst)});
			}
		}
	}

	// KYTY_SRT_VARIANT_READS: whether the read's address (its GetAddressResource or V# handle) can
	// be computed before the dispatch, by the same rules as every other runtime value. A loop-carried
	// pointer (a phi ResolveInvariantPhi cannot reduce) or data the GPU produces cannot: a flat slot
	// for such a read could never be evaluated and would drop the whole dispatch.
	bool EvaluableBeforeDispatch(Value handle) {
		handle           = handle.Resolve();
		const auto* inst = handle.TryInstruction();
		if (handle.IsImmediate() || inst == nullptr) {
			return true;
		}
		if (const auto it = m_evaluable_memo.find(inst); it != m_evaluable_memo.end()) {
			return it->second;
		}
		const bool evaluable = ValidateRuntimeValue(m_program, handle);
		m_evaluable_memo.emplace(inst, evaluable);
		return evaluable;
	}

	Program&           m_program;
	std::vector<Inst*> m_visiting;
	std::vector<Inst*> m_visited;
	std::vector<Patch> m_patches;
	bool               m_variant_reads = false;
	std::unordered_map<const Inst*, bool> m_evaluable_memo;
};

} // namespace

void BuildSrtReadRuns(ResourcePlan& program) {
	struct Shape {
		const Inst* handle = nullptr;
		ValueOpcode opcode = ValueOpcode::Void;
		int64_t offset = 0;
		bool clean = false;
	};
	const auto shape = [&](uint32_t index, Shape& result) {
		const auto& read = program.srt_reads[index];
		auto* inst = read.value.Resolve().TryInstruction();
		if (inst == nullptr || !IsRawRead(program, *inst)) {
			return false;
		}
		const bool buffer = inst->GetOpcode() == ValueOpcode::ReadConstBuffer;
		if (inst->NumArgs() != (buffer ? 2u : 4u)) return false;
		const auto& memory = program.memory_info[inst->Flags<MemoryFlags>().index];
		// Scalar-address loads carry the address instruction's unused high offset
		// and active predicate. Accept only the S_LOAD form understood by the CPU
		// evaluator; other address forms retain their scalar path.
		if (!buffer && (inst->Arg(2).Resolve() != Value(0u) ||
		                inst->Arg(3).Resolve() != Value(true) || memory.address_is_full)) {
			return false;
		}
		const auto offset = inst->Arg(1).Resolve();
		const auto* handle = inst->Arg(0).ResolveInstruction();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32 || handle == nullptr ||
		    handle->GetOpcode() != (buffer ? ValueOpcode::GetBufferResource
		                                  : ValueOpcode::GetAddressResource) ||
		    handle->NumArgs() != (buffer ? 4u : 2u)) {
			return false;
		}
		const auto immediate = static_cast<int64_t>(static_cast<int32_t>(memory.offset));
		if (buffer && immediate < 0) return false;
		result = {handle, inst->GetOpcode(),
		          buffer ? (immediate + offset.U32()) & ~int64_t {3}
		                 : (immediate & ~int64_t {3}) + (offset.U32() & ~3u),
		          read.flat_offset < program.clean_flat_slots.size() &&
		              program.clean_flat_slots[read.flat_offset] != 0u};
		// Stable indices let runtime inspect current memo state without value dispatch.
		(void)inst->EvaluationIndex(program.evaluation_value_count);
		return true;
	};
	EXIT_IF(program.srt_reads.size() > UINT32_MAX);
	program.srt_read_run_ends.assign(program.srt_reads.size(), 0u);
	const auto count = static_cast<uint32_t>(program.srt_reads.size());
	for (uint32_t first = 0; first < count;) {
		Shape previous;
		if (!shape(first, previous)) {
			++first;
			continue;
		}
		uint32_t end = first + 1u;
		while (end < count && end - first < MaxFlatReadRun) {
			Shape next;
			if (!shape(end, next) || next.handle != previous.handle ||
			    next.opcode != previous.opcode || next.clean != previous.clean ||
			    next.offset != previous.offset + 4) {
				break;
			}
			previous = next;
			++end;
		}
		for (auto i = first; i < end; ++i) program.srt_read_run_ends[i] = end;
		first = end;
	}
}

// Exact arities of side-effect-free operations implemented by EvaluateInst.
// Reads, runtime sources, selects and context-changing operations are boundaries.
uint32_t ArithmeticArity(const ResourcePlan::EvaluationRecipe& recipe) {
	using Kind = ResourcePlan::EvaluationRecipe::Kind;
	if (recipe.kind == Kind::Forward || recipe.kind == Kind::Extract) return 1;
	if (recipe.kind == Kind::ExtractCarry) return 2;
	if (recipe.kind != Kind::Operation || recipe.instruction == nullptr) return 0;
	uint32_t arity = 0;
	switch (recipe.instruction->GetOpcode()) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPTrunc32: arity = 1; break;
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMin32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPMul32: arity = 2; break;
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract: arity = 3; break;
		case ValueOpcode::BitFieldInsert: arity = 4; break;
		default: return 0;
	}
	return recipe.instruction->NumArgs() == arity ? arity : 0;
}

void BuildSrtEvaluationRecipes(ResourcePlan& program) {
	static const bool enabled = [] {
		const auto* setting = std::getenv("KYTY_SRT_COMPILED_RECIPES");
		return setting != nullptr && std::strcmp(setting, "1") == 0;
	}();
	if (!enabled) return;
	using Operand = ResourcePlan::EvaluationOperand;
	using Recipe = ResourcePlan::EvaluationRecipe;
	using Kind = Recipe::Kind;
	std::unordered_set<const Inst*> owned;
	// Assign every retained node before resolving operands, including forward edges.
	for (const auto& inst: program.value_storage) {
		owned.insert(&inst);
		(void)inst.EvaluationIndex(program.evaluation_value_count);
	}
	program.evaluation_recipes.resize(program.evaluation_value_count);
	const auto compile = [&](Value value) -> Operand {
		value = value.Resolve();
		if (value.IsImmediate()) {
			uint64_t bits = 0;
			switch (value.GetType()) {
				case Type::U1: bits = value.U1(); break;
				case Type::U8: bits = value.U8(); break;
				case Type::U16: bits = value.U16(); break;
				case Type::U32: bits = value.U32(); break;
				case Type::U64: bits = value.U64(); break;
				case Type::F32: bits = std::bit_cast<uint32_t>(value.F32Value()); break;
				default: return {};
			}
			return {bits, 0, Operand::Kind::Immediate};
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || !owned.contains(inst)) return {};
		const auto index = inst->EvaluationIndex(program.evaluation_value_count);
		if (index >= program.evaluation_recipes.size()) return {};
		return {0, index, Operand::Kind::Node};
	};
	for (const auto& inst: program.value_storage) {
		auto& recipe = program.evaluation_recipes[inst.EvaluationIndex(program.evaluation_value_count)];
		recipe.instruction = &inst;
		const auto op = inst.GetOpcode();
		if (IsRuntimeSelect(op) && inst.NumArgs() == 3) {
			recipe.selection_mask = inst.Arg(0).Resolve();
		}
		const auto operands = [&](size_t count) {
			if (count > recipe.operands.size() || inst.NumArgs() != count) return false;
			for (size_t i = 0; i < count; ++i) {
				recipe.operands[i] = compile(inst.Arg(i));
				if (recipe.operands[i].kind == Operand::Kind::Invalid) return false;
			}
			return true;
		};
		if (op == ValueOpcode::GetUserData && inst.Arg(0).IsImmediate() &&
		    inst.Arg(0).GetType() == Type::ScalarReg) {
			recipe.kind = Kind::UserData;
			recipe.parameter = RegIndex(inst.Arg(0).ScalarRegister());
		} else if (op == ValueOpcode::ReadConst) {
			const auto slot = inst.Arg(1).Resolve();
			if (slot.IsImmediate() && slot.GetType() == Type::U32 &&
			    slot.U32() < program.srt_reads.size()) {
				recipe.parameter = slot.U32();
				recipe.operands[0] = compile(program.srt_reads[slot.U32()].value);
				if (recipe.operands[0].kind != Operand::Kind::Invalid) recipe.kind = Kind::FlatRead;
			}
		} else if (op == ValueOpcode::CompositeExtractU64 ||
		           op == ValueOpcode::CompositeExtractU32x2) {
			const auto component = inst.Arg(1).Resolve();
			if (!component.IsImmediate() || component.GetType() != Type::U32 ||
			    component.U32() >= 2) continue;
			recipe.parameter = component.U32();
			if (op == ValueOpcode::CompositeExtractU64) {
				recipe.operands[0] = compile(inst.Arg(0));
				if (recipe.operands[0].kind != Operand::Kind::Invalid) recipe.kind = Kind::Extract;
			} else if (const auto* source = inst.Arg(0).Resolve().TryInstruction()) {
				if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
					recipe.operands[0] = compile(source->Arg(recipe.parameter));
					if (recipe.operands[0].kind != Operand::Kind::Invalid) recipe.kind = Kind::Forward;
				} else if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
					recipe.operands[0] = compile(source->Arg(0));
					recipe.operands[1] = compile(source->Arg(1));
					if (recipe.operands[0].kind != Operand::Kind::Invalid &&
					    recipe.operands[1].kind != Operand::Kind::Invalid) recipe.kind = Kind::ExtractCarry;
				}
			}
		} else if (IsRawRead(program, inst)) {
			const auto* handle = inst.Arg(0).Resolve().TryInstruction();
			const bool buffer = op == ValueOpcode::ReadConstBuffer;
			if (handle == nullptr || handle->NumArgs() != (buffer ? 4u : 2u)) continue;
			recipe.offset = static_cast<int32_t>(program.memory_info[inst.Flags<MemoryFlags>().index].offset);
			recipe.operands[0] = compile(handle->Arg(0));
			recipe.operands[1] = compile(handle->Arg(1));
			recipe.operands[2] = compile(inst.Arg(1));
			if (buffer) {
				recipe.operands[3] = compile(handle->Arg(2));
				recipe.operands[4] = compile(handle->Arg(3));
			}
			bool valid = true;
			for (uint32_t i = 0; i < (buffer ? 5u : 3u); ++i) {
				valid &= recipe.operands[i].kind != Operand::Kind::Invalid;
			}
			if (valid) recipe.kind = buffer ? Kind::RawBuffer : Kind::RawAddress;
		} else if (IsRuntimeSelect(op)) {
			if (operands(3)) recipe.kind = Kind::Select;
		} else if (op == ValueOpcode::GetShaderBase ||
		           (IsRuntimeUniformOp(op) && op != ValueOpcode::CompositeConstructU32x2 &&
		            op != ValueOpcode::IAddCarry32)) {
			if (operands(inst.NumArgs())) recipe.kind = Kind::Operation;
		}
	}
	static const bool decode_roots = [] {
		const auto* setting = std::getenv("KYTY_SRT_DECODED_ROOTS");
		return setting != nullptr && std::strcmp(setting, "1") == 0;
	}();
	if (!decode_roots) return;
	program.descriptor_roots.resize(program.descriptor_sources.size());
	for (size_t source = 0; source < program.descriptor_sources.size(); ++source) {
		const auto& descriptor = program.descriptor_sources[source];
		for (uint32_t word = 0; word < descriptor.dword_count; ++word) {
			program.descriptor_roots[source][word] = compile(descriptor.dwords[word]);
		}
	}
	program.flat_read_roots.resize(program.srt_reads.size());
	for (size_t slot = 0; slot < program.srt_reads.size(); ++slot) {
		program.flat_read_roots[slot] = compile(program.srt_reads[slot].value);
	}
	program.condition_roots.resize(program.control_flow.size());
	program.initial_active_sources.assign(program.descriptor_sources.size(), 1u);
	for (size_t block = 0; block < program.control_flow.size(); ++block) {
		const auto& flow = program.control_flow[block];
		program.condition_roots[block] = compile(flow.condition);
		for (const auto source: flow.sources) program.initial_active_sources.at(source) = 0u;
	}
}

void BuildSrtArithmeticTapes(ResourcePlan& program) {
	static const bool enabled = [] {
		const auto* setting = std::getenv("KYTY_SRT_ARITHMETIC_TAPES");
		return setting != nullptr && std::strcmp(setting, "1") == 0;
	}();
	if (!enabled || program.evaluation_recipes.empty()) return;
	using Operand = ResourcePlan::EvaluationOperand;
	using Recipe = ResourcePlan::EvaluationRecipe;
	using TapeOperand = ResourcePlan::ArithmeticTapeOperand;
	using Instruction = ResourcePlan::ArithmeticTapeInstruction;
	constexpr uint32_t NoIndex = UINT32_MAX;
	constexpr uint32_t MaxOperations = 64;
	constexpr uint32_t MaxPlanInstructions = 16384;
	const auto count = program.evaluation_recipes.size();
	if (count > UINT32_MAX) return;
	std::unordered_map<const Inst*, uint32_t> indices;
	std::vector<uint32_t> arities(count), uses(count), parent(count, NoIndex);
	std::vector<uint8_t> roots(count), covered(count), visiting(count);
	for (uint32_t i = 0; i < count; ++i) {
		const auto& recipe = program.evaluation_recipes[i];
		if (recipe.instruction != nullptr) indices.emplace(recipe.instruction, i);
		arities[i] = ArithmeticArity(recipe);
		if ((recipe.kind == Recipe::Kind::Extract || recipe.kind == Recipe::Kind::ExtractCarry) &&
		    recipe.parameter >= 2) arities[i] = 0;
		for (uint32_t operand = 0; operand < arities[i]; ++operand) {
			const auto& value = recipe.operands[operand];
			if (value.kind == Operand::Kind::Invalid ||
			    (value.kind == Operand::Kind::Node && value.index >= count)) arities[i] = 0;
		}
	}
	const auto index_of = [&](Value value) {
		const auto found = indices.find(value.Resolve().TryInstruction());
		return found != indices.end() ? found->second : NoIndex;
	};
	// Count the union of raw and decoded edges, including repeated operands.
	// Raw fallback/extract paths can expose edges absent from a decoded recipe.
	for (uint32_t i = 0; i < count; ++i) {
		const auto& recipe = program.evaluation_recipes[i];
		std::unordered_map<uint32_t, std::array<uint32_t, 2>> references;
		if (recipe.instruction != nullptr) {
			for (size_t a = 0; a < recipe.instruction->NumArgs(); ++a) {
				const auto child = index_of(recipe.instruction->Arg(a));
				if (child != NoIndex) ++references[child][0];
			}
		}
		for (const auto& operand: recipe.operands) {
			if (operand.kind == Operand::Kind::Node && operand.index < count) {
				++references[operand.index][1];
			}
		}
		for (const auto& [child, reference_count]: references) {
			uses[child] = std::min(2u, uses[child] + std::max(reference_count[0], reference_count[1]));
			parent[child] = i;
		}
	}
	const auto mark_root = [&](Value value) {
		const auto index = index_of(value);
		if (index != NoIndex) roots[index] = 1;
	};
	for (const auto& descriptor: program.descriptor_sources) {
		for (uint32_t i = 0; i < descriptor.dword_count; ++i) mark_root(descriptor.dwords[i]);
		if (descriptor.indirect_image.has_value()) {
			mark_root(descriptor.indirect_image->selector_mask);
			mark_root(descriptor.indirect_image->key_count);
		}
	}
	for (const auto& read: program.srt_reads) mark_root(read.value);
	for (const auto& block: program.control_flow) mark_root(block.condition);
	for (uint32_t i = 0; i < program.uniform_fill.fill.words; ++i) mark_root(program.uniform_fill.values[i]);

	std::vector<ResourcePlan::ArithmeticTape> tapes(count);
	std::vector<Instruction> instructions;
	// Start at natural region roots; the second pass handles bounded region cuts.
	for (uint32_t pass = 0; pass < 2; ++pass) {
		for (uint32_t root = 0; root < count; ++root) {
			if (arities[root] == 0 || covered[root]) continue;
			const bool natural_root = roots[root] || uses[root] != 1 || parent[root] == NoIndex ||
			                          arities[parent[root]] == 0;
			if (natural_root != (pass == 0)) continue;
			std::vector<Instruction> region;
			std::vector<uint32_t> interior;
			bool valid = true;
			const auto emit = [&](auto&& self, const Operand& value, uint32_t depth) -> TapeOperand {
				if (!valid) return {};
				if (value.kind == Operand::Kind::Immediate) return {value.value, true};
				if (value.kind != Operand::Kind::Node || value.index >= count) {
					valid = false;
					return {};
				}
				const auto node = value.index;
				if (visiting[node]) {
					valid = false;
					return {};
				}
				const bool inline_node = node == root ||
				    (arities[node] != 0 && uses[node] == 1 && !roots[node] && !covered[node] &&
				     interior.size() < MaxOperations && depth < MaxOperations);
				Instruction instruction;
				if (!inline_node) {
					instruction.parameter = node;
				} else {
					visiting[node] = 1;
					interior.push_back(node);
					const auto& recipe = program.evaluation_recipes[node];
					instruction.instruction = recipe.instruction;
					instruction.parameter = recipe.parameter;
					instruction.operand_count = static_cast<uint8_t>(arities[node]);
					switch (recipe.kind) {
						case Recipe::Kind::Operation: instruction.kind = Instruction::Kind::Operation; break;
						case Recipe::Kind::Forward: instruction.kind = Instruction::Kind::Forward; break;
						case Recipe::Kind::Extract: instruction.kind = Instruction::Kind::Extract; break;
						case Recipe::Kind::ExtractCarry: instruction.kind = Instruction::Kind::ExtractCarry; break;
						default: valid = false; break;
					}
					for (uint32_t i = 0; valid && i < arities[node]; ++i) {
						instruction.operands[i] = self(self, recipe.operands[i], depth + 1u);
					}
					visiting[node] = 0;
				}
				const auto result_index = static_cast<uint32_t>(region.size());
				if (valid) region.push_back(instruction);
				return {result_index, false};
			};
			(void)emit(emit, Operand {0, root, Operand::Kind::Node}, 0);
			// Tiny regions retain the original evaluator. Rejected compilation has
			// no runtime effects, and does not install a partially compiled tape.
			if (!valid || interior.size() < 2 || region.size() > MaxPlanInstructions - instructions.size()) continue;
			tapes[root] = {static_cast<uint32_t>(instructions.size()), static_cast<uint32_t>(region.size())};
			instructions.insert(instructions.end(), region.begin(), region.end());
			for (const auto node: interior) covered[node] = 1;
		}
	}
	if (!instructions.empty()) {
		program.arithmetic_tapes = std::move(tapes);
		program.arithmetic_tape_instructions = std::move(instructions);
	}
}

void SealEvaluationIndices(ResourcePlan& program) {
	// Value storage owns every instruction reachable from the plan's roots, so no
	// later walk can meet an unassigned node. Earlier builders keep their indices.
	for (const auto& inst: program.value_storage) {
		(void)inst.EvaluationIndex(program.evaluation_value_count);
	}
	program.evaluation_sealed = true;
}

EvaluationScratch& ThreadEvaluationScratch() {
	thread_local EvaluationScratch scratch;
	return scratch;
}

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     std::span<const uint8_t> clean_flat_slots, SrtWalker* clean_evaluator,
                     Value active_mask)
    : SrtWalker(program,
                clean_evaluator != nullptr ? clean_evaluator->m_scratch : ThreadEvaluationScratch(),
                runtime, clean_flat_slots, clean_evaluator, active_mask) {}

SrtWalker::SrtWalker(const ResourcePlan& program, EvaluationScratch& scratch,
                     const SrtRuntime& runtime, std::span<const uint8_t> clean_flat_slots,
                     SrtWalker* clean_evaluator, Value active_mask)
    : m_program(program), m_scratch(scratch), m_runtime(runtime),
      m_clean_flat_slots(clean_flat_slots), m_clean_evaluator(clean_evaluator),
      m_active_mask(active_mask.Resolve()), m_context(AcquireContext(scratch)),
      m_count_recipes(!program.evaluation_recipes.empty() && Profiler::AggregateEnabled()) {
	// CleanRuntime preserves these identities and replaces only the reader. Custom
	// ordinary readers may represent another memory domain and cannot borrow.
	m_share_clean_values = runtime.share_clean_values && clean_evaluator != nullptr &&
	    runtime.read_memory == nullptr && runtime.read_specialization_memory != nullptr &&
	    clean_evaluator->m_runtime.read_memory == runtime.read_specialization_memory &&
	    runtime.userdata == clean_evaluator->m_runtime.userdata &&
	    runtime.user_data.data() == clean_evaluator->m_runtime.user_data.data() &&
	    runtime.user_data.size() == clean_evaluator->m_runtime.user_data.size() &&
	    runtime.shader_base == clean_evaluator->m_runtime.shader_base &&
	    &program == &clean_evaluator->m_program &&
	    runtime.observe_read == clean_evaluator->m_runtime.observe_read &&
	    runtime.observer_userdata == clean_evaluator->m_runtime.observer_userdata &&
	    m_active_mask == clean_evaluator->m_active_mask;
}

SrtWalker::~SrtWalker() {
	if (m_count_recipes) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::SrtRecipeSessions);
		Profiler::CountFrameEvent(Profiler::FrameEvent::SrtRecipeCompiledNodes, m_compiled_nodes);
		Profiler::CountFrameEvent(Profiler::FrameEvent::SrtRecipeFallbackNodes, m_fallback_nodes);
		Profiler::CountFrameEvent(Profiler::FrameEvent::SrtRecipeMemoHits, m_memo_hits);
		if (m_tape_executions != 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::SrtTapeExecutions, m_tape_executions);
			Profiler::CountFrameEvent(Profiler::FrameEvent::SrtTapeOperations, m_tape_operations);
			Profiler::CountFrameEvent(Profiler::FrameEvent::SrtTapeBoundaryCalls, m_tape_boundary_calls);
		}
	}
	EXIT_IF(m_scratch.evaluation_depth == 0u ||
	        &m_scratch.evaluation_contexts[m_scratch.evaluation_depth - 1u] != &m_context);
	--m_scratch.evaluation_depth;
	if (m_shared_clean_hits != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::SrtSharedCleanMemoHits, m_shared_clean_hits);
	}
}

bool SrtWalker::BorrowCleanValue(uint32_t index, uint64_t& result) {
	if (!m_share_clean_values) return false;
	const auto& clean = m_clean_evaluator->m_context;
	if (index >= clean.values.size() || clean.values[index].generation != clean.generation) return false;
	// A completed strict result has already checked GPU ownership and contributed
	// its exact semantic reads to the shared observer. Copy its value, not its
	// generation. Failed/in-progress nodes and other EXEC contexts never cross over.
	result = clean.values[index].value;
	if (index >= m_context.values.size()) m_context.values.resize(m_program.evaluation_value_count);
	m_context.values[index] = {result, m_context.generation};
	++m_shared_clean_hits;
	return true;
}

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext(EvaluationScratch& scratch) {
	if (scratch.evaluation_depth == scratch.evaluation_contexts.size()) {
		scratch.evaluation_contexts.emplace_back();
	}
	// A new generation hides every entry left by an earlier walk, of any plan.
	auto& context = scratch.evaluation_contexts[scratch.evaluation_depth++];
	context.generation += 2;
	return context;
}

uint32_t SrtWalker::MemoIndex(const Inst& inst) const {
	// Sealed plans are shared read-only. Unsealed programs (translation-time passes
	// and tests) still assign slots lazily on their only evaluating thread.
	return m_program.evaluation_sealed ? inst.SealedEvaluationIndex()
	                                   : inst.EvaluationIndex(m_program.evaluation_value_count);
}

float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool SrtWalker::EvaluateWide(Value value, uint64_t& result) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); return true;
			case Type::U8: result = value.U8(); return true;
			case Type::U16: result = value.U16(); return true;
			case Type::U32: result = value.U32(); return true;
			case Type::U64: result = value.U64(); return true;
			case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); return true;
			default: return false;
		}
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (!m_program.evaluation_recipes.empty()) {
		const auto index = MemoIndex(*inst);
		if (index < m_program.evaluation_recipes.size()) {
			return EvaluateRecipeNode(index, result);
		}
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
	    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = MemoIndex(*inst);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		return true;
	}
	// The low generation bit marks an instruction that is still being evaluated.
	if (m_context.values[index].generation == (m_context.generation | 1u)) {
		return false;
	}
	if (BorrowCleanValue(index, result)) return true;
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t out = 0;
	const bool evaluated = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		memo.generation = 0;
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

bool SrtWalker::EvaluateRoot(Value value, const ResourcePlan::EvaluationOperand& root,
                             uint32_t& result) {
	if (root.kind == ResourcePlan::EvaluationOperand::Kind::Invalid) {
		return Evaluate(value, result);
	}
	uint64_t wide = 0;
	if (!EvaluateOperand(root, wide)) return false;
	result = static_cast<uint32_t>(wide);
	return true;
}

const Inst* SrtWalker::FlatReadInstruction(uint32_t slot, uint32_t& memo_index) const {
	if (slot < m_program.flat_read_roots.size()) {
		const auto& root = m_program.flat_read_roots[slot];
		if (root.kind == ResourcePlan::EvaluationOperand::Kind::Node &&
		    root.index < m_program.evaluation_recipes.size()) {
			memo_index = root.index;
			return m_program.evaluation_recipes[root.index].instruction;
		}
	}
	const auto* inst = m_program.srt_reads[slot].value.ResolveInstruction();
	memo_index = MemoIndex(*inst);
	return inst;
}

bool SrtWalker::EvaluateOperand(const ResourcePlan::EvaluationOperand& operand,
                                uint64_t& result) {
	using Kind = ResourcePlan::EvaluationOperand::Kind;
	switch (operand.kind) {
		case Kind::Immediate: result = operand.value; return true;
		case Kind::Node: return EvaluateRecipeNode(operand.index, result);
		case Kind::Invalid: return false;
	}
	return false;
}

bool SrtWalker::EvaluateRecipeNode(uint32_t index, uint64_t& result) {
	using Kind = ResourcePlan::EvaluationRecipe::Kind;
	if (index >= m_program.evaluation_recipes.size()) return false;
	const auto& recipe = m_program.evaluation_recipes[index];
	// EXEC overrides precede memo lookup just as in EvaluateWide. Do not evaluate
	// a predicate, or cache this select under a different active-mask context.
	if (!m_active_mask.IsEmpty() && recipe.selection_mask == m_active_mask) {
		return recipe.kind == Kind::Select
		           ? EvaluateOperand(recipe.operands[1], result)
		           : EvaluateWide(recipe.instruction->Arg(1), result);
	}
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		if (m_count_recipes) ++m_memo_hits;
		result = m_context.values[index].value;
		return true;
	}
	if (m_context.values[index].generation == (m_context.generation | 1u)) return false;
	if (BorrowCleanValue(index, result)) return true;
	m_context.values[index].generation = m_context.generation | 1u;
	if (m_count_recipes) {
		if (recipe.kind == Kind::Fallback) ++m_fallback_nodes;
		else ++m_compiled_nodes;
	}
	uint64_t out = 0;
	// Fallback is selected before evaluating any operand. A failed recipe is a
	// semantic failure and must never retry a partially executed guest read.
	const auto* tape = index < m_program.arithmetic_tapes.size() &&
	                           m_program.arithmetic_tapes[index].count != 0
	                       ? &m_program.arithmetic_tapes[index] : nullptr;
	const bool evaluated = tape != nullptr ? EvaluateArithmeticTape(*tape, out)
	                       : recipe.kind == Kind::Fallback ? EvaluateInst(*recipe.instruction, out)
	                                                       : EvaluateRecipe(recipe, out);
	auto& memo = m_context.values[index];
	if (!evaluated) {
		memo.generation = 0;
		return false;
	}
	memo.value = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

bool SrtWalker::EvaluateArithmeticTape(const ResourcePlan::ArithmeticTape& tape,
                                       uint64_t& result) {
	using Kind = ResourcePlan::ArithmeticTapeInstruction::Kind;
	const auto base = m_context.tape_values_used;
	const auto end = base + tape.count;
	if (end < base || tape.count == 0 ||
	    static_cast<size_t>(tape.first) + tape.count > m_program.arithmetic_tape_instructions.size()) return false;
	if (m_context.tape_values.size() < end) m_context.tape_values.resize(end);
	struct ScratchScope {
		ResourcePlan::EvaluationContext& context;
		size_t previous;
		~ScratchScope() { context.tape_values_used = previous; }
	} scope {m_context, base};
	m_context.tape_values_used = end;
	if (m_count_recipes) ++m_tape_executions;
	for (uint32_t i = 0; i < tape.count; ++i) {
		const auto& instruction = m_program.arithmetic_tape_instructions[tape.first + i];
		uint64_t value = 0;
		if (instruction.kind == Kind::Boundary) {
			if (m_count_recipes) ++m_tape_boundary_calls;
			if (!EvaluateRecipeNode(instruction.parameter, value)) return false;
		} else {
			std::array<uint64_t, 4> operands {};
			if (instruction.operand_count > operands.size()) return false;
			for (uint32_t arg = 0; arg < instruction.operand_count; ++arg) {
				const auto& operand = instruction.operands[arg];
				// Only earlier results are legal, including under nested tape calls.
				if (!operand.immediate && operand.value >= i) return false;
				operands[arg] = operand.immediate ? operand.value : m_context.tape_values[base + operand.value];
			}
			if (m_count_recipes) ++m_tape_operations;
			switch (instruction.kind) {
				case Kind::Operation:
					if (!EvaluateInstWithOperands(*instruction.instruction, value,
					        [&](size_t arg, uint64_t& out) {
						        if (arg >= instruction.operand_count) return false;
						        out = operands[arg];
						        return true;
					        })) return false;
					break;
				case Kind::Forward: value = operands[0]; break;
				case Kind::Extract: value = static_cast<uint32_t>(operands[0] >> (instruction.parameter * 32u)); break;
				case Kind::ExtractCarry:
					value = static_cast<uint64_t>(static_cast<uint32_t>(operands[0])) +
					        static_cast<uint32_t>(operands[1]);
					value = static_cast<uint32_t>(value >> (instruction.parameter * 32u));
					break;
				case Kind::Boundary: return false;
			}
		}
		// A boundary can recursively resize scratch. Do not retain vector references
		// or pointers across that call, and never retry a failed tape through legacy.
		m_context.tape_values[base + i] = value;
	}
	result = m_context.tape_values[end - 1u];
	return true;
}

bool SrtWalker::EvaluateRecipe(const ResourcePlan::EvaluationRecipe& recipe,
                               uint64_t& result) {
	using Kind = ResourcePlan::EvaluationRecipe::Kind;
	uint64_t a = 0;
	uint64_t b = 0;
	switch (recipe.kind) {
		case Kind::Operation:
			// Share arithmetic bodies while directly passing their decoded operands.
			return EvaluateInst(*recipe.instruction, result, &recipe);
		case Kind::UserData:
			if (recipe.parameter < m_program.user_data_base ||
			    recipe.parameter - m_program.user_data_base >= m_runtime.user_data.size()) return false;
			result = m_runtime.user_data[recipe.parameter - m_program.user_data_base];
			return true;
		case Kind::FlatRead:
			if (recipe.parameter < m_clean_flat_slots.size() &&
			    m_clean_flat_slots[recipe.parameter] != 0 && m_clean_evaluator != nullptr) {
				return m_clean_evaluator->EvaluateOperand(recipe.operands[0], result);
			}
			return EvaluateOperand(recipe.operands[0], result);
		case Kind::Forward: return EvaluateOperand(recipe.operands[0], result);
		case Kind::Extract:
			if (!EvaluateOperand(recipe.operands[0], a)) return false;
			result = static_cast<uint32_t>(a >> (recipe.parameter * 32u));
			return true;
		case Kind::ExtractCarry:
			if (!EvaluateOperand(recipe.operands[0], a) || !EvaluateOperand(recipe.operands[1], b)) return false;
			a = static_cast<uint64_t>(static_cast<uint32_t>(a)) + static_cast<uint32_t>(b);
			result = static_cast<uint32_t>(a >> (recipe.parameter * 32u));
			return true;
		case Kind::RawAddress:
		case Kind::RawBuffer:
			if (!ResolveRecipeReadAddress(recipe, a, b)) {
				ObserveSrtRead(m_runtime, a, {}, false);
				return false;
			}
			return ReadRawWord(a, result);
		case Kind::Select: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (!predicate.EvaluateOperand(recipe.operands[0], a)) return false;
			return EvaluateOperand(recipe.operands[a != 0u ? 1u : 2u], result);
		}
		case Kind::Fallback: break;
	}
	return false;
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32) {
		return false;
	}
	const auto component = index.U32();
	if (component >= 2u) {
		return false;
	}
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		uint64_t packed = 0;
		if (!Arg(inst, 0, packed)) {
			return false;
		}
		result = static_cast<uint32_t>(packed >> (component * 32u));
		return true;
	}
	const auto* source = inst.Arg(0).ResolveInstruction();
	if (source == nullptr) {
		return false;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
		return EvaluateWide(source->Arg(component), result);
	}
	if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
		uint64_t lhs = 0;
		uint64_t rhs = 0;
		if (!Arg(*source, 0, lhs) || !Arg(*source, 1, rhs)) {
			return false;
		}
		const auto sum =
		    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
		result =
		    component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
		return true;
	}
	return false;
}

bool SrtWalker::ResolveRawReadAddress(const Inst& inst, uint64_t& address,
	                                  uint64_t& available) {
	if (!m_program.evaluation_recipes.empty()) {
		const auto index = MemoIndex(inst);
		if (index < m_program.evaluation_recipes.size()) {
			const auto& recipe = m_program.evaluation_recipes[index];
			if (recipe.kind == ResourcePlan::EvaluationRecipe::Kind::RawAddress ||
			    recipe.kind == ResourcePlan::EvaluationRecipe::Kind::RawBuffer) {
				return ResolveRecipeReadAddress(recipe, address, available);
			}
		}
	}
	const auto flags = inst.Flags<MemoryFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr) {
		return false;
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	available = UINT64_MAX;
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
		const auto aligned = byte_offset & ~uint64_t {3};
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (aligned > size || size - aligned < sizeof(uint32_t)) {
			return false;
		}
		available = size - aligned;
		address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	return true;
}

bool SrtWalker::ResolveRecipeReadAddress(const ResourcePlan::EvaluationRecipe& recipe,
                                        uint64_t& address, uint64_t& available) {
	uint64_t low = 0, high = 0, offset = 0;
	if (!EvaluateOperand(recipe.operands[0], low) ||
	    !EvaluateOperand(recipe.operands[1], high) ||
	    !EvaluateOperand(recipe.operands[2], offset)) return false;
	const auto base = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	available = UINT64_MAX;
	if (recipe.kind == ResourcePlan::EvaluationRecipe::Kind::RawBuffer) {
		uint64_t records = 0, word3 = 0;
		// Word 3 is semantically evaluated even though the address math does not use it.
		if (!EvaluateOperand(recipe.operands[3], records) ||
		    !EvaluateOperand(recipe.operands[4], word3)) return false;
		if (recipe.offset < 0) return false;
		const auto byte_offset = static_cast<uint64_t>(recipe.offset) + static_cast<uint32_t>(offset);
		const auto aligned = byte_offset & ~uint64_t {3};
		const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		const auto size = stride == 0u ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                              : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (aligned > size || size - aligned < sizeof(uint32_t)) return false;
		available = size - aligned;
		address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
	} else {
		const auto relative = (recipe.offset & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) return false;
	}
	return true;
}

bool SrtWalker::ReadRawWord(uint64_t address, uint64_t& result, bool allow_probe) {
	uint32_t word = 0;
	if (m_runtime.read_memory != nullptr) {
		if (!m_runtime.read_memory(m_runtime.userdata, address, {&word, 1})) {
			ObserveSrtRead(m_runtime, address, {&word, 1}, false);
			// A null guest pointer (an unset table, because this path is not taken) is a null
			// descriptor: read it as zero. Only the first page is unmapped. Checked after the
			// read, so a reader that does back low addresses still returns its data.
			if (address < 0x1000u) {
				result = 0;
				return true;
			}
			return false;
		}
	} else {
		constexpr uint64_t gpu_limit = uint64_t {1} << 40u;
		// Read the exact clean bytes without faulting on unrelated dirty bytes in
		// the same protected page. A failed probe leaves the original read intact.
		// The coherence checks add work even on already-readable pages, so this
		// path remains controlled by the optional runtime probe callback.
		const bool probed = allow_probe && m_runtime.try_read_clean_backing != nullptr &&
		    address != 0 && address < gpu_limit && sizeof(word) < gpu_limit - address &&
		    m_runtime.try_read_clean_backing(m_runtime.userdata, address, {&word, 1});
		if (!probed) {
			if (!InPlaceReadable(address)) {
				// No guest page backs the address, so the in-place read would fault with nothing
				// to resolve it. Read 0, as the GPU does from an unmapped page. The read is not
				// certifiable: a prepared-read capture of this materialization is rejected.
				NoteUnmappedRead(address);
				ObserveSrtRead(m_runtime, address, {&word, 1}, false);
				result = 0;
				return true;
			}
			std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
		}
	}
	ObserveSrtRead(m_runtime, address, {&word, 1}, true);
	result = word;
	return true;
}

// Every flat SRT read of a plan is evaluated before the dispatch, including loads the shader only
// executes on some paths. Astro Bot's tiled lighting (78af8e26) loads a TLAS pointer from its SRT
// and then header fields through it, behind an EXEC branch taken only by lanes that trace a
// shadow ray; the pointer is null while no TLAS exists. The in-place read goes through the guest
// mapping, where a page the tracker protects faults into its handler and is read back, but an
// address outside every guest mapping faults with nothing to resolve it.
bool SrtWalker::InPlaceReadable(uint64_t address) {
	if (NeverMappedAddress(address, sizeof(uint32_t))) {
		return false;
	}
	if (m_runtime.is_guest_mapped == nullptr || (address >> 12u) == m_mapped_page) {
		return true;
	}
	// Reads are dword-aligned, so a dword never crosses the page.
	if (!m_runtime.is_guest_mapped(address, sizeof(uint32_t))) {
		return false;
	}
	m_mapped_page = address >> 12u;
	return true;
}

void SrtWalker::NoteUnmappedRead(uint64_t address) const {
	Profiler::CountFrameEvent(Profiler::FrameEvent::SrtUnmappedReads);
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> logged;
	{
		std::scoped_lock lock(mutex);
		const auto key = m_program.shader_hash ^ (static_cast<uint64_t>(m_program.stage) << 58u);
		if (!logged.insert(key).second) {
			return;
		}
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "SRT: {} shader 0x{:016x} reads unmapped address 0x{:x} before its dispatch; the read "
	    "returns 0, as the GPU reads an unmapped page, and the dispatch runs.\n",
	    StageName(m_program.stage), m_program.shader_hash, address));
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	uint64_t address = 0;
	uint64_t available = 0;
	if (!ResolveRawReadAddress(inst, address, available)) {
		ObserveSrtRead(m_runtime, address, {}, false);
		return false;
	}
	return ReadRawWord(address, result);
}

bool SrtWalker::EvaluateFlatRun(uint32_t first, uint32_t end,
	                           std::vector<uint32_t>& flat, uint32_t& consumed) {
	consumed = 0;
	const bool strict = m_runtime.read_specialization_memory != nullptr &&
	                    m_runtime.read_memory == m_runtime.read_specialization_memory;
	if (m_runtime.try_read_clean_backing == nullptr || !m_active_mask.IsEmpty() ||
	    (m_runtime.read_memory != nullptr && !strict)) {
		return true;
	}
	uint32_t memo_index = 0;
	const auto* inst = FlatReadInstruction(first, memo_index);
	if (m_context.values.size() < m_program.evaluation_value_count) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[memo_index].generation == m_context.generation) return true;
	if (m_context.values[memo_index].generation == (m_context.generation | 1u)) return false;
	uint64_t shared_value = 0;
	if (BorrowCleanValue(memo_index, shared_value)) return true;
	// Preserve the normal recursion guard and operand-read order. Address resolution
	// can grow memo storage or populate another flat read, so retain no Entry reference.
	m_context.values[memo_index].generation = m_context.generation | 1u;
	uint64_t address = 0;
	uint64_t available = 0;
	if (!ResolveRawReadAddress(*inst, address, available)) {
		ObserveSrtRead(m_runtime, address, {}, false);
		m_context.values[memo_index].generation = 0;
		return false;
	}
	uint32_t words = 1;
	while (first + words < end && words < MaxFlatReadRun &&
	       static_cast<uint64_t>(words + 1u) * sizeof(uint32_t) <= available &&
	       (address & 4095u) + static_cast<uint64_t>(words + 1u) * sizeof(uint32_t) <= 4096u) {
		if (m_program.srt_reads[first + words].flat_offset >= flat.size()) break;
		uint32_t index = 0;
		(void)FlatReadInstruction(first + words, index);
		// All run indices were assigned at compilation. A resolved handle dependency
		// may have evaluated a later member, which must keep its earlier snapshot.
		if (index >= m_context.values.size() ||
		    m_context.values[index].generation == m_context.generation ||
		    m_context.values[index].generation == (m_context.generation | 1u)) {
			break;
		}
		++words;
	}
	std::array<uint32_t, MaxFlatReadRun> values {};
	constexpr uint64_t gpu_limit = uint64_t {1} << 40u;
	const auto bytes = static_cast<uint64_t>(words) * sizeof(uint32_t);
	// Do not turn a scalar GPU range into a union outside the predicate's GPU domain.
	const bool probed = address != 0 && address < gpu_limit && bytes < gpu_limit - address &&
	    m_runtime.try_read_clean_backing(m_runtime.userdata, address, {values.data(), words});
	if (probed) {
		ObserveSrtRead(m_runtime, address, {values.data(), words}, true);
	} else {
		// The silent probe committed nothing. Perform exactly this scalar read now;
		// the next flat slot remains untouched and follows its original evaluation.
		// A failed union may still have a clean first word; do not repeat an
		// identical failed single-word probe.
		uint64_t value = 0;
		if (!ReadRawWord(address, value, words > 1u)) {
			m_context.values[memo_index].generation = 0;
			return false;
		}
		values[0] = static_cast<uint32_t>(value);
		words = 1;
	}
	for (uint32_t i = 0; i < words; ++i) {
		const auto& read = m_program.srt_reads[first + i];
		uint32_t index = 0;
		(void)FlatReadInstruction(first + i, index);
		m_context.values[index] = {values[i], m_context.generation};
		flat[read.flat_offset] = values[i];
	}
	consumed = words;
	return true;
}

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result,
                             const ResourcePlan::EvaluationRecipe* recipe) {
	return EvaluateInstWithOperands(inst, result, [&](size_t index, uint64_t& value) {
		return recipe != nullptr ? EvaluateOperand(recipe->operands[index], value)
		                         : Arg(inst, index, value);
	});
}

template <typename ReadOperand>
bool SrtWalker::EvaluateInstWithOperands(const Inst& inst, uint64_t& result, ReadOperand&& arg) {
	uint64_t a = 0;
	uint64_t b = 0;
	uint64_t c = 0;
	const auto binary  = [&]() { return arg(0, a) && arg(1, b); };
	const auto ternary = [&]() {
		return arg(0, a) && arg(1, b) && arg(2, c);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[reg - m_program.user_data_base];
			return true;
		}
		case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
		case ValueOpcode::Phi: return EvaluatePhi(inst, result);
		case ValueOpcode::ReadFirstLane: {
			const auto clean_runtime = CleanRuntime(m_runtime);
			SrtWalker  clean_active(m_program, clean_runtime, {}, nullptr, inst.Arg(1));
			SrtWalker  active(m_program, m_runtime, m_clean_flat_slots, &clean_active,
			                  inst.Arg(1));
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return arg(0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
		case ValueOpcode::CompositeConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) |
			         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::ReadConst: {
			const auto slot = inst.Arg(1).Resolve();
			if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return false;
			}
			if (slot.U32() < m_clean_flat_slots.size() &&
			    m_clean_flat_slots[slot.U32()] != 0u && m_clean_evaluator != nullptr) {
				return m_clean_evaluator->EvaluateWide(m_program.srt_reads[slot.U32()].value,
				                                       result);
			}
			return EvaluateWide(m_program.srt_reads[slot.U32()].value, result);
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
			if (IsRawRead(m_program, inst)) {
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::IAdd32:
			if (binary()) {
				result = static_cast<uint32_t>(a + b);
				return true;
			}
			return false;
		case ValueOpcode::IAdd64:
			if (binary()) {
				result = a + b;
				return true;
			}
			return false;
		case ValueOpcode::ISub32:
			if (binary()) {
				result = static_cast<uint32_t>(a - b);
				return true;
			}
			return false;
		case ValueOpcode::ISub64:
			if (binary()) {
				result = a - b;
				return true;
			}
			return false;
		case ValueOpcode::IMul32:
			if (binary()) {
				result = static_cast<uint32_t>(a * b);
				return true;
			}
			return false;
		case ValueOpcode::IMul64:
			if (binary()) {
				result = a * b;
				return true;
			}
			return false;
		case ValueOpcode::UMin32:
			if (binary()) {
				result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::ConvertF32U32:
			if (arg(0, a)) {
				result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
				return true;
			}
			return false;
		case ValueOpcode::ConvertU32F32:
			if (arg(0, a)) {
				const auto value = Float32(a);
				if (!std::isfinite(value) || value < 0.0f ||
				    static_cast<double>(value) > UINT32_MAX) {
					return false;
				}
				result = static_cast<uint32_t>(value);
				return true;
			}
			return false;
		case ValueOpcode::FPMul32:
			if (binary()) {
				result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
				return true;
			}
			return false;
		case ValueOpcode::FPTrunc32:
			if (arg(0, a)) {
				result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
				return true;
			}
			return false;
		case ValueOpcode::FPIsNan32:
			if (arg(0, a)) {
				result = std::isnan(Float32(a));
				return true;
			}
			return false;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (binary()) {
				const auto operand = [&](uint64_t bits) {
					if (inst.Flags<FPCompareFlags>().flush_input_denorms &&
					    (bits & 0x7fffffffu) < 0x00800000u) {
						bits &= 0x80000000u;
					}
					return Float32(bits);
				};
				result = inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32
				             ? operand(a) <= operand(b)
				             : operand(a) >= operand(b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd32:
			if (binary()) {
				result = static_cast<uint32_t>(a & b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd64:
			if (binary()) {
				result = a & b;
				return true;
			}
			return false;
		case ValueOpcode::BitwiseOr32:
			if (binary()) {
				result = static_cast<uint32_t>(a | b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseXor32:
			if (binary()) {
				result = static_cast<uint32_t>(a ^ b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseNot32:
			if (arg(0, a)) {
				result = ~static_cast<uint32_t>(a);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) << (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical64:
			if (binary()) {
				result = a << (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) >> (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical64:
			if (binary()) {
				result = a >> (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic32:
			if (binary()) {
				result = static_cast<uint32_t>(
				    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic64:
			if (binary()) {
				result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
				return true;
			}
			return false;
		case ValueOpcode::BitFieldUExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				const auto mask = width == 32u  ? UINT32_MAX
				                  : width == 0u ? 0u
				                                : (uint32_t {1} << width) - 1u;
				result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldSExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = 0;
					return true;
				}
				const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
				auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
				if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
					bits |= ~mask;
				}
				result = bits;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !arg(3, d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask =
			    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) |
			         ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (predicate.EvaluateWide(inst.Arg(0), a)) {
				return arg(a != 0u ? 1u : 2u, result);
			}
			return false;
		}
		case ValueOpcode::IEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::INotEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::UGreaterThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::SGreaterThanEqual32:
			if (binary()) {
				result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
				         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::LogicalAnd:
			if (binary()) {
				result = (a != 0u) && (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::LogicalOr:
			if (binary()) {
				result = (a != 0u) || (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::LogicalXor:
			if (binary()) {
				result = (a != 0u) != (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::LogicalNot:
			if (arg(0, a)) {
				result = a == 0u;
				return true;
			}
			return false;
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU8:
		case ValueOpcode::UndefU16:
		case ValueOpcode::UndefU32:
		case ValueOpcode::UndefU64: return false;
		default: break;
	}
	return false;
}
bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	result = {};
	result.dword_count = descriptor.dword_count;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		const bool ok = source < m_program.descriptor_roots.size()
		    ? EvaluateRoot(descriptor.dwords[index], m_program.descriptor_roots[source][index],
		                   result.dwords[index])
		    : Evaluate(descriptor.dwords[index], result.dwords[index]);
		if (!ok) {
			return false;
		}
	}
	return true;
}

std::span<const uint8_t> SrtWalker::FindActiveSources() {
	if (m_program.control_flow.empty()) {
		return {};
	}
	auto& active = m_scratch.active_sources;
	const bool decoded = m_program.condition_roots.size() == m_program.control_flow.size();
	const bool compact = m_program.flow_aliases.size() == m_program.control_flow.size();
	if (compact) {
		active = m_program.flow_initial_sources;
	} else if (decoded) {
		active = m_program.initial_active_sources;
	} else {
		active.assign(m_program.descriptor_sources.size(), 1u);
		for (const auto& block: m_program.control_flow) {
			for (const auto source: block.sources) {
				active.at(source) = 0u;
			}
		}
	}
	auto& visited = m_scratch.visited_blocks;
	auto& pending = m_scratch.pending_blocks;
	auto& visit_tags = m_scratch.flow_visit_tags;
	uint32_t visit_epoch = 0;
	if (compact) {
		// Tags are epoch-stamped per scratch, so tags left by other plans never match.
		if (visit_tags.size() < m_program.control_flow.size()) {
			visit_tags.resize(m_program.control_flow.size(), 0u);
		}
		visit_epoch = ++m_scratch.flow_visit_epoch;
		if (visit_epoch == 0u) {
			std::ranges::fill(visit_tags, 0u);
			visit_epoch = ++m_scratch.flow_visit_epoch;
		}
	} else {
		visited.assign(m_program.control_flow.size(), 0u);
	}
	pending.clear();
	pending.push_back(0u);
	while (!pending.empty()) {
		const auto requested = pending.back();
		const auto index = compact ? m_program.flow_aliases.at(requested) : requested;
		pending.pop_back();
		if (compact) {
			if (visit_tags.at(index) == visit_epoch) continue;
			visit_tags[index] = visit_epoch;
		} else {
			if (visited.at(index)) continue;
			visited[index] = 1u;
		}
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) {
			active[source] = 1u;
		}
		uint32_t condition = 0;
		if (!block.condition.IsEmpty() && m_runtime.read_specialization_memory != nullptr &&
		    (decoded ? EvaluateRoot(block.condition, m_program.condition_roots[index], condition)
		             : Evaluate(block.condition, condition))) {
			pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	return active;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) {
		return false;
	}
	flat.resize(m_program.srt_reads.size());
	for (uint32_t index = 0; index < m_program.srt_reads.size();) {
		const auto& read = m_program.srt_reads[index];
		const bool clean = read.flat_offset < m_clean_flat_slots.size() &&
		                   m_clean_flat_slots[read.flat_offset] != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr)) {
			return false;
		}
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		if (read.flat_offset >= flat.size()) return false;
		if (evaluator.m_runtime.try_read_clean_backing != nullptr &&
		    index < m_program.srt_read_run_ends.size() &&
		    m_program.srt_read_run_ends[index] > index) {
			auto end = m_program.srt_read_run_ends[index];
			// A caller can supply a different clean-slot view than the compiled plan.
			// Never batch across the evaluator it selects for an individual flat slot.
			for (auto next = index + 1u; next < end; ++next) {
				const auto slot = m_program.srt_reads[next].flat_offset;
				if ((slot < m_clean_flat_slots.size() && m_clean_flat_slots[slot] != 0u) != clean) {
					end = next;
					break;
				}
			}
			uint32_t consumed = 0;
			if (!evaluator.EvaluateFlatRun(index, end, flat, consumed)) return false;
			if (consumed != 0u) {
				index += consumed;
				continue;
			}
		}
		const bool ok = index < m_program.flat_read_roots.size()
		    ? evaluator.EvaluateRoot(read.value, m_program.flat_read_roots[index], flat[read.flat_offset])
		    : evaluator.Evaluate(read.value, flat[read.flat_offset]);
		if (!ok) {
			return false;
		}
		++index;
	}
	return true;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

bool FindVariantFlatRead(const ResourcePlan& program, uint32_t& pc) {
	std::unordered_set<const Inst*> visited;
	std::vector<Value>              pending;
	for (const auto& read: program.srt_reads) {
		const auto* read_inst = read.value.Resolve().TryInstruction();
		if (read_inst == nullptr) {
			continue;
		}
		// The read's address operands; the value it reads does not matter.
		pending.clear();
		for (size_t index = 0; index < read_inst->NumArgs(); index++) {
			pending.push_back(read_inst->Arg(index));
		}
		while (!pending.empty()) {
			const auto value = pending.back().Resolve();
			pending.pop_back();
			const auto* inst = value.IsImmediate() ? nullptr : value.TryInstruction();
			if (inst == nullptr || !visited.insert(inst).second) {
				continue;
			}
			if (inst->GetOpcode() == ValueOpcode::Phi) {
				const auto invariant = ResolveInvariantPhi(program, value);
				if (invariant.IsEmpty()) {
					pc = read_inst->Flags<MemoryFlags>().pc;
					return true;
				}
				pending.push_back(invariant);
				continue;
			}
			for (size_t index = 0; index < inst->NumArgs(); index++) {
				pending.push_back(inst->Arg(index));
			}
		}
	}
	return false;
}

void BuildSrtPlan(Program& program, bool variant_reads) {
	if (program.resource_tracking_complete) {
		EXIT("shader SRT planning failed: cannot rebuild SRT after resource tracking");
	}
	program.srt_plan_complete = false;
	PlanBuilder(program, variant_reads).Run();
	program.srt_plan_complete = true;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
