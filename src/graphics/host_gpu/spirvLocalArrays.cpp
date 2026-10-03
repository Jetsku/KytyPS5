#include "graphics/host_gpu/spirvLocalArrays.h"

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::SpirvLocalArrays {

namespace {

constexpr uint32_t Magic = 0x07230203u;

std::atomic<uint32_t> g_max_subgroup_size {128};

// An inclusive range of 32-bit unsigned values.
struct Range {
	uint64_t lo = 0;
	uint64_t hi = UINT32_MAX;
};

constexpr Range Unknown {0, UINT32_MAX};

Range Union(Range a, Range b) {
	return {std::min(a.lo, b.lo), std::max(a.hi, b.hi)};
}

// All values with no bit above the highest set bit of `value`.
uint64_t BitCeilMask(uint64_t value) {
	uint64_t mask = 0;
	while (mask < value) {
		mask = (mask << 1u) | 1u;
	}
	return mask;
}

struct Instruction {
	uint32_t offset = 0; // word offset of the instruction in the module
	uint16_t opcode = 0;
	uint16_t count  = 0;
};

class Analysis {
public:
	explicit Analysis(std::span<const uint32_t> words): m_words(words) {}

	bool Parse() {
		if (m_words.size() < 5 || m_words[0] != Magic) {
			return false;
		}
		size_t offset = 5;
		while (offset < m_words.size()) {
			const auto word  = m_words[offset];
			const auto count = static_cast<uint16_t>(word >> 16u);
			const auto op    = static_cast<uint16_t>(word & 0xffffu);
			if (count == 0 || offset + count > m_words.size()) {
				return false;
			}
			const Instruction inst {static_cast<uint32_t>(offset), op, count};
			bool has_result = false;
			bool has_type   = false;
			spv::HasResultAndType(static_cast<spv::Op>(op), &has_result, &has_type);
			if (has_result) {
				const auto index = has_type ? 2u : 1u;
				if (index >= count) {
					return false;
				}
				m_defs[m_words[offset + index]] = inst;
			}
			if (op == spv::OpDecorate && count >= 4 && m_words[offset + 2] == spv::DecorationBuiltIn) {
				m_builtins[m_words[offset + 1]] = m_words[offset + 3];
			}
			if (op == spv::OpName && count >= 3) {
				const auto* text = reinterpret_cast<const char*>(&m_words[offset + 2]);
				m_names[m_words[offset + 1]] =
				    std::string(text, strnlen(text, (count - 2u) * sizeof(uint32_t)));
			}
			m_insts.push_back(inst);
			offset += count;
		}
		return true;
	}

	[[nodiscard]] const std::vector<Instruction>& Instructions() const { return m_insts; }
	[[nodiscard]] uint32_t Word(const Instruction& inst, uint32_t index) const {
		return m_words[inst.offset + index];
	}
	[[nodiscard]] const Instruction* Def(uint32_t id) const {
		const auto found = m_defs.find(id);
		return found != m_defs.end() ? &found->second : nullptr;
	}
	[[nodiscard]] std::string Name(uint32_t id) const {
		const auto found = m_names.find(id);
		return found != m_names.end() ? found->second : std::string {};
	}

	// The value of an integer constant of at most 32 bits.
	bool Constant(uint32_t id, uint32_t& value) const {
		const auto* def = Def(id);
		if (def == nullptr || def->opcode != spv::OpConstant || def->count != 4) {
			return false;
		}
		const auto* type = Def(Word(*def, 1));
		if (type == nullptr || type->opcode != spv::OpTypeInt || Word(*type, 2) > 32) {
			return false;
		}
		value = Word(*def, 3);
		return true;
	}

	// Bytes of a scalar, vector or array type (0: unknown).
	uint64_t TypeBytes(uint32_t id, int depth = 0) const {
		const auto* def = Def(id);
		if (def == nullptr || depth > 8) {
			return 0;
		}
		switch (def->opcode) {
			case spv::OpTypeInt:
			case spv::OpTypeFloat: return Word(*def, 2) / 8u;
			case spv::OpTypeVector: return TypeBytes(Word(*def, 2), depth + 1) * Word(*def, 3);
			case spv::OpTypeArray: {
				uint32_t length = 0;
				return Constant(Word(*def, 3), length) ? TypeBytes(Word(*def, 2), depth + 1) * length : 0;
			}
			default: return 0;
		}
	}

	// Values an integer result can take (Unknown when not provable).
	Range ValueRange(uint32_t id, int depth = 0) {
		if (const auto cached = m_ranges.find(id); cached != m_ranges.end()) {
			return cached->second;
		}
		if (depth > 96 || !m_visiting.insert(id).second) {
			return Unknown; // too deep, or a cycle through phis
		}
		const auto range = Compute(id, depth);
		m_visiting.erase(id);
		m_ranges[id] = range;
		return range;
	}

private:
	Range Compute(uint32_t id, int depth) {
		const auto* def = Def(id);
		if (def == nullptr) {
			return Unknown;
		}
		const auto  op  = static_cast<spv::Op>(def->opcode);
		const auto& d   = *def;
		const auto  arg = [&](uint32_t index) { return ValueRange(Word(d, index), depth + 1); };
		uint32_t    constant = 0;
		switch (op) {
			case spv::OpConstant:
				if (Constant(id, constant)) {
					return {constant, constant};
				}
				return Unknown;
			case spv::OpLoad: {
				// Loads of the subgroup builtins.
				const auto pointer = Word(d, 3);
				const auto builtin = m_builtins.find(pointer);
				if (builtin == m_builtins.end()) {
					return Unknown;
				}
				const auto max = g_max_subgroup_size.load(std::memory_order_relaxed);
				if (builtin->second == spv::BuiltInSubgroupLocalInvocationId) {
					return {0, max - 1u};
				}
				if (builtin->second == spv::BuiltInSubgroupSize) {
					return {1, max};
				}
				return Unknown;
			}
			case spv::OpCopyObject:
			case spv::OpBitcast: {
				// Same bits; only through 32-bit integer types.
				const auto* type = Def(Word(d, 1));
				if (type == nullptr || type->opcode != spv::OpTypeInt || Word(*type, 2) != 32) {
					return Unknown;
				}
				const auto* source = Def(Word(d, 3));
				if (source == nullptr) {
					return Unknown;
				}
				bool has_result = false;
				bool has_type   = false;
				spv::HasResultAndType(static_cast<spv::Op>(source->opcode), &has_result, &has_type);
				const auto* source_type = has_type ? Def(Word(*source, 1)) : nullptr;
				if (source_type == nullptr || source_type->opcode != spv::OpTypeInt ||
				    Word(*source_type, 2) != 32) {
					return Unknown;
				}
				return arg(3);
			}
			case spv::OpIAdd: {
				const auto a = arg(3);
				const auto b = arg(4);
				return a.hi + b.hi <= UINT32_MAX ? Range {a.lo + b.lo, a.hi + b.hi} : Unknown;
			}
			case spv::OpISub: {
				const auto a = arg(3);
				const auto b = arg(4);
				return a.lo >= b.hi ? Range {a.lo - b.hi, a.hi - b.lo} : Unknown;
			}
			case spv::OpIMul: {
				const auto a = arg(3);
				const auto b = arg(4);
				return a.hi * b.hi <= UINT32_MAX ? Range {a.lo * b.lo, a.hi * b.hi} : Unknown;
			}
			case spv::OpUDiv: {
				const auto a = arg(3);
				const auto b = arg(4);
				return b.lo != 0 ? Range {a.lo / b.hi, a.hi / b.lo} : Unknown;
			}
			case spv::OpUMod: {
				const auto a = arg(3);
				const auto b = arg(4);
				return b.lo != 0 ? Range {0, std::min(a.hi, b.hi - 1u)} : Unknown;
			}
			case spv::OpShiftLeftLogical: {
				const auto a = arg(3);
				const auto s = arg(4);
				if (s.hi >= 32 || (a.hi << s.hi) > UINT32_MAX) {
					return Unknown;
				}
				return {a.lo << s.lo, a.hi << s.hi};
			}
			case spv::OpShiftRightLogical: {
				const auto a = arg(3);
				const auto s = arg(4);
				if (s.hi >= 32) {
					return Unknown;
				}
				return {a.lo >> s.hi, a.hi >> s.lo};
			}
			case spv::OpShiftRightArithmetic: {
				const auto a = arg(3);
				const auto s = arg(4);
				if (s.hi >= 32 || a.hi > INT32_MAX) {
					return Unknown;
				}
				return {a.lo >> s.hi, a.hi >> s.lo};
			}
			case spv::OpBitwiseAnd: {
				const auto a = arg(3);
				const auto b = arg(4);
				return {0, std::min(a.hi, b.hi)};
			}
			case spv::OpBitwiseOr:
			case spv::OpBitwiseXor: {
				const auto a = arg(3);
				const auto b = arg(4);
				return {op == spv::OpBitwiseOr ? std::max(a.lo, b.lo) : 0, BitCeilMask(std::max(a.hi, b.hi))};
			}
			case spv::OpBitFieldUExtract: {
				const auto a     = arg(3);
				const auto count = arg(5);
				if (count.hi >= 32) {
					return {0, a.hi};
				}
				return {0, std::min<uint64_t>(a.hi, (uint64_t {1} << count.hi) - 1u)};
			}
			case spv::OpSelect: return Union(arg(4), arg(5));
			case spv::OpPhi: {
				Range range {UINT32_MAX, 0};
				for (uint32_t index = 3; index + 1 < d.count; index += 2) {
					range = Union(range, arg(index));
				}
				return range.lo <= range.hi ? range : Unknown;
			}
			default: return Unknown;
		}
	}

	std::span<const uint32_t>                      m_words;
	std::vector<Instruction>                       m_insts;
	std::unordered_map<uint32_t, Instruction>      m_defs;
	std::unordered_map<uint32_t, uint32_t>         m_builtins; // variable id -> BuiltIn
	std::unordered_map<uint32_t, std::string>      m_names;
	std::unordered_map<uint32_t, Range>            m_ranges;
	std::unordered_set<uint32_t>                   m_visiting;
};

// Instructions that cannot use a Function variable (debug, annotations, modes, types, constants):
// their literal operands must not be read as ids.
bool CannotUseVariable(uint16_t op) {
	switch (op) {
		case spv::OpSourceContinued:
		case spv::OpSource:
		case spv::OpSourceExtension:
		case spv::OpName:
		case spv::OpMemberName:
		case spv::OpString:
		case spv::OpLine:
		case spv::OpNoLine:
		case spv::OpModuleProcessed:
		case spv::OpDecorate:
		case spv::OpMemberDecorate:
		case spv::OpDecorateId:
		case spv::OpDecorateString:
		case spv::OpMemberDecorateString:
		case spv::OpDecorationGroup:
		case spv::OpGroupDecorate:
		case spv::OpGroupMemberDecorate:
		case spv::OpExtension:
		case spv::OpExtInstImport:
		case spv::OpMemoryModel:
		case spv::OpEntryPoint: // interfaces list Input/Output/global variables, never Function ones
		case spv::OpExecutionMode:
		case spv::OpExecutionModeId:
		case spv::OpCapability:
		case spv::OpTypeVoid:
		case spv::OpTypeBool:
		case spv::OpTypeInt:
		case spv::OpTypeFloat:
		case spv::OpTypeVector:
		case spv::OpTypeMatrix:
		case spv::OpTypeImage:
		case spv::OpTypeSampler:
		case spv::OpTypeSampledImage:
		case spv::OpTypeArray:
		case spv::OpTypeRuntimeArray:
		case spv::OpTypeStruct:
		case spv::OpTypePointer:
		case spv::OpTypeFunction:
		case spv::OpConstantTrue:
		case spv::OpConstantFalse:
		case spv::OpConstant:
		case spv::OpConstantComposite:
		case spv::OpConstantNull:
		case spv::OpSpecConstantTrue:
		case spv::OpSpecConstantFalse:
		case spv::OpSpecConstant:
		case spv::OpSpecConstantComposite:
		case spv::OpSpecConstantOp:
		case spv::OpVariable:
		case spv::OpLabel:
		case spv::OpBranch:
		case spv::OpSelectionMerge:
		case spv::OpLoopMerge:
		case spv::OpSwitch:
		case spv::OpFunction:
		case spv::OpFunctionEnd: return true;
		default: return false;
	}
}

// The operand words of `op` that may be ids of a value or pointer: [first, end). Literal tails
// (memory access masks, composite indices, shuffle components) are excluded.
std::pair<uint32_t, uint32_t> IdOperands(uint16_t op, uint32_t count) {
	switch (op) {
		case spv::OpLoad: return {3, std::min(count, 4u)};               // pointer
		case spv::OpStore: return {1, std::min(count, 3u)};              // pointer, object
		case spv::OpCopyMemory: return {1, std::min(count, 3u)};         // target, source
		case spv::OpCopyMemorySized: return {1, std::min(count, 4u)};    // target, source, size
		case spv::OpCompositeExtract: return {3, std::min(count, 4u)};   // composite
		case spv::OpCompositeInsert: return {3, std::min(count, 5u)};    // object, composite
		case spv::OpVectorShuffle: return {3, std::min(count, 5u)};      // vectors
		case spv::OpExtInst: return {5, count};                          // operands
		default: {
			bool has_result = false;
			bool has_type   = false;
			spv::HasResultAndType(static_cast<spv::Op>(op), &has_result, &has_type);
			return {(has_type ? 1u : 0u) + (has_result ? 1u : 0u) + 1u, count};
		}
	}
}

// Whether any Function-storage pointer type points to an array type. Types are declared before the
// first function, so this reads only the declarations: most modules have no such type and need no
// analysis.
bool HasFunctionArrayPointer(std::span<const uint32_t> module) {
	if (module.size() < 5 || module[0] != Magic) {
		return false;
	}
	std::unordered_set<uint32_t> arrays;
	for (size_t offset = 5; offset < module.size();) {
		const auto count = module[offset] >> 16u;
		const auto op    = module[offset] & 0xffffu;
		if (count == 0 || offset + count > module.size()) {
			return true; // malformed: let the full parse decide
		}
		if (op == spv::OpFunction) {
			return false;
		}
		if (op == spv::OpTypeArray && count >= 4) {
			arrays.insert(module[offset + 1]);
		} else if (op == spv::OpTypePointer && count >= 4 && module[offset + 2] == spv::StorageClassFunction &&
		           arrays.contains(module[offset + 3])) {
			return true;
		}
		offset += count;
	}
	return false;
}

} // namespace

void SetMaxSubgroupSize(uint32_t size) noexcept {
	if (size >= 1 && size <= 128) {
		g_max_subgroup_size.store(size, std::memory_order_relaxed);
	}
}

Result Shrink(std::span<const uint32_t> module, std::vector<uint32_t>& out, Init init) {
	Result result;
	out.clear();
	if (!HasFunctionArrayPointer(module)) {
		return result;
	}
	Analysis analysis(module);
	if (!analysis.Parse()) {
		return result;
	}
	struct Candidate {
		const Instruction* variable  = nullptr;
		uint32_t           array     = 0; // OpTypeArray id
		uint32_t           element   = 0; // element type id
		uint32_t           length    = 0;
		uint32_t           int_type  = 0; // type of the length constant
		uint64_t           max_index = 0;
		bool               bounded   = true;
		bool               used      = false;
	};
	std::unordered_map<uint32_t, Candidate> candidates; // variable id -> candidate
	for (const auto& inst: analysis.Instructions()) {
		if (inst.opcode != spv::OpVariable || inst.count < 4 ||
		    analysis.Word(inst, 3) != spv::StorageClassFunction) {
			continue;
		}
		const auto* pointer = analysis.Def(analysis.Word(inst, 1));
		if (pointer == nullptr || pointer->opcode != spv::OpTypePointer) {
			continue;
		}
		const auto* array = analysis.Def(analysis.Word(*pointer, 3));
		if (array == nullptr || array->opcode != spv::OpTypeArray) {
			continue;
		}
		Candidate candidate;
		candidate.variable = &inst;
		candidate.array    = analysis.Word(*pointer, 3);
		candidate.element  = analysis.Word(*array, 2);
		const auto* length = analysis.Def(analysis.Word(*array, 3));
		if (length == nullptr || !analysis.Constant(analysis.Word(*array, 3), candidate.length)) {
			continue;
		}
		candidate.int_type = analysis.Word(*length, 1);
		// An initializer has the old array type.
		candidate.bounded = inst.count == 4;
		const auto bytes  = analysis.TypeBytes(candidate.array);
		result.bytes_before += bytes;
		candidates.emplace(analysis.Word(inst, 2), candidate);
	}
	if (candidates.empty()) {
		return result;
	}
	// Every use of a candidate must be an access chain with a bounded first index.
	for (const auto& inst: analysis.Instructions()) {
		if (CannotUseVariable(inst.opcode)) {
			continue;
		}
		const bool chain = inst.opcode == spv::OpAccessChain || inst.opcode == spv::OpInBoundsAccessChain;
		const auto [first, end] = IdOperands(inst.opcode, inst.count);
		for (uint32_t index = first; index < end; index++) {
			const auto found = candidates.find(analysis.Word(inst, index));
			if (found == candidates.end()) {
				continue;
			}
			auto& candidate = found->second;
			if (!chain || index != 3 || inst.count < 5) {
				candidate.bounded = false;
				continue;
			}
			const auto range = analysis.ValueRange(analysis.Word(inst, 4));
			if (range.hi >= candidate.length) {
				candidate.bounded = false;
				continue;
			}
			candidate.used      = true;
			candidate.max_index = std::max(candidate.max_index, range.hi);
		}
	}
	// New lengths (at least 1 element), and the instructions that declare them.
	uint32_t                                        bound = module[3];
	std::unordered_map<uint32_t, std::vector<uint32_t>> insert_after; // offset -> words
	std::unordered_map<uint32_t, uint32_t>              new_type_of;   // variable id -> pointer id
	std::unordered_map<uint32_t, uint32_t>              init_of;       // variable id -> initializer id
	for (auto& [id, candidate]: candidates) {
		if (!candidate.bounded) {
			result.unbounded_arrays++;
			result.bytes_after += analysis.TypeBytes(candidate.array);
			continue;
		}
		const auto new_length = static_cast<uint32_t>(candidate.used ? candidate.max_index + 1u : 1u);
		const auto element_bytes = analysis.TypeBytes(candidate.element);
		if (new_length >= candidate.length) {
			result.bytes_after += analysis.TypeBytes(candidate.array);
			continue;
		}
		const auto constant_id = bound++;
		const auto array_id    = bound++;
		const auto pointer_id  = bound++;
		// After the variable's pointer type: everything it needs is declared before it.
		const auto* pointer = analysis.Def(analysis.Word(*candidate.variable, 1));
		auto&       words   = insert_after[pointer->offset];
		words.insert(words.end(), {(4u << 16u) | spv::OpConstant, candidate.int_type, constant_id, new_length});
		words.insert(words.end(), {(4u << 16u) | spv::OpTypeArray, array_id, candidate.element, constant_id});
		words.insert(words.end(),
		             {(4u << 16u) | spv::OpTypePointer, pointer_id, spv::StorageClassFunction, array_id});
		new_type_of[id] = pointer_id;
		const auto* element_type = analysis.Def(candidate.element);
		const bool  scalar32     = element_type != nullptr &&
		                      (element_type->opcode == spv::OpTypeInt || element_type->opcode == spv::OpTypeFloat) &&
		                      analysis.Word(*element_type, 2) == 32;
		if (init == Init::Poison && scalar32) {
			const auto poison_id    = bound++;
			const auto composite_id = bound++;
			words.insert(words.end(), {(4u << 16u) | spv::OpConstant, candidate.element, poison_id, 0x7fc00000u});
			words.push_back(((3u + new_length) << 16u) | spv::OpConstantComposite);
			words.push_back(array_id);
			words.push_back(composite_id);
			words.insert(words.end(), new_length, poison_id);
			init_of[id] = composite_id;
		} else if (init != Init::None) {
			const auto null_id = bound++;
			words.insert(words.end(), {(3u << 16u) | spv::OpConstantNull, array_id, null_id});
			init_of[id] = null_id;
		}
		result.arrays.push_back({id, analysis.Name(id), candidate.length, new_length,
		                         static_cast<uint32_t>(element_bytes)});
		result.bytes_after += element_bytes * new_length;
	}
	if (new_type_of.empty()) {
		return result;
	}
	out.reserve(module.size() + insert_after.size() * 12u);
	out.insert(out.end(), module.begin(), module.begin() + 5);
	out[3] = bound;
	for (const auto& inst: analysis.Instructions()) {
		const auto begin = module.begin() + inst.offset;
		if (inst.opcode == spv::OpVariable) {
			const auto variable = analysis.Word(inst, 2);
			const auto retyped  = new_type_of.find(variable);
			if (retyped != new_type_of.end()) {
				// Candidates have no initializer (4 words); Init adds one.
				const auto initializer = init_of.find(variable);
				if (initializer != init_of.end()) {
					out.insert(out.end(), {(5u << 16u) | spv::OpVariable, retyped->second, variable,
					                       spv::StorageClassFunction, initializer->second});
				} else {
					out.insert(out.end(), {(4u << 16u) | spv::OpVariable, retyped->second, variable,
					                       spv::StorageClassFunction});
				}
				continue;
			}
		}
		out.insert(out.end(), begin, begin + inst.count);
		if (const auto extra = insert_after.find(inst.offset); extra != insert_after.end()) {
			out.insert(out.end(), extra->second.begin(), extra->second.end());
		}
	}
	result.changed = true;
	return result;
}

} // namespace Libs::Graphics::SpirvLocalArrays
