#pragma once

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/Reg.h"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"

#include <bit>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Block;
class Inst;
struct Program;

// ir/ProgramClone.cpp: deep copy of a translated program (copies every private member).
[[nodiscard]] bool CloneProgram(const Program& source, Program& target);
// ir/ProgramCodec.cpp: byte encoding of resource plans (reads and writes every private member).
struct ProgramCodecAccess;

class Value {
public:
	Value() = default;
	explicit Value(Inst* value);
	explicit Value(ScalarReg value);
	explicit Value(VectorReg value);
	explicit Value(bool value);
	explicit Value(uint8_t value);
	explicit Value(uint16_t value);
	explicit Value(uint32_t value);
	explicit Value(uint64_t value);

	static Value F16(uint16_t bits);
	static Value F32(float value);

	[[nodiscard]] bool IsEmpty() const;
	[[nodiscard]] bool IsImmediate() const;
	[[nodiscard]] bool IsIdentity() const;
	[[nodiscard]] bool IsPhi() const;
	[[nodiscard]] Type GetType() const;

	[[nodiscard]] Inst*     Instruction() const;
	[[nodiscard]] Inst*     TryInstruction() const;
	[[nodiscard]] Inst*     ResolveInstruction() const;
	[[nodiscard]] Value     Resolve() const;
	[[nodiscard]] ScalarReg ScalarRegister() const;
	[[nodiscard]] VectorReg VectorRegister() const;
	[[nodiscard]] bool      U1() const;
	[[nodiscard]] uint8_t   U8() const;
	[[nodiscard]] uint16_t  U16() const;
	[[nodiscard]] uint32_t  U32() const;
	[[nodiscard]] uint64_t  U64() const;
	[[nodiscard]] uint16_t  F16Bits() const;
	[[nodiscard]] float     F32Value() const;

	bool operator==(const Value& other) const;

private:
	friend struct ProgramCodecAccess;

	Type type = Type::Void;
	union {
		Inst*     inst;
		ScalarReg scalar_reg;
		VectorReg vector_reg;
		bool      imm_u1;
		uint8_t   imm_u8;
		uint16_t  imm_u16;
		uint32_t  imm_u32;
		uint64_t  imm_u64;
	};

	explicit Value(Type type, uint64_t bits);
};
static_assert(std::is_trivially_copyable_v<Value>);

template <Type type_>
class TypedValue: public Value {
public:
	TypedValue() = default;

	explicit TypedValue(Value value): Value(value) {
		EXIT_IF(!AreTypesCompatible(value.GetType(), type_));
	}
};

using U1  = TypedValue<Type::U1>;
using U16 = TypedValue<Type::U16>;
using U32 = TypedValue<Type::U32>;
using U64 = TypedValue<Type::U64>;
using F16 = TypedValue<Type::F16>;
using F32 = TypedValue<Type::F32>;

struct Use {
	Inst*  user    = nullptr;
	size_t operand = 0;

	bool operator==(const Use&) const = default;
};

class Inst {
public:
	explicit Inst(ValueOpcode opcode, uint64_t flags = 0);
	~Inst();

	Inst(const Inst&)            = delete;
	Inst& operator=(const Inst&) = delete;
	Inst(Inst&&)                 = delete;
	Inst& operator=(Inst&&)      = delete;

	[[nodiscard]] ValueOpcode             GetOpcode() const;
	[[nodiscard]] Type                    GetType() const;
	[[nodiscard]] bool                    MayHaveSideEffects() const;
	[[nodiscard]] bool                    HasUses() const;
	[[nodiscard]] size_t                  UseCount() const;
	[[nodiscard]] size_t                  NumArgs() const;
	[[nodiscard]] size_t                  NumPhiBlocks() const;
	[[nodiscard]] Value                   Arg(size_t index) const;
	[[nodiscard]] Block*                  PhiBlock(size_t index) const;
	[[nodiscard]] Block*                  Parent() const;
	[[nodiscard]] const std::vector<Use>& Uses() const;
	// Runtime indices belong to the resource plan that owns this instruction.
	// Lazy assignment is for plan construction and unsealed, single-threaded programs.
	[[nodiscard]] uint32_t EvaluationIndex(uint32_t& count) const {
		if (evaluation_index == UINT32_MAX) {
			evaluation_index = count++;
		}
		return evaluation_index;
	}
	// A sealed plan assigned every index at extraction; reading one never writes.
	[[nodiscard]] uint32_t SealedEvaluationIndex() const {
		EXIT_IF(evaluation_index == UINT32_MAX);
		return evaluation_index;
	}

	void SetParent(Block* block);
	void SetArg(size_t index, Value value);
	void AddPhiOperand(Block* predecessor, Value value);
	void ReplaceUsesWith(Value replacement, bool preserve = true);
	void ReplaceOpcode(ValueOpcode opcode);
	void Invalidate();
	// KYTY_IR_LINEAR_USES (CodegenOptions::ir_linear_uses), Senaxx 5145dc1f9:
	// Detaches without maintaining other instructions' use lists: only when every instruction
	// that refers to this one is destroyed with it (Program::~Program).
	void DropForDestruction();
	// RemoveIdentities: ReplaceUsesWith(replacement, false) for an instruction about to be erased,
	// leaving its own entries in its arguments' use lists; the caller drops those for every
	// removed instruction at once (DropRemovedUses), which keeps every list in the same order.
	// Uses by instructions already in `removed` (erased) are skipped. Arguments go to `touched`.
	void ReplaceUsesForRemoval(Value replacement, const std::unordered_set<const Inst*>& removed,
	                           std::vector<Inst*>& touched);
	static void DropRemovedUses(std::span<Inst* const>              touched,
	                            const std::unordered_set<const Inst*>& removed);

	template <typename T>
	requires(sizeof(T) <= sizeof(uint64_t) && std::is_trivially_copyable_v<T>)
	[[nodiscard]] T Flags() const {
		T result {};
		std::memcpy(&result, &flags, sizeof(result));
		return result;
	}

	template <typename T>
	requires(sizeof(T) <= sizeof(uint64_t) && std::is_trivially_copyable_v<T>)
	void SetFlags(T value) {
		flags = 0;
		std::memcpy(&flags, &value, sizeof(value));
	}

private:
	// Copy and encode every member below; update both (and their layout checks) when adding one.
	friend bool CloneProgram(const Program& source, Program& target);
	friend struct ProgramCodecAccess;

	void AddUse(Inst* used, size_t operand);
	void RemoveUse(Inst* used, size_t operand);
	void ClearArgs();

	ValueOpcode         opcode;
	uint64_t            flags;
	Block*              parent = nullptr;
	std::vector<Value>  args;
	std::vector<Block*> phi_blocks;
	std::vector<Use>    uses;
	mutable uint32_t    evaluation_index = UINT32_MAX;
};

// The accessors the IR passes and the resource walker call for every operand, inline (Senaxx
// c73447398 via usa-ratt/KytyPS5-GT7): out of line they were a fifth of the GPU thread's resource
// walk in builds without link-time optimization. Same checks.
inline bool Value::IsEmpty() const {
	return type == Type::Void;
}

inline bool Value::IsImmediate() const {
	return type != Type::Opaque;
}

inline bool Value::IsIdentity() const {
	return type == Type::Opaque && inst->GetOpcode() == ValueOpcode::Identity;
}

inline bool Value::IsPhi() const {
	return type == Type::Opaque && inst->GetOpcode() == ValueOpcode::Phi;
}

inline Inst* Value::TryInstruction() const {
	return type == Type::Opaque ? inst : nullptr;
}

inline Inst* Value::ResolveInstruction() const {
	EXIT_IF(type != Type::Opaque);
	return IsIdentity() ? inst->Arg(0).ResolveInstruction() : inst;
}

inline Value Value::Resolve() const {
	return IsIdentity() ? inst->Arg(0).Resolve() : *this;
}

inline uint32_t Value::U32() const {
	EXIT_IF(type != Type::U32);
	return imm_u32;
}

inline ValueOpcode Inst::GetOpcode() const {
	return opcode;
}

inline size_t Inst::NumArgs() const {
	return args.size();
}

inline Value Inst::Arg(size_t index) const {
	EXIT_IF(index >= args.size());
	return args[index];
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
