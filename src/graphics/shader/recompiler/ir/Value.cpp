#include "graphics/shader/recompiler/ir/Value.h"

#include "common/config.h"
#include "graphics/shader/recompiler/CodegenOptions.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>

namespace Libs::Graphics::ShaderRecompiler::IR {

Value::Value(Inst* value): type(Type::Opaque), inst(value) {}
Value::Value(ScalarReg value): type(Type::ScalarReg), scalar_reg(value) {}
Value::Value(VectorReg value): type(Type::VectorReg), vector_reg(value) {}
Value::Value(bool value): type(Type::U1), imm_u1(value) {}
Value::Value(uint8_t value): type(Type::U8), imm_u8(value) {}
Value::Value(uint16_t value): type(Type::U16), imm_u16(value) {}
Value::Value(uint32_t value): type(Type::U32), imm_u32(value) {}
Value::Value(uint64_t value): type(Type::U64), imm_u64(value) {}

Value::Value(Type value_type, uint64_t bits): type(value_type), imm_u64(bits) {}

Value Value::F16(uint16_t bits) {
	return Value(Type::F16, bits);
}

Value Value::F32(float value) {
	return Value(Type::F32, std::bit_cast<uint32_t>(value));
}

Type Value::GetType() const {
	if (IsPhi()) {
		return inst->Flags<Type>();
	}
	if (IsIdentity()) {
		return inst->Arg(0).GetType();
	}
	return type == Type::Opaque ? inst->GetType() : type;
}

Inst* Value::Instruction() const {
	EXIT_IF(type != Type::Opaque);
	return inst;
}

ScalarReg Value::ScalarRegister() const {
	EXIT_IF(type != Type::ScalarReg);
	return scalar_reg;
}

VectorReg Value::VectorRegister() const {
	EXIT_IF(type != Type::VectorReg);
	return vector_reg;
}

bool Value::U1() const {
	EXIT_IF(type != Type::U1);
	return imm_u1;
}

uint8_t Value::U8() const {
	EXIT_IF(type != Type::U8);
	return imm_u8;
}

uint16_t Value::U16() const {
	EXIT_IF(type != Type::U16);
	return imm_u16;
}

uint64_t Value::U64() const {
	EXIT_IF(type != Type::U64);
	return imm_u64;
}

uint16_t Value::F16Bits() const {
	EXIT_IF(type != Type::F16);
	return imm_u16;
}

float Value::F32Value() const {
	EXIT_IF(type != Type::F32);
	return std::bit_cast<float>(imm_u32);
}

bool Value::operator==(const Value& other) const {
	if (type != other.type) {
		return false;
	}
	switch (type) {
		case Type::Void: return true;
		case Type::Opaque: return inst == other.inst;
		case Type::ScalarReg: return scalar_reg == other.scalar_reg;
		case Type::VectorReg: return vector_reg == other.vector_reg;
		case Type::U1: return imm_u1 == other.imm_u1;
		case Type::U8: return imm_u8 == other.imm_u8;
		case Type::U16:
		case Type::F16: return imm_u16 == other.imm_u16;
		case Type::U32:
		case Type::F32: return imm_u32 == other.imm_u32;
		case Type::U64: return imm_u64 == other.imm_u64;
		default: return false;
	}
}

Inst::Inst(ValueOpcode value_opcode, uint64_t value_flags)
    : opcode(value_opcode), flags(value_flags) {
	const auto count = NumArgsOf(opcode);
	if (count != std::numeric_limits<size_t>::max()) {
		args.resize(count);
	}
}

Inst::~Inst() {
	ClearArgs();
}

Type Inst::GetType() const {
	if (opcode == ValueOpcode::Phi) {
		return static_cast<Type>(flags);
	}
	if (opcode == ValueOpcode::Identity && !args.empty()) {
		return args.front().GetType();
	}
	return TypeOf(opcode);
}

bool Inst::MayHaveSideEffects() const {
	return HasSideEffects(opcode);
}

bool Inst::HasUses() const {
	return !uses.empty();
}

size_t Inst::UseCount() const {
	return uses.size();
}

size_t Inst::NumPhiBlocks() const {
	return phi_blocks.size();
}

Block* Inst::PhiBlock(size_t index) const {
	EXIT_IF(opcode != ValueOpcode::Phi || index >= phi_blocks.size());
	return phi_blocks[index];
}

Block* Inst::Parent() const {
	return parent;
}

const std::vector<Use>& Inst::Uses() const {
	return uses;
}

void Inst::SetParent(Block* block) {
	parent = block;
}

void Inst::SetArg(size_t index, Value value) {
	if (index >= args.size()) {
		EXIT_IF(NumArgsOf(opcode) != std::numeric_limits<size_t>::max());
		args.resize(index + 1);
	}
	const auto old = args[index];
	if (auto* old_inst = old.TryInstruction(); old_inst != nullptr) {
		RemoveUse(old_inst, index);
	}
	args[index] = value;
	if (auto* new_inst = value.TryInstruction(); new_inst != nullptr) {
		AddUse(new_inst, index);
	}
}

void Inst::AddPhiOperand(Block* predecessor, Value value) {
	EXIT_IF(opcode != ValueOpcode::Phi);
	const auto index = args.size();
	args.push_back(value);
	phi_blocks.push_back(predecessor);
	if (auto* value_inst = value.TryInstruction(); value_inst != nullptr) {
		AddUse(value_inst, index);
	}
}

void Inst::ReplaceUsesWith(Value replacement, bool preserve) {
	if (GetCodegenOptions().ir_linear_uses) {
		// KYTY_IR_LINEAR_USES (Senaxx 5145dc1f9): every use goes, so the list is taken over at
		// once, rewriting each user's slot and appending it to the replacement's uses in the same
		// order as SetArg would, without searching and erasing this instruction's list once per use
		// (quadratic for widely used values; the SSA rewrite and identity removal spent most of a
		// big shader's IR passes there). The argument lists are left first, while this list is
		// intact (a phi can use itself; Invalidate below then finds nothing to remove): when the
		// replacement is one of this instruction's own arguments, as for every identity, removing
		// this instruction from its list before the moved uses are appended shifts only the older
		// entries, and the list ends up in the same order.
		ClearArgs();
		auto  old_uses         = std::move(uses);
		auto* replacement_inst = replacement.TryInstruction();
		uses.clear();
		for (const auto& use: old_uses) {
			EXIT_IF(use.operand >= use.user->args.size() ||
			        use.user->args[use.operand].TryInstruction() != this);
			use.user->args[use.operand] = replacement;
			if (replacement_inst != nullptr) {
				// That slot held this instruction until now, so the replacement cannot list it yet.
				replacement_inst->uses.push_back({use.user, use.operand});
			}
		}
	} else {
		const auto old_uses = uses;
		for (const auto& use: old_uses) {
			use.user->SetArg(use.operand, replacement);
		}
	}
	Invalidate();
	if (preserve) {
		ReplaceOpcode(ValueOpcode::Identity);
		args.resize(1);
		SetArg(0, replacement);
	}
}

void Inst::ReplaceOpcode(ValueOpcode value_opcode) {
	opcode           = value_opcode;
	const auto count = NumArgsOf(opcode);
	if (count != std::numeric_limits<size_t>::max()) {
		EXIT_IF(!args.empty() && args.size() != count);
		args.resize(count);
	}
}

void Inst::Invalidate() {
	ClearArgs();
	opcode = ValueOpcode::Void;
}

void Inst::AddUse(Inst* used, size_t operand) {
	// The duplicate search is linear in the use count, so quadratic for widely used values: with
	// KYTY_IR_LINEAR_USES only debug builds run it.
	if (KYTY_BUILD == KYTY_BUILD_DEBUG || !GetCodegenOptions().ir_linear_uses) {
		const auto found = std::ranges::find_if(
		    used->uses, [&](const Use& use) { return use.user == this && use.operand == operand; });
		EXIT_IF(found != used->uses.end());
	}
	used->uses.push_back({this, operand});
}

void Inst::RemoveUse(Inst* used, size_t operand) {
	auto& list = used->uses;
	if (GetCodegenOptions().ir_linear_uses) {
		// Uses are unique, and the one removed is usually among the most recent: search from the
		// end (KYTY_IR_LINEAR_USES).
		const auto found = std::find_if(list.rbegin(), list.rend(), [&](const Use& use) {
			return use.user == this && use.operand == operand;
		});
		EXIT_IF(found == list.rend());
		list.erase(std::next(found).base());
		return;
	}
	const auto found = std::ranges::find_if(
	    list, [&](const Use& use) { return use.user == this && use.operand == operand; });
	EXIT_IF(found == list.end());
	list.erase(found);
}

void Inst::ReplaceUsesForRemoval(Value replacement, const std::unordered_set<const Inst*>& removed,
                                 std::vector<Inst*>& touched) {
	auto  old_uses         = std::move(uses);
	auto* replacement_inst = replacement.TryInstruction();
	uses.clear();
	for (const auto& use: old_uses) {
		if (use.user == this || removed.contains(use.user)) {
			continue;
		}
		EXIT_IF(use.operand >= use.user->args.size() ||
		        use.user->args[use.operand].TryInstruction() != this);
		use.user->args[use.operand] = replacement;
		if (replacement_inst != nullptr) {
			replacement_inst->uses.push_back({use.user, use.operand});
		}
	}
	for (const auto& arg: args) {
		if (auto* target = arg.TryInstruction(); target != nullptr && target != this) {
			touched.push_back(target);
		}
	}
	args.clear();
	phi_blocks.clear();
	opcode = ValueOpcode::Void;
}

void Inst::DropRemovedUses(std::span<Inst* const>              touched,
                           const std::unordered_set<const Inst*>& removed) {
	// Each target once: a value many identities folded into would be filtered once per identity.
	std::vector<Inst*> targets(touched.begin(), touched.end());
	std::ranges::sort(targets);
	targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
	for (auto* target: targets) {
		// A removed target is erased already; the others drop their removed users in one pass.
		if (!removed.contains(target)) {
			std::erase_if(target->uses, [&](const Use& use) { return removed.contains(use.user); });
		}
	}
}

void Inst::DropForDestruction() {
	// Every instruction that could refer to this one is being destroyed with it.
	args.clear();
	phi_blocks.clear();
	uses.clear();
	opcode = ValueOpcode::Void;
}

void Inst::ClearArgs() {
	for (size_t index = 0; index < args.size(); index++) {
		if (auto* value_inst = args[index].TryInstruction(); value_inst != nullptr) {
			RemoveUse(value_inst, index);
		}
	}
	args.clear();
	phi_blocks.clear();
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
