#include "graphics/shader/recompiler/CallSplit.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"

#include <cstring>
#include <initializer_list>
#include <vector>

// S_KYTY_CALL_STATE (CallSplit.h): the entry loads, the saves and the probes of call-split
// programs.
namespace Libs::Graphics::ShaderRecompiler::Frontend {
namespace {

Decoder::Operand CodeOperand(uint16_t code) {
	Decoder::Operand operand;
	switch (code) {
		case CallSplit::CodeVccLo: operand.kind = Decoder::OperandKind::VccLo; break;
		case CallSplit::CodeVccHi: operand.kind = Decoder::OperandKind::VccHi; break;
		case CallSplit::CodeM0: operand.kind = Decoder::OperandKind::M0; break;
		case CallSplit::CodeExecLo: operand.kind = Decoder::OperandKind::ExecLo; break;
		case CallSplit::CodeExecHi: operand.kind = Decoder::OperandKind::ExecHi; break;
		case CallSplit::CodeScc: operand.kind = Decoder::OperandKind::Scc; break;
		default:
			EXIT_IF(code >= IR::NumScalarRegs);
			operand.kind = Decoder::OperandKind::Sgpr;
			operand.reg  = code;
			break;
	}
	return operand;
}

IR::U32 CallStateBuiltin(IR::IREmitter& ir, IR::StageInputKind kind, uint32_t component) {
	return IR::U32(ir.Emit(IR::ValueOpcode::GetBuiltin,
	                       {IR::Value(static_cast<uint32_t>(kind)), IR::Value(component)}));
}

IR::U32 U32Imm(uint32_t value) {
	return IR::U32(IR::Value(value));
}

} // namespace

IR::Value Translator::CallStateAddress(IR::Value base, IR::U32 dword) {
	const auto offset = ir.Emit(IR::ValueOpcode::ShiftLeftLogical64,
	                            {ir.ConstructU64(dword, U32Imm(0)), IR::Value(2u)});
	return ir.Emit(IR::ValueOpcode::IAdd64, {base, offset});
}

IR::U32 Translator::ReadCallStateCode(uint16_t code) {
	if (code >= CallSplit::VectorCode) {
		return ir.GetVectorReg(static_cast<IR::VectorReg>(code - CallSplit::VectorCode));
	}
	return ReadRawU32(CodeOperand(code));
}

void Translator::WriteCallStateCode(uint16_t code, IR::U32 value) {
	if (code >= CallSplit::VectorCode) {
		// Every lane, whatever EXEC: the state of inactive lanes is restored too.
		ir.SetVectorReg(static_cast<IR::VectorReg>(code - CallSplit::VectorCode), value);
		return;
	}
	WriteRawU32(CodeOperand(code), value);
}

void Translator::EmitCallStateRecipe(const void* recipe_pointer, std::vector<IR::Value>& values) {
	const auto& recipe = *static_cast<const CallSplit::SiteRecipe*>(recipe_pointer);
	// Nodes are in dependency order. Every node is rebuilt once (unused ones are removed as dead
	// code).
	values.assign(recipe.nodes.size(), {});
	for (int32_t index = 0; index < static_cast<int32_t>(recipe.nodes.size()); index++) {
		const auto&            node = recipe.nodes[static_cast<size_t>(index)];
		std::vector<IR::Value> args;
		args.reserve(node.args.size());
		for (const auto arg: node.args) {
			args.push_back(arg < 0 ? recipe.imms.at(static_cast<size_t>(-1 - arg))
			                       : values.at(static_cast<size_t>(arg)));
		}
		uint64_t flags = node.flags;
		if (node.has_memory) {
			const auto memory_flags = AddMemoryInfo(node.memory, node.memory_pc);
			flags                   = 0;
			std::memcpy(&flags, &memory_flags, sizeof(memory_flags));
		}
		auto& value = values[static_cast<size_t>(index)];
		switch (args.size()) {
			case 0: value = ir.Emit(node.opcode, {}, flags); break;
			case 1: value = ir.Emit(node.opcode, {args[0]}, flags); break;
			case 2: value = ir.Emit(node.opcode, {args[0], args[1]}, flags); break;
			case 3: value = ir.Emit(node.opcode, {args[0], args[1], args[2]}, flags); break;
			case 4: value = ir.Emit(node.opcode, {args[0], args[1], args[2], args[3]}, flags); break;
			case 5:
				value = ir.Emit(node.opcode, {args[0], args[1], args[2], args[3], args[4]}, flags);
				break;
			default: EXIT("call-split recipe node with %zu arguments", args.size());
		}
	}
}

void Translator::S_KYTY_CALL_STATE(const Decoder::Instruction& inst) {
	if (const auto partition_op = static_cast<Decoder::CallStateOp>(inst.offset);
	    partition_op == Decoder::CallStateOp::PartitionCompare ||
	    partition_op == Decoder::CallStateOp::PartitionDrop ||
	    partition_op == Decoder::CallStateOp::MissReport) {
		EmitCallPartition(inst);
		return;
	}
	if (const auto split_op = static_cast<Decoder::CallStateOp>(inst.offset);
	    split_op == Decoder::CallStateOp::SplitTargetCompare ||
	    split_op == Decoder::CallStateOp::ShaderAddress) {
		EmitCallSplitAddress(inst);
		return;
	}
	auto* split = CallSplit::Current();
	EXIT_IF(split == nullptr || program.wave_size != 64u);
	const auto& header   = split->header;
	const bool  liveness = (header.flags & CallSplit::FlagLiveness) != 0;
	const auto  op       = static_cast<Decoder::CallStateOp>(inst.offset);
	const auto  site_of  = [&]() {
		const auto found = split->site_at_pc.find(inst.pc);
		EXIT_IF(found == split->site_at_pc.end() || found->second == 0 ||
		        found->second > header.sites.size());
		return found->second;
	};
	const auto recipe_of = [&](uint32_t site) -> const CallSplit::SiteRecipe* {
		if (split->recipe == nullptr || site == 0 || site > split->recipe->sites.size()) {
			return nullptr;
		}
		return &split->recipe->sites[site - 1u];
	};
	const auto marker = [&](uint16_t code) {
		split->marker_codes.push_back(code);
		return IR::U32(ir.Emit(
		    IR::ValueOpcode::CallStateParam,
		    {IR::Value(CallSplit::MarkerBase + static_cast<uint32_t>(split->marker_codes.size() - 1u))}));
	};
	const auto lane_dword = [&](const CallSplit::SiteLayout& layout, uint32_t slot, IR::U32 lane) {
		return ir.IAdd(IR::U32(split->wave_dword), ir.IAdd(U32Imm(layout.VectorBase() + slot * 64u), lane));
	};
	switch (op) {
		case Decoder::CallStateOp::Header: {
			EXIT_IF(inst.pc != 0 || split->prepared);
			split->prepared = true;
			if ((header.flags & CallSplit::FlagProbe) != 0) {
				return;
			}
			const auto param = [&](uint32_t index) {
				return IR::U32(ir.Emit(IR::ValueOpcode::CallStateParam, {IR::Value(index)}));
			};
			split->state_base =
			    ir.ConstructU64(param(CallSplit::ParamStateLo), param(CallSplit::ParamStateHi));
			split->header_base =
			    ir.ConstructU64(param(CallSplit::ParamHeaderLo), param(CallSplit::ParamHeaderHi));
			IR::U32 wave;
			if (header.entry_site == 0) {
				// One wave per workgroup (the planner requires it): the flat workgroup index.
				const auto id = [&](uint32_t axis) {
					return CallStateBuiltin(ir, IR::StageInputKind::WorkgroupId, axis);
				};
				const auto count = [&](uint32_t axis) {
					return CallStateBuiltin(ir, IR::StageInputKind::NumWorkgroups, axis);
				};
				wave = ir.IAdd(id(0), ir.IMul(count(0), ir.IAdd(id(1), ir.IMul(count(1), id(2)))));
			} else {
				// A wave list of the site (the bin pass): workgroup x runs the wave it names.
				const auto list =
				    ir.ConstructU64(param(CallSplit::ParamListLo), param(CallSplit::ParamListHi));
				wave = IR::U32(ir.Emit(
				    IR::ValueOpcode::CallStateLoad,
				    {CallStateAddress(list, CallStateBuiltin(ir, IR::StageInputKind::WorkgroupId, 0))}));
			}
			split->wave       = wave;
			split->wave_dword = ir.IMul(wave, param(CallSplit::ParamStride));
			if (header.entry_site == 0) {
				return;
			}
			const auto& layout = header.sites[header.entry_site - 1u];
			const auto* recipe = recipe_of(header.entry_site);
			const auto  lane   = IR::U32(ir.Emit(IR::ValueOpcode::LaneId));
			std::vector<IR::Value> rebuilt;
			if (!liveness && recipe != nullptr) {
				EmitCallStateRecipe(recipe, rebuilt);
			}
			for (uint32_t i = 0; i < layout.uniform.size(); i++) {
				const auto code = layout.uniform[i];
				IR::U32    value;
				if (liveness) {
					value = marker(code);
				} else if (recipe != nullptr && recipe->roots.contains(code)) {
					const auto root = recipe->roots.at(code);
					value = IR::U32(root < 0 ? recipe->imms.at(static_cast<size_t>(-1 - root))
					                         : rebuilt.at(static_cast<size_t>(root)));
				} else {
					value = IR::U32(ir.Emit(
					    IR::ValueOpcode::CallStateLoad,
					    {CallStateAddress(split->state_base, ir.IAdd(IR::U32(split->wave_dword), U32Imm(i)))}));
				}
				WriteCallStateCode(code, value);
			}
			split->loaded.clear();
			for (uint32_t j = 0; j < layout.vector.size(); j++) {
				const auto value =
				    liveness ? marker(layout.vector[j])
				             : IR::U32(ir.Emit(IR::ValueOpcode::CallStateLoad,
				                               {CallStateAddress(split->state_base,
				                                                 lane_dword(layout, j, lane))}));
				split->loaded.push_back(value);
				WriteCallStateCode(layout.vector[j], value);
			}
			return;
		}
		case Decoder::CallStateOp::Save: {
			const auto  site   = site_of();
			const auto& layout = header.sites[site - 1u];
			const auto* recipe = recipe_of(site);
			const auto  remat  = [&](uint16_t code) {
				return recipe != nullptr && recipe->roots.contains(code);
			};
			if (liveness) {
				// The next site's registers pass through this program: keep them alive.
				for (const auto code: layout.uniform) {
					if (!remat(code)) {
						ir.Emit(IR::ValueOpcode::CallStateProbe,
						        {IR::Value(0xffffffffu), ReadCallStateCode(code)});
					}
				}
				for (const auto code: layout.vector) {
					ir.Emit(IR::ValueOpcode::CallStateProbe,
					        {IR::Value(0xffffffffu), ReadCallStateCode(code)});
				}
				return;
			}
			EXIT_IF((header.flags & CallSplit::FlagProbe) != 0);
			ir.Emit(IR::ValueOpcode::CallStateStore,
			        {CallStateAddress(split->header_base, IR::U32(split->wave)), IR::Value(site)});
			for (uint32_t i = 0; i < layout.uniform.size(); i++) {
				if (remat(layout.uniform[i])) {
					continue;
				}
				ir.Emit(IR::ValueOpcode::CallStateStore,
				        {CallStateAddress(split->state_base, ir.IAdd(IR::U32(split->wave_dword), U32Imm(i))),
				         ReadCallStateCode(layout.uniform[i])});
			}
			const auto lane = IR::U32(ir.Emit(IR::ValueOpcode::LaneId));
			for (uint32_t j = 0; j < layout.vector.size(); j++) {
				ir.Emit(IR::ValueOpcode::CallStateStore,
				        {CallStateAddress(split->state_base, lane_dword(layout, j, lane)),
				         ReadCallStateCode(layout.vector[j])});
			}
			return;
		}
		case Decoder::CallStateOp::SaveChanged: {
			if (liveness) {
				return;
			}
			EXIT_IF(header.entry_site == 0);
			const auto& layout = header.sites[header.entry_site - 1u];
			EXIT_IF(split->loaded.size() != layout.vector.size());
			// Payload word 1 = shaded only: a lane whose call named another callee ran the skip path
			// here (its loop flags changed as if it had been called). Its registers stay as loaded, so
			// the callee pass of its own target still finds it pending, and the callee passes of one
			// site store disjoint lanes.
			const bool shaded_only = inst.word_count >= 2u && inst.raw[1] == 1u;
			const auto shaded      = shaded_only ? ir.GetGotoVariable(IR::SplitShadedVariable) : IR::U1 {};
			const auto lane        = IR::U32(ir.Emit(IR::ValueOpcode::LaneId));
			for (uint32_t j = 0; j < layout.vector.size(); j++) {
				const auto loaded = IR::U32(split->loaded[j]);
				auto       value  = ReadCallStateCode(layout.vector[j]);
				if (shaded_only) {
					value = ir.Select(shaded, value, loaded);
				}
				ir.Emit(IR::ValueOpcode::CallStateStoreChanged,
				        {CallStateAddress(split->state_base, lane_dword(layout, j, lane)), value, loaded});
			}
			return;
		}
		case Decoder::CallStateOp::Probe: {
			const auto site  = site_of();
			const auto probe = [&](uint16_t code) {
				ir.Emit(IR::ValueOpcode::CallStateProbe,
				        {IR::Value((site << 16u) | code), ReadCallStateCode(code)});
			};
			for (uint16_t code = 0; code < IR::NumScalarRegs; code++) {
				probe(code);
			}
			for (const uint16_t code: {CallSplit::CodeVccLo, CallSplit::CodeVccHi, CallSplit::CodeM0,
			                           CallSplit::CodeExecLo, CallSplit::CodeExecHi}) {
				probe(code);
			}
			return;
		}
	}
	EXIT("unknown S_KYTY_CALL_STATE operation %u at pc 0x%08x", inst.offset, inst.pc);
}

// Call-split callee programs (CallSplitPlan.cpp BuildCode): SplitTargetCompare sets SCC to whether the
// target pair s[reg:reg+1] is the callee the dispatch runs (push parameters ParamTargetLo/Hi);
// ShaderAddress writes the shader base plus an offset into a scalar pair (S_MOV_B32: SCC kept).
void Translator::EmitCallSplitAddress(const Decoder::Instruction& inst) {
	const auto op       = static_cast<Decoder::CallStateOp>(inst.offset);
	const auto pair_reg = inst.word_count >= 2u ? inst.raw[1] : 0xffffffffu;
	EXIT_IF(op == Decoder::CallStateOp::SplitTargetCompare ? inst.word_count != 2u : inst.word_count != 3u);
	Decoder::Operand lo_operand;
	Decoder::Operand hi_operand;
	if (pair_reg == 106u) {
		lo_operand.kind = Decoder::OperandKind::VccLo;
		hi_operand.kind = Decoder::OperandKind::VccHi;
	} else {
		EXIT_IF(pair_reg + 1u >= 106u);
		lo_operand.kind = Decoder::OperandKind::Sgpr;
		lo_operand.reg  = pair_reg;
		hi_operand      = lo_operand;
		hi_operand.reg  = pair_reg + 1u;
	}
	if (op == Decoder::CallStateOp::SplitTargetCompare) {
		EXIT_IF(CallSplit::Current() == nullptr);
		const auto param = [&](uint32_t index) {
			return IR::U32(ir.Emit(IR::ValueOpcode::CallStateParam, {IR::Value(index)}));
		};
		const auto match = ir.LogicalAnd(ir.IEqual(ReadRawU32(lo_operand), param(CallSplit::ParamTargetLo)),
		                                 ir.IEqual(ReadRawU32(hi_operand), param(CallSplit::ParamTargetHi)));
		// The active lanes are the call's lanes (the waterfall narrowed EXEC to one record's lanes).
		ir.SetGotoVariable(IR::SplitShadedVariable,
		                   ir.LogicalOr(ir.GetGotoVariable(IR::SplitShadedVariable), ir.LogicalAnd(ir.GetExec(), match)));
		ir.SetScc(match);
		return;
	}
	const auto base    = IR::U64(ir.Emit(IR::ValueOpcode::GetShaderBase));
	const auto address = IR::U64(
	    ir.Emit(IR::ValueOpcode::IAdd64, {base, IR::Value(static_cast<uint64_t>(inst.raw[2]))}));
	const auto words = ExtractU64(address);
	WriteOperand(lo_operand, words[0]);
	WriteOperand(hi_operand, words[1]);
}

// KYTY_CALL_PARTITIONS (ShaderCalls.h): the partition markers of an expanded call. The target pair
// is s[reg:reg+1]; push-data parameters 0/1 are the report set's address, 2/3 the dispatch's target.
void Translator::EmitCallPartition(const Decoder::Instruction& inst) {
	EXIT_IF(inst.word_count != 2u);
	const uint32_t pair_reg = inst.raw[1]; // the payload (DecodeSopp keeps it)
	EXIT_IF(pair_reg + 1u >= IR::NumScalarRegs);
	Decoder::Operand lo_operand;
	lo_operand.kind = Decoder::OperandKind::Sgpr;
	lo_operand.reg  = pair_reg;
	Decoder::Operand hi_operand = lo_operand;
	hi_operand.reg             = pair_reg + 1u;
	const auto lo    = ReadRawU32(lo_operand);
	const auto hi    = ReadRawU32(hi_operand);
	const auto param = [&](uint32_t index) {
		return IR::U32(ir.Emit(IR::ValueOpcode::CallStateParam, {IR::Value(index)}));
	};
	if (static_cast<Decoder::CallStateOp>(inst.offset) == Decoder::CallStateOp::PartitionCompare) {
		ir.SetScc(ir.LogicalAnd(ir.IEqual(lo, param(IR::CallPartitionParamTargetLo)),
		                        ir.IEqual(hi, param(IR::CallPartitionParamTargetHi))));
		return;
	}
	// One report per wave: the first active lane inserts the target's low word (with bit 0 set,
	// so it is never zero) into the dispatch's report set.
	const auto exec   = ir.GetExec();
	const auto lane   = IR::U32(ir.Emit(IR::ValueOpcode::LaneId));
	const auto first  = IR::U32(ir.Emit(IR::ValueOpcode::ReadFirstLane, {lane, exec}));
	const auto report_lo = param(IR::CallPartitionParamReportLo);
	const auto report_hi = param(IR::CallPartitionParamReportHi);
	// No report set (a dispatch that did not pass one): the lanes still leave, nothing is written.
	const auto has_report =
	    ir.LogicalNot(ir.IEqual(ir.BitwiseOr(report_lo, report_hi), IR::U32(IR::Value(0u))));
	const auto leader = ir.LogicalAnd(ir.LogicalAnd(exec, ir.IEqual(lane, first)), has_report);
	const auto base   = ir.ConstructU64(report_lo, report_hi);
	ir.Emit(IR::ValueOpcode::CallStateReport, {base, IR::U32(ir.BitwiseOr(lo, IR::U32(IR::Value(1u)))), leader});
	if (static_cast<Decoder::CallStateOp>(inst.offset) == Decoder::CallStateOp::MissReport) {
		return; // KYTY_CALL_LEARN: the lanes stay and take the skip path
	}
	// The active lanes leave the wave: IREmitter::SetExec keeps them out of every later EXEC.
	ir.SetGotoVariable(IR::PartitionDropVariable,
	                   ir.LogicalOr(ir.GetGotoVariable(IR::PartitionDropVariable), exec));
	ir.SetExec(IR::U1(IR::Value(false)));
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
