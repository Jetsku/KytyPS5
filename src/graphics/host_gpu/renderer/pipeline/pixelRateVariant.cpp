#include "graphics/host_gpu/renderer/pipeline/pixelRateVariant.h"

#include <cstdlib>
#include <initializer_list>
#include <mutex>
#include <spirv-tools/libspirv.h>
#include <spirv/unified1/spirv.hpp>
#include <unordered_map>

namespace Libs::Graphics {

namespace {

// What SpirvWithoutSampleId learns from parsing a module. SPIRV-Tools tells literal operands from
// <id> operands, which a word-by-word scan cannot: the variable's id is small, and so are storage
// classes, extended instruction numbers and memory operands.
struct SampleIdScan {
	uint32_t sample_id   = 0;
	bool     per_sample  = false; // another per-sample input (SamplePosition, a Sample decoration)
	bool     unsupported = false; // a form the rewrite does not handle (decoration groups, ...)
	bool     other_use   = false; // the variable named other than by a load, decoration, name or
	                              // the entry point's interface
};

bool IsIdOperand(spv_operand_type_t type) {
	switch (type) {
		case SPV_OPERAND_TYPE_ID:
		case SPV_OPERAND_TYPE_TYPE_ID:
		case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
		case SPV_OPERAND_TYPE_SCOPE_ID:
		case SPV_OPERAND_TYPE_OPTIONAL_ID:
		case SPV_OPERAND_TYPE_VARIABLE_ID: return true;
		default: return false;
	}
}

// First pass: the variable decorated BuiltIn SampleId, and anything else that shades per sample.
spv_result_t ScanSampleIdDecorations(void* user_data, const spv_parsed_instruction_t* inst) {
	auto&       scan  = *static_cast<SampleIdScan*>(user_data);
	const auto* words = inst->words;
	switch (static_cast<spv::Op>(inst->opcode)) {
		case spv::OpDecorate:
			if (inst->num_words >= 3 && words[2] == spv::DecorationSample) {
				scan.per_sample = true;
			}
			if (inst->num_words >= 4 && words[2] == spv::DecorationBuiltIn) {
				if (words[3] == spv::BuiltInSampleId) {
					if (scan.sample_id != 0 && scan.sample_id != words[1]) {
						scan.unsupported = true;
					}
					scan.sample_id = words[1];
				} else if (words[3] == spv::BuiltInSamplePosition) {
					scan.per_sample = true;
				}
			}
			break;
		case spv::OpMemberDecorate:
			if ((inst->num_words >= 4 && words[3] == spv::DecorationSample) ||
			    (inst->num_words >= 5 && words[3] == spv::DecorationBuiltIn &&
			     (words[4] == spv::BuiltInSampleId || words[4] == spv::BuiltInSamplePosition))) {
				scan.per_sample = true;
			}
			break;
		// A decoration group could decorate the variable (or another input) without naming it.
		case spv::OpGroupDecorate:
		case spv::OpGroupMemberDecorate: scan.unsupported = true; break;
		default: break;
	}
	return SPV_SUCCESS;
}

// Second pass: every <id> operand naming the variable, other than the forms that are rewritten.
spv_result_t ScanSampleIdUses(void* user_data, const spv_parsed_instruction_t* inst) {
	auto& scan = *static_cast<SampleIdScan*>(user_data);
	switch (static_cast<spv::Op>(inst->opcode)) {
		case spv::OpDecorate:
		case spv::OpName:
		case spv::OpEntryPoint: return SPV_SUCCESS;
		default: break;
	}
	for (uint16_t i = 0; i < inst->num_operands; i++) {
		const auto& operand = inst->operands[i];
		if (!IsIdOperand(operand.type) || inst->words[operand.offset] != scan.sample_id) {
			continue;
		}
		// A load from the variable (its pointer operand) is what the rewrite keeps.
		if (static_cast<spv::Op>(inst->opcode) == spv::OpLoad && operand.offset == 3) {
			continue;
		}
		scan.other_use = true;
	}
	return SPV_SUCCESS;
}

// One parse of a module by SPIRV-Tools, with operand types; false when it does not parse.
bool ParseSpirv(std::span<const uint32_t> spirv, SampleIdScan& scan,
                spv_parsed_instruction_fn_t callback) {
	static const spv_context context    = spvContextCreate(SPV_ENV_UNIVERSAL_1_6);
	spv_diagnostic           diagnostic = nullptr;
	const auto               result =
	    spvBinaryParse(context, &scan, spirv.data(), spirv.size(), nullptr, callback, &diagnostic);
	spvDiagnosticDestroy(diagnostic);
	return result == SPV_SUCCESS;
}

} // namespace

bool EqaaPerSampleShading() {
	static const bool value = [] {
		const char* text = std::getenv("KYTY_EQAA_PER_SAMPLE");
		return text != nullptr && text[0] == '1';
	}();
	return value;
}

bool SpirvShadesPerSample(std::span<const uint32_t> spirv) {
	SampleIdScan scan;
	if (spirv.size() < 5 || spirv[0] != spv::MagicNumber ||
	    !ParseSpirv(spirv, scan, ScanSampleIdDecorations)) {
		return false;
	}
	return scan.sample_id != 0 || scan.per_sample;
}

std::vector<uint32_t> SpirvWithoutSampleId(std::span<const uint32_t> spirv) {
	constexpr uint32_t HeaderWords = 5;
	if (spirv.size() < HeaderWords || spirv[0] != spv::MagicNumber) {
		return {};
	}
	SampleIdScan scan;
	if (!ParseSpirv(spirv, scan, ScanSampleIdDecorations) || scan.sample_id == 0 ||
	    scan.per_sample || scan.unsupported || !ParseSpirv(spirv, scan, ScanSampleIdUses) ||
	    scan.other_use) {
		return {};
	}
	const uint32_t sample_id = scan.sample_id;

	struct Instruction {
		size_t   offset;
		uint32_t count;
		spv::Op  opcode;
	};
	std::vector<Instruction> instructions;
	for (size_t offset = HeaderWords; offset < spirv.size();) {
		const uint32_t count = spirv[offset] >> spv::WordCountShift;
		if (count == 0 || offset + count > spirv.size()) {
			return {};
		}
		instructions.push_back(
		    {offset, count, static_cast<spv::Op>(spirv[offset] & spv::OpCodeMask)});
		offset += count;
	}
	const auto words = [&](const Instruction& inst) {
		return spirv.subspan(inst.offset, inst.count);
	};
	// Its declaration: an Input variable of a pointer to a 32-bit integer.
	const Instruction* variable = nullptr;
	uint32_t           pointer  = 0;
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpVariable && w.size() == 4 && w[2] == sample_id) {
			if (w[3] != spv::StorageClassInput) {
				return {};
			}
			variable = &inst;
			pointer  = w[1];
		}
	}
	if (variable == nullptr) {
		return {};
	}
	uint32_t integer = 0;
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpTypePointer && w.size() == 4 && w[1] == pointer &&
		    w[2] == spv::StorageClassInput) {
			integer = w[3];
		}
	}
	bool integer_ok = false;
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpTypeInt && w.size() == 4 && w[1] == integer && w[2] == 32) {
			integer_ok = true;
		}
	}
	if (!integer_ok) {
		return {};
	}
	// A Private pointer to the integer and a 0 of it: existing ones declared before the variable
	// where the module has them, new ones (declared just before it) otherwise.
	uint32_t bound        = spirv[3];
	uint32_t private_type = 0;
	uint32_t zero         = 0;
	for (const auto& inst: instructions) {
		if (&inst == variable) {
			break;
		}
		const auto w = words(inst);
		if (inst.opcode == spv::OpTypePointer && w.size() == 4 &&
		    w[2] == spv::StorageClassPrivate && w[3] == integer && private_type == 0) {
			private_type = w[1];
		} else if (inst.opcode == spv::OpConstant && w.size() == 4 && w[1] == integer &&
		           w[3] == 0 && zero == 0) {
			zero = w[2];
		}
	}
	const bool new_private = private_type == 0;
	const bool new_zero    = zero == 0;
	if (new_private) {
		private_type = bound++;
	}
	if (new_zero) {
		zero = bound++;
	}
	// Since SPIR-V 1.4 the entry point's interface lists every global variable it uses.
	const bool interface_lists_private = spirv[1] >= 0x00010400u;

	std::vector<uint32_t> out(spirv.begin(), spirv.begin() + HeaderWords);
	out[3] = bound;
	out.reserve(spirv.size() + 16);
	const auto append = [&](std::initializer_list<uint32_t> list) {
		out.insert(out.end(), list.begin(), list.end());
	};
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpDecorate && w.size() >= 2 && w[1] == sample_id) {
			continue; // BuiltIn, Flat
		}
		if (inst.opcode == spv::OpCapability && w.size() == 2 &&
		    w[1] == spv::CapabilitySampleRateShading) {
			continue; // nothing else needs it (no other per-sample input, see the scan)
		}
		if (inst.opcode == spv::OpEntryPoint && !interface_lists_private) {
			// Before SPIR-V 1.4 the interface lists Input and Output variables only.
			size_t name_end = 3;
			while (name_end < w.size()) {
				const auto word = w[name_end++];
				if ((word & 0xff000000u) == 0) {
					break; // the word holding the name's terminating zero
				}
			}
			std::vector<uint32_t> rewritten(w.begin(),
			                                w.begin() + static_cast<ptrdiff_t>(name_end));
			for (size_t i = name_end; i < w.size(); i++) {
				if (w[i] != sample_id) {
					rewritten.push_back(w[i]);
				}
			}
			rewritten[0] = (static_cast<uint32_t>(rewritten.size()) << spv::WordCountShift) |
			               spv::OpEntryPoint;
			out.insert(out.end(), rewritten.begin(), rewritten.end());
			continue;
		}
		if (&inst == variable) {
			if (new_private) {
				append({(4u << spv::WordCountShift) | spv::OpTypePointer, private_type,
				        spv::StorageClassPrivate, integer});
			}
			if (new_zero) {
				append({(4u << spv::WordCountShift) | spv::OpConstant, integer, zero, 0u});
			}
			append({(5u << spv::WordCountShift) | spv::OpVariable, private_type, sample_id,
			        spv::StorageClassPrivate, zero});
			continue;
		}
		out.insert(out.end(), w.begin(), w.end());
	}
	return out;
}

namespace PixelRate {

namespace {

std::mutex                             g_mutex;
std::unordered_map<uint64_t, Lookup>   g_programs;

} // namespace

void Register(uint64_t program_id, bool shades_per_sample, vk::ShaderModule module) {
	std::scoped_lock lock {g_mutex};
	g_programs[program_id] = {shades_per_sample, module};
}

void Forget(vk::Device device, uint64_t program_id) {
	vk::ShaderModule module = nullptr;
	{
		std::scoped_lock lock {g_mutex};
		const auto       found = g_programs.find(program_id);
		if (found == g_programs.end()) {
			return;
		}
		module = found->second.module;
		g_programs.erase(found);
	}
	if (module != nullptr) {
		device.destroyShaderModule(module, nullptr);
	}
}

Lookup Find(uint64_t program_id) {
	std::scoped_lock lock {g_mutex};
	const auto       found = g_programs.find(program_id);
	return found != g_programs.end() ? found->second : Lookup {};
}

} // namespace PixelRate

} // namespace Libs::Graphics
