#pragma once

#include <spirv-tools/libspirv.h>

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace Libs::Graphics {

// Diagnostic only: rename every SPIR-V ID without changing instructions or literal operands.
// A distinct salt gives the driver distinct input bytes without clearing its global cache.
inline bool SaltSpirvIds(std::span<const uint32_t> code, uint32_t salt,
                         std::vector<uint32_t>& output) {
	output.clear();
	if (code.size() < 5 || code[0] != 0x07230203u ||
	    salt > std::numeric_limits<uint32_t>::max() - code[3]) return false;
	if (salt == 0) {
		output.assign(code.begin(), code.end());
		return true;
	}
	struct State {
		std::vector<uint32_t>& output;
		uint32_t salt;
	};
	State state {output, salt};
	output.assign(code.begin(), code.begin() + 5);
	output[3] += salt;
	auto parse = [](void* data, const spv_parsed_instruction_t* instruction) -> spv_result_t {
		auto& state = *static_cast<State*>(data);
		const auto start = state.output.size();
		state.output.insert(state.output.end(), instruction->words,
		                    instruction->words + instruction->num_words);
		for (uint16_t i = 0; i < instruction->num_operands; ++i) {
			const auto& operand = instruction->operands[i];
			switch (operand.type) {
				case SPV_OPERAND_TYPE_ID:
				case SPV_OPERAND_TYPE_TYPE_ID:
				case SPV_OPERAND_TYPE_RESULT_ID:
				case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
				case SPV_OPERAND_TYPE_SCOPE_ID:
					state.output[start + operand.offset] += state.salt;
					break;
				default: break;
			}
		}
		return SPV_SUCCESS;
	};
	const auto context = spvContextCreate(SPV_ENV_VULKAN_1_3);
	const auto result = spvBinaryParse(context, &state, code.data(), code.size(), nullptr, parse, nullptr);
	spvContextDestroy(context);
	if (result != SPV_SUCCESS) output.clear();
	return result == SPV_SUCCESS;
}

} // namespace Libs::Graphics
