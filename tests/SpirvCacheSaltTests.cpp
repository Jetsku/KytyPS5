#include "graphics/host_gpu/spirvCacheSalt.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <vector>

int main() {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::vector<uint32_t> input;
	if (!tools.Assemble(R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %main "main"
OpName %entry "entry"
OpName %fn "fn"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%scope = OpConstant %uint 2
%semantics = OpConstant %uint 264
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpControlBarrier %scope %scope %semantics
OpReturn
OpFunctionEnd
)", &input) || !tools.Validate(input)) return 1;
	for (const uint32_t salt: {0u, 31u, 4096u}) {
		std::vector<uint32_t> output;
		if (!Libs::Graphics::SaltSpirvIds(input, salt, output) || !tools.Validate(output)) return 2;
		if (salt == 0 && output != input) return 3;
		if (output[3] != input[3] + salt || output.size() != input.size()) return 4;
		std::string before, after;
		tools.Disassemble(input, &before, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
		tools.Disassemble(output, &after, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
		if (before != after) return 5;
	}
	std::vector<uint32_t> output;
	if (Libs::Graphics::SaltSpirvIds(input, UINT32_MAX, output) || !output.empty()) return 6;
	input.pop_back();
	input.push_back(0xffffffffu);
	if (Libs::Graphics::SaltSpirvIds(input, 31, output) || !output.empty()) return 7;
	std::puts("SPIR-V cache salt: identity, ID references, literal preservation, validation and invalid input passed");
	return 0;
}
