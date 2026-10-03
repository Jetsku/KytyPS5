// KYTY_FUNCTION_ARRAY_SHRINK (graphics/host_gpu/spirvLocalArrays.h): Function-storage arrays are
// shrunk only to a proven index bound, the result validates, and unbounded uses are left alone.
// Optional argument: a .spv file to shrink and validate (prints the arrays and the time).
#include "graphics/host_gpu/spirvLocalArrays.h"

#include <spirv-tools/libspirv.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using Libs::Graphics::SpirvLocalArrays::Result;
using Libs::Graphics::SpirvLocalArrays::SetMaxSubgroupSize;
using Libs::Graphics::SpirvLocalArrays::Shrink;

int g_failures = 0;

void Expect(bool condition, const char* what) {
	if (!condition) {
		std::printf("SpirvLocalArraysTests: failed: %s\n", what);
		g_failures++;
	}
}

std::vector<uint32_t> Assemble(const std::string& text) {
	spvtools::SpirvTools  tools(SPV_ENV_VULKAN_1_3);
	std::vector<uint32_t> binary;
	tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
		std::printf("assembler: line %zu: %s\n", position.line, message);
	});
	if (!tools.Assemble(text, &binary, SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS)) {
		std::printf("SpirvLocalArraysTests: failed: assembly\n");
		g_failures++;
	}
	return binary;
}

bool Validate(const std::vector<uint32_t>& binary) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		std::printf("validator: %s\n", message);
	});
	return tools.Validate(binary);
}

std::string Disassemble(const std::vector<uint32_t>& binary) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          text;
	tools.Disassemble(binary, &text, SPV_BINARY_TO_TEXT_OPTION_NO_HEADER | SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
	return text;
}

// The recompiler's pattern for LDS in a pixel shader: a Function array of 8192 dwords indexed by
// (SubgroupLocalInvocationId * 4 + 256) / 4 and SubgroupLocalInvocationId, behind a bounds check.
// `index_source` replaces the lane load for the unbounded variants.
std::string LdsModule(const std::string& index_source, bool whole_array_load) {
	std::string text = R"(
               OpCapability Shader
               OpCapability GroupNonUniform
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %lane_var %out_var %in_var
               OpExecutionMode %main OriginUpperLeft
               OpName %lds "lds_dwords"
               OpDecorate %lane_var BuiltIn SubgroupLocalInvocationId
               OpDecorate %lane_var Flat
               OpDecorate %out_var Location 0
               OpDecorate %in_var Location 0
               OpDecorate %in_var Flat
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_2 = OpConstant %uint 2
   %uint_256 = OpConstant %uint 256
  %uint_8192 = OpConstant %uint 8192
    %arr8192 = OpTypeArray %uint %uint_8192
%ptr_fn_arr = OpTypePointer Function %arr8192
%ptr_fn_uint = OpTypePointer Function %uint
 %ptr_in_uint = OpTypePointer Input %uint
%ptr_out_uint = OpTypePointer Output %uint
   %lane_var = OpVariable %ptr_in_uint Input
     %in_var = OpVariable %ptr_in_uint Input
    %out_var = OpVariable %ptr_out_uint Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
        %lds = OpVariable %ptr_fn_arr Function
       %lane = OpLoad %uint )";
	text += index_source;
	text += R"(
     %shifted = OpShiftLeftLogical %uint %lane %uint_2
        %addr = OpIAdd %uint %shifted %uint_256
       %index = OpShiftRightLogical %uint %addr %uint_2
          %ok = OpULessThan %bool %index %uint_8192
               OpSelectionMerge %store_done None
               OpBranchConditional %ok %store %store_done
      %store = OpLabel
    %element = OpAccessChain %ptr_fn_uint %lds %index
               OpStore %element %lane
               OpBranch %store_done
 %store_done = OpLabel
      %index2 = OpShiftRightLogical %uint %shifted %uint_2
    %element2 = OpAccessChain %ptr_fn_uint %lds %index2
      %value = OpLoad %uint %element2
)";
	if (whole_array_load) {
		text += "      %whole = OpLoad %arr8192 %lds\n";
	}
	text += R"(
               OpStore %out_var %value
               OpReturn
               OpFunctionEnd
)";
	return text;
}

void TestBoundedLds() {
	SetMaxSubgroupSize(32);
	const auto module = Assemble(LdsModule("%lane_var", false));
	Expect(Validate(module), "input module validates");
	std::vector<uint32_t> out;
	const Result          result = Shrink(module, out);
	Expect(result.changed, "bounded LDS array is shrunk");
	Expect(result.arrays.size() == 1, "one array shrunk");
	if (!result.arrays.empty()) {
		// Largest index: (31 * 4 + 256) / 4 = 95.
		Expect(result.arrays[0].old_length == 8192, "old length 8192");
		Expect(result.arrays[0].new_length == 96, "new length 96 (lane < 32, offset 64)");
		Expect(result.arrays[0].name == "lds_dwords", "array name");
	}
	Expect(result.bytes_before == 8192u * 4u, "bytes before");
	Expect(result.bytes_after == 96u * 4u, "bytes after");
	Expect(Validate(out), "shrunk module validates");
	const auto text = Disassemble(out);
	Expect(text.find("OpConstant %uint 96") != std::string::npos, "new length constant present");
	SetMaxSubgroupSize(128);
	std::vector<uint32_t> out128;
	const auto            wide = Shrink(module, out128);
	Expect(wide.changed && !wide.arrays.empty() && wide.arrays[0].new_length == 192,
	       "128-wide subgroups: new length 192");
	Expect(Validate(out128), "128-wide shrunk module validates");
}

void TestInitModes() {
	using Libs::Graphics::SpirvLocalArrays::Init;
	SetMaxSubgroupSize(32);
	const auto module = Assemble(LdsModule("%lane_var", false));
	std::vector<uint32_t> zero;
	const auto            zero_result = Shrink(module, zero, Init::Zero);
	Expect(zero_result.changed && Validate(zero), "zero-filled shrunk module validates");
	Expect(Disassemble(zero).find("OpConstantNull") != std::string::npos, "zero fill uses OpConstantNull");
	std::vector<uint32_t> poison;
	const auto            poison_result = Shrink(module, poison, Init::Poison);
	Expect(poison_result.changed && Validate(poison), "NaN-filled shrunk module validates");
	const auto text = Disassemble(poison);
	Expect(text.find("OpConstant %uint 2143289344") != std::string::npos, "poison constant 0x7fc00000");
	Expect(text.find("OpConstantComposite") != std::string::npos, "poison fill is a constant composite");
	Expect(zero_result.arrays.size() == 1 && zero_result.arrays[0].new_length == 96, "same length with fills");
}

void TestUnboundedIndex() {
	// The index comes from an input varying: no bound, no change.
	const auto            module = Assemble(LdsModule("%in_var", false));
	std::vector<uint32_t> out;
	const Result          result = Shrink(module, out);
	Expect(!result.changed, "unbounded index leaves the module alone");
	Expect(out.empty(), "no output for an unchanged module");
	Expect(result.unbounded_arrays == 1, "array counted as unbounded");
}

void TestWholeArrayUse() {
	// A load of the whole array needs the declared type: no change.
	const auto            module = Assemble(LdsModule("%lane_var", true));
	std::vector<uint32_t> out;
	const Result          result = Shrink(module, out);
	Expect(!result.changed, "whole-array use leaves the module alone");
}

void TestNoFunctionArrays() {
	// No Function-storage array (only a scalar Function variable): the declarations pre-scan returns
	// before any analysis, the module is unchanged and has no Function-array bytes.
	const auto module = Assemble(R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %out_var
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %out_var Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_4 = OpConstant %uint 4
      %arr4 = OpTypeArray %uint %uint_4
%ptr_out_uint = OpTypePointer Output %uint
%ptr_fn_uint = OpTypePointer Function %uint
    %out_var = OpVariable %ptr_out_uint Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
     %scalar = OpVariable %ptr_fn_uint Function
               OpStore %scalar %uint_4
      %value = OpLoad %uint %scalar
               OpStore %out_var %value
               OpReturn
               OpFunctionEnd
)");
	Expect(Validate(module), "no-array module validates");
	std::vector<uint32_t> out;
	const Result          result = Shrink(module, out);
	Expect(!result.changed && out.empty(), "no Function array: unchanged");
	Expect(result.bytes_before == 0 && result.unbounded_arrays == 0, "no Function array: no bytes counted");
}

void TestNotSpirv() {
	const std::vector<uint32_t> garbage {1, 2, 3, 4, 5, 6};
	std::vector<uint32_t>       out;
	Expect(!Shrink(garbage, out).changed, "not a module: unchanged");
}

int ShrinkFile(const char* path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		std::printf("cannot open %s\n", path);
		return 1;
	}
	std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	std::vector<uint32_t> module(bytes.size() / 4);
	std::memcpy(module.data(), bytes.data(), module.size() * 4);
	SetMaxSubgroupSize(32);
	std::vector<uint32_t> out;
	const auto            begin   = std::chrono::steady_clock::now();
	const auto            result  = Shrink(module, out);
	const auto            shrunk  = std::chrono::steady_clock::now();
	const bool            valid   = result.changed ? Validate(out) : true;
	const auto            checked = std::chrono::steady_clock::now();
	std::printf("%s: %zu words, changed %d, %u unbounded, %llu -> %llu bytes per invocation, shrink %.2f ms, "
	            "validate %.2f ms, valid %d\n",
	            path, module.size(), result.changed ? 1 : 0, result.unbounded_arrays,
	            static_cast<unsigned long long>(result.bytes_before),
	            static_cast<unsigned long long>(result.bytes_after),
	            std::chrono::duration<double, std::milli>(shrunk - begin).count(),
	            std::chrono::duration<double, std::milli>(checked - shrunk).count(), valid ? 1 : 0);
	for (const auto& array: result.arrays) {
		std::printf("  %s: %u -> %u elements of %u bytes\n", array.name.c_str(), array.old_length,
		            array.new_length, array.element_bytes);
	}
	// The fill variants must validate too.
	bool fills_valid = true;
	for (const auto init: {Libs::Graphics::SpirvLocalArrays::Init::Zero, Libs::Graphics::SpirvLocalArrays::Init::Poison}) {
		std::vector<uint32_t> filled;
		if (Shrink(module, filled, init).changed && !Validate(filled)) {
			fills_valid = false;
		}
	}
	std::printf("  zero and NaN fills validate: %d\n", fills_valid ? 1 : 0);
	return valid && fills_valid ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
	if (argc > 1) {
		int failures = 0;
		for (int index = 1; index < argc; index++) {
			failures += ShrinkFile(argv[index]);
		}
		return failures == 0 ? 0 : 1;
	}
	TestBoundedLds();
	TestInitModes();
	TestUnboundedIndex();
	TestWholeArrayUse();
	TestNoFunctionArrays();
	TestNotSpirv();
	if (g_failures != 0) {
		std::printf("%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("spirv_local_arrays: all tests passed\n");
	return 0;
}
