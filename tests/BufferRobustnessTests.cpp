#include "graphics/shader/recompiler/backend/spirv/HostBufferRobustness.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

using namespace Libs::Graphics::ShaderRecompiler::Spirv;

static void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		std::exit(1);
	}
}

int main() {
	constexpr uint32_t Amd = 0x1002;
	constexpr uint32_t Nvidia = 0x10de;
	const auto amd = SelectHostBufferRobustness(Amd, true, 4, true, true);
	Check(amd.storage_dword_loads_return_zero, "AMD alignment 4 delegates bounds when opted in");
	Check(amd.null_descriptor_for_short_ranges, "AMD alignment 4 requires null short ranges");
	Check(!SelectHostBufferRobustness(Amd, true, 4, true, false).storage_dword_loads_return_zero,
	      "AMD optimization defaults off");
	Check(!SelectHostBufferRobustness(Nvidia, true, 4, true, true).storage_dword_loads_return_zero,
	      "alignment 4 optimization is AMD only");
	Check(!SelectHostBufferRobustness(Amd, false, 4, true, true).storage_dword_loads_return_zero,
	      "robustBufferAccess2 is required");
	Check(!SelectHostBufferRobustness(Amd, true, 4, false, true).storage_dword_loads_return_zero,
	      "nullDescriptor is required for AMD alignment 4");
	Check(!SelectHostBufferRobustness(Amd, true, 8, true, true).storage_dword_loads_return_zero,
	      "unsupported alignment keeps shader bounds");
	const auto nvidia = SelectHostBufferRobustness(Nvidia, true, 1, false, false);
	Check(nvidia.storage_dword_loads_return_zero && !nvidia.null_descriptor_for_short_ranges,
	      "existing alignment 1 behavior is preserved");
	for (uint64_t bytes = 0; bytes < 260; ++bytes) {
		const auto range = StorageBufferDwordRange(bytes, amd);
		Check(range == (bytes & ~uint64_t {3}), "AMD descriptors contain only complete dwords");
		Check(StorageBufferDwordRange(bytes, {}) == (bytes >= 4 ? range : bytes),
		      "disabled descriptor behavior is preserved");
	}
	// Buffer offset adjustments are dword aligned; a three-byte guest range must not expose
	// another word after a 4/16/252-byte prefix, while a four-byte guest range must expose one.
	for (uint64_t adjustment: {4u, 16u, 252u}) {
		Check(StorageBufferDwordRange(adjustment + 3u, amd) == adjustment,
		      "partial dword after alignment prefix is inaccessible");
		Check(StorageBufferDwordRange(adjustment + 4u, amd) == adjustment + 4u,
		      "complete dword after alignment prefix remains accessible");
	}
	std::puts("Buffer robustness capability and descriptor boundary tests passed");
}
