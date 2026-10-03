#pragma once

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// Device checks must match the renderer's whole-dword descriptor ranges.
struct HostBufferRobustness {
	bool storage_dword_loads_return_zero = false;
	bool null_descriptor_for_short_ranges = false;
};

constexpr HostBufferRobustness SelectHostBufferRobustness(
    uint32_t vendor_id, bool robust_buffer_access2, uint64_t storage_alignment,
    bool null_descriptor, bool amd_bounds_requested) {
	const bool amd_dword_bounds = amd_bounds_requested && vendor_id == 0x1002u &&
	                              robust_buffer_access2 && storage_alignment == 4u && null_descriptor;
	return {.storage_dword_loads_return_zero =
	            robust_buffer_access2 && (storage_alignment == 1u || amd_dword_bounds),
	        .null_descriptor_for_short_ranges = amd_dword_bounds};
}

constexpr uint64_t StorageBufferDwordRange(uint64_t bytes,
                                          const HostBufferRobustness& robustness) {
	return bytes >= 4u || robustness.null_descriptor_for_short_ranges ? bytes & ~uint64_t {3}
	                                                                  : bytes;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
