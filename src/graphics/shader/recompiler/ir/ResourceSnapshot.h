#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_

#include <array>
#include <cstdint>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct DescriptorValue {
	std::array<uint32_t, 8> dwords      = {};
	uint32_t                dword_count = 0;

	bool operator==(const DescriptorValue& other) const {
		return dword_count == other.dword_count && dwords == other.dwords;
	}
};

enum class UniformFillKind { None, Buffer, Image };

struct UniformFill {
	UniformFillKind          kind         = UniformFillKind::None;
	uint32_t                 resource     = 0;
	std::array<uint32_t, 3> group_stride {};
	uint32_t                 words        = 0;
	uint32_t                 value        = 0;

	bool operator==(const UniformFill&) const = default;
};

// A flat SRT slot the GPU fills (KYTY_SRT_GPU_FILL): its guest dword at `address` was written by
// GPU work into a buffer when the refresh ran, and only the shader uses the value, so the
// renderer copies the dword into the uploaded flat buffer on the GPU before the command; the
// snapshot's word at `flat_offset` is a placeholder (0). Reading it on the host instead waits for
// the GPU (a drain on the command processor).
struct SrtGpuFill {
	uint32_t flat_offset = 0;
	uint64_t address     = 0;

	bool operator==(const SrtGpuFill&) const = default;
};

// A guest descriptor heap a bindless image indexes: the host owns its key -> slot translation
// and patches the image's two flattened-SRT words (region base, entry count) before upload.
struct BindlessHeapUse {
	uint64_t base           = 0;
	uint64_t size           = 0;
	uint32_t table_offset   = 0;
	uint32_t image          = 0;
	uint32_t mapping_offset = 0;
	// Keys select records of `stride` bytes. With a record pointer (not UINT32_MAX), key k's T#
	// is at the 64-bit address stored at base + k * stride + record_pointer, plus table_offset;
	// otherwise at base + table_offset + k * 32.
	uint32_t stride         = 32;
	uint32_t record_pointer = UINT32_MAX;
	// Nested heaps (IndirectDescriptor::nested_vsharp): key = record * nested_keys + inner key.
	uint32_t nested_vsharp  = UINT32_MAX;
	uint32_t nested_heap    = 0;
	uint32_t nested_keys    = 0;
	// KYTY_RUNTIME_IMAGES: not a guest heap but the runtime image heap of `image`'s binding (the
	// T#s shaders loaded at run time and requested); nothing to patch, nothing read from guest
	// memory.
	bool     runtime        = false;

	bool operator==(const BindlessHeapUse&) const = default;
};

// KYTY_BINDLESS_SAMPLERS: a guest sampler heap a bindless sampler indexes. The host mirrors its S#
// records (key k's at base + table_offset + k * stride) into the bindless sampler array and
// patches the sampler's two flattened-SRT words (region base, record count); zeros select the
// stand-in sampler in slot 0.
struct BindlessSamplerHeapUse {
	uint64_t base           = 0;
	uint64_t size           = 0;
	uint32_t table_offset   = 0;
	uint32_t stride         = 16;
	uint32_t sampler        = 0;
	uint32_t mapping_offset = 0;

	bool operator==(const BindlessSamplerHeapUse&) const = default;
};

struct ResourceSnapshot {
	std::vector<DescriptorValue> buffers;
	std::vector<DescriptorValue> images;
	std::vector<DescriptorValue> samplers;
	std::vector<uint32_t>        flattened_srt;
	std::vector<uint32_t>        user_data;
	UniformFill                 uniform_fill;
	// Compute refreshes on the command processor only (SrtRuntime::gpu_written).
	std::vector<SrtGpuFill>      gpu_fills;
	std::vector<BindlessHeapUse> bindless_heaps;
	std::vector<BindlessSamplerHeapUse> bindless_sampler_heaps;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
