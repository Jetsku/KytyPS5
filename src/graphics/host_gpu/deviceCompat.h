#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICECOMPAT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICECOMPAT_H_

#include <array>
#include <cstddef>
#include <cstdint>

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES // as in vulkanCommon.h
#endif
#include <vulkan/vulkan_core.h>

// Choices that let the emulator run on devices without every optional Vulkan feature it uses (AMD,
// Intel, older NVIDIA): pure functions of what the device reports, tested without a device
// (tests/DeviceCompatTests.cpp). A device that has everything (NVIDIA RTX) gets what it got before.
namespace Libs::Graphics::DeviceCompat {

// What the device reports about subgroup sizes (VK_EXT_subgroup_size_control, core in Vulkan 1.3).
struct SubgroupSizeControl {
	uint32_t min_size       = 0;     // minSubgroupSize
	uint32_t max_size       = 0;     // maxSubgroupSize
	bool     enabled        = false; // the subgroupSizeControl feature is enabled
	bool     compute        = false; // requiredSubgroupSizeStages includes the compute stage
	bool     compute_wave64 = false; // GraphicContext::compute_subgroup_size_control_enabled
};

// The subgroup size a compute pipeline requires so that one host subgroup holds one guest wave, or
// 0 to require nothing. A program runs a wave on min(wave_size, host_subgroup_size) invocations (a
// wave64 program on a 32-wide subgroup runs two lanes per invocation). A device that can run wave64
// (AMD) requires the wave size, as before. One with several sizes but no 64 (Intel: 8 to 32)
// requires that width, so the driver cannot pick a narrower one (SIMD8 or SIMD16) that splits each
// wave over several subgroups. A device with one size (NVIDIA: 32) requires nothing.
[[nodiscard]] constexpr uint32_t ComputeSubgroupSize(const SubgroupSizeControl& device,
                                                     uint32_t                   wave_size,
                                                     uint32_t host_subgroup_size) noexcept {
	if (device.compute_wave64) {
		return wave_size >= device.min_size && wave_size <= device.max_size ? wave_size : 0u;
	}
	const uint32_t size = wave_size < host_subgroup_size ? wave_size : host_subgroup_size;
	if (!device.enabled || !device.compute || device.min_size >= device.max_size ||
	    size < device.min_size || size > device.max_size) {
		return 0u;
	}
	return size;
}

// The subgroup size a mesh or pixel pipeline stage requires so that one host subgroup holds one
// guest wave, or 0 to require nothing. `default_size` is the device's default subgroup size and
// `stage_required` says requiredSubgroupSizeStages includes the stage. A device that can run wave64
// (AMD) requires the wave size where the default differs from it, as before. One with several sizes
// but no 64 (Intel: 8 to 32) picks the width per shader, so it requires min(wave size, default)
// even when that equals the default (the driver may choose SIMD8 or SIMD16 for a wave32 shader).
// A device with one size (NVIDIA: 32) requires nothing.
[[nodiscard]] constexpr uint32_t GraphicsSubgroupSize(const SubgroupSizeControl& device,
                                                      bool                       stage_required,
                                                      uint32_t                   wave_size,
                                                      uint32_t default_size) noexcept {
	if (!device.enabled || !stage_required) {
		return 0u;
	}
	if (device.compute_wave64) {
		return wave_size != default_size && wave_size >= device.min_size &&
		               wave_size <= device.max_size
		           ? wave_size
		           : 0u;
	}
	const uint32_t size = wave_size < default_size ? wave_size : default_size;
	if (device.min_size >= device.max_size || size < device.min_size || size > device.max_size) {
		return 0u;
	}
	return size;
}

// The rasterizer's depthClampEnable. The DB always clamps depth to the viewport range; with
// VK_EXT_depth_clip_enable Z clipping is set apart, so the clamp is always on. Without it the clamp
// also turns clipping off, so it is on only where the guest turns Z clipping off, and a draw that
// keeps clipping clips without the clamp.
[[nodiscard]] constexpr bool DepthClampEnable(bool depth_clip_enable_extension,
                                              bool guest_z_clip) noexcept {
	return depth_clip_enable_extension || !guest_z_clip;
}

[[nodiscard]] constexpr bool IsBlockCompressedFormat(VkFormat format) noexcept {
	return format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && format <= VK_FORMAT_BC7_SRGB_BLOCK;
}

// Whether the format itself (optimal tiling) supports every usage in `usage`. Usages without a
// matching format feature (e.g. the attachment feedback loop) count as unsupported.
[[nodiscard]] constexpr bool FormatSupportsUsage(VkFormatFeatureFlags features,
                                                 VkImageUsageFlags    usage) noexcept {
	constexpr std::array<std::array<uint32_t, 2>, 6> pairs {{
	    {VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_FORMAT_FEATURE_TRANSFER_SRC_BIT},
	    {VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_FORMAT_FEATURE_TRANSFER_DST_BIT},
	    {VK_IMAGE_USAGE_SAMPLED_BIT, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT},
	    {VK_IMAGE_USAGE_STORAGE_BIT, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT},
	    {VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT},
	    {VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
	     VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT},
	}};
	VkImageUsageFlags                                supported = 0;
	for (const auto& [usage_bit, feature_bit]: pairs) {
		if ((features & feature_bit) != 0) {
			supported |= usage_bit;
		}
	}
	return (usage & ~supported) == 0;
}

struct ImageCreateCandidate {
	VkImageUsageFlags  usage = 0;
	VkImageCreateFlags flags = 0;
};

struct ImageCreateCandidates {
	std::array<ImageCreateCandidate, 4> list {};
	uint32_t                            count = 0;
};

// For an image the device refuses: the image's usage and flags (the first candidate), then the
// same without what no role of the image needs on this device, dropped in turn (each candidate
// keeps the previous drops), until the device accepts one:
//  1. storage, when the format itself has no storage support: the usage only serves storage views
//     of another format (through EXTENDED_USAGE). The emulator's own storage writers (direct tiler
//     uploads, DCC clears, blits) check the image's usage first; a guest storage binding of such an
//     image stops with a message (Image::FindView);
//  2. BLOCK_TEXEL_VIEW_COMPATIBLE, once there is no storage usage: uncompressed views of a
//     block-compressed image serve its storage uploads; a guest binding of one stops likewise;
//  3. EXTENDED_USAGE, once the format itself supports every remaining usage (the flag only allows
//     usages it does not).
// Every usage a binding needs, and MUTABLE_FORMAT (views in other formats), stay.
[[nodiscard]] constexpr ImageCreateCandidates
OptionalImageCreateFallbacks(VkImageUsageFlags usage, VkImageCreateFlags flags,
                             VkFormatFeatureFlags features) noexcept {
	ImageCreateCandidates result;
	result.list[result.count++] = {usage, flags};
	if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0 &&
	    (features & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) == 0) {
		usage &= ~static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_STORAGE_BIT);
		result.list[result.count++] = {usage, flags};
	}
	if ((flags & VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT) != 0 &&
	    (usage & VK_IMAGE_USAGE_STORAGE_BIT) == 0) {
		flags &= ~static_cast<VkImageCreateFlags>(VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT);
		result.list[result.count++] = {usage, flags};
	}
	if ((flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT) != 0 && FormatSupportsUsage(features, usage)) {
		flags &= ~static_cast<VkImageCreateFlags>(VK_IMAGE_CREATE_EXTENDED_USAGE_BIT);
		result.list[result.count++] = {usage, flags};
	}
	return result;
}

// NVIDIA architecture from the PCI device ID. The Vulkan feature, limit and format reports of
// Turing (RTX 20, GTX 16) and Ampere (RTX 30) are identical on the same driver (vulkan.gpuinfo.org
// 51568 vs 52005, driver 616), so the ID is the only thing that tells them apart.
enum class NvidiaArchitecture : uint8_t {
	NotNvidia,
	Unknown,
	PascalOrOlder, // Pascal, Volta and older (below 0x1E00)
	Turing,        // TU102/TU104/TU106/TU117 0x1E00-0x1FFF, TU116 0x2180-0x21FF
	Ampere,        // GA100 0x20B0-0x20FF, GA10x 0x2200-0x25FF (Hopper 0x2321-0x233F included)
	Ada,           // AD10x 0x2680-0x28FF
	Blackwell,     // GB20x 0x2900-0x2FFF
};

constexpr uint32_t NvidiaVendorId = 0x10deu;

constexpr NvidiaArchitecture GuessNvidiaArchitecture(uint32_t vendor_id, uint32_t device_id) {
	if (vendor_id != NvidiaVendorId) {
		return NvidiaArchitecture::NotNvidia;
	}
	if (device_id < 0x1e00u) {
		return NvidiaArchitecture::PascalOrOlder;
	}
	if (device_id <= 0x1fffu || (device_id >= 0x2180u && device_id <= 0x21ffu)) {
		return NvidiaArchitecture::Turing;
	}
	if ((device_id >= 0x20b0u && device_id <= 0x20ffu) ||
	    (device_id >= 0x2200u && device_id <= 0x25ffu)) {
		return NvidiaArchitecture::Ampere;
	}
	if (device_id >= 0x2680u && device_id <= 0x28ffu) {
		return NvidiaArchitecture::Ada;
	}
	if (device_id >= 0x2900u && device_id <= 0x2fffu) {
		return NvidiaArchitecture::Blackwell;
	}
	return NvidiaArchitecture::Unknown;
}

constexpr const char* NvidiaArchitectureName(NvidiaArchitecture architecture) {
	switch (architecture) {
		case NvidiaArchitecture::NotNvidia: return "not NVIDIA";
		case NvidiaArchitecture::Unknown: return "NVIDIA (unknown architecture)";
		case NvidiaArchitecture::PascalOrOlder: return "NVIDIA Pascal/Volta or older";
		case NvidiaArchitecture::Turing: return "NVIDIA Turing (RTX 20 / GTX 16)";
		case NvidiaArchitecture::Ampere: return "NVIDIA Ampere (RTX 30)";
		case NvidiaArchitecture::Ada: return "NVIDIA Ada (RTX 40)";
		case NvidiaArchitecture::Blackwell: return "NVIDIA Blackwell (RTX 50)";
	}
	return "?";
}

// A device workaround switch: unset, empty or "auto" lets the device decide, "1"/"on"/"true"
// forces it on, "0"/"off"/"false" forces it off. Anything else counts as auto.
enum class QuirkMode : uint8_t { Auto, On, Off };

inline QuirkMode ParseQuirkMode(const char* value) {
	if (value == nullptr || value[0] == '\0') {
		return QuirkMode::Auto;
	}
	const auto equals = [&](const char* text) {
		size_t i = 0;
		for (; text[i] != '\0' && value[i] != '\0'; i++) {
			const char c = value[i] >= 'A' && value[i] <= 'Z' ? static_cast<char>(value[i] + 32)
			                                                  : value[i];
			if (c != text[i]) {
				return false;
			}
		}
		return text[i] == '\0' && value[i] == '\0';
	};
	if (equals("1") || equals("on") || equals("true")) {
		return QuirkMode::On;
	}
	if (equals("0") || equals("off") || equals("false")) {
		return QuirkMode::Off;
	}
	return QuirkMode::Auto;
}

constexpr bool ResolveQuirk(QuirkMode mode, bool automatic) {
	return mode == QuirkMode::On || (mode == QuirkMode::Auto && automatic);
}

constexpr const char* QuirkModeName(QuirkMode mode) {
	return mode == QuirkMode::On ? "on" : mode == QuirkMode::Off ? "off" : "auto";
}

// Defined values for geometry outputs a guest shader may leave unwritten (KYTY_VS_OUTPUT_GUARD,
// KYTY_MESH_OUTPUT_GUARD): automatic on NVIDIA Turing only, the architecture with the vertex
// explosions. Ampere and newer keep their code unless it is forced on.
constexpr bool GeometryOutputGuardAutomatic(NvidiaArchitecture architecture) {
	return architecture == NvidiaArchitecture::Turing;
}

} // namespace Libs::Graphics::DeviceCompat

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_DEVICECOMPAT_H_ */
