// Device-capability choices (graphics/host_gpu/deviceCompat.h) for the subgroup sizes, features and
// formats that NVIDIA, AMD and Intel GPUs report.
#include "graphics/host_gpu/breadcrumbLogic.h"
#include "graphics/host_gpu/deviceCompat.h"

#include <cstdint>
#include <cstdio>

namespace {

using namespace Libs::Graphics::DeviceCompat;

int g_failures = 0;

void Expect(bool condition, const char* what) {
	if (!condition) {
		std::printf("DeviceCompatTests: failed: %s\n", what);
		g_failures++;
	}
}

void TestComputeSubgroupSize() {
	// NVIDIA (RTX 3090): one size, so nothing is required, as before.
	const SubgroupSizeControl nvidia {.min_size = 32, .max_size = 32, .enabled = false,
	                                  .compute = true, .compute_wave64 = false};
	Expect(ComputeSubgroupSize(nvidia, 32, 32) == 0, "NVIDIA: wave32 requires nothing");
	Expect(ComputeSubgroupSize(nvidia, 64, 32) == 0, "NVIDIA: wave64 (two lanes) requires nothing");
	// AMD (RDNA, Windows or RADV): the wave size, as before.
	const SubgroupSizeControl amd {.min_size = 32, .max_size = 64, .enabled = true, .compute = true,
	                               .compute_wave64 = true};
	Expect(ComputeSubgroupSize(amd, 32, 64) == 32, "AMD: wave32 requires 32");
	Expect(ComputeSubgroupSize(amd, 64, 64) == 64, "AMD: wave64 requires 64");
	// Intel (Arc, Iris Xe, UHD: 8 to 32): 32 for both wave sizes (wave64 runs two lanes each).
	const SubgroupSizeControl intel {.min_size = 8, .max_size = 32, .enabled = true, .compute = true,
	                                 .compute_wave64 = false};
	Expect(ComputeSubgroupSize(intel, 32, 32) == 32, "Intel: wave32 requires 32");
	Expect(ComputeSubgroupSize(intel, 64, 32) == 32, "Intel: wave64 on 32 lanes requires 32");
	// Intel Xe2 (16 to 32).
	const SubgroupSizeControl xe2 {.min_size = 16, .max_size = 32, .enabled = true, .compute = true,
	                               .compute_wave64 = false};
	Expect(ComputeSubgroupSize(xe2, 32, 32) == 32, "Xe2: wave32 requires 32");
	// Without the feature, or without the compute stage, nothing can be required.
	auto no_feature    = intel;
	no_feature.enabled = false;
	Expect(ComputeSubgroupSize(no_feature, 32, 32) == 0, "no subgroupSizeControl: nothing");
	auto no_stage    = intel;
	no_stage.compute = false;
	Expect(ComputeSubgroupSize(no_stage, 32, 32) == 0, "compute not in the required stages: nothing");
	// A size outside the device's range is never required.
	const SubgroupSizeControl narrow {.min_size = 4, .max_size = 16, .enabled = true, .compute = true,
	                                  .compute_wave64 = false};
	Expect(ComputeSubgroupSize(narrow, 32, 32) == 0, "32 above the maximum: nothing");
	// GCN on AMD's Windows driver (64 only) keeps its wave64 requirement and cannot narrow wave32.
	const SubgroupSizeControl gcn {.min_size = 64, .max_size = 64, .enabled = true, .compute = true,
	                               .compute_wave64 = true};
	Expect(ComputeSubgroupSize(gcn, 64, 64) == 64, "GCN: wave64 requires 64");
	Expect(ComputeSubgroupSize(gcn, 32, 64) == 0, "GCN: wave32 cannot be required");
}

void TestGraphicsSubgroupSize() {
	// NVIDIA: one size, nothing is required for either stage.
	const SubgroupSizeControl nvidia {.min_size = 32, .max_size = 32, .enabled = false};
	Expect(GraphicsSubgroupSize(nvidia, true, 32, 32) == 0, "NVIDIA: wave32 requires nothing");
	Expect(GraphicsSubgroupSize(nvidia, true, 64, 32) == 0, "NVIDIA: wave64 requires nothing");
	auto nvidia_enabled    = nvidia;
	nvidia_enabled.enabled = true;
	Expect(GraphicsSubgroupSize(nvidia_enabled, true, 32, 32) == 0, "min == max: nothing");
	// AMD (default 64): the wave size when it differs from the default, as before.
	const SubgroupSizeControl amd {.min_size = 32, .max_size = 64, .enabled = true, .compute = true,
	                               .compute_wave64 = true};
	Expect(GraphicsSubgroupSize(amd, true, 32, 64) == 32, "AMD: wave32 requires 32");
	Expect(GraphicsSubgroupSize(amd, true, 64, 64) == 0, "AMD: wave64 is the default, nothing");
	Expect(GraphicsSubgroupSize(amd, true, 32, 32) == 0, "AMD: wave32 on default 32, nothing");
	Expect(GraphicsSubgroupSize(amd, false, 32, 64) == 0, "AMD: stage not required: nothing");
	const SubgroupSizeControl gcn {.min_size = 64, .max_size = 64, .enabled = true, .compute = true,
	                               .compute_wave64 = true};
	Expect(GraphicsSubgroupSize(gcn, true, 32, 64) == 0, "GCN: wave32 cannot be required");
	Expect(GraphicsSubgroupSize(gcn, true, 64, 64) == 0, "GCN: wave64 is the default, nothing");
	// Intel (8 to 32, default 32): the driver picks the width, so 32 is required for wave32 and
	// for wave64 (two lanes per invocation); a wave16 or wave8 program requires its own size.
	const SubgroupSizeControl intel {.min_size = 8, .max_size = 32, .enabled = true, .compute = true};
	Expect(GraphicsSubgroupSize(intel, true, 32, 32) == 32, "Intel: wave32 requires 32");
	Expect(GraphicsSubgroupSize(intel, true, 64, 32) == 32, "Intel: wave64 requires 32");
	Expect(GraphicsSubgroupSize(intel, true, 16, 32) == 16, "Intel: wave16 requires 16");
	Expect(GraphicsSubgroupSize(intel, false, 32, 32) == 0, "Intel: stage not required: nothing");
	auto no_feature    = intel;
	no_feature.enabled = false;
	Expect(GraphicsSubgroupSize(no_feature, true, 32, 32) == 0, "Intel: no feature: nothing");
	const SubgroupSizeControl xe2 {.min_size = 16, .max_size = 32, .enabled = true, .compute = true};
	Expect(GraphicsSubgroupSize(xe2, true, 32, 32) == 32, "Xe2: wave32 requires 32");
	Expect(GraphicsSubgroupSize(xe2, true, 8, 32) == 0, "Xe2: size below the minimum: nothing");
}

constexpr VkImageUsageFlags Transfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkFormatFeatureFlags TransferFeatures =
    VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;

bool Same(const ImageCreateCandidate& candidate, VkImageUsageFlags usage, VkImageCreateFlags flags) {
	return candidate.usage == usage && candidate.flags == flags;
}

void TestImageCreateFallbacks() {
	const VkImageCreateFlags mutable_extended =
	    VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
	// An sRGB colour image (no storage format feature): storage through UNORM views is dropped
	// first, then EXTENDED_USAGE, which no remaining usage needs. MUTABLE_FORMAT stays.
	{
		const VkImageUsageFlags usage = Transfer | VK_IMAGE_USAGE_SAMPLED_BIT |
		                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		                                VK_IMAGE_USAGE_STORAGE_BIT;
		const VkFormatFeatureFlags features = TransferFeatures | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
		                                      VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
		const auto c = OptionalImageCreateFallbacks(usage, mutable_extended, features);
		Expect(c.count == 3, "sRGB: three candidates");
		Expect(Same(c.list[0], usage, mutable_extended), "sRGB: the request comes first");
		Expect(Same(c.list[1], usage & ~VK_IMAGE_USAGE_STORAGE_BIT, mutable_extended),
		       "sRGB: then without storage");
		Expect(Same(c.list[2], usage & ~VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT),
		       "sRGB: then without EXTENDED_USAGE, keeping MUTABLE_FORMAT");
	}
	// A format with storage support keeps storage; EXTENDED_USAGE is all that can go.
	{
		const VkImageUsageFlags usage =
		    Transfer | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		const VkFormatFeatureFlags features = TransferFeatures | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
		                                      VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
		const auto c = OptionalImageCreateFallbacks(usage, mutable_extended, features);
		Expect(c.count == 2 && Same(c.list[1], usage, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT),
		       "storage format: only EXTENDED_USAGE is dropped");
	}
	// A usage the format lacks that is not optional (sampling a format without the sampled feature)
	// keeps EXTENDED_USAGE.
	{
		const VkImageUsageFlags    usage    = Transfer | VK_IMAGE_USAGE_SAMPLED_BIT;
		const VkFormatFeatureFlags features = TransferFeatures;
		const auto c = OptionalImageCreateFallbacks(usage, mutable_extended, features);
		Expect(c.count == 1, "a needed usage the format lacks: nothing to drop");
	}
	// A block-compressed 2D texture: block-texel (uncompressed) views go, then EXTENDED_USAGE.
	{
		const VkImageUsageFlags  usage = Transfer | VK_IMAGE_USAGE_SAMPLED_BIT;
		const VkImageCreateFlags flags = mutable_extended | VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
		const VkFormatFeatureFlags features = TransferFeatures | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
		const auto c = OptionalImageCreateFallbacks(usage, flags, features);
		Expect(c.count == 3, "BC: three candidates");
		Expect(Same(c.list[1], usage, mutable_extended), "BC: without block-texel views");
		Expect(Same(c.list[2], usage, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT),
		       "BC: then without EXTENDED_USAGE");
	}
	// Block-texel views stay while the image keeps storage usage (its uploads write through them).
	{
		const VkImageUsageFlags  usage = Transfer | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		const VkImageCreateFlags flags = mutable_extended | VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
		const VkFormatFeatureFlags features = TransferFeatures | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
		                                      VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
		const auto c = OptionalImageCreateFallbacks(usage, flags, features);
		Expect(c.count == 2 && Same(c.list[1], usage, flags & ~VK_IMAGE_CREATE_EXTENDED_USAGE_BIT),
		       "BC with storage support: block-texel views stay");
	}
	// A depth image has no optional flags: its request is the only candidate.
	{
		const VkImageUsageFlags usage =
		    Transfer | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		const auto c = OptionalImageCreateFallbacks(usage, 0, 0);
		Expect(c.count == 1 && Same(c.list[0], usage, 0), "depth: nothing to drop");
	}
	Expect(FormatSupportsUsage(TransferFeatures | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,
	                           Transfer | VK_IMAGE_USAGE_SAMPLED_BIT),
	       "format usage: sampled and transfer supported");
	Expect(!FormatSupportsUsage(TransferFeatures | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,
	                            Transfer | VK_IMAGE_USAGE_SAMPLED_BIT |
	                                VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT),
	       "format usage: a usage without a format feature is not supported");
	Expect(IsBlockCompressedFormat(VK_FORMAT_BC1_RGB_UNORM_BLOCK) &&
	           IsBlockCompressedFormat(VK_FORMAT_BC7_SRGB_BLOCK) &&
	           !IsBlockCompressedFormat(VK_FORMAT_R32G32_UINT) &&
	           !IsBlockCompressedFormat(VK_FORMAT_ASTC_4x4_UNORM_BLOCK),
	       "block-compressed formats are BC1 to BC7");
}

void TestComputeWave64Split() {
	// KYTY_COMPUTE_WAVE64=0 on AMD: compute_wave64 off and host subgroup 32, so a wave64 program
	// requires 32 lanes (two guest lanes per invocation, as on NVIDIA) and a wave32 one still 32.
	const SubgroupSizeControl split {.min_size = 32, .max_size = 64, .enabled = true, .compute = true,
	                                 .compute_wave64 = false};
	Expect(ComputeSubgroupSize(split, 64, 32) == 32, "AMD split: wave64 requires 32");
	Expect(ComputeSubgroupSize(split, 32, 32) == 32, "AMD split: wave32 requires 32");
	// Pixel and mesh stages keep one host subgroup per guest wave (the default 64 for wave64).
	Expect(GraphicsSubgroupSize(split, true, 32, 64) == 32, "AMD split: wave32 pixel requires 32");
	Expect(GraphicsSubgroupSize(split, true, 64, 64) == 64, "AMD split: wave64 pixel requires 64");
}

void TestBreadcrumbs() {
	using namespace Libs::Graphics::Breadcrumbs;
	Expect(SequenceFromMarker(0, 100) == 0, "breadcrumbs: no marker");
	Expect(SequenceFromMarker(42, 100) == 42, "breadcrumbs: marker below the newest");
	Expect(SequenceFromMarker(100, 100) == 100, "breadcrumbs: the newest marker");
	Expect(SequenceFromMarker(101, 100) == 0, "breadcrumbs: a marker above the newest in the first window");
	const uint64_t window = uint64_t {1} << 32u;
	Expect(SequenceFromMarker(5, window + 10) == window + 5, "breadcrumbs: second window");
	Expect(SequenceFromMarker(0xfffffff0u, window + 10) == 0xfffffff0u,
	       "breadcrumbs: a marker of the previous window");
	// Begun 7, everything before 5 completed: 5..7 may still run, 5 is the oldest unfinished.
	const auto a = QueueInFlight(7, 5);
	Expect(a.first == 5 && a.last == 7 && a.Count() == 3, "breadcrumbs: in flight 5..7");
	// Begun and done-before equal: that one operation was running.
	const auto b = QueueInFlight(9, 9);
	Expect(b.first == 9 && b.last == 9 && b.Count() == 1, "breadcrumbs: one running operation");
	Expect(QueueInFlight(0, 0).Empty(), "breadcrumbs: a queue that never began anything");
	const auto c = QueueInFlight(3, 0);
	Expect(c.first == 1 && c.last == 3, "breadcrumbs: nothing completed yet");
}

void TestDepthClamp() {
	// With VK_EXT_depth_clip_enable (NVIDIA, as before): always clamped, clipping set apart.
	Expect(DepthClampEnable(true, true) && DepthClampEnable(true, false),
	       "depth clip extension: the clamp is always on");
	// Without it: the clamp only where the guest turns Z clipping off (the clamp disables clipping).
	Expect(!DepthClampEnable(false, true), "no extension, Z clipping on: clip without the clamp");
	Expect(DepthClampEnable(false, false), "no extension, Z clipping off: clamp");
}

} // namespace

int main() {
	TestComputeSubgroupSize();
	TestGraphicsSubgroupSize();
	TestImageCreateFallbacks();
	TestDepthClamp();
	TestComputeWave64Split();
	TestBreadcrumbs();
	if (g_failures != 0) {
		std::printf("DeviceCompatTests: failed: %d check(s)\n", g_failures);
		return 1;
	}
	std::printf("DeviceCompatTests: ok\n");
	return 0;
}
