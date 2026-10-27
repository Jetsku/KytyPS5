#include "graphics/host_gpu/renderer/cache/samplerCache.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/skipInvalid.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Libs::Graphics {

// KYTY_SAMPLER_REDUCTION=0 ignores the S# FILTER_MODE (min/max filtering), as before U50.
static bool SamplerReductionEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_SAMPLER_REDUCTION");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// S# fields that the host sampler does not model (RDNA2 ISA 8.2.7), logged for the first
// distinct samplers that use them so a run shows whether a title depends on them.
static void LogUnmodelledSamplerFields(const ShaderSamplerResource& r, bool reduction_dropped) {
	static std::atomic<uint32_t> logged {0};
	constexpr uint32_t           MaxLogged = 32;
	if (logged.load(std::memory_order_relaxed) >= MaxLogged) {
		return;
	}
	const bool aniso = r.MaxAnisoRatio() != 0u;
	std::string fields;
	const auto add = [&](const char* name, uint32_t value, bool present) {
		if (!present) {
			return;
		}
		fields += fields.empty() ? "" : " ";
		fields += name;
		fields += "=" + std::to_string(value);
	};
	add("filter_mode", r.FilterMode(), reduction_dropped);
	add("force_degamma", r.ForceSrgb() ? 1u : 0u, r.ForceSrgb());
	add("skip_degamma", r.DisableDegamma() ? 1u : 0u, r.DisableDegamma());
	add("trunc_coord", r.TruncCoord() ? 1u : 0u, r.TruncCoord());
	add("mc_coord_trunc", 1u, ((r.fields[0] >> 19u) & 1u) != 0u);
	add("aniso_threshold", r.AnisoThreshold(), aniso && r.AnisoThreshold() != 0u);
	add("aniso_bias", r.AnisoBias(), aniso && r.AnisoBias() != 0u);
	add("perf_mip", r.PerfMip(), r.PerfMip() != 0u);
	add("perf_z", r.PerfZ(), r.PerfZ() != 0u);
	add("lod_bias_sec", r.LodBiasSec(), r.LodBiasSec() != 0u);
	add("mip_point_preclamp", 1u, r.PointPreclamp());
	add("blend_zero_prt", 1u, r.BlendZeroPrt());
	// Modelled for implicit LOD only: Vulkan also adds mipLodBias to explicit-LOD samples
	// (IMAGE_SAMPLE_L/_LZ), which the texture unit most likely does not.
	add("lod_bias_s6.8", r.LodBias(), r.LodBias() != 0u);
	if (fields.empty() || logged.fetch_add(1, std::memory_order_relaxed) >= MaxLogged) {
		return;
	}
	LOGF("Sampler: S# fields not modelled by the host sampler: %s dwords=%08x,%08x,%08x,%08x\n",
	     fields.c_str(), r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
}

SamplerCache::~SamplerCache() {
	// The S# map's samplers are these (or stand-ins among them): each is destroyed once.
	for (const auto& [key, sampler]: m_vulkan_samplers) {
		(void)key;
		m_graphics.device.destroySampler(sampler, nullptr);
	}
}

uint64_t SamplerCache::NextInstance() noexcept {
	static std::atomic<uint64_t> instances {0};
	return instances.fetch_add(1, std::memory_order_relaxed) + 1;
}

vk::Sampler SamplerCache::FindSampler(const ShaderSamplerResource& r, bool integer_border) {
	Common::LockGuard lock(m_mutex);
	const SamplerKey key {r.fields[0], r.fields[1], r.fields[2], r.fields[3], integer_border};
	const auto       iter = m_samplers.find(key);
	return iter != m_samplers.end() ? iter->second : vk::Sampler {};
}

vk::Sampler SamplerCache::GetSampler(const ShaderSamplerResource& r, bool integer_border) {
	Common::LockGuard lock(m_mutex);

	const SamplerKey key {r.fields[0], r.fields[1], r.fields[2], r.fields[3], integer_border};
	if (auto iter = m_samplers.find(key); iter != m_samplers.end()) {
		return iter->second;
	}

	float      aniso_ratio = 1.0f;
	const auto mag_filter  = r.XyMagFilter();
	const auto min_filter  = r.XyMinFilter();

	auto is_aniso_filter = [](uint8_t filter) {
		switch (static_cast<Prospero::SamplerFilter>(filter)) {
			case Prospero::SamplerFilter::kAnisoPoint:
			case Prospero::SamplerFilter::kAnisoLinear: return true;
			case Prospero::SamplerFilter::kPoint:
			case Prospero::SamplerFilter::kBilinear: return false;
			default: EXIT("unknown sampler filter: %u\n", filter);
		}
		return false;
	};

	auto to_vk_filter = [](uint8_t filter) {
		switch (static_cast<Prospero::SamplerFilter>(filter)) {
			case Prospero::SamplerFilter::kPoint:
			case Prospero::SamplerFilter::kAnisoPoint: return vk::Filter::eNearest;
			case Prospero::SamplerFilter::kBilinear:
			case Prospero::SamplerFilter::kAnisoLinear: return vk::Filter::eLinear;
			default: EXIT("unknown sampler filter: %u\n", filter);
		}
		return vk::Filter::eNearest;
	};

	const bool aniso = is_aniso_filter(mag_filter) || is_aniso_filter(min_filter);
	if (aniso) {
		switch (static_cast<Prospero::SamplerAnisoRatio>(r.MaxAnisoRatio())) {
			case Prospero::SamplerAnisoRatio::kOne: aniso_ratio = 1.0f; break;
			case Prospero::SamplerAnisoRatio::kTwo: aniso_ratio = 2.0f; break;
			case Prospero::SamplerAnisoRatio::kFour: aniso_ratio = 4.0f; break;
			case Prospero::SamplerAnisoRatio::kEight: aniso_ratio = 8.0f; break;
			case Prospero::SamplerAnisoRatio::kSixteen:
			// Reserved encodings 5-7 fall back to the maximum ratio.
			default: aniso_ratio = 16.0f; break;
		}
	}

	const auto mip_filter = r.MipFilter();
	float      min_lod    = 0.0f;
	float      max_lod    = 0.0f;
	if (static_cast<Prospero::SamplerMipFilter>(mip_filter) != Prospero::SamplerMipFilter::kNone) {
		min_lod = static_cast<float>(r.MinLod()) / 256.0f;
		max_lod = static_cast<float>(r.MaxLod()) / 256.0f;
	}

	vk::SamplerCreateInfo sampler_info {};

	auto to_vk_address_mode = [](uint8_t clamp) {
		switch (static_cast<Prospero::SamplerClampMode>(clamp)) {
			case Prospero::SamplerClampMode::kWrap: return vk::SamplerAddressMode::eRepeat;
			case Prospero::SamplerClampMode::kMirror:
				return vk::SamplerAddressMode::eMirroredRepeat;
			case Prospero::SamplerClampMode::kClampLastTexel:
				return vk::SamplerAddressMode::eClampToEdge;
			case Prospero::SamplerClampMode::kMirrorOnceLastTexel:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			case Prospero::SamplerClampMode::kClampHalfBorder:
				return vk::SamplerAddressMode::eClampToBorder;
			case Prospero::SamplerClampMode::kMirrorOnceHalfBorder:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			case Prospero::SamplerClampMode::kClampBorder:
				return vk::SamplerAddressMode::eClampToBorder;
			case Prospero::SamplerClampMode::kMirrorOnceBorder:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			default: EXIT("unknown clamp: %u\n", clamp);
		}
		return vk::SamplerAddressMode::eClampToBorder;
	};

	vk::BorderColor border = integer_border ? vk::BorderColor::eIntTransparentBlack
	                                       : vk::BorderColor::eFloatTransparentBlack;
	switch (static_cast<Prospero::SamplerBorderColor>(r.BorderColorType())) {
		case Prospero::SamplerBorderColor::kTransBlack: break;
		case Prospero::SamplerBorderColor::kOpaqueBlack:
			border = integer_border ? vk::BorderColor::eIntOpaqueBlack
			                        : vk::BorderColor::eFloatOpaqueBlack;
			break;
		case Prospero::SamplerBorderColor::kOpaqueWhite:
			border = integer_border ? vk::BorderColor::eIntOpaqueWhite
			                        : vk::BorderColor::eFloatOpaqueWhite;
			break;
		case Prospero::SamplerBorderColor::kFromTable:
			LOGF(
			    "temporary: approximating table border color as transparent black, index = %" PRIu16
			    "\n",
			    r.BorderColorPtr());
			break;
		default: EXIT("unknown border color: %d", static_cast<int>(r.BorderColorType()));
	}

	sampler_info.magFilter = to_vk_filter(mag_filter);
	sampler_info.minFilter = to_vk_filter(min_filter);
	sampler_info.mipmapMode =
	    (static_cast<Prospero::SamplerMipFilter>(mip_filter) == Prospero::SamplerMipFilter::kLinear
	         ? vk::SamplerMipmapMode::eLinear
	         : vk::SamplerMipmapMode::eNearest);
	sampler_info.addressModeU = to_vk_address_mode(r.ClampX());
	sampler_info.addressModeV = to_vk_address_mode(r.ClampY());
	sampler_info.addressModeW = to_vk_address_mode(r.ClampZ());
	// The S# bias spans +-32; Vulkan requires |mipLodBias| <= maxSamplerLodBias (upstream 59a17604).
	const auto max_lod_bias = m_graphics.GetPhysicalDeviceProperties().limits.maxSamplerLodBias;
	sampler_info.mipLodBias = std::clamp(
	    static_cast<float>(static_cast<int16_t>((r.LodBias() ^ 0x2000u) - 0x2000u)) / 256.0f,
	    -max_lod_bias, max_lod_bias);
	sampler_info.anisotropyEnable        = (aniso ? VK_TRUE : VK_FALSE);
	sampler_info.maxAnisotropy           = aniso_ratio;
	sampler_info.compareEnable           = (r.DepthCompareFunc() != 0 ? VK_TRUE : VK_FALSE);
	sampler_info.compareOp               = static_cast<vk::CompareOp>(r.DepthCompareFunc());
	sampler_info.minLod                  = min_lod;
	sampler_info.maxLod                  = max_lod;
	sampler_info.borderColor             = border;
	sampler_info.unnormalizedCoordinates = (r.ForceUnormCoords() ? VK_TRUE : VK_FALSE);

	if (r.ForceUnormCoords()) {
		sampler_info.addressModeU     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.addressModeV     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.addressModeW     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.mipmapMode       = vk::SamplerMipmapMode::eNearest;
		sampler_info.minLod           = 0.0f;
		sampler_info.maxLod           = 0.0f;
		sampler_info.anisotropyEnable = VK_FALSE;
		sampler_info.maxAnisotropy    = 1.0f;
		sampler_info.compareEnable    = VK_FALSE;
		sampler_info.mipLodBias       = 0.0f;
	}

	// FILTER_MODE (S# bits 29..30): 0 blends the filter footprint, 1 and 2 return its minimum or
	// maximum texel (RDNA2 ISA 8.2.7). Vulkan's reduction modes replace the weighted average the
	// same way; they exclude depth comparison.
	vk::SamplerReductionModeCreateInfo reduction {};
	uint32_t                           reduction_key     = 0;
	bool                               reduction_dropped = false;
	if (const auto filter_mode = r.FilterMode(); filter_mode == 1u || filter_mode == 2u) {
		if (SamplerReductionEnabled() && m_graphics.sampler_filter_minmax_enabled &&
		    sampler_info.compareEnable == VK_FALSE) {
			reduction.reductionMode = filter_mode == 1u ? vk::SamplerReductionMode::eMin
			                                            : vk::SamplerReductionMode::eMax;
			sampler_info.pNext = &reduction;
			reduction_key      = filter_mode;
		} else {
			reduction_dropped = true;
		}
	}
	LogUnmodelledSamplerFields(r, reduction_dropped);

	const auto vk_sampler = FindOrCreate(sampler_info, reduction_key);
	m_samplers.emplace(key, vk_sampler);
	return vk_sampler;
}

namespace {

uint32_t FloatBits(float value) {
	uint32_t bits = 0;
	std::memcpy(&bits, &value, sizeof(bits));
	return bits;
}

// Bits 25 to 31 of a discrete key's first word: compare enable, unnormalized coordinates, the
// compare op and the reduction mode, which a sampler standing in for another must share.
constexpr uint32_t SamplerKindBits = 0xfe000000u;

} // namespace

// usa-ratt/KytyPS5-GT7 81ec2a939: a VkSampler for each distinct create info, not for each distinct
// S#. S#s that differ only where the Vulkan sampler does not (or by a LOD bias the game moves) piled
// up past the driver's limit (maxSamplerAllocationCount, 4000 on NVIDIA), and creating the next one
// ended the emulator after ten minutes of GT7 Circuit Experience retries. Samplers of equal create
// info are interchangeable, so sharing one changes nothing. With KYTY_SKIP_INVALID, past a budget
// under the limit (or when vkCreateSampler fails) the nearest in LOD of the same kind stands in.
vk::Sampler SamplerCache::FindOrCreate(const vk::SamplerCreateInfo& info, uint32_t reduction) {
	const DiscreteKey discrete {
	    static_cast<uint32_t>(info.magFilter) | (static_cast<uint32_t>(info.minFilter) << 4u) |
	        (static_cast<uint32_t>(info.mipmapMode) << 8u) |
	        (static_cast<uint32_t>(info.addressModeU) << 12u) |
	        (static_cast<uint32_t>(info.addressModeV) << 16u) |
	        (static_cast<uint32_t>(info.addressModeW) << 20u) |
	        (static_cast<uint32_t>(info.anisotropyEnable) << 24u) |
	        (static_cast<uint32_t>(info.compareEnable) << 25u) |
	        (static_cast<uint32_t>(info.unnormalizedCoordinates) << 26u) |
	        ((static_cast<uint32_t>(info.compareOp) & 7u) << 27u) | ((reduction & 3u) << 30u),
	    static_cast<uint32_t>(info.borderColor), FloatBits(info.maxAnisotropy)};
	const VulkanKey key {discrete[0],
	                     discrete[1],
	                     discrete[2],
	                     FloatBits(info.mipLodBias),
	                     FloatBits(info.minLod),
	                     FloatBits(info.maxLod)};
	if (const auto found = m_vulkan_samplers.find(key); found != m_vulkan_samplers.end()) {
		return found->second;
	}
	const bool skip = Common::SkipInvalidEnabled();
	if (m_budget == 0) {
		// Room for the samplers made elsewhere (the blit helper's, the presenter's, the overlay's).
		const auto limit =
		    m_graphics.GetPhysicalDeviceProperties().limits.maxSamplerAllocationCount;
		m_budget = limit > 1024u ? limit - 256u : std::max(limit / 2u, 1u);
	}
	auto&      variants = m_variants[discrete];
	const auto nearest  = [&info](const std::vector<LodVariant>& list) -> vk::Sampler {
		const LodVariant* best          = nullptr;
		float             best_distance = 0.0f;
		for (const auto& variant: list) {
			const float distance = std::abs(variant.bias - info.mipLodBias) +
			                       std::abs(variant.min_lod - info.minLod) +
			                       std::abs(variant.max_lod - info.maxLod);
			if (best == nullptr || distance < best_distance) {
				best          = &variant;
				best_distance = distance;
			}
		}
		return best != nullptr ? best->sampler : nullptr;
	};
	const auto note_reuse = [this](const char* why) {
		m_nearest_reuses++;
		if (m_nearest_reuses <= 8 || (m_nearest_reuses & (m_nearest_reuses - 1u)) == 0) {
			LOGF("Samplers: %s; the nearest of the same kind stands in (%" PRIu64
			     " so far, %zu Vulkan samplers, %zu kinds)\n",
			     why, m_nearest_reuses, m_vulkan_samplers.size(), m_variants.size());
		}
	};
	if (skip && m_vulkan_samplers.size() >= m_budget) {
		if (const auto stand_in = nearest(variants); stand_in != nullptr) {
			note_reuse("the sampler budget is used up");
			return stand_in;
		}
	}
	vk::Sampler vk_sampler = nullptr;
	const auto  result     = m_graphics.device.createSampler(&info, nullptr, &vk_sampler);
	if (result != vk::Result::eSuccess || vk_sampler == nullptr) {
		vk::Sampler stand_in = nullptr;
		if (skip) {
			// The nearest of the same discrete state, or else any of the same kind.
			stand_in = nearest(variants);
			for (auto it = m_variants.begin(); stand_in == nullptr && it != m_variants.end(); ++it) {
				if ((it->first[0] & SamplerKindBits) == (discrete[0] & SamplerKindBits)) {
					stand_in = nearest(it->second);
				}
			}
		}
		if (stand_in == nullptr) {
			EXIT("vkCreateSampler failed: %s (%zu samplers made)\n", vk::to_string(result).c_str(),
			     m_vulkan_samplers.size());
		}
		note_reuse("vkCreateSampler failed");
		return stand_in;
	}
	m_vulkan_samplers.emplace(key, vk_sampler);
	variants.push_back({info.mipLodBias, info.minLod, info.maxLod, vk_sampler});
	if (m_vulkan_samplers.size() >= m_next_count_log) {
		LOGF("Samplers: %zu Vulkan samplers made (%zu kinds), for %zu S# variants (budget %u)\n",
		     m_vulkan_samplers.size(), m_variants.size(), m_samplers.size() + 1u, m_budget);
		m_next_count_log *= 2u;
	}
	return vk_sampler;
}

} // namespace Libs::Graphics
