#include "graphics/host_gpu/renderer/cache/samplerCache.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <atomic>
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
	for (const auto& [key, sampler]: m_samplers) {
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
	bool                               reduction_dropped = false;
	if (const auto filter_mode = r.FilterMode(); filter_mode == 1u || filter_mode == 2u) {
		if (SamplerReductionEnabled() && m_graphics.sampler_filter_minmax_enabled &&
		    sampler_info.compareEnable == VK_FALSE) {
			reduction.reductionMode = filter_mode == 1u ? vk::SamplerReductionMode::eMin
			                                            : vk::SamplerReductionMode::eMax;
			sampler_info.pNext = &reduction;
		} else {
			reduction_dropped = true;
		}
	}
	LogUnmodelledSamplerFields(r, reduction_dropped);

	vk::Sampler vk_sampler = nullptr;
	const auto  result     = m_graphics.device.createSampler(&sampler_info, nullptr, &vk_sampler);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || vk_sampler == nullptr);

	m_samplers.emplace(key, vk_sampler);
	return vk_sampler;
}

} // namespace Libs::Graphics
