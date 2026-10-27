#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

class SamplerCache {
public:
	explicit SamplerCache(GraphicContext& graphics)
	    : m_graphics(graphics), m_instance(NextInstance()) {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	}
	~SamplerCache();
	KYTY_CLASS_NO_COPY(SamplerCache);

	// integer_border: the sampler reads integer images (IR::SamplerResource::integer_border), so a
	// border colour must be the integer variant.
	vk::Sampler GetSampler(const ShaderSamplerResource& r, bool integer_border);
	// The sampler GetSampler returns for these dwords and border class when it was created
	// already, null otherwise (lookup only: nothing is created). Samplers are never evicted, so a
	// handle found stays the answer for the cache's lifetime. Any thread (DrawPrep binding plans).
	[[nodiscard]] vk::Sampler FindSampler(const ShaderSamplerResource& r, bool integer_border);
	// Unique per cache object for the process lifetime (per-thread lookup memos key on it).
	[[nodiscard]] uint64_t InstanceId() const noexcept { return m_instance; }

private:
	using SamplerKey = std::array<uint32_t, 5>;
	// A Vulkan sampler's create info, packed: the discrete state (filters, address modes,
	// anisotropy, compare, reduction, border, unnormalized coordinates) in the first three words,
	// then the LOD bias, the minimum and the maximum LOD as float bits.
	using VulkanKey   = std::array<uint32_t, 6>;
	using DiscreteKey = std::array<uint32_t, 3>;

	struct SamplerKeyHash {
		template <size_t N>
		std::size_t operator()(const std::array<uint32_t, N>& key) const {
			std::size_t hash = 0;
			for (auto value: key) {
				hash ^= static_cast<std::size_t>(value) +
				        static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
				        (hash >> 2u);
			}
			return hash;
		}
	};

	struct LodVariant {
		float       bias    = 0.0f;
		float       min_lod = 0.0f;
		float       max_lod = 0.0f;
		vk::Sampler sampler = nullptr;
	};

	static uint64_t NextInstance() noexcept;

	// The sampler for a create info (m_mutex held): one already made for the same state, a new one
	// while the budget allows, or else (KYTY_SKIP_INVALID) the nearest in LOD of the same kind.
	vk::Sampler FindOrCreate(const vk::SamplerCreateInfo& info, uint32_t reduction);

	GraphicContext&                                             m_graphics;
	const uint64_t                                              m_instance;
	Common::Mutex                                               m_mutex;
	std::unordered_map<SamplerKey, vk::Sampler, SamplerKeyHash> m_samplers;
	// The Vulkan samplers made, by create info (each made once, destroyed with the cache), and
	// those of each discrete state.
	std::unordered_map<VulkanKey, vk::Sampler, SamplerKeyHash>               m_vulkan_samplers;
	std::unordered_map<DiscreteKey, std::vector<LodVariant>, SamplerKeyHash> m_variants;
	uint32_t                                                                 m_budget         = 0;
	uint64_t                                                                 m_nearest_reuses = 0;
	size_t                                                                   m_next_count_log = 256;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
