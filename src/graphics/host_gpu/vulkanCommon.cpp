#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/spirvCacheSalt.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/aftermath.h"
#include "graphics/host_gpu/gpuBreadcrumbs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/spirvLocalArrays.h"
#include "graphics/host_gpu/vramStats.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include <spirv-tools/libspirv.hpp>

#include <xxhash.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace Libs::Graphics {
namespace {

struct FormatMapping {
	Prospero::BufferFormat guest;
	vk::Format             host;
};

constexpr FormatMapping kFormatMappings[] = {
    {Prospero::BufferFormat::k8UNorm, vk::Format::eR8Unorm},
    {Prospero::BufferFormat::k8SNorm, vk::Format::eR8Snorm},
    {Prospero::BufferFormat::k8UScaled, vk::Format::eR8Uscaled},
    {Prospero::BufferFormat::k8SScaled, vk::Format::eR8Sscaled},
    {Prospero::BufferFormat::k8UInt, vk::Format::eR8Uint},
    {Prospero::BufferFormat::k8SInt, vk::Format::eR8Sint},
    {Prospero::BufferFormat::k16UNorm, vk::Format::eR16Unorm},
    {Prospero::BufferFormat::k16SNorm, vk::Format::eR16Snorm},
    {Prospero::BufferFormat::k16UScaled, vk::Format::eR16Uscaled},
    {Prospero::BufferFormat::k16SScaled, vk::Format::eR16Sscaled},
    {Prospero::BufferFormat::k16UInt, vk::Format::eR16Uint},
    {Prospero::BufferFormat::k16SInt, vk::Format::eR16Sint},
    {Prospero::BufferFormat::k16Float, vk::Format::eR16Sfloat},
    {Prospero::BufferFormat::k8_8UNorm, vk::Format::eR8G8Unorm},
    {Prospero::BufferFormat::k8_8SNorm, vk::Format::eR8G8Snorm},
    {Prospero::BufferFormat::k8_8UScaled, vk::Format::eR8G8Uscaled},
    {Prospero::BufferFormat::k8_8SScaled, vk::Format::eR8G8Sscaled},
    {Prospero::BufferFormat::k8_8UInt, vk::Format::eR8G8Uint},
    {Prospero::BufferFormat::k8_8SInt, vk::Format::eR8G8Sint},
    {Prospero::BufferFormat::k32UInt, vk::Format::eR32Uint},
    {Prospero::BufferFormat::k32SInt, vk::Format::eR32Sint},
    {Prospero::BufferFormat::k32Float, vk::Format::eR32Sfloat},
    {Prospero::BufferFormat::k16_16UNorm, vk::Format::eR16G16Unorm},
    {Prospero::BufferFormat::k16_16SNorm, vk::Format::eR16G16Snorm},
    {Prospero::BufferFormat::k16_16UScaled, vk::Format::eR16G16Uscaled},
    {Prospero::BufferFormat::k16_16SScaled, vk::Format::eR16G16Sscaled},
    {Prospero::BufferFormat::k16_16UInt, vk::Format::eR16G16Uint},
    {Prospero::BufferFormat::k16_16SInt, vk::Format::eR16G16Sint},
    {Prospero::BufferFormat::k16_16Float, vk::Format::eR16G16Sfloat},
    {Prospero::BufferFormat::k11_11_10Float, vk::Format::eB10G11R11UfloatPack32},
    // 2_10_10_10 keeps its 2-bit component at the bottom, which no Vulkan format does: the host
    // image holds it in alpha (ResolveHostFormat in textureCommon.cpp maps the components;
    // usa-ratt/KytyPS5-GT7 a33e0a273).
    {Prospero::BufferFormat::k2_10_10_10UNorm, vk::Format::eA2B10G10R10UnormPack32},
    {Prospero::BufferFormat::k2_10_10_10UInt, vk::Format::eA2B10G10R10UintPack32},
    {Prospero::BufferFormat::k10_10_10_2UNorm, vk::Format::eA2B10G10R10UnormPack32},
    {Prospero::BufferFormat::k10_10_10_2SNorm, vk::Format::eA2B10G10R10SnormPack32},
    {Prospero::BufferFormat::k10_10_10_2UScaled, vk::Format::eA2B10G10R10UscaledPack32},
    {Prospero::BufferFormat::k10_10_10_2UInt, vk::Format::eA2B10G10R10UintPack32},
    {Prospero::BufferFormat::k8_8_8_8UNorm, vk::Format::eR8G8B8A8Unorm},
    {Prospero::BufferFormat::k8_8_8_8SNorm, vk::Format::eR8G8B8A8Snorm},
    {Prospero::BufferFormat::k8_8_8_8UScaled, vk::Format::eR8G8B8A8Uscaled},
    {Prospero::BufferFormat::k8_8_8_8SScaled, vk::Format::eR8G8B8A8Sscaled},
    {Prospero::BufferFormat::k8_8_8_8UInt, vk::Format::eR8G8B8A8Uint},
    {Prospero::BufferFormat::k8_8_8_8SInt, vk::Format::eR8G8B8A8Sint},
    {Prospero::BufferFormat::k32_32UInt, vk::Format::eR32G32Uint},
    {Prospero::BufferFormat::k32_32SInt, vk::Format::eR32G32Sint},
    {Prospero::BufferFormat::k32_32Float, vk::Format::eR32G32Sfloat},
    {Prospero::BufferFormat::k16_16_16_16UNorm, vk::Format::eR16G16B16A16Unorm},
    {Prospero::BufferFormat::k16_16_16_16SNorm, vk::Format::eR16G16B16A16Snorm},
    {Prospero::BufferFormat::k16_16_16_16UScaled, vk::Format::eR16G16B16A16Uscaled},
    {Prospero::BufferFormat::k16_16_16_16SScaled, vk::Format::eR16G16B16A16Sscaled},
    {Prospero::BufferFormat::k16_16_16_16UInt, vk::Format::eR16G16B16A16Uint},
    {Prospero::BufferFormat::k16_16_16_16SInt, vk::Format::eR16G16B16A16Sint},
    {Prospero::BufferFormat::k16_16_16_16Float, vk::Format::eR16G16B16A16Sfloat},
    {Prospero::BufferFormat::k32_32_32UInt, vk::Format::eR32G32B32Uint},
    {Prospero::BufferFormat::k32_32_32SInt, vk::Format::eR32G32B32Sint},
    {Prospero::BufferFormat::k32_32_32Float, vk::Format::eR32G32B32Sfloat},
    {Prospero::BufferFormat::k32_32_32_32UInt, vk::Format::eR32G32B32A32Uint},
    {Prospero::BufferFormat::k32_32_32_32SInt, vk::Format::eR32G32B32A32Sint},
    {Prospero::BufferFormat::k32_32_32_32Float, vk::Format::eR32G32B32A32Sfloat},
    // 8 and 8_8 sRGB views are optional in Vulkan, so these images store UNorm.
    {Prospero::BufferFormat::k8Srgb, vk::Format::eR8Unorm},
    {Prospero::BufferFormat::k8_8Srgb, vk::Format::eR8G8Unorm},
    {Prospero::BufferFormat::k8_8_8_8Srgb, vk::Format::eR8G8B8A8Srgb},
    {Prospero::BufferFormat::k9_9_9_5Float, vk::Format::eE5B9G9R9UfloatPack32},
    {Prospero::BufferFormat::k5_6_5UNorm, vk::Format::eB5G6R5UnormPack16},
    {Prospero::BufferFormat::k5_5_5_1UNorm, vk::Format::eA1R5G5B5UnormPack16},
    {Prospero::BufferFormat::k1_5_5_5UNorm, vk::Format::eR5G5B5A1UnormPack16},
    {Prospero::BufferFormat::k4_4_4_4UNorm, vk::Format::eR4G4B4A4UnormPack16},
    {Prospero::BufferFormat::kBc1UNorm, vk::Format::eBc1RgbaUnormBlock},
    {Prospero::BufferFormat::kBc1Srgb, vk::Format::eBc1RgbaSrgbBlock},
    {Prospero::BufferFormat::kBc2UNorm, vk::Format::eBc2UnormBlock},
    {Prospero::BufferFormat::kBc2Srgb, vk::Format::eBc2SrgbBlock},
    {Prospero::BufferFormat::kBc3UNorm, vk::Format::eBc3UnormBlock},
    {Prospero::BufferFormat::kBc3Srgb, vk::Format::eBc3SrgbBlock},
    {Prospero::BufferFormat::kBc4UNorm, vk::Format::eBc4UnormBlock},
    {Prospero::BufferFormat::kBc4SNorm, vk::Format::eBc4SnormBlock},
    {Prospero::BufferFormat::kBc5UNorm, vk::Format::eBc5UnormBlock},
    {Prospero::BufferFormat::kBc5SNorm, vk::Format::eBc5SnormBlock},
    {Prospero::BufferFormat::kBc6UFloat, vk::Format::eBc6HUfloatBlock},
    {Prospero::BufferFormat::kBc6SFloat, vk::Format::eBc6HSfloatBlock},
    {Prospero::BufferFormat::kBc7UNorm, vk::Format::eBc7UnormBlock},
    {Prospero::BufferFormat::kBc7Srgb, vk::Format::eBc7SrgbBlock},
};

constexpr auto MakeFormatLookup() {
	constexpr auto kMaxFormat = static_cast<size_t>(Prospero::BufferFormat::kBc7Srgb);
	std::array<vk::Format, kMaxFormat + 1> lookup {};
	lookup.fill(vk::Format::eUndefined);
	for (const auto& mapping: kFormatMappings) {
		lookup[static_cast<size_t>(mapping.guest)] = mapping.host;
	}
	return lookup;
}

constexpr auto kFormatLookup = MakeFormatLookup();

} // namespace

vk::Format VulkanFormat(Prospero::BufferFormat guest_format) {
	const auto index = static_cast<size_t>(guest_format);
	return index < kFormatLookup.size() ? kFormatLookup[index] : vk::Format::eUndefined;
}

void RequireVulkanSuccess(vk::Result result, const char* operation) {
	if (result != vk::Result::eSuccess) {
		if (result == vk::Result::eErrorDeviceLost) {
			// Another thread may see the loss first (the upload DMA worker, a side readback).
			GpuBreadcrumbs::Dump();
			Aftermath::WaitForDump(); // KYTY_AFTERMATH: until the driver handed over the crash dump
			// The thread reporting the loss (DumpDeviceLossDiagnostics) finishes its report and
			// the device-fault dump file before this exit ends the process.
			WaitForDeviceLossDiagnostics();
		}
		EXIT("%s failed: %s (%d)\n", operation, vk::to_string(result).c_str(),
		     static_cast<int>(result));
	}
}

namespace {

// KYTY_FUNCTION_ARRAY_SHRINK (default on since AMD test 3; =0 turns it off): shader modules get
// their per-invocation
// (Function storage) arrays shrunk to the indices they can reach (spirvLocalArrays.h). The driver
// reserves local memory for the largest per-thread footprint of any pipeline times every thread the
// GPU keeps resident and never returns it; the recompiler's 8192-dword LDS emulation in vertex and
// pixel shaders made that 3.9 GiB on an RTX 3090 (Astro Bot's galaxy map), where the pixel shaders
// only reach 96 dwords. A rewritten module that fails spirv-val is not used (logged once).
// =zero also zero-fills the shrunk arrays at function entry (OpConstantNull initializer), =poison
// fills them with float NaNs (diagnostic: makes reads of elements no path wrote visible).
// Default on: two RX 6900 XT players whose launcher did not apply u59-preset.json (no shrink line
// in their logs) crashed inside the driver a minute into play, shortly after a pixel shader with
// the unshrunk 32 KiB per-invocation LDS array was compiled; AMD backs it with scratch for every
// resident lane (about 2 MiB per wave64 wave).
struct ShrinkMode {
	bool                   enabled = false;
	SpirvLocalArrays::Init init    = SpirvLocalArrays::Init::None;
};

const ShrinkMode& FunctionArrayShrinkMode() {
	static const ShrinkMode mode = [] {
		ShrinkMode  result;
		const auto* value = std::getenv("KYTY_FUNCTION_ARRAY_SHRINK");
		if (value == nullptr || *value == '\0') {
			value = "1";
		}
		result.enabled = std::strcmp(value, "0") != 0;
		if (!result.enabled) {
			Log::WriteToConsoleAndLog("Kyty Function-storage arrays: not shrunk (KYTY_FUNCTION_ARRAY_SHRINK=0)\n");
		}
		if (result.enabled) {
			if (std::strcmp(value, "zero") == 0) {
				result.init = SpirvLocalArrays::Init::Zero;
			} else if (std::strcmp(value, "poison") == 0) {
				result.init = SpirvLocalArrays::Init::Poison;
			}
			char text[192];
			std::snprintf(text, sizeof(text), "Kyty Function-storage arrays: shrunk to their proven index bound%s "
			            "(KYTY_FUNCTION_ARRAY_SHRINK=%s)\n",
			            result.init == SpirvLocalArrays::Init::Zero     ? ", zero-filled"
			            : result.init == SpirvLocalArrays::Init::Poison ? ", filled with NaNs (diagnostic)"
			                                                            : "",
			            value);
			Log::WriteToConsoleAndLog(text);
		}
		return result;
	}();
	return mode;
}

// CompileSpvTimes accounting (KYTY_SHADER_PRECOMPILE_STATS=1).
bool StatsOn() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_SHADER_PRECOMPILE_STATS");
		return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

uint64_t ClockNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

thread_local CompileSpvTimes t_spv_times;
struct {
	std::atomic<uint64_t> calls {0}, shrunk {0}, validated {0}, validate_skipped {0};
	std::atomic<uint64_t> shrink_ns {0}, validate_ns {0}, create_ns {0};
} g_spv_times;

void AddSpvTimes(const CompileSpvTimes& d) {
	auto& t = t_spv_times;
	t.calls += d.calls;
	t.shrunk += d.shrunk;
	t.validated += d.validated;
	t.validate_skipped += d.validate_skipped;
	t.shrink_ns += d.shrink_ns;
	t.validate_ns += d.validate_ns;
	t.create_ns += d.create_ns;
	g_spv_times.calls.fetch_add(d.calls, std::memory_order_relaxed);
	g_spv_times.shrunk.fetch_add(d.shrunk, std::memory_order_relaxed);
	g_spv_times.validated.fetch_add(d.validated, std::memory_order_relaxed);
	g_spv_times.validate_skipped.fetch_add(d.validate_skipped, std::memory_order_relaxed);
	g_spv_times.shrink_ns.fetch_add(d.shrink_ns, std::memory_order_relaxed);
	g_spv_times.validate_ns.fetch_add(d.validate_ns, std::memory_order_relaxed);
	g_spv_times.create_ns.fetch_add(d.create_ns, std::memory_order_relaxed);
}

// Shrunk modules known to pass spirv-val (UseShrinkVerdictFile). A verdict is a pure function of
// the module's words and the validator (its build and SPV_ENV_VULKAN_1_3 with default options), so
// equal words need no second run. Keys are XXH3-128 of the words.
class ShrinkVerdicts {
public:
	static ShrinkVerdicts& Instance() {
		static ShrinkVerdicts instance;
		return instance;
	}

	uint64_t Open(const std::string& path) {
		std::scoped_lock lock(m_mutex);
		if (m_file != nullptr || path.empty()) return m_known.size();
		const auto header = Header();
		std::vector<uint8_t> bytes;
		if (std::FILE* in = std::fopen(path.c_str(), "rb"); in != nullptr) {
			std::array<uint8_t, 65536> chunk {};
			for (size_t got; (got = std::fread(chunk.data(), 1, chunk.size(), in)) != 0;) {
				bytes.insert(bytes.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(got));
			}
			std::fclose(in);
		}
		if (bytes.size() >= header.size() && std::equal(header.begin(), header.end(), bytes.begin())) {
			for (size_t at = header.size(); at + sizeof(Key) <= bytes.size(); at += sizeof(Key)) {
				Key key {};
				std::memcpy(&key, bytes.data() + at, sizeof(Key));
				m_known.insert(key);
			}
		}
		// Rewritten whole: a torn record at the end (a killed run) or another validator's file is
		// dropped, and later appends stay aligned.
		m_file = std::fopen(path.c_str(), "wb");
		if (m_file == nullptr) return m_known.size();
		std::fwrite(header.data(), 1, header.size(), m_file);
		for (const auto& key: m_known) std::fwrite(&key, sizeof(Key), 1, m_file);
		std::fflush(m_file);
		m_enabled.store(true, std::memory_order_release);
		return m_known.size();
	}

	[[nodiscard]] bool Enabled() const { return m_enabled.load(std::memory_order_acquire); }

	[[nodiscard]] bool Known(std::span<const uint32_t> words) {
		const auto key = KeyOf(words);
		std::scoped_lock lock(m_mutex);
		return m_known.contains(key);
	}

	void Remember(std::span<const uint32_t> words) {
		const auto key = KeyOf(words);
		std::scoped_lock lock(m_mutex);
		if (!m_known.insert(key).second || m_file == nullptr) return;
		std::fwrite(&key, sizeof(Key), 1, m_file);
		std::fflush(m_file);
	}

private:
	struct Key {
		uint64_t low  = 0;
		uint64_t high = 0;
		bool     operator==(const Key&) const = default;
	};
	struct KeyHash {
		size_t operator()(const Key& key) const { return static_cast<size_t>(key.low ^ (key.high * 31u)); }
	};
	static Key KeyOf(std::span<const uint32_t> words) {
		const auto hash = XXH3_128bits(words.data(), words.size_bytes());
		return {.low = hash.low64, .high = hash.high64};
	}
	// Magic, then the XXH3-64 of the validator's identity (32 bytes in all).
	static std::array<uint8_t, 32> Header() {
		std::array<uint8_t, 32> header {};
		constexpr char          magic[16] = "KYTY SHRINKVAL1";
		std::memcpy(header.data(), magic, sizeof(magic));
		const std::string identity = std::string(spvSoftwareVersionDetailsString()) + "; vulkan1.3 default options";
		const uint64_t    hash     = XXH3_64bits(identity.data(), identity.size());
		std::memcpy(header.data() + 16, &hash, sizeof(hash));
		return header;
	}

	std::mutex                        m_mutex;
	std::unordered_set<Key, KeyHash>  m_known;
	std::FILE*                        m_file = nullptr;
	std::atomic<bool>                 m_enabled {false};
};

// Returns the rewritten module, or nothing (unchanged or invalid). `declared` / `created`: the
// module's Function-storage bytes per invocation before and as given to the driver.
std::vector<uint32_t> ShrinkFunctionArrays(std::span<const uint32_t> code, uint64_t& declared,
                                           uint64_t& created) {
	const bool            stats       = StatsOn();
	CompileSpvTimes       times;
	const auto            shrink_from = stats ? ClockNs() : 0;
	std::vector<uint32_t> shrunk;
	const auto            result = SpirvLocalArrays::Shrink(code, shrunk, FunctionArrayShrinkMode().init);
	declared                     = result.bytes_before;
	created                      = result.bytes_before;
	if (stats) times.shrink_ns = ClockNs() - shrink_from;
	if (!result.changed) {
		if (stats) AddSpvTimes(times);
		return {};
	}
	times.shrunk = 1;
	auto&       verdicts = ShrinkVerdicts::Instance();
	const bool  memo     = verdicts.Enabled();
	std::string messages;
	bool        valid = memo && verdicts.Known(shrunk);
	if (valid) {
		times.validate_skipped = 1;
	} else {
		const auto           validate_from = stats ? ClockNs() : 0;
		spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
		tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&,
		                                     const char* message) {
			if (messages.size() < 512) {
				messages += message;
				messages += "; ";
			}
		});
		// Uniform blocks of dword arrays (KYTY_UNIFORM_BUFFERS, KYTY_SRT_UNIFORM) need the standard
		// layout rules the device enables for them (as the pipeline cache validates).
		spvtools::ValidatorOptions options;
		options.SetUniformBufferStandardLayout(
		    Libs::Graphics::ShaderRecompiler::Spirv::GetHostBufferRobustness().uniform_dword_loads_return_zero);
		valid = tools.Validate(shrunk.data(), shrunk.size(), options);
		if (stats) {
			times.validated   = 1;
			times.validate_ns = ClockNs() - validate_from;
		}
		if (valid && memo) verdicts.Remember(shrunk);
	}
	if (stats) AddSpvTimes(times);
	static std::mutex                   log_mutex;
	static std::set<std::string>        logged;
	std::string                         arrays;
	for (const auto& array: result.arrays) {
		char part[160];
		std::snprintf(part, sizeof(part), "%s%s %u -> %u", arrays.empty() ? "" : ", ",
		              array.name.empty() ? "array" : array.name.c_str(), array.old_length, array.new_length);
		arrays += part;
	}
	{
		std::scoped_lock lock(log_mutex);
		if (logged.insert(arrays + (valid ? "" : " invalid")).second) {
			char text[1024];
			std::snprintf(text, sizeof(text), "Kyty Function-storage arrays (KYTY_FUNCTION_ARRAY_SHRINK): %s elements, %llu -> %llu "
			            "bytes per invocation%s%s\n",
			            arrays.c_str(), static_cast<unsigned long long>(result.bytes_before),
			            static_cast<unsigned long long>(result.bytes_after),
			            valid ? "" : "; the rewritten module failed validation, the original is used: ",
			            valid ? "" : messages.c_str());
			Log::WriteToConsoleAndLog(text);
		}
	}
	if (valid) {
		created = result.bytes_after;
	}
	return valid ? shrunk : std::vector<uint32_t> {};
}

} // namespace

vk::ShaderModule CompileSPV(std::span<const uint32_t> code, vk::Device device) {
	std::vector<uint32_t> shrunk;
	uint64_t              declared = 0;
	uint64_t              created  = 0;
	if (FunctionArrayShrinkMode().enabled) {
		shrunk = ShrinkFunctionArrays(code, declared, created);
		if (!shrunk.empty()) {
			code = shrunk;
		}
	} else if (VramStats::Enabled()) {
		// The analysis only measures here (the module is not changed).
		std::vector<uint32_t> unused;
		declared = SpirvLocalArrays::Shrink(code, unused).bytes_before;
		created  = declared;
	}
	if (VramStats::Enabled()) {
		VramStats::NoteFunctionStorage(declared, created);
	}
	// Startup diagnostic; the program cache retains unsalted code. Only driver input changes.
	static const uint32_t salt = [] {
		const auto* value = std::getenv("KYTY_PIPELINE_COLD_SALT");
		return value == nullptr ? 0u : static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
	}();
	std::vector<uint32_t> salted;
	if (salt != 0) {
		EXIT_IF(!SaltSpirvIds(code, salt, salted));
		code = salted;
	}
	vk::ShaderModuleCreateInfo create_info {};
	create_info.codeSize    = code.size_bytes();
	create_info.pCode       = code.data();
	vk::ShaderModule module = nullptr;
	const auto       create_from = StatsOn() ? ClockNs() : 0;
	RequireVulkanSuccess(device.createShaderModule(&create_info, nullptr, &module),
	                     "create SPIR-V shader module");
	if (StatsOn()) AddSpvTimes({.calls = 1, .create_ns = ClockNs() - create_from});
	return module;
}

bool CompileSpvStatsEnabled() {
	return StatsOn();
}

CompileSpvTimes CompileSpvThreadTimes() {
	return t_spv_times;
}

CompileSpvTimes CompileSpvTotals() {
	const auto load = [](const std::atomic<uint64_t>& v) { return v.load(std::memory_order_relaxed); };
	return {.calls            = load(g_spv_times.calls),
	        .shrunk           = load(g_spv_times.shrunk),
	        .validated        = load(g_spv_times.validated),
	        .validate_skipped = load(g_spv_times.validate_skipped),
	        .shrink_ns        = load(g_spv_times.shrink_ns),
	        .validate_ns      = load(g_spv_times.validate_ns),
	        .create_ns        = load(g_spv_times.create_ns)};
}

uint64_t UseShrinkVerdictFile(const std::string& path) {
	if (const auto* value = std::getenv("KYTY_SHRINK_VERDICTS"); value != nullptr && std::strcmp(value, "0") == 0) {
		return 0;
	}
	return ShrinkVerdicts::Instance().Open(path);
}

vk::Pipeline CreateComputePipeline(vk::Device device, std::span<const uint32_t> code,
                                   vk::PipelineLayout layout, const char* operation,
                                   const vk::SpecializationInfo* specialization) {
	const auto                    shader = CompileSPV(code, device);
	vk::ComputePipelineCreateInfo create {};
	create.stage.stage    = vk::ShaderStageFlagBits::eCompute;
	create.stage.module   = shader;
	create.stage.pName    = "main";
	create.stage.pSpecializationInfo = specialization;
	create.layout         = layout;
	vk::Pipeline pipeline = nullptr;
	const auto   result   = device.createComputePipelines(nullptr, 1, &create, nullptr, &pipeline);
	device.destroyShaderModule(shader, nullptr);
	RequireVulkanSuccess(result, operation);
	return pipeline;
}

} // namespace Libs::Graphics
