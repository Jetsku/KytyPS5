#include "common/hostException.h"
#include "common/sehGuard.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineOptPolicy.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace Libs::Graphics;

static int g_failed = 0;
#define CHECK(expr)                                                                          \
	do {                                                                                     \
		if (!(expr)) {                                                                       \
			std::printf("PipelineOptPolicyTests: FAILED: %s (line %d)\n", #expr, __LINE__); \
			++g_failed;                                                                      \
		}                                                                                    \
	} while (0)

static std::atomic<int> g_handler_calls {0};
[[maybe_unused]] static bool CountingHandler(const Common::HostException::ExceptionInfo& /*info*/) {
	// Kyty's real handler ends the process for a fault it cannot resolve; this one just counts.
	g_handler_calls.fetch_add(1);
	return false;
}

static volatile uintptr_t g_null_address = 0;
#if defined(_MSC_VER) && !defined(__clang__)
#define TEST_NOINLINE __declspec(noinline)
#else
#define TEST_NOINLINE __attribute__((noinline))
#endif
[[maybe_unused]] TEST_NOINLINE static void ReadNull(void* context) {
	*static_cast<uint32_t*>(context) = *reinterpret_cast<volatile const uint32_t*>(g_null_address);
}
static void NoFault(void* context) {
	*static_cast<uint32_t*>(context) = 7;
}

int main() {
	// Hashes: hex with or without 0x, never zero; lists split at ',', ';' and spaces.
	{
		CHECK(ParseShaderHash("0xa6d0a69b25e2707a") == 0xa6d0a69b25e2707aull);
		CHECK(ParseShaderHash(" A6D0A69B25E2707A ") == 0xa6d0a69b25e2707aull);
		CHECK(!ParseShaderHash("0x"));
		CHECK(!ParseShaderHash("0"));
		CHECK(!ParseShaderHash("xyz"));
		CHECK(!ParseShaderHash("0x11112222333344445"));
		std::vector<std::string> bad;
		const auto list = ParseShaderHashList("0x1, 2;bogus  0x1 3", &bad);
		CHECK(list.size() == 3 && list[0] == 1 && list[1] == 2 && list[2] == 3);
		CHECK(bad.size() == 1 && bad[0] == "bogus");
	}

	// Settings: the built-in list only on AMD; "none" clears it and the learned entries; the
	// fault guard defaults on for AMD on Windows only.
	{
		const auto amd = ParsePipelineOptSettings(nullptr, nullptr, nullptr, 0x1002u);
		CHECK(amd.optimize);
		CHECK(amd.source == PipelineOptSettings::ListSource::BuiltIn);
		CHECK(amd.shaders.size() == AmdNoOptShaders.size());
		CHECK(amd.use_learned);
#ifdef _WIN32
		CHECK(amd.fault_guard);
#else
		CHECK(!amd.fault_guard);
#endif
		const auto nv = ParsePipelineOptSettings(nullptr, nullptr, nullptr, 0x10deu);
		CHECK(nv.optimize && !nv.fault_guard && nv.shaders.empty());
		CHECK(nv.source == PipelineOptSettings::ListSource::None);
		const auto off = ParsePipelineOptSettings("0", nullptr, "0", 0x1002u);
		CHECK(!off.optimize && !off.fault_guard);
		const auto none = ParsePipelineOptSettings(nullptr, "none", nullptr, 0x1002u);
		CHECK(none.shaders.empty() && !none.use_learned);
		CHECK(none.source == PipelineOptSettings::ListSource::Cleared);
		const auto custom = ParsePipelineOptSettings("1", "0xabc", nullptr, 0x10deu);
		CHECK(custom.shaders.size() == 1 && custom.shaders[0] == 0xabc);
		CHECK(custom.source == PipelineOptSettings::ListSource::Env);
	}

	// List: a single-hash entry matches any pipeline with that shader; a learned set needs all.
	{
		PipelineNoOptList list;
		const uint64_t    ps = 0xa6d0a69b25e2707aull;
		CHECK(list.Add({&ps, 1}));
		CHECK(!list.Add({&ps, 1}));
		const PipelineShaderHashes pair = {0x1111, 0, 0, 0x2222};
		CHECK(list.Add(pair));
		CHECK(!list.Add(PipelineShaderHashes {0x1111, 0, 0, 0x2222}));
		CHECK(list.Size() == 2);
		CHECK(list.Match(PipelineShaderHashes {0x5, 0, 0, ps}).has_value());
		CHECK(list.Match(PipelineShaderHashes {0x1111, 0, 0, 0x2222}).has_value());
		CHECK(!list.Match(PipelineShaderHashes {0x1111, 0, 0, 0x3333}).has_value());
		CHECK(!list.Match(PipelineShaderHashes {0x4444, 0, 0, 0x2222}).has_value());
		CHECK(!list.Add(PipelineShaderHashes {0, 0, 0, 0}));
	}

	// File: lines of other devices and malformed ones are skipped; a written line reads back.
	{
		const auto key  = PipelineOptDeviceKey(0x1002u, 0x73bfu, 0x00800123u);
		CHECK(key == "1002:73bf:00800123");
		const auto line = FormatNoOptLine(key, PipelineShaderHashes {0x1e83ec6088d0f320ull, 0, 0,
		                                                             0xa6d0a69b25e2707aull});
		CHECK(line == "1002:73bf:00800123 0x1e83ec6088d0f320+0xa6d0a69b25e2707a");
		const std::string text = "# comment\n" + line + "  # driver fault\r\n" +
		                         "10de:2204:00000001 0x1\n" + key + " 0xzz\n" + key + " 0x5\n";
		const auto entries = ParseNoOptFile(text, key);
		CHECK(entries.size() == 2);
		CHECK(entries.size() == 2 && entries[0].size() == 2 &&
		      entries[0][0] == 0x1e83ec6088d0f320ull && entries[0][1] == 0xa6d0a69b25e2707aull);
		CHECK(entries.size() == 2 && entries[1].size() == 1 && entries[1][0] == 5);
	}

	// The fault guard: a fault inside it returns its code, even with Kyty's host handler
	// installed (vectored handlers run before __except; the probe makes the handler pass).
	{
		uint32_t value = 0;
		CHECK(Common::CallCatchingStructuredException(NoFault, &value) == 0);
		CHECK(value == 7);
#ifdef _WIN32
		CHECK(Common::HostException::InstallHandler(CountingHandler));
		uint64_t   address = 0;
		const auto code    = Common::CallCatchingStructuredException(ReadNull, &value, &address);
		CHECK(code == 0xC0000005u);
		CHECK(address != 0);
		CHECK(g_handler_calls.load() == 0);
		CHECK(Common::DescribeCodeAddress(address).find('+') != std::string::npos);
		// The guard nests and leaves the probe depth balanced: a second call works the same.
		CHECK(Common::CallCatchingStructuredException(ReadNull, &value) == 0xC0000005u);
		CHECK(Common::CallCatchingStructuredException(NoFault, &value) == 0);
#endif
	}

	if (g_failed == 0) std::printf("PipelineOptPolicyTests: all passed\n");
	return g_failed == 0 ? 0 : 1;
}
