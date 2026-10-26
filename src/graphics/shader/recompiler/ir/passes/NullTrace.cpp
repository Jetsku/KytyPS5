#include "graphics/shader/recompiler/ir/passes/NullTrace.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

thread_local uint64_t t_shader = 0;

struct State {
	std::mutex                                                               mutex;
	std::array<std::unordered_map<uint64_t, uint64_t>, static_cast<size_t>(NullTraceKind::Count)> counts;
	std::chrono::steady_clock::time_point window_start = std::chrono::steady_clock::now();
};

State& GetState() {
	static State state;
	return state;
}

const char* KindName(size_t kind) {
	switch (static_cast<NullTraceKind>(kind)) {
		case NullTraceKind::NullBuffer: return "null V#";
		case NullTraceKind::NullImage: return "null T#";
		case NullTraceKind::SkippedProgram: return "skipped program";
		case NullTraceKind::SkippedResources: return "resources failed";
		default: return "?";
	}
}

} // namespace

bool NullTraceEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_NULL_TRACE");
		return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

void NullTraceSetShader(uint64_t shader_hash) {
	t_shader = shader_hash;
}

void NullTraceNote(NullTraceKind kind, uint64_t shader_hash) {
	if (!NullTraceEnabled()) return;
	if (shader_hash == 0) shader_hash = t_shader;
	auto&       state = GetState();
	std::string lines;
	{
		std::scoped_lock lock(state.mutex);
		state.counts[static_cast<size_t>(kind)][shader_hash]++;
		const auto now = std::chrono::steady_clock::now();
		if (now - state.window_start < std::chrono::seconds(10)) return;
		state.window_start = now;
		lines = "Kyty null trace (last 10 s, shader hash x count):\n";
		for (size_t k = 0; k < state.counts.size(); k++) {
			auto& map = state.counts[k];
			if (map.empty()) continue;
			std::vector<std::pair<uint64_t, uint64_t>> top(map.begin(), map.end());
			std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
			char buffer[64];
			lines += "  ";
			lines += KindName(k);
			lines += ":";
			for (size_t i = 0; i < top.size() && i < 16; i++) {
				std::snprintf(buffer, sizeof(buffer), " %016" PRIx64 "x%" PRIu64, top[i].first, top[i].second);
				lines += buffer;
			}
			if (top.size() > 16) lines += " +" + std::to_string(top.size() - 16) + " more";
			lines += "\n";
			map.clear();
		}
	}
	std::fputs(lines.c_str(), stderr);
	std::fflush(stderr);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
