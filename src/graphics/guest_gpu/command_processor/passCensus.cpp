#include "graphics/guest_gpu/command_processor/passCensus.h"
#include "common/staticOnce.h"

#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/graphicsRun.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::PassCensus {

namespace {

struct Pass {
	uint64_t                                             draws      = 0; // CP packets
	uint64_t                                             dispatches = 0;
	uint64_t                                             exec_draws = 0; // reached the renderer
	uint64_t                                             exec_cs    = 0;
	std::map<std::tuple<char, uint64_t, uint64_t>, uint64_t> shaders;
	std::map<std::tuple<char, uint64_t, std::string>, uint64_t> drops;
	std::map<std::pair<uint64_t, std::string>, uint64_t>  nulls;
	uint32_t                                             first_window = 0;
};

struct Census {
	std::mutex                                  mutex;
	std::unordered_map<std::string, Pass>       passes;
	std::chrono::steady_clock::time_point       window_start = std::chrono::steady_clock::now();
	uint32_t                                    window_frame = 0;
	uint32_t                                    window       = 0;
	FILE*                                       file         = nullptr;
};

Census& Global() {
	static Census c;
	return c;
}

std::atomic<uint32_t> g_frames {0};

thread_local std::unordered_map<const void*, std::vector<std::string>> t_stacks;
thread_local std::string                                             t_key = "-";
thread_local uint64_t                                                t_shader = 0;

std::string KeyOf(const std::vector<std::string>& stack) {
	const auto n = stack.size();
	return n == 0 ? std::string("-") : n == 1 ? stack.back() : stack[n - 2] + ">" + stack.back();
}

void Report(Census& c) {
	const auto now = std::chrono::steady_clock::now();
	if (now - c.window_start < std::chrono::seconds(5)) return;
	const uint32_t frame  = g_frames.load(std::memory_order_relaxed);
	const uint32_t frames = frame > c.window_frame ? frame - c.window_frame : 1u;
	if (c.file == nullptr) {
		const char* path = std::getenv("KYTY_TRACE_PASS_CENSUS_FILE");
		c.file           = std::fopen(path != nullptr && path[0] != '\0' ? path : "_pass-census.txt", "w");
		if (c.file == nullptr) return;
	}
	const double f = frames;
	std::vector<std::pair<std::string, const Pass*>> rows;
	for (const auto& [k, p]: c.passes) rows.emplace_back(k, &p);
	std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
		return a.second->draws + a.second->dispatches > b.second->draws + b.second->dispatches;
	});
	std::fprintf(c.file, "WINDOW %u frames=%u frame=%u passes=%zu\n", c.window, frames, frame, rows.size());
	for (const auto& [k, p]: rows) {
		std::fprintf(c.file,
		             "PASS \"%s\" draws/f=%.1f dispatches/f=%.1f exec_draws/f=%.1f exec_cs/f=%.1f drops/f=%.1f nulls/f=%.1f\n",
		             k.c_str(), p->draws / f, p->dispatches / f, p->exec_draws / f, p->exec_cs / f,
		             [&] {
			             uint64_t n = 0;
			             for (const auto& [dk, dn]: p->drops) n += dn;
			             return n / f;
		             }(),
		             [&] {
			             uint64_t n = 0;
			             for (const auto& [nk, nn]: p->nulls) n += nn;
			             return n / f;
		             }());
		for (const auto& [sk, sn]: p->shaders) {
			std::fprintf(c.file, "  SH %c %016" PRIx64 " %016" PRIx64 " n/f=%.2f\n", std::get<0>(sk), std::get<1>(sk),
			             std::get<2>(sk), sn / f);
		}
		for (const auto& [dk, dn]: p->drops) {
			std::fprintf(c.file, "  DROP %c %016" PRIx64 " n/f=%.2f %s\n", std::get<0>(dk), std::get<1>(dk), dn / f,
			             std::get<2>(dk).c_str());
		}
		for (const auto& [nk, nn]: p->nulls) {
			std::fprintf(c.file, "  NULL shader=%016" PRIx64 " n/f=%.2f %s\n", nk.first, nn / f, nk.second.c_str());
		}
	}
	std::fflush(c.file);
	c.passes.clear();
	c.window_start = now;
	c.window_frame = frame;
	c.window++;
}

bool OnGpuThread() {
	return GuestGpu::IsGpuThread();
}

} // namespace

bool Enabled() {
	static constexpr auto on_init = [] {
		const char* value = std::getenv("KYTY_TRACE_PASS_CENSUS");
		const bool  o     = value != nullptr && value[0] == '1';
		if (o) std::printf("KYTY_TRACE_PASS_CENSUS enabled (diagnostic)\n");
		return o;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(on_init())>> on_once;
	const bool& on = on_once.Get(on_init);
	return on;
}

bool LabelsEnabled() {
	static constexpr auto on_init = [] {
		const char* value = std::getenv("KYTY_TRACE_VK_LABELS");
		const bool  o     = value != nullptr && value[0] != '\0' ? value[0] != '0' : Config::RenderDocEnabled();
		if (o) std::printf("Vulkan debug labels: on (game markers and shader hashes; KYTY_TRACE_VK_LABELS)\n");
		return o;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(on_init())>> on_once;
	const bool& on = on_once.Get(on_init);
	return on;
}

void LabelText(char* out, size_t size, char kind, uint64_t a, uint64_t b) {
	const char* key = t_key.c_str();
	if (kind == 'c') {
		std::snprintf(out, size, "dispatch %016" PRIx64 " | %s", a, key);
	} else {
		std::snprintf(out, size, "draw %016" PRIx64 " %016" PRIx64 " | %s", a, b, key);
	}
}

void Marker(const void* cp, const char* push_name) {
	if (!Tracking() || !OnGpuThread()) return;
	auto& stack = t_stacks[cp];
	if (push_name != nullptr) {
		stack.emplace_back(push_name);
	} else if (!stack.empty()) {
		stack.pop_back();
	}
	t_key = KeyOf(stack);
}

void Packet(const void* cp, bool dispatch) {
	if (!Tracking() || !OnGpuThread()) return;
	t_key   = KeyOf(t_stacks[cp]);
	if (!Enabled()) return; // labels only
	auto& c = Global();
	std::scoped_lock lock(c.mutex);
	auto& p = c.passes[t_key];
	(dispatch ? p.dispatches : p.draws)++;
	Report(c);
}

void Exec(char kind, uint64_t a, uint64_t b) {
	if (!Enabled() || !OnGpuThread()) return;
	auto&            c = Global();
	std::scoped_lock lock(c.mutex);
	auto&            p = c.passes[t_key];
	(kind == 'c' ? p.exec_cs : p.exec_draws)++;
	p.shaders[{kind, a, b}]++;
}

void Drop(char kind, uint64_t hash, const char* reason) {
	if (!Enabled() || !OnGpuThread()) return;
	auto&            c = Global();
	std::scoped_lock lock(c.mutex);
	// A drop whose shader is not known keeps hash 0; the thread's last materialized shader is only named in the
	// reason (it is often a different, earlier shader: the census used to attribute such drops to it).
	std::string text(reason);
	if (hash == 0 && t_shader != 0) {
		char buffer[64];
		std::snprintf(buffer, sizeof(buffer), " [hash unknown; last materialized %016" PRIx64 "]", t_shader);
		text += buffer;
	}
	c.passes[t_key].drops[{kind, hash, text}]++;
}

void Null(const char* reason) {
	if (!Enabled() || !OnGpuThread()) return;
	auto&            c = Global();
	std::scoped_lock lock(c.mutex);
	std::string      r(reason);
	if (r.size() > 160) r.resize(160);
	c.passes[t_key].nulls[{t_shader, r}]++;
}

void SetShader(uint64_t hash) {
	if (!Enabled()) return;
	t_shader = hash;
}

std::string CurrentKey() {
	return t_key;
}

void Frame() {
	if (!Enabled()) return;
	g_frames.fetch_add(1, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::PassCensus
