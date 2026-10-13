#include "libs/audioPropagation.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string_view>
#include <unordered_map>

namespace Libs::AudioPropagation {

namespace {

struct Object {
	Kind         kind        = Kind::System;
	const float* input       = nullptr;
	uint32_t     input_bytes = 0;
};

std::mutex                         g_mutex;
std::unordered_map<Handle, Object> g_objects;
std::atomic<uint64_t>              g_next_handle {1};

RenderMode ModeFromEnvironment() {
	const char* value = std::getenv("KYTY_AUDIO_PROPAGATION_RENDER");
	if (value != nullptr && std::string_view(value) == "silent") {
		return RenderMode::Silent;
	}
	return RenderMode::Direct;
}

std::atomic<RenderMode>& Mode() {
	static std::atomic<RenderMode> mode {ModeFromEnvironment()};
	return mode;
}

} // namespace

RenderMode GetRenderMode() {
	return Mode().load(std::memory_order_relaxed);
}

void SetRenderMode(RenderMode mode) {
	Mode().store(mode, std::memory_order_relaxed);
}

Handle Create(Kind kind) {
	const Handle     handle = g_next_handle.fetch_add(1, std::memory_order_relaxed);
	std::scoped_lock lock(g_mutex);
	g_objects[handle] = Object {kind};
	return handle;
}

bool Destroy(Handle handle) {
	std::scoped_lock lock(g_mutex);
	return g_objects.erase(handle) != 0;
}

bool IsLive(Handle handle, Kind kind) {
	std::scoped_lock lock(g_mutex);
	const auto       it = g_objects.find(handle);
	return it != g_objects.end() && it->second.kind == kind;
}

size_t LiveCount() {
	std::scoped_lock lock(g_mutex);
	return g_objects.size();
}

void Reset() {
	std::scoped_lock lock(g_mutex);
	g_objects.clear();
}

void SetSourceAttributes(Handle source, const Attribute* attributes, uint32_t count) {
	if (attributes == nullptr) {
		return;
	}
	std::scoped_lock lock(g_mutex);
	const auto       it = g_objects.find(source);
	if (it == g_objects.end() || it->second.kind != Kind::Source) {
		return;
	}
	for (uint32_t i = 0; i < count; i++) {
		const auto& attribute = attributes[i];
		if (attribute.id == SourceAttributeInput && attribute.value != nullptr &&
		    attribute.size >= sizeof(InputBuffer::samples) + sizeof(InputBuffer::size_bytes)) {
			const auto* input       = static_cast<const InputBuffer*>(attribute.value);
			it->second.input        = input->samples;
			it->second.input_bytes  = input->samples != nullptr ? input->size_bytes : 0;
		}
	}
}

int32_t Render(const RenderParams& params) {
	if (params.output == nullptr) {
		return params.output_size_bytes == 0 ? 0 : ErrorInvalidParam;
	}
	const size_t output_floats = params.output_size_bytes / sizeof(float);
	std::memset(params.output, 0, output_floats * sizeof(float));
	if (GetRenderMode() == RenderMode::Silent) {
		return 0;
	}
	const float* input       = nullptr;
	uint32_t     input_bytes = 0;
	{
		std::scoped_lock lock(g_mutex);
		const auto       it = g_objects.find(params.source);
		if (it == g_objects.end() || it->second.kind != Kind::Source) {
			return 0;
		}
		input       = it->second.input;
		input_bytes = it->second.input_bytes;
		// The game sets the input right before each render (its buffer may not outlive the call):
		// a render without a new input renders silence.
		it->second.input       = nullptr;
		it->second.input_bytes = 0;
	}
	const size_t frames = input_bytes / sizeof(float);
	if (input == nullptr || frames == 0) {
		return 0;
	}
	// Interleaved bed: the channel count follows from the sizes (16 for Astro Bot's format 2).
	const size_t channels = output_floats / frames;
	if (channels == 0) {
		return 0;
	}
	for (size_t frame = 0; frame < frames; frame++) {
		params.output[frame * channels] = input[frame];
	}
	return 0;
}

} // namespace Libs::AudioPropagation
