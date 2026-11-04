#include "graphics/shader/shader.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdlib>

namespace Libs::Graphics {

namespace {

constexpr uint32_t PsInputOffsetMask = 0x0000001fu;
constexpr uint32_t PsInputFlatShade  = 0x00000400u;

} // namespace

bool ShaderCbShaderMaskExports() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CB_SHADER_MASK_EXPORTS");
		return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

bool ShaderObjprimIdEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_OBJPRIM_ID");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

uint32_t ShaderPixelExportTarget(uint32_t shader_mask, uint32_t export_index) {
	for (uint32_t target = 0; target < 8u; target++) {
		if (((shader_mask >> (target * 4u)) & 0xfu) == 0u) {
			continue;
		}
		if (export_index == 0u) {
			return target;
		}
		export_index--;
	}
	return UINT32_MAX;
}

uint32_t ShaderPixelParameterMappedLocation(const ShaderPixelInputInfo& info, uint32_t input) {
	return input < info.input_num ? info.interpolator_settings[input] & PsInputOffsetMask : input;
}

uint32_t ShaderPixelParameterLocation(const ShaderPixelInputInfo& info,
                                      std::span<const uint32_t> active_inputs, uint32_t input) {
	// Outside rectangle lists, flat and smooth aliases of one vertex export read raw vertices at
	// its location (ShaderInfoCollection). KYTY_PS_PER_VERTEX=0 has no raw-vertex inputs, so
	// differing aliases keep separate locations there as rectangles do.
	if (info.parameter_mode != ShaderPixelParameterMode::Rectangle &&
	    ShaderRecompiler::GetCodegenOptions().ps_per_vertex) {
		return ShaderPixelParameterMappedLocation(info, input);
	}
	std::array<bool, 32> used_locations {};
	for (const auto active_input: active_inputs) {
		used_locations[ShaderPixelParameterMappedLocation(info, active_input)] = true;
	}
	std::array<uint32_t, 64> group_locations;
	group_locations.fill(UINT32_MAX);
	for (const auto active_input: active_inputs) {
		const auto mapped = ShaderPixelParameterMappedLocation(info, active_input);
		const auto group  = mapped * 2u + ShaderPixelParameterIsFlat(info, active_input);
		auto&      location = group_locations[group];
		if (location == UINT32_MAX) {
			location = mapped;
			// Rectangle expansion broadcasts flat aliases through separate outputs.
			if (group_locations[group ^ 1u] != UINT32_MAX) {
				location = 0;
				while (location < used_locations.size() && used_locations[location]) {
					location++;
				}
				EXIT_NOT_IMPLEMENTED(location >= used_locations.size());
			}
			used_locations[location] = true;
		}

		if (active_input == input) {
			return location;
		}
	}
	return ShaderPixelParameterMappedLocation(info, input);
}

void ShaderLinkVertexPixelParameters(ShaderVertexInputInfo& vertex,
                                    const ShaderPixelInputInfo& pixel) {
	std::array<uint32_t, 32> active {};
	uint32_t count = 0;
	if (pixel.stage.program != nullptr) {
		for (const auto& input: pixel.stage.program->info.inputs) {
			if (input.kind == ShaderRecompiler::IR::StageInputKind::Parameter) {
				active[count++] = input.location;
			}
		}
	}
	ShaderLinkVertexPixelParameters(vertex, pixel, std::span(active).first(count));
}

void ShaderLinkVertexPixelParameters(ShaderVertexInputInfo& vertex,
                                     const ShaderPixelInputInfo& pixel,
                                     std::span<const uint32_t> active_inputs) {
	// Runs for every draw. Without a location read both flat and smooth no input moves
	// (ShaderPixelParameterLocation returns the mapped location of each), so no aliases.
	uint32_t flat_locations   = 0;
	uint32_t smooth_locations = 0;
	for (const auto input: active_inputs) {
		const auto bit = 1u << (ShaderPixelParameterMappedLocation(pixel, input) & 31u);
		(ShaderPixelParameterIsFlat(pixel, input) ? flat_locations : smooth_locations) |= bit;
	}
	vertex.param_alias_mask = 0;
	if ((flat_locations & smooth_locations) == 0) {
		return;
	}
	vertex.param_alias_source.fill(0);
	for (const auto input: active_inputs) {
		const auto source = ShaderPixelParameterMappedLocation(pixel, input);
		const auto destination = ShaderPixelParameterLocation(pixel, active_inputs, input);
		if (destination != source) {
			vertex.param_alias_mask |= 1u << destination;
			vertex.param_alias_source[destination] = source;
		}
	}
}

bool ShaderPixelParameterIsFlat(const ShaderPixelInputInfo& info, uint32_t input) {
	return input < info.input_num && (info.interpolator_settings[input] & PsInputFlatShade) != 0 &&
	       !ShaderPixelParameterIsCustom(info, input);
}

bool ShaderPixelParameterIsCustom(const ShaderPixelInputInfo& info, uint32_t input) {
	return input < 32u && (info.custom_interpolation_mask & (1u << input)) != 0;
}

} // namespace Libs::Graphics
