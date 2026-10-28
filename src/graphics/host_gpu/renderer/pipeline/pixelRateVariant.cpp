#include "graphics/host_gpu/renderer/pipeline/pixelRateVariant.h"

#include <cstdlib>
#include <initializer_list>
#include <mutex>
#include <spirv-tools/libspirv.h>
#include <spirv/unified1/spirv.hpp>
#include <unordered_map>

namespace Libs::Graphics {

bool                  SampleInterpolationDropped();
std::vector<uint32_t> WithoutSampleInterpolation(std::span<const uint32_t> spirv);

namespace {

// What SpirvWithoutSampleId learns from parsing a module. SPIRV-Tools tells literal operands from
// <id> operands, which a word-by-word scan cannot: the variable's id is small, and so are storage
// classes, extended instruction numbers and memory operands.
struct SampleIdScan {
	uint32_t sample_id   = 0;
	bool     per_sample  = false; // another per-sample input (SamplePosition, a member decoration)
	bool     sample_decorated = false; // an input decorated Sample (interpolated per sample)
	bool     unsupported = false; // a form the rewrite does not handle (decoration groups, ...)
	bool     other_use   = false; // the variable named other than by a load, decoration, name or
	                              // the entry point's interface
};

bool IsIdOperand(spv_operand_type_t type) {
	switch (type) {
		case SPV_OPERAND_TYPE_ID:
		case SPV_OPERAND_TYPE_TYPE_ID:
		case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
		case SPV_OPERAND_TYPE_SCOPE_ID:
		case SPV_OPERAND_TYPE_OPTIONAL_ID:
		case SPV_OPERAND_TYPE_VARIABLE_ID: return true;
		default: return false;
	}
}

// First pass: the variable decorated BuiltIn SampleId, and anything else that shades per sample.
spv_result_t ScanSampleIdDecorations(void* user_data, const spv_parsed_instruction_t* inst) {
	auto&       scan  = *static_cast<SampleIdScan*>(user_data);
	const auto* words = inst->words;
	switch (static_cast<spv::Op>(inst->opcode)) {
		case spv::OpDecorate:
			if (inst->num_words >= 3 && words[2] == spv::DecorationSample) {
				scan.sample_decorated = true;
				if (!SampleInterpolationDropped()) scan.per_sample = true;
			}
			if (inst->num_words >= 4 && words[2] == spv::DecorationBuiltIn) {
				if (words[3] == spv::BuiltInSampleId) {
					if (scan.sample_id != 0 && scan.sample_id != words[1]) {
						scan.unsupported = true;
					}
					scan.sample_id = words[1];
				} else if (words[3] == spv::BuiltInSamplePosition) {
					scan.per_sample = true;
				}
			}
			break;
		case spv::OpMemberDecorate:
			if ((inst->num_words >= 4 && words[3] == spv::DecorationSample) ||
			    (inst->num_words >= 5 && words[3] == spv::DecorationBuiltIn &&
			     (words[4] == spv::BuiltInSampleId || words[4] == spv::BuiltInSamplePosition))) {
				scan.per_sample = true;
			}
			break;
		// A decoration group could decorate the variable (or another input) without naming it.
		case spv::OpGroupDecorate:
		case spv::OpGroupMemberDecorate: scan.unsupported = true; break;
		default: break;
	}
	return SPV_SUCCESS;
}

// Second pass: every <id> operand naming the variable, other than the forms that are rewritten.
spv_result_t ScanSampleIdUses(void* user_data, const spv_parsed_instruction_t* inst) {
	auto& scan = *static_cast<SampleIdScan*>(user_data);
	switch (static_cast<spv::Op>(inst->opcode)) {
		case spv::OpDecorate:
		case spv::OpName:
		case spv::OpEntryPoint: return SPV_SUCCESS;
		default: break;
	}
	for (uint16_t i = 0; i < inst->num_operands; i++) {
		const auto& operand = inst->operands[i];
		if (!IsIdOperand(operand.type) || inst->words[operand.offset] != scan.sample_id) {
			continue;
		}
		// A load from the variable (its pointer operand) is what the rewrite keeps.
		if (static_cast<spv::Op>(inst->opcode) == spv::OpLoad && operand.offset == 3) {
			continue;
		}
		scan.other_use = true;
	}
	return SPV_SUCCESS;
}

// One parse of a module by SPIRV-Tools, with operand types; false when it does not parse.
bool ParseSpirv(std::span<const uint32_t> spirv, SampleIdScan& scan,
                spv_parsed_instruction_fn_t callback) {
	static const spv_context context    = spvContextCreate(SPV_ENV_UNIVERSAL_1_6);
	spv_diagnostic           diagnostic = nullptr;
	const auto               result =
	    spvBinaryParse(context, &scan, spirv.data(), spirv.size(), nullptr, callback, &diagnostic);
	spvDiagnosticDestroy(diagnostic);
	return result == SPV_SUCCESS;
}

} // namespace

bool EqaaPerSampleShading() {
	static const bool value = [] {
		const char* text = std::getenv("KYTY_EQAA_PER_SAMPLE");
		return text != nullptr && text[0] == '1';
	}();
	return value;
}

bool SpirvShadesPerSample(std::span<const uint32_t> spirv) {
	SampleIdScan scan;
	if (spirv.size() < 5 || spirv[0] != spv::MagicNumber ||
	    !ParseSpirv(spirv, scan, ScanSampleIdDecorations)) {
		return false;
	}
	return scan.sample_id != 0 || scan.per_sample;
}

// KYTY_EQAA_DROP_SAMPLE_INTERP=1 (default off; on in the GT7 preset): the per-pixel copy of a
// pixel shader (KYTY_EQAA_MIXED) may also drop per-sample interpolation (Sample decorations), so
// GT7's pixel shaders that only interpolate per sample get a copy and their EQAA draws run
// instead of being skipped (mirrors and side windows drew garbage).
bool SampleInterpolationDropped() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_EQAA_DROP_SAMPLE_INTERP");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return enabled;
}

// The module without its Sample decorations and the SampleRateShading capability.
std::vector<uint32_t> WithoutSampleInterpolation(std::span<const uint32_t> spirv) {
	constexpr uint32_t    HeaderWords = 5;
	std::vector<uint32_t> out(spirv.begin(), spirv.begin() + HeaderWords);
	for (size_t offset = HeaderWords; offset < spirv.size();) {
		const uint32_t count = spirv[offset] >> spv::WordCountShift;
		if (count == 0 || offset + count > spirv.size()) {
			return {};
		}
		const auto opcode = static_cast<spv::Op>(spirv[offset] & spv::OpCodeMask);
		const bool drop =
		    (opcode == spv::OpDecorate && count == 3 && spirv[offset + 2] == spv::DecorationSample) ||
		    (opcode == spv::OpCapability && count == 2 && spirv[offset + 1] == spv::CapabilitySampleRateShading);
		if (!drop) {
			out.insert(out.end(), spirv.begin() + static_cast<ptrdiff_t>(offset),
			           spirv.begin() + static_cast<ptrdiff_t>(offset + count));
		}
		offset += count;
	}
	return out;
}

std::vector<uint32_t> SpirvWithoutSampleId(std::span<const uint32_t> spirv) {
	constexpr uint32_t HeaderWords = 5;
	if (spirv.size() < HeaderWords || spirv[0] != spv::MagicNumber) {
		return {};
	}
	SampleIdScan scan;
	if (!ParseSpirv(spirv, scan, ScanSampleIdDecorations) || scan.per_sample || scan.unsupported) {
		return {};
	}
	// KYTY_EQAA_DROP_SAMPLE_INTERP: a shader that shades per sample only because inputs are
	// interpolated per sample (no SampleId): the copy interpolates them at the pixel instead.
	if (scan.sample_id == 0) {
		return scan.sample_decorated ? WithoutSampleInterpolation(spirv) : std::vector<uint32_t> {};
	}
	if (!ParseSpirv(spirv, scan, ScanSampleIdUses) || scan.other_use) {
		return {};
	}
	const uint32_t sample_id = scan.sample_id;

	struct Instruction {
		size_t   offset;
		uint32_t count;
		spv::Op  opcode;
	};
	std::vector<Instruction> instructions;
	for (size_t offset = HeaderWords; offset < spirv.size();) {
		const uint32_t count = spirv[offset] >> spv::WordCountShift;
		if (count == 0 || offset + count > spirv.size()) {
			return {};
		}
		instructions.push_back(
		    {offset, count, static_cast<spv::Op>(spirv[offset] & spv::OpCodeMask)});
		offset += count;
	}
	const auto words = [&](const Instruction& inst) {
		return spirv.subspan(inst.offset, inst.count);
	};
	// Its declaration: an Input variable of a pointer to a 32-bit integer.
	const Instruction* variable = nullptr;
	uint32_t           pointer  = 0;
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpVariable && w.size() == 4 && w[2] == sample_id) {
			if (w[3] != spv::StorageClassInput) {
				return {};
			}
			variable = &inst;
			pointer  = w[1];
		}
	}
	if (variable == nullptr) {
		return {};
	}
	uint32_t integer = 0;
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpTypePointer && w.size() == 4 && w[1] == pointer &&
		    w[2] == spv::StorageClassInput) {
			integer = w[3];
		}
	}
	bool integer_ok = false;
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpTypeInt && w.size() == 4 && w[1] == integer && w[2] == 32) {
			integer_ok = true;
		}
	}
	if (!integer_ok) {
		return {};
	}
	// A Private pointer to the integer and a 0 of it: existing ones declared before the variable
	// where the module has them, new ones (declared just before it) otherwise.
	uint32_t bound        = spirv[3];
	uint32_t private_type = 0;
	uint32_t zero         = 0;
	for (const auto& inst: instructions) {
		if (&inst == variable) {
			break;
		}
		const auto w = words(inst);
		if (inst.opcode == spv::OpTypePointer && w.size() == 4 &&
		    w[2] == spv::StorageClassPrivate && w[3] == integer && private_type == 0) {
			private_type = w[1];
		} else if (inst.opcode == spv::OpConstant && w.size() == 4 && w[1] == integer &&
		           w[3] == 0 && zero == 0) {
			zero = w[2];
		}
	}
	const bool new_private = private_type == 0;
	const bool new_zero    = zero == 0;
	if (new_private) {
		private_type = bound++;
	}
	if (new_zero) {
		zero = bound++;
	}
	// Since SPIR-V 1.4 the entry point's interface lists every global variable it uses.
	const bool interface_lists_private = spirv[1] >= 0x00010400u;

	std::vector<uint32_t> out(spirv.begin(), spirv.begin() + HeaderWords);
	out[3] = bound;
	out.reserve(spirv.size() + 16);
	const auto append = [&](std::initializer_list<uint32_t> list) {
		out.insert(out.end(), list.begin(), list.end());
	};
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpDecorate && w.size() >= 2 && w[1] == sample_id) {
			continue; // BuiltIn, Flat
		}
		if (inst.opcode == spv::OpDecorate && w.size() == 3 && w[2] == spv::DecorationSample) {
			continue; // KYTY_EQAA_DROP_SAMPLE_INTERP: interpolated at the pixel
		}
		if (inst.opcode == spv::OpCapability && w.size() == 2 &&
		    w[1] == spv::CapabilitySampleRateShading) {
			continue; // nothing else needs it (no other per-sample input, see the scan)
		}
		if (inst.opcode == spv::OpEntryPoint && !interface_lists_private) {
			// Before SPIR-V 1.4 the interface lists Input and Output variables only.
			size_t name_end = 3;
			while (name_end < w.size()) {
				const auto word = w[name_end++];
				if ((word & 0xff000000u) == 0) {
					break; // the word holding the name's terminating zero
				}
			}
			std::vector<uint32_t> rewritten(w.begin(),
			                                w.begin() + static_cast<ptrdiff_t>(name_end));
			for (size_t i = name_end; i < w.size(); i++) {
				if (w[i] != sample_id) {
					rewritten.push_back(w[i]);
				}
			}
			rewritten[0] = (static_cast<uint32_t>(rewritten.size()) << spv::WordCountShift) |
			               spv::OpEntryPoint;
			out.insert(out.end(), rewritten.begin(), rewritten.end());
			continue;
		}
		if (&inst == variable) {
			if (new_private) {
				append({(4u << spv::WordCountShift) | spv::OpTypePointer, private_type,
				        spv::StorageClassPrivate, integer});
			}
			if (new_zero) {
				append({(4u << spv::WordCountShift) | spv::OpConstant, integer, zero, 0u});
			}
			append({(5u << spv::WordCountShift) | spv::OpVariable, private_type, sample_id,
			        spv::StorageClassPrivate, zero});
			continue;
		}
		out.insert(out.end(), w.begin(), w.end());
	}
	return out;
}

namespace PixelRate {

namespace {

std::mutex                             g_mutex;
std::unordered_map<uint64_t, Lookup>   g_programs;

} // namespace

void Register(uint64_t program_id, bool shades_per_sample, vk::ShaderModule module) {
	std::scoped_lock lock {g_mutex};
	g_programs[program_id] = {shades_per_sample, module};
}

void Forget(vk::Device device, uint64_t program_id) {
	vk::ShaderModule module = nullptr;
	{
		std::scoped_lock lock {g_mutex};
		const auto       found = g_programs.find(program_id);
		if (found == g_programs.end()) {
			return;
		}
		module = found->second.module;
		g_programs.erase(found);
	}
	if (module != nullptr) {
		device.destroyShaderModule(module, nullptr);
	}
}

Lookup Find(uint64_t program_id) {
	std::scoped_lock lock {g_mutex};
	const auto       found = g_programs.find(program_id);
	return found != g_programs.end() ? found->second : Lookup {};
}

} // namespace PixelRate

bool UnnormalizedSamplingFix() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_UNNORMALIZED_FIX");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return enabled;
}

namespace {

// The result type of every result id of a module, from one SPIRV-Tools parse.
spv_result_t ScanResultTypes(void* user_data, const spv_parsed_instruction_t* inst) {
	auto& types = *static_cast<std::unordered_map<uint32_t, uint32_t>*>(user_data);
	if (inst->result_id != 0 && inst->type_id != 0) {
		types[inst->result_id] = inst->type_id;
	}
	return SPV_SUCCESS;
}

} // namespace

std::vector<uint32_t> SpirvNormalizedExplicitLod(std::span<const uint32_t> spirv) {
	constexpr uint32_t HeaderWords = 5;
	if (spirv.size() < HeaderWords || spirv[0] != spv::MagicNumber) {
		return {};
	}
	std::unordered_map<uint32_t, uint32_t> result_types;
	{
		static const spv_context context    = spvContextCreate(SPV_ENV_UNIVERSAL_1_6);
		spv_diagnostic           diagnostic = nullptr;
		const auto result = spvBinaryParse(context, &result_types, spirv.data(), spirv.size(),
		                                   nullptr, ScanResultTypes, &diagnostic);
		spvDiagnosticDestroy(diagnostic);
		if (result != SPV_SUCCESS) {
			return {};
		}
	}
	struct Image {
		uint32_t dim     = 0;
		bool     arrayed = false;
		bool     ms      = false;
	};
	std::unordered_map<uint32_t, Image>    images;         // OpTypeImage
	std::unordered_map<uint32_t, uint32_t> sampled_images; // OpSampledImage result -> image
	std::unordered_map<uint64_t, uint32_t> vectors;        // component type << 32 | count
	uint32_t                               glsl = 0, int_type = 0, float_type = 0;
	bool                                   image_query      = false;
	size_t                                 capabilities_end = 0, globals_end = 0;
	std::vector<size_t> samples; // offsets of OpImageSampleExplicitLod to rewrite
	for (size_t offset = HeaderWords; offset < spirv.size();) {
		const uint32_t count = spirv[offset] >> spv::WordCountShift;
		if (count == 0 || offset + count > spirv.size()) {
			return {};
		}
		const auto w = spirv.subspan(offset, count);
		switch (static_cast<spv::Op>(spirv[offset] & spv::OpCodeMask)) {
			case spv::OpCapability:
				image_query      = image_query || (count >= 2 && w[1] == spv::CapabilityImageQuery);
				capabilities_end = offset + count;
				break;
			case spv::OpExtInstImport:
				if (count >= 3 && std::string_view(reinterpret_cast<const char*>(&w[2])) ==
				                      std::string_view("GLSL.std.450")) {
					glsl = w[1];
				}
				break;
			case spv::OpTypeInt:
				if (count == 4 && w[2] == 32 && w[3] == 1) int_type = w[1];
				break;
			case spv::OpTypeFloat:
				if (count >= 3 && w[2] == 32) float_type = w[1];
				break;
			case spv::OpTypeVector:
				if (count == 4) vectors[(uint64_t {w[2]} << 32u) | w[3]] = w[1];
				break;
			case spv::OpTypeImage:
				if (count >= 9) images[w[1]] = {w[3], w[5] != 0, w[6] != 0};
				break;
			case spv::OpSampledImage:
				if (count == 5) sampled_images[w[2]] = w[3];
				break;
			case spv::OpFunction:
				if (globals_end == 0) globals_end = offset;
				break;
			case spv::OpImageSampleExplicitLod:
				// Lod alone or with offsets (no bias, gradients or other operands).
				if (count >= 7 && (w[5] & spv::ImageOperandsLodMask) != 0 &&
				    (w[5] & ~(spv::ImageOperandsLodMask | spv::ImageOperandsConstOffsetMask |
				              spv::ImageOperandsOffsetMask)) == 0) {
					samples.push_back(offset);
				} else {
					return {};
				}
				break;
			// Any other use of a sampler: the copy's samplers are made normalized, which only
			// the rewritten samples are ready for.
			case spv::OpImageSampleImplicitLod:
			case spv::OpImageSampleDrefImplicitLod:
			case spv::OpImageSampleDrefExplicitLod:
			case spv::OpImageSampleProjImplicitLod:
			case spv::OpImageSampleProjExplicitLod:
			case spv::OpImageSampleProjDrefImplicitLod:
			case spv::OpImageSampleProjDrefExplicitLod:
			case spv::OpImageGather:
			case spv::OpImageDrefGather:
			case spv::OpImageSparseSampleImplicitLod:
			case spv::OpImageSparseSampleExplicitLod:
			case spv::OpImageSparseSampleDrefImplicitLod:
			case spv::OpImageSparseSampleDrefExplicitLod:
			case spv::OpImageSparseGather:
			case spv::OpImageSparseDrefGather:
			case spv::OpImageQueryLod: return {};
			default: break;
		}
		offset += count;
	}
	if (samples.empty() || glsl == 0 || float_type == 0 || globals_end == 0 ||
	    capabilities_end == 0) {
		return {};
	}
	uint32_t              bound = spirv[3];
	std::vector<uint32_t> globals;
	const auto            op = [](spv::Op opcode, uint32_t words) {
		return (words << spv::WordCountShift) | static_cast<uint32_t>(opcode);
	};
	if (int_type == 0) {
		int_type = bound++;
		globals.insert(globals.end(), {op(spv::OpTypeInt, 4), int_type, 32u, 1u});
	}
	const auto vector = [&](uint32_t component, uint32_t components) {
		if (components == 1) return component;
		auto& id = vectors[(uint64_t {component} << 32u) | components];
		if (id == 0) {
			id = bound++;
			globals.insert(globals.end(), {op(spv::OpTypeVector, 4), id, component, components});
		}
		return id;
	};
	const uint32_t half = bound++, zero = bound++, one = bound++;
	globals.insert(globals.end(),
	               {op(spv::OpConstant, 4), float_type, half, 0x3f000000u, op(spv::OpConstant, 4),
	                int_type, zero, 0u, op(spv::OpConstant, 4), int_type, one, 1u});

	// Before each sample: the coordinates divided by the size of the mip the LOD names (the
	// nearest one, within the image's), the array layer as it is.
	std::unordered_map<size_t, std::vector<uint32_t>> inserted;
	std::unordered_map<size_t, uint32_t>              coordinates;
	for (const auto offset: samples) {
		const auto w = spirv.subspan(offset, spirv[offset] >> spv::WordCountShift);
		// Every sample is rewritten, or none (the copy's samplers are all normalized).
		const auto found = sampled_images.find(w[3]);
		if (found == sampled_images.end()) return {};
		const uint32_t image      = found->second;
		const auto     image_type = result_types.find(image);
		if (image_type == result_types.end()) return {};
		const auto info = images.find(image_type->second);
		if (info == images.end() || info->second.ms || info->second.dim > spv::Dim3D) return {};
		const uint32_t spatial    = info->second.dim == spv::Dim1D   ? 1u
		                            : info->second.dim == spv::Dim2D ? 2u
		                                                             : 3u;
		const uint32_t components = spatial + (info->second.arrayed ? 1u : 0u);
		const auto coordinate_type = result_types.find(w[4]);
		if (coordinate_type == result_types.end() ||
		    coordinate_type->second != vector(float_type, components)) {
			return {};
		}
		const uint32_t lod  = w[6];
		auto&          code = inserted[offset];
		const uint32_t lod_half = bound++, lod_floor = bound++, lod_int = bound++, levels = bound++,
		               last = bound++, level = bound++, size = bound++, size_float = bound++;
		const auto size_type = vector(int_type, components);
		const auto float_vec = vector(float_type, components);
		code.insert(code.end(),
		            {op(spv::OpFAdd, 5), float_type, lod_half, lod, half,
		             op(spv::OpExtInst, 6), float_type, lod_floor, glsl, 8u /* Floor */, lod_half,
		             op(spv::OpConvertFToS, 4), int_type, lod_int, lod_floor,
		             op(spv::OpImageQueryLevels, 4), int_type, levels, image,
		             op(spv::OpISub, 5), int_type, last, levels, one,
		             op(spv::OpExtInst, 8), int_type, level, glsl, 45u /* SClamp */, lod_int, zero,
		             last,
		             op(spv::OpImageQuerySizeLod, 5), size_type, size, image, level,
		             op(spv::OpConvertSToF, 4), float_vec, size_float, size});
		uint32_t normalized = 0;
		if (components == 1) {
			normalized = bound++;
			code.insert(code.end(), {op(spv::OpFDiv, 5), float_type, normalized, w[4], size_float});
		} else {
			std::vector<uint32_t> parts;
			for (uint32_t k = 0; k < components; k++) {
				const uint32_t part = bound++;
				code.insert(code.end(),
				            {op(spv::OpCompositeExtract, 5), float_type, part, w[4], k});
				if (k < spatial) {
					const uint32_t extent = bound++, divided = bound++;
					code.insert(code.end(),
					            {op(spv::OpCompositeExtract, 5), float_type, extent, size_float, k,
					             op(spv::OpFDiv, 5), float_type, divided, part, extent});
					parts.push_back(divided);
				} else {
					parts.push_back(part);
				}
			}
			normalized = bound++;
			code.push_back(op(spv::OpCompositeConstruct, 3 + components));
			code.push_back(float_vec);
			code.push_back(normalized);
			code.insert(code.end(), parts.begin(), parts.end());
		}
		coordinates[offset] = normalized;
	}
	if (coordinates.empty()) {
		return {};
	}
	std::vector<uint32_t> out(spirv.begin(), spirv.begin() + HeaderWords);
	out[3] = bound;
	out.reserve(spirv.size() + globals.size() + 64u * coordinates.size());
	for (size_t offset = HeaderWords; offset < spirv.size();) {
		const uint32_t count = spirv[offset] >> spv::WordCountShift;
		if (offset == globals_end) {
			out.insert(out.end(), globals.begin(), globals.end());
		}
		if (const auto code = inserted.find(offset); code != inserted.end()) {
			out.insert(out.end(), code->second.begin(), code->second.end());
		}
		const size_t first = out.size();
		out.insert(out.end(), spirv.begin() + static_cast<ptrdiff_t>(offset),
		           spirv.begin() + static_cast<ptrdiff_t>(offset + count));
		if (const auto coordinate = coordinates.find(offset); coordinate != coordinates.end()) {
			out[first + 4] = coordinate->second;
		}
		if (!image_query && offset + count == capabilities_end) {
			out.insert(out.end(), {op(spv::OpCapability, 2), spv::CapabilityImageQuery});
		}
		offset += count;
	}
	return out;
}

namespace NormalizedLod {

namespace {

std::mutex                           g_mutex;
std::unordered_map<uint64_t, Lookup> g_programs;

} // namespace

void Register(uint64_t program_id, vk::ShaderModule module, uint64_t variant_id) {
	std::scoped_lock lock {g_mutex};
	g_programs[program_id] = {module, variant_id};
}

void Forget(vk::Device device, uint64_t program_id) {
	vk::ShaderModule module = nullptr;
	{
		std::scoped_lock lock {g_mutex};
		const auto       found = g_programs.find(program_id);
		if (found == g_programs.end()) {
			return;
		}
		module = found->second.module;
		g_programs.erase(found);
	}
	if (module != nullptr) {
		device.destroyShaderModule(module, nullptr);
	}
}

Lookup Find(uint64_t program_id) {
	std::scoped_lock lock {g_mutex};
	const auto       found = g_programs.find(program_id);
	return found != g_programs.end() ? found->second : Lookup {};
}

} // namespace NormalizedLod

} // namespace Libs::Graphics
