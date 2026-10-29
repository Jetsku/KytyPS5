#include "graphics/host_gpu/renderer/pipeline/aaMaskCoverage.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/pipeline/pixelRateVariant.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <mutex>
#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <zstd.h>

namespace Libs::Graphics {

namespace {

struct Instruction {
	size_t   offset = 0;
	uint32_t count  = 0;
	spv::Op  opcode = spv::OpNop;
};

[[nodiscard]] bool IsAnnotation(spv::Op op) {
	switch (op) {
		case spv::OpDecorate:
		case spv::OpMemberDecorate:
		case spv::OpDecorationGroup:
		case spv::OpGroupDecorate:
		case spv::OpGroupMemberDecorate:
		case spv::OpDecorateId:
		case spv::OpDecorateString:
		case spv::OpMemberDecorateString: return true;
		default: return false;
	}
}

[[nodiscard]] bool IsTypeSection(spv::Op op) {
	if (op >= spv::OpTypeVoid && op <= spv::OpTypeForwardPointer) return true;
	switch (op) {
		case spv::OpConstantTrue:
		case spv::OpConstantFalse:
		case spv::OpConstant:
		case spv::OpConstantComposite:
		case spv::OpConstantSampler:
		case spv::OpConstantNull:
		case spv::OpSpecConstantTrue:
		case spv::OpSpecConstantFalse:
		case spv::OpSpecConstant:
		case spv::OpSpecConstantComposite:
		case spv::OpSpecConstantOp:
		case spv::OpVariable:
		case spv::OpUndef: return true;
		default: return false;
	}
}

[[nodiscard]] uint64_t NowMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

} // namespace

std::vector<uint32_t> SpirvWithAaMask(std::span<const uint32_t> spirv, uint32_t mask01,
                                      uint32_t mask23) {
	constexpr size_t HeaderWords = 5;
	if (spirv.size() < HeaderWords || spirv[0] != spv::MagicNumber) {
		return {};
	}
	std::vector<Instruction> instructions;
	for (size_t offset = HeaderWords; offset < spirv.size();) {
		const uint32_t count = spirv[offset] >> spv::WordCountShift;
		if (count == 0 || offset + count > spirv.size()) {
			return {};
		}
		instructions.push_back({offset, count, static_cast<spv::Op>(spirv[offset] & spv::OpCodeMask)});
		offset += count;
	}
	const auto words = [&](const Instruction& inst) { return spirv.subspan(inst.offset, inst.count); };

	// The fragment entry point (one), the FragCoord input it may have, and no SampleMask of its
	// own: a shader that exports its coverage is left as it is.
	size_t   entry_index = SIZE_MAX;
	uint32_t entry_fn    = 0;
	uint32_t frag_coord  = 0;
	for (size_t i = 0; i < instructions.size(); i++) {
		const auto& inst = instructions[i];
		const auto  w    = words(inst);
		if (inst.opcode == spv::OpEntryPoint && w.size() >= 4) {
			if (w[1] != spv::ExecutionModelFragment || entry_index != SIZE_MAX) {
				return {};
			}
			entry_index = i;
			entry_fn    = w[2];
		} else if (inst.opcode == spv::OpDecorate && w.size() >= 4 && w[2] == spv::DecorationBuiltIn) {
			if (w[3] == spv::BuiltInSampleMask) return {};
			if (w[3] == spv::BuiltInFragCoord) frag_coord = w[1];
		} else if (inst.opcode == spv::OpMemberDecorate && w.size() >= 5 &&
		           w[3] == spv::DecorationBuiltIn && w[4] == spv::BuiltInSampleMask) {
			return {};
		}
	}
	if (entry_index == SIZE_MAX) {
		return {};
	}
	// Whether the entry point's interface (after its name) lists the FragCoord variable already.
	bool frag_coord_listed = false;
	if (frag_coord != 0) {
		const auto w        = words(instructions[entry_index]);
		size_t     name_end = 3;
		while (name_end < w.size()) {
			if ((w[name_end++] & 0xff000000u) == 0) {
				break; // the word holding the name's terminating zero
			}
		}
		frag_coord_listed =
		    std::find(w.begin() + static_cast<std::ptrdiff_t>(name_end), w.end(), frag_coord) != w.end();
	}
	// Types: a 32-bit uint, and the FragCoord variable's vec4 of 32-bit floats when it has one.
	uint32_t uint_type = 0, float_type = 0, vec4_type = 0;
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpTypeInt && w.size() == 4 && w[2] == 32 && w[3] == 0) {
			uint_type = w[1];
		} else if (inst.opcode == spv::OpTypeFloat && w.size() == 3 && w[2] == 32) {
			float_type = w[1];
		}
	}
	for (const auto& inst: instructions) {
		const auto w = words(inst);
		if (inst.opcode == spv::OpTypeVector && w.size() == 4 && float_type != 0 && w[2] == float_type &&
		    w[3] == 4) {
			vec4_type = w[1];
		}
	}
	if (frag_coord != 0) {
		// Its pointer must be an Input vec4 of floats.
		uint32_t pointer = 0;
		for (const auto& inst: instructions) {
			const auto w = words(inst);
			if (inst.opcode == spv::OpVariable && w.size() >= 4 && w[2] == frag_coord) {
				pointer = w[1];
			}
		}
		bool ok = false;
		for (const auto& inst: instructions) {
			const auto w = words(inst);
			if (inst.opcode == spv::OpTypePointer && w.size() == 4 && w[1] == pointer &&
			    w[2] == spv::StorageClassInput && vec4_type != 0 && w[3] == vec4_type) {
				ok = true;
			}
		}
		if (!ok) {
			return {};
		}
	}
	// Where things go: decorations after the last annotation (before the types without one),
	// declarations before the first function, the code at the top of the entry point's first
	// block, after its variables.
	size_t last_annotation = SIZE_MAX, first_type = SIZE_MAX, first_function = SIZE_MAX, code_at = SIZE_MAX;
	for (size_t i = 0; i < instructions.size(); i++) {
		const auto op = instructions[i].opcode;
		if (IsAnnotation(op)) last_annotation = i;
		if (first_type == SIZE_MAX && IsTypeSection(op)) first_type = i;
		if (first_function == SIZE_MAX && op == spv::OpFunction) first_function = i;
		if (op == spv::OpFunction && code_at == SIZE_MAX && words(instructions[i]).size() >= 3 &&
		    words(instructions[i])[2] == entry_fn) {
			size_t j = i + 1;
			while (j < instructions.size() && instructions[j].opcode == spv::OpFunctionParameter) j++;
			if (j >= instructions.size() || instructions[j].opcode != spv::OpLabel) {
				return {};
			}
			j++;
			while (j < instructions.size() &&
			       (instructions[j].opcode == spv::OpVariable || instructions[j].opcode == spv::OpLine ||
			        instructions[j].opcode == spv::OpNoLine)) {
				j++;
			}
			code_at = j;
		}
	}
	if (first_function == SIZE_MAX || code_at >= instructions.size() ||
	    (last_annotation == SIZE_MAX && first_type == SIZE_MAX)) {
		return {};
	}

	uint32_t   next      = spirv[3];
	const auto id        = [&] { return next++; };
	const bool new_uint  = uint_type == 0;
	const bool new_float = frag_coord == 0 && float_type == 0;
	const bool new_vec4  = frag_coord == 0 && vec4_type == 0;
	if (new_uint) uint_type = id();
	if (new_float) float_type = id();
	if (new_vec4) vec4_type = id();
	const bool     new_frag_coord = frag_coord == 0;
	const uint32_t c0 = id(), c1 = id(), c16 = id(), c_low = id(), c_m01 = id(), c_m23 = id();
	const uint32_t mask_array = id(), out_array = id(), out_uint = id();
	const uint32_t in_vec4 = new_frag_coord ? id() : 0u;
	if (new_frag_coord) frag_coord = id();
	const uint32_t sample_mask = id();

	const auto op = [](std::vector<uint32_t>& out, spv::Op opcode, std::initializer_list<uint32_t> operands) {
		out.push_back((static_cast<uint32_t>(operands.size() + 1u) << spv::WordCountShift) |
		              static_cast<uint32_t>(opcode));
		out.insert(out.end(), operands.begin(), operands.end());
	};
	std::vector<uint32_t> decorations;
	op(decorations, spv::OpDecorate, {sample_mask, spv::DecorationBuiltIn, spv::BuiltInSampleMask});
	if (new_frag_coord) {
		op(decorations, spv::OpDecorate, {frag_coord, spv::DecorationBuiltIn, spv::BuiltInFragCoord});
	}
	std::vector<uint32_t> declarations;
	if (new_uint) op(declarations, spv::OpTypeInt, {uint_type, 32, 0});
	if (new_float) op(declarations, spv::OpTypeFloat, {float_type, 32});
	if (new_vec4) op(declarations, spv::OpTypeVector, {vec4_type, float_type, 4});
	op(declarations, spv::OpConstant, {uint_type, c0, 0});
	op(declarations, spv::OpConstant, {uint_type, c1, 1});
	op(declarations, spv::OpConstant, {uint_type, c16, 16});
	op(declarations, spv::OpConstant, {uint_type, c_low, 0xffffu});
	op(declarations, spv::OpConstant, {uint_type, c_m01, mask01});
	op(declarations, spv::OpConstant, {uint_type, c_m23, mask23});
	op(declarations, spv::OpTypeArray, {mask_array, uint_type, c1});
	op(declarations, spv::OpTypePointer, {out_array, spv::StorageClassOutput, mask_array});
	op(declarations, spv::OpTypePointer, {out_uint, spv::StorageClassOutput, uint_type});
	if (new_frag_coord) {
		op(declarations, spv::OpTypePointer, {in_vec4, spv::StorageClassInput, vec4_type});
		op(declarations, spv::OpVariable, {in_vec4, frag_coord, spv::StorageClassInput});
	}
	op(declarations, spv::OpVariable, {out_array, sample_mask, spv::StorageClassOutput});
	// mask = (y odd ? mask23 : mask01) >> (x odd ? 16 : 0) & 0xffff, as integer arithmetic.
	const uint32_t position = id(), fx = id(), fy = id(), ux = id(), uy = id(), bx = id(), by = id(),
	               diff = id(), scaled = id(), word = id(), shift = id(), shifted = id(), mask = id(),
	               pointer = id();
	std::vector<uint32_t> code;
	op(code, spv::OpLoad, {vec4_type, position, frag_coord});
	op(code, spv::OpCompositeExtract, {float_type, fx, position, 0});
	op(code, spv::OpCompositeExtract, {float_type, fy, position, 1});
	op(code, spv::OpConvertFToU, {uint_type, ux, fx});
	op(code, spv::OpConvertFToU, {uint_type, uy, fy});
	op(code, spv::OpBitwiseAnd, {uint_type, bx, ux, c1});
	op(code, spv::OpBitwiseAnd, {uint_type, by, uy, c1});
	op(code, spv::OpISub, {uint_type, diff, c_m23, c_m01});
	op(code, spv::OpIMul, {uint_type, scaled, diff, by});
	op(code, spv::OpIAdd, {uint_type, word, c_m01, scaled});
	op(code, spv::OpIMul, {uint_type, shift, bx, c16});
	op(code, spv::OpShiftRightLogical, {uint_type, shifted, word, shift});
	op(code, spv::OpBitwiseAnd, {uint_type, mask, shifted, c_low});
	op(code, spv::OpAccessChain, {out_uint, pointer, sample_mask, c0});
	op(code, spv::OpStore, {pointer, mask});

	std::vector<uint32_t> out(spirv.begin(), spirv.begin() + HeaderWords);
	out[3] = next;
	out.reserve(spirv.size() + decorations.size() + declarations.size() + code.size() + 4u);
	for (size_t i = 0; i < instructions.size(); i++) {
		const auto& inst = instructions[i];
		const auto  w    = words(inst);
		if (last_annotation == SIZE_MAX && i == first_type) {
			out.insert(out.end(), decorations.begin(), decorations.end());
		}
		if (i == first_function) {
			out.insert(out.end(), declarations.begin(), declarations.end());
		}
		if (i == code_at) {
			out.insert(out.end(), code.begin(), code.end());
		}
		if (inst.opcode == spv::OpExecutionMode && w.size() >= 3 && w[1] == entry_fn &&
		    w[2] == spv::ExecutionModeEarlyFragmentTests) {
			// Not in the copy: tests before the shader would write depth and stencil where the
			// mask leaves the pixel out (the guest's rasterizer applies the mask before them).
			continue;
		}
		if (i == entry_index) {
			// The interface lists the new variables too (every SPIR-V version), and FragCoord.
			const bool     list_frag_coord = !frag_coord_listed;
			const uint32_t added           = list_frag_coord ? 2u : 1u;
			out.push_back(((inst.count + added) << spv::WordCountShift) | static_cast<uint32_t>(spv::OpEntryPoint));
			out.insert(out.end(), w.begin() + 1, w.end());
			out.push_back(sample_mask);
			if (list_frag_coord) out.push_back(frag_coord);
		} else {
			out.insert(out.end(), w.begin(), w.end());
		}
		if (i == last_annotation) {
			out.insert(out.end(), decorations.begin(), decorations.end());
		}
	}
	return out;
}

std::vector<uint32_t> CoverageOnlyFragmentSpirv(uint32_t mask01, uint32_t mask23) {
	// OpCapability Shader; OpMemoryModel Logical GLSL450; OpEntryPoint Fragment %3 "main";
	// OpExecutionMode %3 OriginUpperLeft; %1 = OpTypeVoid; %2 = OpTypeFunction %1;
	// %3 = OpFunction %1 None %2; %4 = OpLabel; OpReturn; OpFunctionEnd.
	const auto word = [](spv::Op opcode, uint32_t count) {
		return (count << spv::WordCountShift) | static_cast<uint32_t>(opcode);
	};
	const std::vector<uint32_t> minimal = {spv::MagicNumber,
	                                       0x00010000u,
	                                       0u,
	                                       5u,
	                                       0u,
	                                       word(spv::OpCapability, 2),
	                                       spv::CapabilityShader,
	                                       word(spv::OpMemoryModel, 3),
	                                       spv::AddressingModelLogical,
	                                       spv::MemoryModelGLSL450,
	                                       word(spv::OpEntryPoint, 5),
	                                       spv::ExecutionModelFragment,
	                                       3u,
	                                       0x6e69616du,
	                                       0u,
	                                       word(spv::OpExecutionMode, 3),
	                                       3u,
	                                       spv::ExecutionModeOriginUpperLeft,
	                                       word(spv::OpTypeVoid, 2),
	                                       1u,
	                                       word(spv::OpTypeFunction, 3),
	                                       2u,
	                                       1u,
	                                       word(spv::OpFunction, 5),
	                                       1u,
	                                       3u,
	                                       spv::FunctionControlMaskNone,
	                                       2u,
	                                       word(spv::OpLabel, 2),
	                                       4u,
	                                       word(spv::OpReturn, 1),
	                                       word(spv::OpFunctionEnd, 1)};
	return SpirvWithAaMask(minimal, mask01, mask23);
}

namespace AaMaskCoverage {

namespace {

struct Key {
	uint64_t program_id = 0;
	uint32_t mask01     = 0;
	uint32_t mask23     = 0;
	bool     pixel_rate = false;
	bool     operator==(const Key&) const = default;
};
struct KeyHash {
	std::size_t operator()(const Key& key) const noexcept {
		return static_cast<std::size_t>((key.program_id * 0x9e3779b97f4a7c15ull) ^
		                                (uint64_t {key.mask01} << 32u | key.mask23) ^
		                                (key.pixel_rate ? 0x5bd1e995ull : 0ull));
	}
};
enum class State { Queued, Ready, Failed };
struct Entry {
	State            state       = State::Queued;
	vk::ShaderModule module      = nullptr;
	uint64_t         shader_hash = 0;
};
struct Source {
	uint64_t             shader_hash = 0;
	uint32_t             words       = 0;
	std::vector<uint8_t> compressed;
};

class Variants {
public:
	~Variants() { Stop(); }

	void Register(uint64_t program_id, uint64_t shader_hash, std::span<const uint32_t> spirv) {
		Source source;
		source.shader_hash = shader_hash;
		source.words       = static_cast<uint32_t>(spirv.size());
		source.compressed.resize(ZSTD_compressBound(spirv.size_bytes()));
		const auto size = ZSTD_compress(source.compressed.data(), source.compressed.size(), spirv.data(),
		                                spirv.size_bytes(), 1);
		if (ZSTD_isError(size) != 0u) return;
		source.compressed.resize(size);
		source.compressed.shrink_to_fit();
		std::scoped_lock lock(m_mutex);
		m_kept_bytes += size;
		m_sources[program_id] = std::move(source);
	}

	void Forget(vk::Device device, uint64_t program_id) {
		std::vector<vk::ShaderModule> doomed;
		{
			std::scoped_lock lock(m_mutex);
			if (const auto found = m_sources.find(program_id); found != m_sources.end()) {
				m_kept_bytes -= found->second.compressed.size();
				m_sources.erase(found);
			}
			for (auto it = m_entries.begin(); it != m_entries.end();) {
				if (it->first.program_id == program_id && it->second.state != State::Queued) {
					if (it->second.module != nullptr) doomed.push_back(it->second.module);
					it = m_entries.erase(it);
				} else {
					++it;
				}
			}
		}
		for (auto module: doomed) device.destroyShaderModule(module, nullptr);
	}

	vk::ShaderModule Request(vk::Device device, const Key& key) {
		std::scoped_lock lock(m_mutex);
		if (m_stopped) return nullptr;
		auto [it, inserted] = m_entries.try_emplace(key);
		if (!inserted) {
			return it->second.state == State::Ready ? it->second.module : nullptr;
		}
		if (const auto source = m_sources.find(key.program_id); source != m_sources.end()) {
			it->second.shader_hash = source->second.shader_hash;
		} else if (key.program_id != 0) {
			it->second.state = State::Failed; // not kept (compiled before the switch's registration)
			return nullptr;
		}
		m_device = device;
		m_queue.push_back(key);
		if (!m_worker.joinable()) {
			m_worker = std::thread([this] { Run(); });
		}
		m_wake.notify_one();
		return nullptr;
	}

	vk::ShaderModule Lookup(const Key& key) {
		std::scoped_lock lock(m_mutex);
		const auto       found = m_entries.find(key);
		return found != m_entries.end() && found->second.state == State::Ready ? found->second.module : nullptr;
	}

	void Stop() {
		{
			std::scoped_lock lock(m_mutex);
			m_stopped = true;
			m_queue.clear();
		}
		m_wake.notify_all();
		if (m_worker.joinable()) m_worker.join();
	}

	void DestroyAll(vk::Device device) {
		std::scoped_lock lock(m_mutex);
		for (auto& [key, entry]: m_entries) {
			(void)key;
			if (entry.module != nullptr) device.destroyShaderModule(entry.module, nullptr);
			entry.module = nullptr;
			entry.state  = State::Failed;
		}
	}

private:
	void Run() {
		std::unique_lock lock(m_mutex);
		for (;;) {
			m_wake.wait(lock, [this] { return m_stopped || !m_queue.empty(); });
			if (m_stopped) return;
			const auto key = m_queue.front();
			m_queue.pop_front();
			std::vector<uint8_t> compressed;
			uint32_t             words = 0;
			if (key.program_id != 0) {
				const auto source = m_sources.find(key.program_id);
				if (source != m_sources.end()) {
					compressed = source->second.compressed;
					words      = source->second.words;
				}
			}
			const auto device = m_device;
			const auto hash   = m_entries[key].shader_hash;
			lock.unlock();
			const auto module = Make(device, key, hash, compressed, words);
			lock.lock();
			auto& entry  = m_entries[key];
			entry.module = module;
			entry.state  = module != nullptr ? State::Ready : State::Failed;
			if (m_logged < 48u) {
				m_logged++;
				LOGF("AA mask: %s 0x%016" PRIx64 "%s with masks %08x %08x: %s (KYTY_AA_MASK_COVERAGE; %" PRIu64
				     " KiB of pixel SPIR-V kept)\n",
				     key.program_id == 0 ? "draws without a pixel shader," : "pixel shader", hash,
				     key.pixel_rate ? " (EQAA per-pixel copy)" : "", key.mask01, key.mask23,
				     module != nullptr      ? "its copy is ready"
				     : key.program_id == 0 ? "the coverage-only shader failed"
				                           : "no copy (its draws stay as they were)",
				     static_cast<uint64_t>(m_kept_bytes / 1024u));
			}
		}
	}

	vk::ShaderModule Make(vk::Device device, const Key& key, uint64_t hash, const std::vector<uint8_t>& compressed,
	                      uint32_t words) {
		std::vector<uint32_t> variant;
		if (key.program_id == 0) {
			variant = CoverageOnlyFragmentSpirv(key.mask01, key.mask23);
		} else {
			if (compressed.empty() || words == 0) return nullptr;
			std::vector<uint32_t> spirv(words);
			const auto            size =
			    ZSTD_decompress(spirv.data(), spirv.size() * sizeof(uint32_t), compressed.data(), compressed.size());
			if (ZSTD_isError(size) != 0u || size != spirv.size() * sizeof(uint32_t)) return nullptr;
			if (key.pixel_rate) {
				// The EQAA per-pixel copy (sample index 0) the draw runs, then the mask.
				spirv = SpirvWithoutSampleId(spirv);
				if (spirv.empty()) return nullptr;
			}
			variant = SpirvWithAaMask(spirv, key.mask01, key.mask23);
		}
		if (variant.empty() || !Validate(hash, variant)) return nullptr;
		return CompileSPV(variant, device);
	}

	bool Validate(uint64_t hash, const std::vector<uint32_t>& spirv) {
		// The rewrite is mechanical: the first copies are checked, and all of them after a failure.
		{
			std::scoped_lock lock(m_mutex);
			if (m_validated >= 8u && !m_validation_failed) return true;
			m_validated++;
		}
		spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
		std::string          messages;
		tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&,
		                                     const char* message) {
			if (messages.size() < 2048u) {
				messages += message;
				messages += '\n';
			}
		});
		spvtools::ValidatorOptions options;
		options.SetUniformBufferStandardLayout(
		    ShaderRecompiler::Spirv::GetHostBufferRobustness().uniform_dword_loads_return_zero);
		if (tools.Validate(spirv.data(), spirv.size(), options)) return true;
		{
			std::scoped_lock lock(m_mutex);
			m_validation_failed = true;
		}
		LOGF("AA mask: the copy of shader 0x%016" PRIx64 " is not valid SPIR-V; not used:\n%s", hash,
		     messages.c_str());
		return false;
	}

	std::mutex                                  m_mutex;
	std::condition_variable                     m_wake;
	std::unordered_map<uint64_t, Source>        m_sources;
	std::unordered_map<Key, Entry, KeyHash>     m_entries;
	std::deque<Key>                             m_queue;
	vk::Device                                  m_device = nullptr;
	uint64_t                                    m_kept_bytes        = 0;
	uint32_t                                    m_validated         = 0;
	bool                                        m_validation_failed = false;
	uint32_t                                    m_logged            = 0;
	bool                                        m_stopped           = false;
	std::thread                                 m_worker;
};

Variants& Instance() {
	static Variants variants;
	return variants;
}

} // namespace

bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_AA_MASK_COVERAGE");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return on;
}

void Register(uint64_t program_id, uint64_t shader_hash, std::span<const uint32_t> spirv) {
	if (Enabled() && program_id != 0 && !spirv.empty()) Instance().Register(program_id, shader_hash, spirv);
}

void Forget(vk::Device device, uint64_t program_id) {
	if (Enabled() && program_id != 0) Instance().Forget(device, program_id);
}

vk::ShaderModule Request(vk::Device device, uint64_t program_id, bool pixel_rate, uint32_t mask01,
                         uint32_t mask23) {
	if (!Enabled()) return nullptr;
	return Instance().Request(device, Key {program_id, mask01, mask23, pixel_rate && program_id != 0});
}

vk::ShaderModule Lookup(uint64_t program_id, bool pixel_rate, uint32_t mask01, uint32_t mask23) {
	if (!Enabled()) return nullptr;
	return Instance().Lookup(Key {program_id, mask01, mask23, pixel_rate && program_id != 0});
}

void Shutdown(vk::Device device) {
	if (!Enabled()) return;
	Instance().Stop();
	Instance().DestroyAll(device);
}

} // namespace AaMaskCoverage

} // namespace Libs::Graphics
