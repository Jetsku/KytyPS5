#include "graphics/shader/recompiler/ShaderCalls.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <xxhash.h>

namespace Libs::Graphics::ShaderRecompiler {

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

const char* StageLabel(ShaderType stage) {
	switch (stage) {
		case ShaderType::Compute: return "CS";
		case ShaderType::Pixel: return "PS";
		case ShaderType::Vertex: return "VS";
		default: return "shader";
	}
}

// KYTY_SHADER_CALLS_DUMP=<folder> (diagnostics, default off): each program with calls is written
// there once as calls_<hash>.bin (its code) and calls_<hash>.txt (its disassembly), and each
// expansion as calls_<hash>_<digest>.bin/.txt. The files stay on this machine.
const char* DumpFolder() {
	static const char* folder = [] {
		const char* value = std::getenv("KYTY_SHADER_CALLS_DUMP");
		return value != nullptr && *value != '\0' ? value : nullptr;
	}();
	return folder;
}

void Dump(std::string_view name, std::span<const uint32_t> code) {
	const auto* folder = DumpFolder();
	if (folder == nullptr) return;
	std::error_code error;
	std::filesystem::create_directories(folder, error);
	const auto base = std::filesystem::path(folder) / std::string(name);
	if (std::ofstream bin(std::filesystem::path(base).replace_extension(".bin"), std::ios::binary); bin) {
		bin.write(reinterpret_cast<const char*>(code.data()),
		          static_cast<std::streamsize>(code.size_bytes()));
	}
	Decoder::Program decoded;
	Decoder::DecodeProgram(code, decoded, true);
	if (std::ofstream text(std::filesystem::path(base).replace_extension(".txt")); text) {
		text << Decoder::ProgramToString(decoded);
	}
}

} // namespace

bool EvaluateCallTable(const CallTable& table, std::span<const uint32_t> user_data,
                       uint32_t user_data_base, const Decoder::ShaderCodeReader& read,
                       CallTableView& view, std::string& failure) {
	std::vector<uint32_t> values;
	if (table.user_sgpr < user_data_base || table.user_sgpr - user_data_base >= user_data.size()) {
		failure = fmt::format("s{} is not user data", table.user_sgpr);
		return false;
	}
	const auto first = user_data.subspan(table.user_sgpr - user_data_base);
	values.assign(first.begin(), first.end());
	for (const auto& step: table.steps) {
		if (step.base_index + 1u >= values.size()) {
			failure = "a load base outside the loaded value";
			return false;
		}
		const auto low     = static_cast<uint64_t>(values[step.base_index]);
		const auto high    = static_cast<uint64_t>(values[step.base_index + 1u]);
		auto       address = (step.buffer ? low | ((high & 0xffffu) << 32u) : low | (high << 32u)) &
		               AddressMask;
		address += static_cast<uint64_t>(static_cast<int64_t>(step.offset));
		if (address == 0 || step.dwords == 0 || step.dwords > 16u) {
			failure = fmt::format("a load step from 0x{:x}", address);
			return false;
		}
		values.assign(step.dwords, 0u);
		if (!read(address, values)) {
			failure = fmt::format("unreadable 0x{:x}", address);
			return false;
		}
	}
	if (table.table_index + 4u > values.size()) {
		failure = "no table descriptor";
		return false;
	}
	const auto* vsharp = &values[table.table_index];
	const auto  base   = (static_cast<uint64_t>(vsharp[0]) |
                       (static_cast<uint64_t>(vsharp[1] & 0xffffu) << 32u)) &
	                  AddressMask;
	const auto stride = (vsharp[1] >> 16u) & 0x3fffu;
	const auto bytes  = stride != 0 ? static_cast<uint64_t>(vsharp[2]) * stride : uint64_t {vsharp[2]};
	const auto record_bytes = 1u << std::min(table.record_shift, 31u);
	view.base          = base;
	view.bytes         = std::min(bytes, MaxCallTableBytes) & ~uint64_t {3};
	view.record_bytes  = record_bytes;
	view.target_offset = table.record_offset + table.target_word * 4u;
	if ((base & 3u) != 0 || record_bytes < 8u || (view.target_offset & 3u) != 0) {
		failure = fmt::format("table 0x{:x} with {}-byte records is not usable", base, record_bytes);
		return false;
	}
	return true;
}

bool ReadCallTable(const CallTableView& view, const Decoder::ShaderCodeReader& read,
                   std::vector<uint32_t>& words, std::string& failure) {
	words.resize(view.bytes / sizeof(uint32_t));
	if (!words.empty() && !read(view.base, words)) {
		failure = fmt::format("unreadable table 0x{:x} (0x{:x} bytes)", view.base, view.bytes);
		return false;
	}
	return true;
}

void CallTableTargets(const CallTableView& view, std::span<const uint32_t> words,
                      std::set<uint64_t>& targets) {
	if (view.record_bytes == 0) return;
	const uint64_t bytes = words.size() * sizeof(uint32_t);
	uint64_t       last  = 0;
	for (uint64_t offset = view.target_offset; offset + 8u <= bytes; offset += view.record_bytes) {
		const auto value = static_cast<uint64_t>(words[offset / 4u]) |
		                   (static_cast<uint64_t>(words[offset / 4u + 1u]) << 32u);
		if (value == last || (value & AddressMask) == 0) continue;
		last = value;
		targets.insert(value);
	}
}

bool ShaderCallExpander::Expand(std::span<const uint32_t> code, uint64_t base,
                                const CompileOptions& options, const Decoder::ShaderCodeReader& read,
                                Expanded& expanded) {
	std::scoped_lock lock(m_mutex);
	const auto       key = base ^ (options.shader_hash * 0x9e3779b97f4a7c15ull) ^ code.size();
	auto [entry, inserted] = m_programs.try_emplace(key);
	auto& program          = entry->second;
	if (inserted) {
		Dump(fmt::format("calls_{:016x}", options.shader_hash), code);
		const auto analysis = AnalyzeShaderCalls(code, options);
		if (analysis.calls.empty() || !analysis.failures.empty()) {
			program.usable = false;
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_SHADER_CALLS: {} shader 0x{:016x} is not expanded: {}.\n",
			    StageLabel(options.stage), options.shader_hash,
			    analysis.calls.empty() ? std::string("no call found") : analysis.failures));
		}
		for (const auto& call: analysis.calls) {
			program.calls.push_back({.trace = call});
		}
	}
	if (!program.usable) return false;
	Refresh(program, code, base, options, read);
	if (program.current.code == nullptr) return false;
	expanded = program.current;
	return true;
}

bool ShaderCallExpander::Refresh(Program& program, std::span<const uint32_t> code, uint64_t base,
                                 const CompileOptions& options,
                                 const Decoder::ShaderCodeReader& read) {
	const bool periodic = rescan_interval != 0 && program.dispatches % rescan_interval == 0;
	program.dispatches++;
	bool        changed = program.current.code == nullptr;
	std::string failures;
	const auto  fail = [&](const Call& call, std::string_view why) {
        failures += fmt::format("{}pc 0x{:x}: {}", failures.empty() ? "" : "; ", call.trace.pc, why);
	};
	const auto add = [&](Call& call, uint64_t target) {
		if (call.targets.contains(target)) return;
		if (call.targets.size() >= MaxCallSiteTargets) {
			fail(call, "more targets than the bound");
			return;
		}
		call.targets.insert(target);
		changed = true;
	};
	for (auto& call: program.calls) {
		if (call.trace.user_sgpr != UINT32_MAX) {
			const auto index = call.trace.user_sgpr - options.user_data_base;
			if (index + 2u > options.user_data.size()) {
				fail(call, "user data too short");
				continue;
			}
			const auto target = static_cast<uint64_t>(options.user_data[index]) |
			                    (static_cast<uint64_t>(options.user_data[index + 1u]) << 32u);
			if ((target & AddressMask) != 0) add(call, target);
			continue;
		}
		CallTableView view;
		std::string   why;
		if (!EvaluateCallTable(*call.trace.table, options.user_data, options.user_data_base, read, view,
		                       why)) {
			fail(call, why);
			continue;
		}
		const bool moved = !call.view_known || view != call.view;
		if (moved || periodic) {
			if (!ReadCallTable(view, read, m_table_words, why)) {
				fail(call, why);
				continue;
			}
			const auto digest = XXH3_64bits(m_table_words.data(), m_table_words.size() * sizeof(uint32_t));
			if (moved || digest != call.table_digest) {
				std::set<uint64_t> found;
				CallTableTargets(view, m_table_words, found);
				for (const auto target: found) {
					add(call, target);
				}
			}
			call.view         = view;
			call.view_known   = true;
			call.table_digest = digest;
		}
	}
	if (periodic && !changed) {
		std::vector<uint32_t> words;
		for (const auto& [address, expected]: program.expansion.reads) {
			words.resize(expected.size());
			if (!read(address, words) || words != expected) {
				changed = true;
				break;
			}
		}
	}
	if (!failures.empty() && failures != program.last_failure) {
		program.last_failure = failures;
		Log::WriteToConsoleAndLog(fmt::format("KYTY_SHADER_CALLS: {} shader 0x{:016x}: {}.\n",
		                                      StageLabel(options.stage), options.shader_hash, failures));
	}
	if (!changed) return true;

	std::vector<Decoder::ShaderCallSite> sites;
	size_t                               targets = 0;
	for (const auto& call: program.calls) {
		sites.push_back({.pc         = call.trace.pc,
		                 .source_reg = call.trace.source_reg,
		                 .return_reg = call.trace.return_reg,
		                 .addresses  = {call.targets.begin(), call.targets.end()}});
		targets += call.targets.size();
	}
	Decoder::ShaderCallExpansion expansion;
	std::string                  reason;
	if (!Decoder::ExpandShaderCalls(code, base, sites, read, options.wave_size, expansion, reason)) {
		if (reason != program.last_failure) {
			program.last_failure = reason;
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_SHADER_CALLS: {} shader 0x{:016x} is not expanded: {}.\n",
			    StageLabel(options.stage), options.shader_hash, reason));
		}
		return false;
	}
	const auto digest =
	    XXH3_64bits(expansion.code.data(), expansion.code.size() * sizeof(uint32_t));
	Log::WriteToConsoleAndLog(fmt::format(
	    "KYTY_SHADER_CALLS: {} shader 0x{:016x}: {} call(s), {} target(s), {} function bod{} placed, "
	    "{} words (from {}), expansion 0x{:016x}{}{}.\n",
	    StageLabel(options.stage), options.shader_hash, program.calls.size(), targets,
	    expansion.inlined, expansion.inlined == 1 ? "y" : "ies", expansion.code.size(), code.size(),
	    digest, expansion.dropped.empty() ? "" : fmt::format("; {} left out, first: ", expansion.dropped.size()),
	    expansion.dropped.empty() ? std::string() : expansion.dropped.front()));
	program.current = {.code = std::make_shared<const std::vector<uint32_t>>(std::move(expansion.code)),
	                   .digest = digest};
	m_retained.push_back(program.current.code);
	Dump(fmt::format("calls_{:016x}_{:016x}", options.shader_hash, digest), *program.current.code);
	expansion.code.clear();
	program.expansion = std::move(expansion);
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler
