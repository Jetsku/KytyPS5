#include "graphics/shader/recompiler/ShaderCalls.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <magic_enum.hpp>
#include <set>
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

// KYTY_SHADER_CALLS_TOPK=N (default 4; 0: every target): each call table keeps the N targets most
// of its records name; calls to the others take the skip path. With every callee, an expansion of
// Miles Morales' ray-tracing kernels is millions of SPIR-V words (2 callees: a 12 s driver
// compile, 6: 44 s), so only a few compile in seconds.
size_t MaxTargetsPerTable() {
	static const size_t cap = [] {
		const char* value = std::getenv("KYTY_SHADER_CALLS_TOPK");
		return value != nullptr ? static_cast<size_t>(std::strtoull(value, nullptr, 0)) : size_t {4};
	}();
	return cap;
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

// Read on each expander construction (one per pipeline cache, and in tests).
bool StickyExpansionsDefault() {
	const char* value = std::getenv("KYTY_SHADER_CALLS_STICKY");
	return value == nullptr || std::strcmp(value, "0") != 0;
}

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

void ShaderCallExpander::DropTarget(uint64_t address, std::string_view why) {
	std::scoped_lock lock(m_mutex);
	if (!m_dropped.insert(address & AddressMask).second) return;
	Log::WriteToConsoleAndLog(fmt::format(
	    "KYTY_SHADER_CALLS: callee 0x{:x} left out of the expansions ({}); its calls are skipped.\n",
	    address & AddressMask, why));
}

bool ShaderCallExpander::Refresh(Program& program, std::span<const uint32_t> code, uint64_t base,
                                 const CompileOptions& options,
                                 const Decoder::ShaderCodeReader& read) {
	const auto  now = program.dispatches++;
	const auto  due = [&](uint32_t since) { return rescan_interval != 0 && now - since >= rescan_interval; };
	std::string failures;
	const auto  fail = [&](const Call& call, std::string_view why) {
        failures += fmt::format("{}pc 0x{:x}: {}", failures.empty() ? "" : "; ", call.trace.pc, why);
	};
	// The targets of each call for this dispatch.
	std::vector<Decoder::ShaderCallSite> sites;
	size_t                               targets = 0;
	uint64_t                             key     = 0x5bd1e995u;
	for (auto& call: program.calls) {
		auto& site = sites.emplace_back(Decoder::ShaderCallSite {.pc         = call.trace.pc,
		                                                         .source_reg = call.trace.source_reg,
		                                                         .return_reg = call.trace.return_reg});
		if (call.trace.null_callee) {
			// KYTY_RAY_HIT_CALLS_RETURN: no target; the call takes the skip path (returns at once).
		} else if (call.trace.user_sgpr != UINT32_MAX) {
			const auto index = call.trace.user_sgpr - options.user_data_base;
			if (index + 2u > options.user_data.size()) {
				fail(call, "user data too short");
			} else {
				const auto target = static_cast<uint64_t>(options.user_data[index]) |
				                    (static_cast<uint64_t>(options.user_data[index + 1u]) << 32u);
				if ((target & AddressMask) != 0) site.addresses.push_back(target);
			}
		} else {
			CallTableView view;
			std::string   why;
			if (!EvaluateCallTable(*call.trace.table, options.user_data, options.user_data_base, read,
			                       view, why)) {
				fail(call, why);
			} else {
				auto [state, fresh] = call.tables.try_emplace(
				    std::tuple {view.base, view.bytes, view.record_bytes, view.target_offset});
				if (fresh || due(state->second.scanned_at)) {
					if (!ReadCallTable(view, read, m_table_words, why)) {
						fail(call, why);
					} else {
						const auto digest =
						    XXH3_64bits(m_table_words.data(), m_table_words.size() * sizeof(uint32_t));
						if (fresh || digest != state->second.digest) {
							std::set<uint64_t> found;
							CallTableTargets(view, m_table_words, found);
							if (found.size() > MaxCallSiteTargets) {
								fail(call, fmt::format("{} targets, more than the bound", found.size()));
								found.clear();
							}
							state->second.targets.assign(found.begin(), found.end());
							if (const auto cap = MaxTargetsPerTable();
							    cap != 0 && state->second.targets.size() > cap) {
								// KYTY_SHADER_CALLS_TOPK: the targets most
								// records name; the others take the skip path.
								std::map<uint64_t, uint32_t> records;
								const uint64_t bytes = m_table_words.size() * sizeof(uint32_t);
								for (uint64_t offset = view.target_offset; offset + 8u <= bytes;
								     offset += view.record_bytes) {
									const auto value =
									    static_cast<uint64_t>(m_table_words[offset / 4u]) |
									    (static_cast<uint64_t>(m_table_words[offset / 4u + 1u]) << 32u);
									if ((value & AddressMask) != 0) records[value]++;
								}
								auto& kept = state->second.targets;
								std::ranges::sort(kept, [&](uint64_t x, uint64_t y) {
									return records[x] != records[y] ? records[x] > records[y] : x < y;
								});
								kept.resize(cap);
								std::ranges::sort(kept);
							}
						}
						state->second.digest     = digest;
						state->second.scanned_at = now;
					}
				}
				site.addresses = state->second.targets;
			}
		}
		std::erase_if(site.addresses, [&](uint64_t address) {
			return m_dropped.contains(address & AddressMask);
		});
		targets += site.addresses.size();
		key = XXH3_64bits_withSeed(site.addresses.data(), site.addresses.size() * sizeof(uint64_t), key);
	}
	if (!failures.empty() && failures != program.last_failure) {
		program.last_failure = failures;
		Log::WriteToConsoleAndLog(fmt::format("KYTY_SHADER_CALLS: {} shader 0x{:016x}: {}.\n",
		                                      StageLabel(options.stage), options.shader_hash, failures));
	}
	// KYTY_SHADER_CALLS_STICKY (default on): once every call reaches two targets, the program keeps
	// that expansion for every table (targets outside it take the skip path), so tables that change
	// do not compile new pipelines (a ray-tracing kernel's takes the driver 10 s and more).
	if (program.pinned != 0) {
		const auto pinned = program.expansions.find(program.pinned);
		const bool usable = pinned != program.expansions.end() &&
		                    std::ranges::none_of(*pinned->second.expanded.bodies, [&](const auto& body) {
			                    return m_dropped.contains(body.address & AddressMask);
		                    });
		if (usable) {
			key = program.pinned;
		} else {
			program.pinned = 0;
		}
	}
	const bool full = MaxTargetsPerTable() != 0 && std::ranges::all_of(sites, [](const auto& site) {
		return site.addresses.size() >= std::min<size_t>(MaxTargetsPerTable(), 2);
	});
	if (program.pinned == 0 && full && sticky && program.expansions.contains(key)) {
		program.pinned = key;
	}
	// An expansion for these targets, while the callee code it placed reads the same.
	if (auto cached = program.expansions.find(key); cached != program.expansions.end()) {
		auto& entry = cached->second;
		bool  valid = true;
		if (due(entry.checked_at)) {
			std::vector<uint32_t> words;
			for (const auto& [address, expected]: entry.expansion.reads) {
				words.resize(expected.size());
				if (!read(address, words) || words != expected) {
					valid = false;
					break;
				}
			}
			entry.checked_at = now;
		}
		if (valid) {
			entry.used_at   = now;
			program.current = entry.expanded;
			return true;
		}
		program.expansions.erase(cached);
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
		program.current = {};
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
	Expanded result {.code = std::make_shared<const std::vector<uint32_t>>(std::move(expansion.code)),
	                 .digest = digest,
	                 .bodies = std::make_shared<const std::vector<Decoder::ShaderCallBody>>(expansion.bodies)};
	m_retained.push_back(result.code);
	Dump(fmt::format("calls_{:016x}_{:016x}", options.shader_hash, digest), *result.code);
	expansion.code.clear();
	if (program.expansions.size() >= MaxExpansionsPerProgram) {
		const auto oldest = std::ranges::min_element(program.expansions, {}, [](const auto& entry) {
			return entry.second.used_at;
		});
		program.expansions.erase(oldest);
	}
	program.expansions[key] = {.expanded   = result,
	                           .expansion  = std::move(expansion),
	                           .checked_at = now,
	                           .used_at    = now};
	program.current = result;
	if (program.pinned == 0 && full && sticky) {
		program.pinned = key;
	}
	return true;
}

bool CallPartitionsEnabled() {
	static const bool value = [] {
		const char* text = std::getenv("KYTY_CALL_PARTITIONS");
		return text != nullptr && *text != '\0' && std::strcmp(text, "0") != 0;
	}();
	return value;
}

uint32_t CallPartitionMinTargets() {
	static const uint32_t value = [] {
		const char* text = std::getenv("KYTY_CALL_PARTITION_MIN");
		return text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 0)) : 16u;
	}();
	return value;
}

uint32_t CallPartitionMaxParts() {
	static const uint32_t value = [] {
		const char* text = std::getenv("KYTY_CALL_PARTITION_MAX");
		return std::max(text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 0)) : 64u, 1u);
	}();
	return value;
}

bool CallPartitionMemoryEnabled() {
	static const bool value = [] {
		const char* text = std::getenv("KYTY_CALL_PARTITION_MEMORY");
		return text != nullptr && *text != '\0' && std::strcmp(text, "0") != 0;
	}();
	return value;
}

void ShaderCallExpander::SetPartitionMemory(const std::string& path) {
	if (!CallPartitionMemoryEnabled() || path.empty()) return;
	std::scoped_lock lock(m_mutex);
	if (!m_memory_path.empty()) return;
	m_memory_path = path;
	std::ifstream file(path);
	std::string   line;
	while (std::getline(file, line)) {
		char*      end    = nullptr;
		const auto hash   = std::strtoull(line.c_str(), &end, 16);
		const auto target = std::strtoull(end, nullptr, 16);
		if (hash != 0 && target != 0) m_memory.emplace(hash, target);
	}
	if (!m_memory.empty()) {
		Log::WriteToConsoleAndLog(fmt::format("KYTY_CALL_PARTITION_MEMORY: {} remembered targets from {}.\n",
		                                      m_memory.size(), path));
	}
}

// After usa-ratt/KytyPS5-GT7 PartitionUnsafe (round 99): an instruction outside the call that
// would see lanes a partition dropped (they run in another dispatch of the shader).
std::string CallPartitionUnsafe(std::span<const uint32_t> code, uint32_t call_pc) {
	Decoder::Program program;
	Decoder::DecodeProgram(code, program, true);
	using Decoder::Opcode;
	// The first pc a dropped lane could be missing at: the start of the outermost loop around the
	// call (a backward branch after it to a target at or before it), else the call itself.
	uint32_t safe_before = call_pc == UINT32_MAX ? 0u : call_pc;
	if (call_pc != UINT32_MAX) {
		for (const auto& inst: program.instructions) {
			if (Decoder::IsDirectBranch(inst.opcode) && inst.pc > call_pc && inst.branch_target <= call_pc) {
				safe_before = std::min(safe_before, inst.branch_target);
			}
		}
	}
	// LDS reads from the loop on are allowed while nothing writes LDS there (other than a lane's own
	// ADDTID slot): every lane wrote what they read before any lane could leave (PPSA03671's
	// material kernels gather a tile's data that way).
	const auto is_lds_read = [](const Decoder::Instruction& inst) {
		return magic_enum::enum_name(inst.opcode).starts_with("DS_READ");
	};
	bool lds_writes_late = false;
	for (const auto& inst: program.instructions) {
		lds_writes_late |= inst.pc >= safe_before && inst.family == Decoder::Family::DS &&
		                   inst.opcode != Opcode::DS_WRITE_ADDTID_B32 && !is_lds_read(inst);
	}
	for (const auto& inst: program.instructions) {
		if (inst.pc < safe_before) continue;
		const char* why = nullptr;
		if (inst.src0.dpp) {
			why = "DPP reads other lanes";
		} else if (inst.gds) {
			why = "GDS is shared between dispatches";
		} else if (inst.family == Decoder::Family::DS && inst.opcode != Opcode::DS_READ_ADDTID_B32 &&
		           inst.opcode != Opcode::DS_WRITE_ADDTID_B32 && (lds_writes_late || !is_lds_read(inst))) {
			why = "LDS shares data between lanes";
		} else {
			switch (inst.opcode) {
				case Opcode::S_BARRIER: why = "a barrier waits for the other lanes"; break;
				case Opcode::S_SENDMSG: why = "a message"; break;
				// V_READLANE_B32 / V_WRITELANE_B32 are allowed (unlike the fork): compilers spill SGPRs
				// into fixed lanes of a VGPR with them, and here a dropped lane keeps running with
				// EXEC off, so V_WRITELANE (which ignores EXEC) still stores into it.
				case Opcode::V_PERMLANE16_B32:
				case Opcode::V_PERMLANEX16_B32: why = "a lane access reads other lanes"; break;
				case Opcode::IMAGE_GET_LOD: why = "an implicit level of detail reads other lanes"; break;
				case Opcode::IMAGE_GATHER4:
				case Opcode::IMAGE_SAMPLE:
					if ((inst.image_sample_flags &
					     (Decoder::ImageSampleFlagLod | Decoder::ImageSampleFlagLevelZero |
					      Decoder::ImageSampleFlagDerivative | Decoder::ImageSampleFlagCd)) == 0) {
						why = "an implicit level of detail reads other lanes";
					}
					break;
				default:
					if (magic_enum::enum_name(inst.opcode).find("ATOMIC") != std::string_view::npos) {
						why = "an atomic would run once per partition";
					}
					break;
			}
		}
		if (why != nullptr) return fmt::format("{} (pc 0x{:x})", why, inst.pc);
	}
	return {};
}

bool ShaderCallExpander::Partition(std::span<const uint32_t> code, uint64_t base,
                                   const CompileOptions& options, const Decoder::ShaderCodeReader& read,
                                   CallPartitionRun& run) {
	if ((!CallPartitionsEnabled() && !learn) || options.stage != ShaderType::Compute) return false;
	std::scoped_lock lock(m_mutex);
	const auto       key = base ^ (options.shader_hash * 0x9e3779b97f4a7c15ull) ^ code.size();
	auto [entry, inserted] = m_programs.try_emplace(key);
	auto& program          = entry->second;
	if (inserted) {
		const auto analysis = AnalyzeShaderCalls(code, options);
		if (analysis.calls.empty() || !analysis.failures.empty()) {
			program.usable = false;
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_SHADER_CALLS: {} shader 0x{:016x} is not expanded: {}.\n", StageLabel(options.stage),
			    options.shader_hash, analysis.calls.empty() ? std::string("no call found") : analysis.failures));
		}
		for (const auto& call: analysis.calls) {
			program.calls.push_back({.trace = call});
		}
	}
	if (!program.usable || program.partition_state < 0) return false;
	if (program.partition_state == 0) {
		std::string why;
		if (program.calls.size() != 1u) {
			why = fmt::format("{} calls", program.calls.size());
		} else if (program.calls.front().trace.user_sgpr != UINT32_MAX || !program.calls.front().trace.table) {
			why = "the call's target is user data, not a table";
		}
		// KYTY_CALL_LEARN: one table call, whether or not its program can be partitioned.
		const bool learnable = why.empty() && learn;
		if (why.empty()) {
			why = CallPartitionsEnabled() ? CallPartitionUnsafe(code, program.calls.front().trace.pc)
			                              : std::string("partitions are off");
		}
		program.partition_state = why.empty() ? 1 : learnable ? 2 : -1;
		if (CallPartitionsEnabled()) {
			Log::WriteToConsoleAndLog(
			    why.empty() ? fmt::format("KYTY_CALL_PARTITIONS: CS shader 0x{:016x} can be partitioned.\n",
			                              options.shader_hash)
			                : fmt::format("KYTY_CALL_PARTITIONS: CS shader 0x{:016x} is not partitioned: {}.\n",
			                              options.shader_hash, why));
		}
		if (program.partition_state < 0) return false;
	}
	if (program.partition_state == 2) {
		return Learn(program, key, code, base, options, read, run);
	}
	// The table's targets (read again every rescan_interval dispatches, as Refresh does).
	auto&         call = program.calls.front();
	const auto    now  = program.dispatches++;
	CallTableView view;
	std::string   why;
	if (!EvaluateCallTable(*call.trace.table, options.user_data, options.user_data_base, read, view, why)) {
		return false;
	}
	auto [state, fresh] =
	    call.tables.try_emplace(std::tuple {view.base, view.bytes, view.record_bytes, view.target_offset});
	if (fresh || (rescan_interval != 0 && now - state->second.scanned_at >= rescan_interval)) {
		if (!ReadCallTable(view, read, m_table_words, why)) return false;
		const auto digest = XXH3_64bits(m_table_words.data(), m_table_words.size() * sizeof(uint32_t));
		if (fresh || digest != state->second.digest) {
			std::set<uint64_t> found;
			CallTableTargets(view, m_table_words, found);
			state->second.targets.assign(found.begin(), found.end());
		}
		state->second.digest     = digest;
		state->second.scanned_at = now;
	}
	const auto& targets = state->second.targets;
	if (targets.size() < CallPartitionMinTargets()) return false;
	program.partition_targets = targets;
	program.shader_hash       = options.shader_hash;
	// KYTY_CALL_PARTITION_MEMORY: the targets this shader reported in earlier starts.
	if (!program.memory_seeded && !m_memory.empty()) {
		program.memory_seeded = true;
		uint32_t seeded       = 0;
		for (auto it = m_memory.lower_bound({options.shader_hash, 0}); it != m_memory.end() && it->first == options.shader_hash;
		     ++it) {
			if (std::ranges::binary_search(targets, it->second) &&
			    std::ranges::find(program.partition_known, it->second) == program.partition_known.end()) {
				program.partition_known.push_back(it->second);
				seeded++;
			}
		}
		if (seeded != 0) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_CALL_PARTITION_MEMORY: CS shader 0x{:016x}: {} remembered targets known (of {} in the table).\n",
			    options.shader_hash, seeded, targets.size()));
		}
	}
	run.program               = key;
	run.shader_hash           = options.shader_hash;
	run.parts.clear();
	const auto make_part = [&](uint64_t target) -> const CallPartitionPart* {
		if (const auto found = program.partition_parts.find(target); found != program.partition_parts.end()) {
			return found->second.code != nullptr ? &found->second : nullptr;
		}
		Decoder::ShaderCallSite site {
		    .pc = call.trace.pc, .source_reg = call.trace.source_reg, .return_reg = call.trace.return_reg};
		site.partition = true;
		if (target != 0) site.addresses.push_back(target);
		Decoder::ShaderCallExpansion expansion;
		std::string                  reason;
		auto&                        part = program.partition_parts[target];
		part.target                       = target;
		if (!Decoder::ExpandShaderCalls(code, base, std::span(&site, 1u), read, options.wave_size, expansion,
		                                reason) ||
		    (target != 0 && expansion.inlined == 0)) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_CALL_PARTITIONS: CS shader 0x{:016x}: no partition for target 0x{:x}: {}.\n",
			    options.shader_hash, target & AddressMask,
			    !reason.empty() ? reason
			                    : (expansion.dropped.empty() ? std::string("not placed") : expansion.dropped.front())));
			return nullptr;
		}
		part.digest = XXH3_64bits(expansion.code.data(), expansion.code.size() * sizeof(uint32_t)) | 1u;
		part.code   = std::make_shared<const std::vector<uint32_t>>(std::move(expansion.code));
		m_retained.push_back(part.code);
		Dump(fmt::format("part_{:016x}_{:x}", options.shader_hash, target & AddressMask), *part.code);
		return &part;
	};
	for (const auto target: program.partition_known) {
		if (run.parts.size() >= CallPartitionMaxParts()) break;
		if (!std::ranges::binary_search(targets, target) || m_dropped.contains(target & AddressMask)) continue;
		if (const auto* part = make_part(target); part != nullptr) run.parts.push_back(*part);
	}
	if (run.parts.empty()) {
		if (const auto* part = make_part(0); part != nullptr) run.parts.push_back(*part);
	}
	return !run.parts.empty();
}

bool CallLearnEnabled() {
	const char* text = std::getenv("KYTY_CALL_LEARN");
	return text != nullptr && *text != '\0' && std::strcmp(text, "0") != 0;
}

uint32_t CallLearnVariants() {
	const char* text = std::getenv("KYTY_CALL_VARIANTS");
	return std::max(text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 0)) : 32u, 1u);
}

bool ShaderCallExpander::Learn(Program& program, uint64_t key, std::span<const uint32_t> code, uint64_t base,
                               const CompileOptions& options, const Decoder::ShaderCodeReader& read,
                               CallPartitionRun& run) {
	// The table's targets (read again every rescan_interval dispatches, as Refresh does).
	auto&         call = program.calls.front();
	const auto    now  = program.dispatches++;
	CallTableView view;
	std::string   why;
	if (!EvaluateCallTable(*call.trace.table, options.user_data, options.user_data_base, read, view, why)) {
		return false;
	}
	auto [state, fresh] =
	    call.tables.try_emplace(std::tuple {view.base, view.bytes, view.record_bytes, view.target_offset});
	if (fresh || (rescan_interval != 0 && now - state->second.scanned_at >= rescan_interval)) {
		if (!ReadCallTable(view, read, m_table_words, why)) return false;
		const auto digest = XXH3_64bits(m_table_words.data(), m_table_words.size() * sizeof(uint32_t));
		if (fresh || digest != state->second.digest) {
			std::set<uint64_t> found;
			CallTableTargets(view, m_table_words, found);
			state->second.targets.assign(found.begin(), found.end());
		}
		state->second.digest     = digest;
		state->second.scanned_at = now;
	}
	const auto& targets = state->second.targets;
	// A table an expansion places whole (or with its TOPK targets) expands as before.
	const size_t over = learn_over != 0                ? learn_over
	                    : MaxTargetsPerTable() != 0 ? std::min(MaxTargetsPerTable(), MaxCallSiteTargets)
	                                                : MaxCallSiteTargets;
	if (targets.size() <= over) return false;
	program.partition_targets = targets;
	program.shader_hash       = options.shader_hash;
	// KYTY_CALL_PARTITION_MEMORY: the targets this shader reported in earlier starts.
	if (!program.memory_seeded && !m_memory.empty()) {
		program.memory_seeded = true;
		for (auto it = m_memory.lower_bound({options.shader_hash, 0});
		     it != m_memory.end() && it->first == options.shader_hash; ++it) {
			if (std::ranges::binary_search(targets, it->second) &&
			    std::ranges::find(program.partition_known, it->second) == program.partition_known.end()) {
				program.partition_known.push_back(it->second);
			}
		}
	}
	// The reported targets the table still holds, in the order they were reported.
	std::vector<uint64_t> known;
	for (const auto target: program.partition_known) {
		if (known.size() >= learn_variants) break;
		if (std::ranges::binary_search(targets, target) && !m_dropped.contains(target & AddressMask)) {
			known.push_back(target);
		}
	}
	std::ranges::sort(known);
	const auto set_key = XXH3_64bits_withSeed(known.data(), known.size() * sizeof(uint64_t), 0x1ea4u) | 1u;
	if (set_key != program.learn_key) {
		program.learn_key  = set_key;
		program.learn_part = {};
		Decoder::ShaderCallSite site {
		    .pc = call.trace.pc, .source_reg = call.trace.source_reg, .return_reg = call.trace.return_reg};
		site.addresses     = known;
		site.report_misses = true;
		Decoder::ShaderCallExpansion expansion;
		std::string                  reason;
		if (!Decoder::ExpandShaderCalls(code, base, std::span(&site, 1u), read, options.wave_size, expansion,
		                                reason)) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_CALL_LEARN: CS shader 0x{:016x}: no expansion with {} learned target(s): {}.\n",
			    options.shader_hash, known.size(), reason));
			return false;
		}
		auto& part  = program.learn_part;
		part.target = known.empty() ? 0 : known.front(); // 0: nothing learned yet, read back every dispatch
		part.digest = XXH3_64bits(expansion.code.data(), expansion.code.size() * sizeof(uint32_t)) | 1u;
		part.code   = std::make_shared<const std::vector<uint32_t>>(std::move(expansion.code));
		m_retained.push_back(part.code);
		Dump(fmt::format("learn_{:016x}_{:016x}", options.shader_hash, part.digest), *part.code);
		Log::WriteToConsoleAndLog(fmt::format(
		    "KYTY_CALL_LEARN: CS shader 0x{:016x}: expansion with {} learned target(s) ({} placed) of {} in "
		    "the table, {} words.\n",
		    options.shader_hash, known.size(), expansion.inlined, targets.size(), part.code->size()));
	}
	if (program.learn_part.code == nullptr) return false;
	run.program     = key;
	run.shader_hash = options.shader_hash;
	run.parts.assign(1u, program.learn_part);
	return true;
}

void ShaderCallExpander::ReportPartitionKeys(uint64_t program_key, std::span<const uint32_t> keys) {
	std::scoped_lock lock(m_mutex);
	const auto       found = m_programs.find(program_key);
	if (found == m_programs.end()) return;
	auto&              program = found->second;
	std::set<uint32_t> reported(keys.begin(), keys.end());
	reported.erase(0u);
	for (const auto target: program.partition_targets) {
		if (!reported.contains(static_cast<uint32_t>(target) | 1u)) continue;
		if (std::ranges::find(program.partition_known, target) != program.partition_known.end()) continue;
		program.partition_known.push_back(target);
		if (!m_memory_path.empty() && program.shader_hash != 0 &&
		    m_memory.emplace(program.shader_hash, target).second) {
			if (std::ofstream file(m_memory_path, std::ios::app); file) {
				file << fmt::format("{:016x} {:x}\n", program.shader_hash, target);
			}
		}
		Log::WriteToConsoleAndLog(fmt::format(
		    "KYTY_CALL_PARTITIONS: program 0x{:016x}: target 0x{:x} reported ({} known of {} in the table).\n",
		    program_key, target & AddressMask, program.partition_known.size(), program.partition_targets.size()));
	}
}

} // namespace Libs::Graphics::ShaderRecompiler
