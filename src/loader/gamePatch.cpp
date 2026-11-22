#include "loader/gamePatch.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/virtualMemory.h"
#include "kernel/memory.h"
#include "loader/elf.h"
#include "loader/gamePatchFilter.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace Loader::GamePatch {

namespace {

constexpr uint64_t kPageSize = 0x4000;

struct Write {
	uint64_t             source_address = 0;
	uint64_t             target_address = 0;
	std::vector<uint8_t> off;
	std::vector<uint8_t> on;
};

struct Plan {
	std::string              title_id;
	std::string              version;
	std::string              process;
	std::vector<Write>       writes;
	std::vector<std::string> mod_names;
	std::vector<uint64_t>    cave_pages;
};

using Json = nlohmann::json;

std::unique_ptr<Plan> g_pending_plan;
std::vector<uint64_t> g_applied_cave_pages;

bool Fail(std::string* error, std::string message) {
	*error = std::move(message);
	return false;
}

bool ParseBytes(const std::string& text, std::vector<uint8_t>* bytes) {
	if (text.empty() || (text.size() % 2) != 0) {
		return false;
	}

	bytes->resize(text.size() / 2);
	for (size_t index = 0; index < bytes->size(); index++) {
		(*bytes)[index] = static_cast<uint8_t>(std::stoul(text.substr(index * 2, 2), nullptr, 16));
	}
	return true;
}

bool LoadPlan(const std::filesystem::path& path, Plan* plan, std::string* error) {
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		return Fail(error, "could not read cheat file");
	}

	const auto root = Json::parse(file, nullptr, false);
	if (!root.is_object() || !root.contains("id") || !root.contains("version") ||
	    !root.contains("process") || !root.contains("mods")) {
		return Fail(error, "expected a GoldHEN-style mods JSON file");
	}
	plan->title_id = root["id"].get<std::string>();
	plan->version  = root["version"].get<std::string>();
	plan->process  = root["process"].get<std::string>();

	const auto* disable_env = std::getenv(kDisableEnv);
	const auto  patterns    = ParseDisableList(disable_env != nullptr ? disable_env : "");
	std::vector<ModEntry> entries;
	for (const auto& mod: root["mods"]) {
		ModEntry entry;
		if (mod.contains("name") && mod["name"].is_string()) {
			entry.name = mod["name"].get<std::string>();
		}
		entry.enabled = mod.value("enabled", true);
		entries.push_back(std::move(entry));
	}
	const auto selection = SelectMods(entries, patterns);
	for (size_t index = 0; index < entries.size(); index++) {
		const char* reason = "applying";
		std::string detail;
		if (selection.decisions[index] == ModDecision::SkipDisabledInFile) {
			reason = "skipping";
			detail = " (disabled in the patch file)";
		} else if (selection.decisions[index] == ModDecision::SkipDisabledByEnv) {
			reason = "skipping";
			detail = fmt::format(" ({} matched \"{}\")", kDisableEnv,
			                     patterns[selection.pattern_of_mod[index]]);
		}
		Log::WriteToConsoleAndLog(fmt::format("Game patch: {} mod {} \"{}\"{}\n", reason,
		                                      index + 1, entries[index].name, detail));
	}
	for (size_t index = 0; index < patterns.size(); index++) {
		if (!selection.pattern_matched[index]) {
			Log::WriteToConsoleAndLog(fmt::format("Game patch: {} pattern \"{}\" matched no mod\n",
			                                      kDisableEnv, patterns[index]));
		}
	}

	size_t mod_index = 0;
	for (const auto& mod: root["mods"]) {
		if (selection.decisions[mod_index++] != ModDecision::Apply) {
			continue;
		}
		const auto name = mod["name"].get<std::string>();
		plan->mod_names.push_back(name);
		for (const auto& entry: mod["memory"]) {
			Write write;
			write.source_address = std::stoull(entry["offset"].get<std::string>(), nullptr, 16);
			if (!ParseBytes(entry["off"].get<std::string>(), &write.off) ||
			    !ParseBytes(entry["on"].get<std::string>(), &write.on) ||
			    write.off.size() != write.on.size()) {
				return Fail(error, "invalid mod memory bytes");
			}
			plan->writes.push_back(std::move(write));
		}
	}
	return true;
}

bool ValidateGame(const Plan& plan, const Program* main_program, std::string* error) {
	if (main_program == nullptr || main_program->base_vaddr == 0) {
		return Fail(error, "main executable is not loaded");
	}
	std::string title_id;
	std::string version;
	if (!SystemContentParamSfoGetString("TITLE_ID", &title_id) ||
	    !SystemContentParamSfoGetString("APP_VER", &version) ||
	    !Common::EqualNoCase(main_program->file_name.filename().string(), plan.process) ||
	    !Common::EqualNoCase(title_id, plan.title_id) || version != plan.version) {
		return Fail(error, "cheat file does not match the loaded game");
	}
	return true;
}

void Translate(const Program& program, uint64_t source_base, uint64_t source_address,
               uint64_t* target_address) {
	if (source_address >= source_base) {
		*target_address = program.base_vaddr + source_address - source_base;
		return;
	}
	*target_address = program.base_vaddr - (source_base - source_address);
}

bool IsInsideProgram(const Program& program, uint64_t address, size_t size) {
	return size != 0 && address >= program.base_vaddr &&
	       address + size <= program.base_vaddr + program.mapped_size;
}

bool IsZero(const std::vector<uint8_t>& bytes) {
	return std::all_of(bytes.begin(), bytes.end(), [](uint8_t byte) { return byte == 0; });
}

bool TranslateAndValidate(Plan* plan, const Program& program, uint64_t source_base) {
	for (auto& write: plan->writes) {
		Translate(program, source_base, write.source_address, &write.target_address);
		if (!IsZero(write.off) &&
		    (!IsInsideProgram(program, write.target_address, write.off.size()) ||
		     std::memcmp(reinterpret_cast<const void*>(write.target_address), write.off.data(),
		                 write.off.size()) != 0)) {
			return false;
		}
	}
	return true;
}

bool Matches(Plan* plan, const Program& program, uint64_t* source_base) {
	const auto anchor = std::find_if(plan->writes.begin(), plan->writes.end(),
	                                 [](const Write& write) { return !IsZero(write.off); });
	if (anchor == plan->writes.end() || program.mapped_size < anchor->off.size()) {
		return false;
	}

	const auto* ehdr = program.elf->GetEhdr();
	const auto* phdr = program.elf->GetPhdr();
	for (Elf64_Half index = 0; index < ehdr->e_phnum; index++) {
		const auto& segment = phdr[index];
		if ((segment.p_type != PT_LOAD && segment.p_type != PT_OS_RELRO) ||
		    segment.p_filesz < anchor->off.size()) {
			continue;
		}

		const auto* begin = reinterpret_cast<const uint8_t*>(program.base_vaddr + segment.p_vaddr);
		const auto* end   = begin + segment.p_filesz;
		for (auto* current = begin; current < end;) {
			const auto* found = std::search(current, end, anchor->off.begin(), anchor->off.end());
			if (found == end) {
				break;
			}
			const auto rva = static_cast<uint64_t>(found - begin) + segment.p_vaddr;
			if (anchor->source_address >= rva) {
				*source_base = anchor->source_address - rva;
				if (TranslateAndValidate(plan, program, *source_base)) {
					return true;
				}
			}
			current = found + 1;
		}
	}
	return false;
}

void ReleaseCaves(std::vector<uint64_t>* cave_pages) {
	for (auto page = cave_pages->rbegin(); page != cave_pages->rend(); ++page) {
		EXIT_IF(!Libs::LibKernel::Memory::FreeGuestMemory(*page, kPageSize));
	}
	cave_pages->clear();
}

bool AllocateCave(Plan* plan, const Program& program, std::string* error) {
	std::vector<uint64_t> pages;
	for (const auto& write: plan->writes) {
		if (IsInsideProgram(program, write.target_address, write.off.size())) {
			continue;
		}
		if (!IsZero(write.off)) {
			return Fail(error, "non-zero write is outside the matched module");
		}

		const auto write_end = write.target_address + write.off.size();
		for (auto page = write.target_address & ~(kPageSize - 1); page < write_end;
		     page += kPageSize) {
			if (std::find(pages.begin(), pages.end(), page) == pages.end()) {
				pages.push_back(page);
			}
		}
	}

	for (const auto page: pages) {
		const auto allocated = Libs::LibKernel::Memory::AllocateRuntimeMemory(
		    page, kPageSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "game_cheat_code_cave",
		    true);
		if (allocated != page) {
			if (allocated != 0) {
				Libs::LibKernel::Memory::FreeGuestMemory(allocated, kPageSize);
			}
			ReleaseCaves(&plan->cave_pages);
			return Fail(error, "could not allocate code-cave page");
		}
		plan->cave_pages.push_back(page);
	}
	return true;
}

bool ApplyWrites(Plan* plan, const Program& program, std::string* error) {
	if (!AllocateCave(plan, program, error)) {
		return false;
	}

	for (int external = 1; external >= 0; external--) {
		for (const auto& write: plan->writes) {
			if (static_cast<int>(
			        !IsInsideProgram(program, write.target_address, write.on.size())) != external) {
				continue;
			}
			std::memcpy(reinterpret_cast<void*>(write.target_address), write.on.data(),
			            write.on.size());
			if (!Common::VirtualMemory::FlushInstructionCache(write.target_address,
			                                                  write.on.size())) {
				return Fail(error, "could not finalize cheat write");
			}
		}
	}
	return true;
}

} // namespace

bool Apply(const std::filesystem::path& plan_path, Program* main_program,
           const std::vector<Program*>& programs) {
	Plan        plan;
	std::string error;
	if (!LoadPlan(plan_path, &plan, &error) || !ValidateGame(plan, main_program, &error)) {
		::printf("Game cheat error: %s\n", error.c_str());
		return false;
	}
	if (plan.writes.empty()) {
		Log::WriteToConsoleAndLog("Game patch: no mod writes left to apply\n");
		return true;
	}

	g_pending_plan = std::make_unique<Plan>(std::move(plan));
	for (auto* program: programs) {
		if (!ApplyPending(program)) {
			return false;
		}
		if (g_pending_plan == nullptr) {
			return true;
		}
	}
	::printf("Game cheat: waiting for the matching module\n");
	return true;
}

bool ApplyPending(Program* program) {
	uint64_t source_base = 0;
	if (g_pending_plan == nullptr || program == nullptr || program->elf == nullptr ||
	    !Matches(g_pending_plan.get(), *program, &source_base)) {
		return true;
	}
	::printf("Game cheat: matched %s with source base 0x%llx\n",
	         program->file_name.filename().string().c_str(),
	         static_cast<unsigned long long>(source_base));

	std::string error;
	if (!ApplyWrites(g_pending_plan.get(), *program, &error)) {
		ReleaseCaves(&g_pending_plan->cave_pages);
		g_pending_plan.reset();
		::printf("Game cheat error: %s\n", error.c_str());
		return false;
	}

	for (const auto& name: g_pending_plan->mod_names) {
		::printf("Successfully applied cheat: %s\n", name.c_str());
	}
	g_applied_cave_pages = std::move(g_pending_plan->cave_pages);
	g_pending_plan.reset();
	return true;
}

void ApplyBuiltin(Program* main_program) {
	if (main_program == nullptr || main_program->elf == nullptr || main_program->elf->IsShared()) {
		return;
	}
	// KYTY_ASTRO_CB_SCALE=<1-4> (default 2; 1 = off): Astro Bot (PPSA21567 01.018.000; other versions
	// where the table is found by its contents, such as 01.019.000) sizes its
	// five draw threads' command buffers {7, 7, 4, 4, 6} MiB (table at 0x8491510, read by
	// sub_4A38C0); running out is its "too many objects/batches in this scene" assert (Bug C). In
	// speedrun_inflator the Draw Geometry thread's frame is mostly mesh-deformer dispatches with
	// inline bone palettes (22-48 KiB each, ~5.7 MiB per frame). When that thread falls a frame
	// behind (game threads short of CPU, or blocked on reads of GPU-written memory), one buffer
	// takes two frames' work: 11.4-12.1 MiB, measured flat at that ceiling also with the game on
	// 16 logical processors. The table is scaled before the game runs.
	int scale = 2;
	if (const auto* value = std::getenv("KYTY_ASTRO_CB_SCALE"); value != nullptr) {
		scale = std::clamp(std::atoi(value), 1, 4);
	}
	if (scale == 1) {
		return;
	}
	std::string title_id;
	std::string version;
	if (!SystemContentParamSfoGetString("TITLE_ID", &title_id) ||
	    !SystemContentParamSfoGetString("APP_VER", &version) || title_id != "PPSA21567") {
		return;
	}
	static constexpr uint64_t kTable     = 0x8491510;
	static constexpr uint32_t kSizes[5]  = {7, 7, 4, 4, 6};
	auto                      address    = main_program->base_vaddr + kTable;
	if (version != "01.018.000" || !IsInsideProgram(*main_program, address, sizeof(kSizes)) ||
	    std::memcmp(reinterpret_cast<const void*>(address), kSizes, sizeof(kSizes)) != 0) {
		// Other versions (01.019.000) keep the table elsewhere: use it when the loaded segments hold
		// these 20 bytes exactly once.
		address = 0;
		int         matches = 0;
		const auto* pattern = reinterpret_cast<const uint8_t*>(kSizes);
		const auto* ehdr    = main_program->elf->GetEhdr();
		const auto* phdr    = main_program->elf->GetPhdr();
		for (Elf64_Half index = 0; index < ehdr->e_phnum; index++) {
			const auto& segment = phdr[index];
			if ((segment.p_type != PT_LOAD && segment.p_type != PT_OS_RELRO) ||
			    segment.p_filesz < sizeof(kSizes)) {
				continue;
			}
			const auto* begin = reinterpret_cast<const uint8_t*>(main_program->base_vaddr + segment.p_vaddr);
			const auto* end   = begin + segment.p_filesz;
			for (const auto* current = begin; current < end; current++) {
				current = std::search(current, end, pattern, pattern + sizeof(kSizes));
				if (current == end) {
					break;
				}
				if (reinterpret_cast<uint64_t>(current) % alignof(uint32_t) == 0 &&
				    reinterpret_cast<uint64_t>(current) != address) {
					address = reinterpret_cast<uint64_t>(current);
					matches++;
				}
			}
		}
		if (matches != 1) {
			::printf("Game patch: Astro Bot %s command buffer table not found (%d matches; "
			         "KYTY_ASTRO_CB_SCALE)\n",
			         version.c_str(), matches);
			return;
		}
		::printf("Game patch: Astro Bot %s command buffer table at +0x%" PRIx64 "\n", version.c_str(),
		         address - main_program->base_vaddr);
	}
	uint32_t scaled[5] {};
	for (int i = 0; i < 5; i++) {
		scaled[i] = kSizes[i] * static_cast<uint32_t>(scale);
	}
	std::memcpy(reinterpret_cast<void*>(address), scaled, sizeof(scaled));
	::printf("Game patch: Astro Bot draw-thread command buffers x%d (%u, %u, %u, %u, %u MiB; "
	         "KYTY_ASTRO_CB_SCALE)\n",
	         scale, scaled[0], scaled[1], scaled[2], scaled[3], scaled[4]);
}

void Clear() {
	if (g_pending_plan != nullptr) {
		ReleaseCaves(&g_pending_plan->cave_pages);
	}
	g_pending_plan.reset();
	ReleaseCaves(&g_applied_cave_pages);
}

} // namespace Loader::GamePatch
