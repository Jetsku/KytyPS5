#include "graphics/shader/recompiler/CallSplitPlan.h"
#include "common/staticOnce.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <set>
#include <xxhash.h>

namespace Libs::Graphics::ShaderRecompiler::CallSplit {
namespace {

using Decoder::Opcode;

constexpr uint32_t SEndpgm = 0xbf810000u;

uint32_t SBranch(size_t at, size_t target) {
	const auto offset = static_cast<int64_t>(target) - static_cast<int64_t>(at + 1u);
	EXIT_IF(offset < INT16_MIN || offset > INT16_MAX);
	return 0xbf820000u | (static_cast<uint32_t>(offset) & 0xffffu);
}

uint32_t EndPc(const Decoder::Instruction& inst) {
	return inst.pc + inst.word_count * 4u;
}

bool IsCallInstruction(const Decoder::Instruction& inst) {
	return inst.family == Decoder::Family::SOP1 && inst.opcode_id == 0x21u &&
	       ((inst.raw[0] >> 16u) & 0x7fu) != 125u;
}

// S_GETPC_B64 into the call's destination pair: the skip path of a call (the return address).
uint32_t GetPcWord(uint32_t call_word) {
	return (call_word & 0xffff0000u) | (0x1fu << 8u);
}

void Log(std::string_view text) {
	Log::WriteToConsoleAndLog(fmt::format("KYTY_SHADER_CALLS_SPLIT: {}\n", text));
}

std::array<uint64_t, 2> Digest(std::span<const uint32_t> words) {
	const auto hash = XXH3_128bits(words.data(), words.size_bytes());
	return {hash.low64, hash.high64};
}

// KYTY_SHADER_CALLS_SPLIT_CALLEE_DUMP=<folder> (diagnostic, default off): every callee pass program
// (C pass: the call site region plus the inlined callee) is written there once as
// callee_<kernel hash>_s<site>_<target>.bin/.txt (code and disassembly). Local files only.
void DumpCallee(uint64_t hash, uint32_t site, uint64_t target, std::span<const uint32_t> words) {
	static const char* folder = [] {
		const char* value = std::getenv("KYTY_SHADER_CALLS_SPLIT_CALLEE_DUMP");
		return value != nullptr && *value != '\0' ? value : nullptr;
	}();
	if (folder == nullptr) return;
	const auto name = fmt::format("{}/callee_{:016x}_s{}_{:x}", folder, hash, site, target);
	if (std::FILE* file = std::fopen((name + ".bin").c_str(), "wb"); file != nullptr) {
		std::fwrite(words.data(), sizeof(uint32_t), words.size(), file);
		std::fclose(file);
	}
	Decoder::Program decoded;
	Decoder::DecodeProgram(words, decoded, true);
	const auto text = Decoder::ProgramToString(decoded);
	if (std::FILE* file = std::fopen((name + ".txt").c_str(), "wb"); file != nullptr) {
		std::fwrite(text.data(), 1, text.size(), file);
		std::fclose(file);
	}
}

std::shared_ptr<const Code> MakeCode(std::vector<uint32_t> words) {
	auto code    = std::make_shared<Code>();
	code->digest = XXH3_64bits(words.data(), words.size() * sizeof(uint32_t)) | 1u;
	code->words  = std::move(words);
	return code;
}

SiteLayout Candidates(uint32_t vgprs) {
	SiteLayout layout;
	for (uint16_t code = 0; code < IR::NumScalarRegs; code++) {
		layout.uniform.push_back(code);
	}
	for (const uint16_t code: {CodeVccLo, CodeVccHi, CodeM0, CodeExecLo, CodeExecHi, CodeScc}) {
		layout.uniform.push_back(code);
	}
	for (uint32_t v = 0; v < vgprs; v++) {
		layout.vector.push_back(static_cast<uint16_t>(VectorCode + v));
	}
	return layout;
}

uint32_t KernelReadVgprLimit(const Decoder::Program& decoded) {
	uint32_t   limit   = 1;
	const auto include = [&](const Decoder::Operand& operand, uint32_t count) {
		if (operand.kind == Decoder::OperandKind::Vgpr) {
			limit = std::max(limit, std::min(operand.reg + count, IR::NumVectorRegs));
		}
	};
	for (const auto& inst: decoded.instructions) {
		// Generous: every operand may span the instruction's widest data.
		const uint32_t width = std::max({inst.data_dwords, inst.data_components, 4u}) + (inst.tfe ? 1u : 0u);
		include(inst.dst, width);
		include(inst.dst2, 2u);
		include(inst.src0, inst.family == Decoder::Family::MIMG ? 16u : width);
		include(inst.src1, width);
		include(inst.src2, width);
		include(inst.src3, width);
		for (uint32_t i = 0; i < inst.image_nsa_dwords && i < Decoder::MaxImageNsaAddressComponents; i++) {
			limit = std::max(limit, std::min(inst.image_nsa_addr[i] + 1u, IR::NumVectorRegs));
		}
	}
	return limit;
}

// Read on every build (not cached), so a test can build both variants in one process.
bool ShadedOnly() {
	const char* value = std::getenv("KYTY_SHADER_CALLS_SPLIT_SHADED_ONLY");
	return value == nullptr || value[0] != '0';
}

} // namespace

bool AnalyzeShape(std::span<const uint32_t> code, const CompileOptions& options, Shape& shape,
                  std::string& failure) {
	shape = {};
	if (options.stage != ShaderType::Compute || options.wave_size != 64u ||
	    options.input_info.compute == nullptr) {
		failure = "not a wave64 compute program";
		return false;
	}
	const auto& cs = *options.input_info.compute;
	if (std::max(cs.threads_num[0], 1u) * std::max(cs.threads_num[1], 1u) *
	        std::max(cs.threads_num[2], 1u) != 64u ||
	    cs.dispatch_thread_dimensions) {
		failure = "its workgroup is not one wave";
		return false;
	}
	const auto analysis = AnalyzeShaderCalls(code, options);
	if (analysis.calls.empty() || !analysis.failures.empty()) {
		failure = analysis.calls.empty() ? std::string("no call") : analysis.failures;
		return false;
	}
	Decoder::Program decoded;
	Decoder::DecodeProgram(code, decoded, true);
	const auto& instructions = decoded.instructions;
	const auto  at_pc        = [&](uint32_t pc) -> const Decoder::Instruction* {
		const auto found = std::ranges::lower_bound(instructions, pc, {}, &Decoder::Instruction::pc);
		return found != instructions.end() && found->pc == pc ? &*found : nullptr;
	};
	for (const auto& call: analysis.calls) {
		if (!call.table.has_value()) {
			failure = fmt::format("pc 0x{:x}: the target is not from a table", call.pc);
			return false;
		}
		// The innermost loop around the call: its back edge.
		const Decoder::Instruction* back = nullptr;
		for (const auto& inst: instructions) {
			if (inst.pc > call.pc && Decoder::IsDirectBranch(inst.opcode) &&
			    inst.branch_target <= call.pc &&
			    (back == nullptr || inst.pc - inst.branch_target < back->pc - back->branch_target)) {
				back = &inst;
			}
		}
		if (back == nullptr || back->opcode != Opcode::S_BRANCH || back->branch_target < 4u) {
			failure = fmt::format("pc 0x{:x}: the call is not in a loop", call.pc);
			return false;
		}
		Site site;
		site.call_pc    = call.pc;
		site.source_reg = call.source_reg;
		site.return_reg = call.return_reg;
		site.table      = *call.table;
		site.b = EndPc(*back);
		// The loop is entered by saving EXEC, a few instructions before its head (vector moves of
		// the loop's own flags may sit between).
		const Decoder::Instruction* first = nullptr;
		{
			auto head = std::ranges::lower_bound(instructions, back->branch_target, {},
			                                     &Decoder::Instruction::pc);
			for (int step = 0; step < 4 && head != instructions.begin(); step++) {
				--head;
				if (head->opcode == Opcode::S_MOV_B64 && head->word_count == 1u &&
				    head->src0.kind == Decoder::OperandKind::ExecLo) {
					first = &*head;
					break;
				}
				if (head->family != Decoder::Family::VOP1 || Decoder::IsDirectBranch(head->opcode)) {
					break;
				}
			}
		}
		site.a = first != nullptr ? first->pc : 0u;
		if (first == nullptr) {
			failure = fmt::format("pc 0x{:x}: the loop is not entered by saving EXEC", call.pc);
			return false;
		}
		bool exits = false;
		bool record = false;
		uint32_t calls = 0;
		for (const auto& inst: instructions) {
			const bool inside = inst.pc >= site.a && inst.pc < site.b;
			if (Decoder::IsDirectBranch(inst.opcode)) {
				const bool target_inside = inst.branch_target > site.a && inst.branch_target < site.b;
				if (inside && (inst.branch_target < site.a || inst.branch_target > site.b)) {
					failure = fmt::format("pc 0x{:x}: the loop branches out of itself", inst.pc);
					return false;
				}
				if (!inside && target_inside) {
					failure = fmt::format("pc 0x{:x}: a branch into the loop of pc 0x{:x}", inst.pc, call.pc);
					return false;
				}
				exits |= inside && inst.opcode == Opcode::S_CBRANCH_EXECZ && inst.branch_target == site.b;
			}
			if (inside && inst.opcode == Opcode::V_READFIRSTLANE_B32 &&
			    inst.src0.kind == Decoder::OperandKind::Vgpr && !record) {
				site.record_vgpr = inst.src0.reg;
				record           = true;
			}
			if (inside && (IsCallInstruction(inst) || inst.opcode == Opcode::S_ENDPGM ||
			               inst.opcode == Opcode::S_SETPC_B64)) {
				calls += IsCallInstruction(inst) ? 1u : 100u;
			}
		}
		if (!exits || !record || calls != 1u) {
			failure = fmt::format("pc 0x{:x}: not a waterfall loop with one call", call.pc);
			return false;
		}
		shape.sites.push_back(std::move(site));
	}
	std::ranges::sort(shape.sites, {}, &Site::a);
	for (size_t i = 1; i < shape.sites.size(); i++) {
		if (shape.sites[i].a < shape.sites[i - 1u].b) {
			failure = "two calls share a loop";
			return false;
		}
	}
	// No other loop may enclose a call loop: the sites then run in pc order.
	for (const auto& inst: instructions) {
		if (!Decoder::IsDirectBranch(inst.opcode) || inst.branch_target > inst.pc) continue;
		for (const auto& site: shape.sites) {
			const bool own = inst.pc >= site.a && inst.pc < site.b;
			if (!own && inst.branch_target <= site.a && inst.pc >= site.a) {
				failure = fmt::format("the loop at pc 0x{:x} encloses the call at 0x{:x}", inst.branch_target,
				                      site.call_pc);
				return false;
			}
		}
	}
	// The canonical program: calls take the skip path; it ends at its last decoded instruction.
	const auto end_words = EndPc(instructions.back()) / 4u;
	shape.canonical.assign(code.begin(), code.begin() + end_words);
	for (const auto& site: shape.sites) {
		auto& word = shape.canonical[site.call_pc / 4u];
		word       = GetPcWord(word);
	}
	shape.digest = Digest(shape.canonical);
	shape.vgprs  = KernelReadVgprLimit(decoded);
	return true;
}

bool BuildCode(const Shape& shape, const Header& layouts, Kind kind, uint32_t site, bool liveness,
               std::span<const uint32_t> original, uint64_t base, uint64_t callee,
               const Decoder::ShaderCodeReader& read, std::vector<uint32_t>& code,
               std::string& failure) {
	code.clear();
	Header header  = layouts;
	header.digest  = shape.digest;
	header.vgprs   = shape.vgprs;
	header.entry_site = kind == Kind::Probe ? 0u : site;
	header.flags   = (liveness ? FlagLiveness : 0u) | (kind == Kind::Probe ? FlagProbe : 0u);
	if (liveness && site != 0) {
		header.sites[site - 1u] = Candidates(shape.vgprs);
	}
	EncodeHeader(header, code);
	if (kind == Kind::Callee) {
		const auto& s = shape.sites.at(site - 1u);
		std::vector<uint32_t> region(original.begin() + s.a / 4u, original.begin() + s.b / 4u);
		// KYTY_SHADER_CALLS_SPLIT_SHADED_ONLY (default on; =0 reverts): store only the lanes that ran
		// this callee. Without it every lane of the wave is stored, and a lane of another target leaves
		// this pass marked done by the skip path, so in a wave whose lanes name several callees only
		// the first callee pass shaded anything (Miles Morales: 65% of the reflection hits black).
		if (ShadedOnly()) {
			region.push_back(Decoder::EncodeCallStateWord(Decoder::CallStateOp::SaveChanged, 1));
			region.push_back(1u);
		} else {
			region.push_back(Decoder::EncodeCallStateWord(Decoder::CallStateOp::SaveChanged, 0));
		}
		region.push_back(SEndpgm);
		// The callee is compared with the target the dispatch passes in and the kernel's addresses
		// are written relative to its shader base: the words hold no guest address, so the program
		// (and its pipeline) is the same in every session, wherever the game loaded the callee.
		const Decoder::ShaderCallSite call {.pc           = s.call_pc - s.a,
		                                    .source_reg   = s.source_reg,
		                                    .return_reg   = s.return_reg,
		                                    .addresses    = {callee},
		                                    .split_callee = true,
		                                    .shader_base  = base};
		Decoder::ShaderCallExpansion expansion;
		if (!Decoder::ExpandShaderCalls(region, base + s.a, std::span(&call, 1u), read, 64u, expansion,
		                                failure)) {
			return false;
		}
		if (expansion.inlined == 0) {
			failure = expansion.dropped.empty() ? std::string("the callee was not placed") : expansion.dropped.front();
			return false;
		}
		code.insert(code.end(), expansion.code.begin(), expansion.code.end());
		return true;
	}
	if (site != 0) {
		code.push_back(0); // S_BRANCH a_k, below
	}
	const auto prefix = code.size();
	const auto words  = shape.canonical.size();
	code.insert(code.end(), shape.canonical.begin(), shape.canonical.end());
	for (uint32_t k = 1; k <= shape.sites.size(); k++) {
		const auto& s = shape.sites[k - 1u];
		const auto  a = prefix + s.a / 4u;
		if (kind == Kind::Resume && k == site) {
			continue; // the entered loop runs (every call skipped)
		}
		if (kind == Kind::Resume && k < site) {
			continue; // not reachable from a_site
		}
		const auto block = code.size();
		const auto saved = code[a];
		code[a]          = SBranch(a, block);
		if (kind == Kind::Probe) {
			code.push_back(Decoder::EncodeCallStateWord(Decoder::CallStateOp::Probe, 1));
			code.push_back(k);
			code.push_back(saved);
			code.push_back(SBranch(code.size(), a + 1u));
		} else {
			code.push_back(Decoder::EncodeCallStateWord(Decoder::CallStateOp::Save, 1));
			code.push_back(k);
			code.push_back(SEndpgm);
		}
	}
	if (site != 0) {
		code[prefix - 1u] = SBranch(prefix - 1u, prefix + shape.sites[site - 1u].a / 4u);
	}
	if (kind == Kind::Probe) {
		code.push_back(SEndpgm); // the decoder ends at an S_ENDPGM (a probe block ends in a branch)
	}
	(void)words;
	return true;
}

Planner::Planner() = default;

Planner::~Planner() {
	{
		std::scoped_lock lock(m_mutex);
		m_stop = true;
	}
	m_wake.notify_all();
	for (auto& thread: m_threads) {
		if (thread.joinable()) thread.join();
	}
}

Planner& Planner::Instance() {
	static Planner planner;
	return planner;
}

static std::atomic<int> g_enabled_override {-1};

void Planner::SetEnabledOverride(int enabled) {
	g_enabled_override.store(enabled);
}

bool Planner::Enabled() {
	if (const auto value = g_enabled_override.load(std::memory_order_relaxed); value >= 0) {
		return value != 0;
	}
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_SHADER_CALLS_SPLIT");
		return value != nullptr && std::strcmp(value, "0") != 0 && *value != '\0';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

bool Planner::Translate(const Kernel& kernel, std::span<const uint32_t> code, Report& report) {
	ShaderComputeInputInfo info = kernel.info;
	ShaderStageInputInfo   stage_input {};
	stage_input.compute = &info;
	CompileOptions options;
	options.stage          = ShaderType::Compute;
	options.wave_size      = 64;
	options.shader_hash    = kernel.hash;
	options.user_data_base = kernel.user_data_base;
	options.user_data      = kernel.user_data;
	options.input_info     = stage_input;
	options.dump_ir        = false;
	const auto result      = TranslateProgram(code, options);
	if (result.call_split == nullptr) {
		return false;
	}
	report = *result.call_split;
	return true;
}

void Planner::Work() {
	for (;;) {
		Job job;
		{
			std::unique_lock lock(m_mutex);
			m_wake.wait(lock, [&] { return m_stop || !m_jobs.empty(); });
			if (m_stop) return;
			job = m_jobs.front();
			m_jobs.pop_front();
		}
		if (job.site == 0) {
			PlanKernel(job.kernel);
		} else {
			PlanTarget(job.kernel, job.site, job.target);
		}
	}
}

// Probe, layouts (last site first: a site's registers pass through the programs before it),
// then the codes of R_0, R_ak and the callees known so far.
void Planner::PlanKernel(uint64_t id) {
	const auto begin = std::chrono::steady_clock::now();
	Kernel     kernel;
	std::vector<std::vector<uint64_t>> targets;
	{
		std::scoped_lock lock(m_mutex);
		auto&            k = m_kernels.at(id);
		kernel.shape       = k.shape;
		kernel.original    = k.original;
		kernel.base        = k.base;
		kernel.user_data   = k.user_data;
		kernel.info        = k.info;
		kernel.hash        = k.hash;
		kernel.user_data_base = k.user_data_base;
		for (const auto& site: k.sites) targets.push_back(site.targets);
	}
	const auto fail = [&](std::string why) {
		Log(fmt::format("CS shader 0x{:016x} is not split: {}.", kernel.hash, why));
		std::scoped_lock lock(m_mutex);
		auto&            k = m_kernels.at(id);
		k.state            = Kernel::State::Failed;
		k.failure          = std::move(why);
	};
	const auto& shape = kernel.shape;
	const auto  n     = static_cast<uint32_t>(shape.sites.size());
	Header      layouts;
	layouts.sites.resize(n);
	std::string           why;
	std::vector<uint32_t> code;
	Report                report;
	if (!BuildCode(shape, layouts, Kind::Probe, 0, false, {}, 0, 0, m_read, code, why) ||
	    !Translate(kernel, code, report)) {
		return fail("the probe translation failed" + (why.empty() ? std::string() : ": " + why));
	}
	auto recipe = std::make_shared<Recipe>(std::move(report.recipe));
	RegisterRecipe(shape.digest, recipe);
	std::vector<std::vector<bool>> usable(n);
	for (uint32_t k = n; k >= 1; k--) {
		std::set<uint16_t> live;
		if (!BuildCode(shape, layouts, Kind::Resume, k, true, {}, 0, 0, m_read, code, why) ||
		    !Translate(kernel, code, report)) {
			return fail(fmt::format("the liveness translation of site {} failed {}", k, why));
		}
		live.insert(report.live.begin(), report.live.end());
		usable[k - 1u].assign(targets[k - 1u].size(), false);
		for (size_t t = 0; t < targets[k - 1u].size(); t++) {
			why.clear();
			if (BuildCode(shape, layouts, Kind::Callee, k, true, kernel.original, kernel.base,
			              targets[k - 1u][t], m_read, code, why) &&
			    Translate(kernel, code, report)) {
				live.insert(report.live.begin(), report.live.end());
				usable[k - 1u][t] = true;
			} else {
				Log(fmt::format("CS shader 0x{:016x} site {}: callee 0x{:x} is skipped: {}.", kernel.hash, k,
				                targets[k - 1u][t], why.empty() ? std::string("its translation failed") : why));
			}
		}
		auto& layout = layouts.sites[k - 1u];
		layout       = {};
		for (const auto code_value: live) {
			(code_value >= VectorCode ? layout.vector : layout.uniform).push_back(code_value);
		}
		const auto& s = shape.sites[k - 1u];
		const bool  exec =
		    std::ranges::count(layout.uniform, CodeExecLo) == 1 && std::ranges::count(layout.uniform, CodeExecHi) == 1;
		const bool record = std::ranges::count(layout.vector, static_cast<uint16_t>(VectorCode + s.record_vgpr)) == 1;
		if (!exec || !record) {
			return fail(fmt::format("site {} does not keep EXEC and its record register", k));
		}
	}
	auto start = std::make_shared<std::vector<uint32_t>>();
	if (!BuildCode(shape, layouts, Kind::Resume, 0, false, {}, 0, 0, m_read, *start, why)) {
		return fail("R_0: " + why);
	}
	std::vector<std::shared_ptr<const Code>>              resumes;
	std::vector<std::vector<std::shared_ptr<const Code>>> callees(n);
	for (uint32_t k = 1; k <= n; k++) {
		if (!BuildCode(shape, layouts, Kind::Resume, k, false, {}, 0, 0, m_read, code, why)) {
			return fail(fmt::format("R_a{}: {}", k, why));
		}
		resumes.push_back(MakeCode(code));
		for (size_t t = 0; t < targets[k - 1u].size(); t++) {
			std::shared_ptr<const Code> callee;
			if (usable[k - 1u][t] && BuildCode(shape, layouts, Kind::Callee, k, false, kernel.original,
			                                   kernel.base, targets[k - 1u][t], m_read, code, why)) {
				DumpCallee(kernel.hash, k, targets[k - 1u][t], code);
				callee = MakeCode(code);
			}
			callees[k - 1u].push_back(std::move(callee));
		}
	}
	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                    std::chrono::steady_clock::now() - begin)
	                    .count();
	std::string sizes;
	for (uint32_t k = 1; k <= n; k++) {
		const auto& layout = layouts.sites[k - 1u];
		sizes += fmt::format("{}site {}: {} scalar + {} vector registers ({} scalars rebuilt), {} callees",
		                     k == 1 ? "" : "; ", k, layout.uniform.size(), layout.vector.size(),
		                     recipe->sites[k - 1u].roots.size(), callees[k - 1u].size());
	}
	Log(fmt::format("CS shader 0x{:016x} splits at {} call loop(s), planned in {} ms, {} dwords per wave: {}.",
	                kernel.hash, n, ms, layouts.WaveDwords(), sizes));
	std::scoped_lock lock(m_mutex);
	auto&            k = m_kernels.at(id);
	k.layouts          = layouts;
	k.start            = MakeCode(std::move(*start));
	for (uint32_t s = 0; s < n; s++) {
		auto& site  = k.sites[s];
		site.resume = resumes[s];
		for (size_t t = 0; t < callees[s].size(); t++) {
			site.state[t].code     = callees[s][t];
			site.state[t].unusable = callees[s][t] == nullptr;
		}
		// Targets that appeared while planning.
		for (size_t t = callees[s].size(); t < site.targets.size(); t++) {
			m_jobs.push_back({id, s + 1u, site.targets[t]});
		}
	}
	k.state = Kernel::State::Ready;
	m_wake.notify_all();
}

// A target found after the plan: usable when its inputs fit the site's layout.
void Planner::PlanTarget(uint64_t id, uint32_t site, uint64_t target) {
	Kernel kernel;
	{
		std::scoped_lock lock(m_mutex);
		auto&            k = m_kernels.at(id);
		if (k.state != Kernel::State::Ready) return;
		kernel.shape          = k.shape;
		kernel.layouts        = k.layouts;
		kernel.original       = k.original;
		kernel.base           = k.base;
		kernel.user_data      = k.user_data;
		kernel.info           = k.info;
		kernel.hash           = k.hash;
		kernel.user_data_base = k.user_data_base;
	}
	std::vector<uint32_t> code;
	std::string           why;
	Report                report;
	std::shared_ptr<const Code> result;
	if (BuildCode(kernel.shape, kernel.layouts, Kind::Callee, site, true, kernel.original, kernel.base,
	              target, m_read, code, why) &&
	    Translate(kernel, code, report)) {
		const auto& layout = kernel.layouts.sites[site - 1u];
		const auto  fits   = std::ranges::all_of(report.live, [&](uint16_t c) {
			return std::ranges::count(layout.uniform, c) != 0 || std::ranges::count(layout.vector, c) != 0;
		});
		if (!fits) {
			why = "its inputs are not in the site's layout";
		} else if (BuildCode(kernel.shape, kernel.layouts, Kind::Callee, site, false, kernel.original,
		                     kernel.base, target, m_read, code, why)) {
			DumpCallee(kernel.hash, site, target, code);
			result = MakeCode(code);
		}
	} else if (why.empty()) {
		why = "its translation failed";
	}
	if (result == nullptr) {
		Log(fmt::format("CS shader 0x{:016x} site {}: callee 0x{:x} is skipped: {}.", kernel.hash, site, target,
		                why));
	}
	std::scoped_lock lock(m_mutex);
	auto&            k     = m_kernels.at(id);
	auto&            plan  = k.sites[site - 1u];
	const auto       found = plan.slot.find(target);
	if (found == plan.slot.end()) return;
	plan.state[found->second].code     = result;
	plan.state[found->second].unusable = result == nullptr;
}

Planner::Status Planner::GetStatus() {
	std::scoped_lock lock(m_mutex);
	Status           status;
	status.jobs = static_cast<uint32_t>(m_jobs.size());
	for (const auto& [id, kernel]: m_kernels) {
		(void)id;
		switch (kernel.state) {
			case Kernel::State::Planning: status.planning++; break;
			case Kernel::State::Failed: status.failed++; break;
			case Kernel::State::Ready:
				status.ready++;
				for (const auto& site: kernel.sites) {
					for (const auto& target: site.state) {
						status.targets++;
						if (target.code != nullptr) {
							status.built++;
						} else if (target.unusable) {
							status.unusable++;
						} else {
							status.pending++;
						}
					}
				}
				break;
		}
	}
	return status;
}

Planner::Result Planner::Plan(std::span<const uint32_t> code, uint64_t base,
                              const CompileOptions& options, const ShaderComputeInputInfo& info,
                              const Decoder::ShaderCodeReader& read, Run& run) {
	std::unique_lock lock(m_mutex);
	if (!m_read) {
		m_read = read;
	}
	const auto id = base ^ (options.shader_hash * 0x9e3779b97f4a7c15ull) ^ code.size();
	auto [entry, inserted] = m_kernels.try_emplace(id);
	auto& kernel           = entry->second;
	if (inserted) {
		std::string failure;
		if (!AnalyzeShape(code, options, kernel.shape, failure)) {
			kernel.state   = Kernel::State::Failed;
			kernel.failure = failure;
			Log(fmt::format("CS shader 0x{:016x} is not split: {}.", options.shader_hash, failure));
			return Result::NotSplit;
		}
		kernel.original.assign(code.begin(), code.end());
		kernel.base           = base;
		kernel.user_data.assign(options.user_data.begin(), options.user_data.end());
		kernel.info           = info;
		kernel.info.stage     = {};
		kernel.hash           = options.shader_hash;
		kernel.user_data_base = options.user_data_base;
		kernel.sites.resize(kernel.shape.sites.size());
	}
	if (kernel.state == Kernel::State::Failed) {
		return Result::NotSplit;
	}
	const auto now = kernel.dispatches++;
	// Each site's table: its targets and record map (read when new and every 64 dispatches).
	for (uint32_t s = 0; s < kernel.sites.size(); s++) {
		auto&         plan = kernel.sites[s];
		CallTableView view;
		std::string   why;
		if (!EvaluateCallTable(kernel.shape.sites[s].table, options.user_data, options.user_data_base, read, view,
		                       why)) {
			plan.record_slots.clear();
			continue;
		}
		if (view == plan.view && now - plan.scanned_at < 64u && !plan.record_slots.empty()) {
			continue;
		}
		plan.view       = view;
		plan.scanned_at = now;
		if (!ReadCallTable(view, read, m_table_words, why)) {
			plan.record_slots.clear();
			continue;
		}
		const auto digest = XXH3_64bits(m_table_words.data(), m_table_words.size() * sizeof(uint32_t));
		if (digest == plan.table_digest && !plan.record_slots.empty()) {
			continue;
		}
		plan.table_digest   = digest;
		const auto records  = view.record_bytes != 0 ? view.bytes / view.record_bytes : 0u;
		plan.record_slots.assign(records, Run::NoSlot);
		for (uint64_t r = 0; r < records; r++) {
			const auto offset = r * view.record_bytes + view.target_offset;
			if (offset + 8u > m_table_words.size() * 4u) break;
			const auto target = static_cast<uint64_t>(m_table_words[offset / 4u]) |
			                    (static_cast<uint64_t>(m_table_words[offset / 4u + 1u]) << 32u);
			if ((target & 0x0000ffffffffffffull) == 0) continue;
			auto [slot, fresh] = plan.slot.try_emplace(target, static_cast<uint32_t>(plan.targets.size()));
			if (fresh) {
				plan.targets.push_back(target);
				plan.state.emplace_back();
				if (kernel.state == Kernel::State::Ready) {
					m_jobs.push_back({id, s + 1u, target});
				}
			}
			plan.record_slots[r] = slot->second;
		}
	}
	if (inserted) {
		m_jobs.push_back({id, 0, 0});
		const auto threads = std::clamp<uint64_t>(
		    std::getenv("KYTY_SHADER_CALLS_SPLIT_THREADS") != nullptr
		        ? std::strtoull(std::getenv("KYTY_SHADER_CALLS_SPLIT_THREADS"), nullptr, 0)
		        : 2u,
		    1u, 8u);
		while (m_threads.size() < threads) {
			m_threads.emplace_back([this] { Work(); });
		}
	}
	m_wake.notify_all();
	if (kernel.state != Kernel::State::Ready) {
		return Result::Pending;
	}
	run        = {};
	run.kernel = id;
	run.hash   = kernel.hash;
	run.start  = kernel.start;
	run.wave_dwords = kernel.layouts.WaveDwords();
	run.layouts     = kernel.layouts.sites;
	for (uint32_t s = 0; s < kernel.sites.size(); s++) {
		const auto& plan   = kernel.sites[s];
		const auto& layout = kernel.layouts.sites[s];
		run.targets.push_back(plan.targets);
		auto&       site   = run.sites.emplace_back();
		site.resume        = plan.resume;
		site.targets       = plan.targets;
		for (const auto& state: plan.state) {
			site.callees.push_back(state.code);
			site.pending.push_back(state.code == nullptr && !state.unusable);
		}
		site.record_slots = plan.record_slots;
		const auto index  = [&](uint16_t code_value) {
			return static_cast<uint32_t>(std::ranges::find(layout.uniform, code_value) - layout.uniform.begin());
		};
		site.exec_lo_slot = index(CodeExecLo);
		site.exec_hi_slot = index(CodeExecHi);
		site.record_dword =
		    layout.VectorBase() +
		    static_cast<uint32_t>(std::ranges::find(layout.vector, static_cast<uint16_t>(
		                                                               VectorCode + kernel.shape.sites[s].record_vgpr)) -
		                          layout.vector.begin()) *
		        64u;
	}
	return Result::Ready;
}

} // namespace Libs::Graphics::ShaderRecompiler::CallSplit
