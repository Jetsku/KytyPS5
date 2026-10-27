#include "graphics/shader/recompiler/CallSplit.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <algorithm>
#include <mutex>
#include <set>
#include <unordered_map>

namespace Libs::Graphics::ShaderRecompiler::CallSplit {
namespace {

thread_local Translation* t_current = nullptr;

struct DigestHash {
	size_t operator()(const std::array<uint64_t, 2>& digest) const {
		return static_cast<size_t>(digest[0] ^ (digest[1] * 0x9e3779b97f4a7c15ull));
	}
};

std::mutex& RecipeMutex() {
	static std::mutex mutex;
	return mutex;
}

std::unordered_map<std::array<uint64_t, 2>, std::shared_ptr<const Recipe>, DigestHash>& Recipes() {
	static std::unordered_map<std::array<uint64_t, 2>, std::shared_ptr<const Recipe>, DigestHash>
	    recipes;
	return recipes;
}

// Values that are a function of the user data and memory alone: what a loading program may
// rebuild instead of reading the state.
bool Rebuildable(const IR::Inst& inst) {
	using O = IR::ValueOpcode;
	switch (inst.GetOpcode()) {
		case O::GetUserData:
		case O::GetShaderBase:
		case O::ReadConstBuffer:
		case O::GetBufferResource:
		case O::GetAddressResource:
		case O::CompositeConstructU64:
		case O::CompositeExtractU64:
		case O::CompositeConstructU32x2:
		case O::CompositeConstructU32x3:
		case O::CompositeConstructU32x4:
		case O::CompositeExtractU32x2:
		case O::CompositeExtractU32x3:
		case O::CompositeExtractU32x4:
		case O::IAdd32:
		case O::IAdd64:
		case O::ISub32:
		case O::IMul32:
		case O::UMin32:
		case O::ShiftLeftLogical32:
		case O::ShiftLeftLogical64:
		case O::ShiftRightLogical32:
		case O::ShiftRightLogical64:
		case O::ShiftRightArithmetic32:
		case O::BitwiseAnd32:
		case O::BitwiseAnd64:
		case O::BitwiseOr32:
		case O::BitwiseXor32:
		case O::BitwiseNot32:
		case O::BitFieldInsert:
		case O::BitFieldUExtract:
		case O::BitFieldSExtract:
		case O::SelectU32:
		case O::IEqual32:
		case O::INotEqual32:
		case O::ULessThan32:
		case O::UGreaterThan32:
		case O::LogicalAnd:
		case O::LogicalOr:
		case O::LogicalNot: return true;
		// A scalar load (S_LOAD): active is the immediate true.
		case O::LoadAddressU32:
			return inst.Arg(3).IsImmediate() && inst.Arg(3).GetType() == IR::Type::U1 &&
			       inst.Arg(3).U1();
		default: return false;
	}
}

bool HasMemoryInfo(IR::ValueOpcode opcode) {
	return opcode == IR::ValueOpcode::ReadConstBuffer || opcode == IR::ValueOpcode::LoadAddressU32;
}

class RecipeBuilder {
public:
	RecipeBuilder(const IR::Program& program, SiteRecipe& recipe): m_program(program), m_recipe(recipe) {}

	// The argument for `value`, or false when it is not rebuildable.
	bool Build(IR::Value value, int32_t& result, uint32_t depth = 0) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			m_recipe.imms.push_back(value);
			result = -static_cast<int32_t>(m_recipe.imms.size());
			return true;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || depth > 48u) {
			return false;
		}
		if (const auto found = m_memo.find(inst); found != m_memo.end()) {
			result = found->second;
			return result != Failed;
		}
		if (!Rebuildable(*inst)) {
			m_memo.emplace(inst, Failed);
			return false;
		}
		RecipeNode node;
		node.opcode = inst->GetOpcode();
		node.flags  = inst->Flags<uint64_t>();
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			int32_t arg = 0;
			if (!Build(inst->Arg(index), arg, depth + 1u)) {
				m_memo.emplace(inst, Failed);
				return false;
			}
			node.args.push_back(arg);
		}
		if (HasMemoryInfo(node.opcode)) {
			const auto flags = inst->Flags<IR::MemoryFlags>();
			if (flags.index >= m_program.memory_info.size()) {
				m_memo.emplace(inst, Failed);
				return false;
			}
			node.has_memory = true;
			node.memory     = m_program.memory_info[flags.index];
			node.memory_pc  = flags.pc;
		}
		result = static_cast<int32_t>(m_recipe.nodes.size());
		m_recipe.nodes.push_back(std::move(node));
		m_memo.emplace(inst, result);
		return true;
	}

private:
	static constexpr int32_t Failed = INT32_MIN;

	const IR::Program&                            m_program;
	SiteRecipe&                                   m_recipe;
	std::unordered_map<const IR::Inst*, int32_t>  m_memo;
};

} // namespace

uint32_t Header::WaveDwords() const {
	uint32_t dwords = 0;
	for (const auto& site: sites) {
		dwords = std::max(dwords, site.WaveDwords());
	}
	return dwords;
}

void EncodeHeader(const Header& header, std::vector<uint32_t>& out) {
	std::vector<uint32_t> payload;
	payload.push_back(HeaderMagic);
	payload.push_back(header.entry_site | header.flags);
	payload.push_back(static_cast<uint32_t>(header.digest[0]));
	payload.push_back(static_cast<uint32_t>(header.digest[0] >> 32u));
	payload.push_back(static_cast<uint32_t>(header.digest[1]));
	payload.push_back(static_cast<uint32_t>(header.digest[1] >> 32u));
	payload.push_back(header.vgprs);
	payload.push_back(static_cast<uint32_t>(header.sites.size()));
	for (const auto& site: header.sites) {
		payload.push_back(static_cast<uint32_t>(site.uniform.size()) |
		                  (static_cast<uint32_t>(site.vector.size()) << 16u));
		std::vector<uint16_t> codes = site.uniform;
		codes.insert(codes.end(), site.vector.begin(), site.vector.end());
		for (size_t i = 0; i < codes.size(); i += 2u) {
			payload.push_back(codes[i] | (i + 1u < codes.size() ? uint32_t {codes[i + 1u]} << 16u : 0u));
		}
	}
	EXIT_IF(payload.size() > 0xfffu);
	out.push_back(Decoder::EncodeCallStateWord(Decoder::CallStateOp::Header,
	                                           static_cast<uint32_t>(payload.size())));
	out.insert(out.end(), payload.begin(), payload.end());
}

bool DecodeHeader(std::span<const uint32_t> payload, Header& header) {
	header = {};
	if (payload.size() < 8u || payload[0] != HeaderMagic) {
		return false;
	}
	header.entry_site = payload[1] & 0xffu;
	header.flags      = payload[1] & ~0xffu;
	header.digest[0]  = payload[2] | (uint64_t {payload[3]} << 32u);
	header.digest[1]  = payload[4] | (uint64_t {payload[5]} << 32u);
	header.vgprs      = payload[6];
	const auto count  = payload[7];
	size_t     at     = 8;
	for (uint32_t s = 0; s < count; s++) {
		if (at >= payload.size()) return false;
		const auto uniform = payload[at] & 0xffffu;
		const auto vector  = payload[at] >> 16u;
		at++;
		std::vector<uint16_t> codes;
		for (uint32_t i = 0; i < uniform + vector; i++) {
			const auto word = at + i / 2u;
			if (word >= payload.size()) return false;
			codes.push_back(static_cast<uint16_t>((i & 1u) != 0 ? payload[word] >> 16u : payload[word]));
		}
		at += (uniform + vector + 1u) / 2u;
		SiteLayout site;
		site.uniform.assign(codes.begin(), codes.begin() + uniform);
		site.vector.assign(codes.begin() + uniform, codes.end());
		header.sites.push_back(std::move(site));
	}
	return header.entry_site <= header.sites.size();
}

void RegisterRecipe(const std::array<uint64_t, 2>& digest, std::shared_ptr<const Recipe> recipe) {
	std::scoped_lock lock(RecipeMutex());
	Recipes()[digest] = std::move(recipe);
}

std::shared_ptr<const Recipe> FindRecipe(const std::array<uint64_t, 2>& digest) {
	std::scoped_lock lock(RecipeMutex());
	const auto       found = Recipes().find(digest);
	return found != Recipes().end() ? found->second : nullptr;
}

Translation* Current() {
	return t_current;
}

void SetCurrent(Translation* translation) {
	t_current = translation;
}

void CollectReport(const Translation& translation, const IR::Program& program, Report& report) {
	report = {};
	std::set<uint16_t> live;
	report.recipe.sites.resize(translation.header.sites.size());
	std::vector<RecipeBuilder> builders;
	builders.reserve(report.recipe.sites.size());
	for (auto& site: report.recipe.sites) {
		builders.emplace_back(program, site);
	}
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == IR::ValueOpcode::CallStateParam && inst.HasUses() &&
			    inst.Arg(0).IsImmediate() && inst.Arg(0).U32() >= MarkerBase) {
				const auto index = inst.Arg(0).U32() - MarkerBase;
				if (index < translation.marker_codes.size()) {
					live.insert(static_cast<uint16_t>(translation.marker_codes[index]));
				}
			}
			if (inst.GetOpcode() == IR::ValueOpcode::CallStateProbe && inst.Arg(0).IsImmediate() &&
			    inst.Arg(0).U32() != 0xffffffffu) {
				const auto key  = inst.Arg(0).U32();
				const auto site = key >> 16u;
				const auto code = static_cast<uint16_t>(key & 0xffffu);
				if (site == 0 || site > builders.size()) continue;
				int32_t root = 0;
				if (builders[site - 1u].Build(inst.Arg(1), root)) {
					report.recipe.sites[site - 1u].roots[code] = root;
				}
			}
		}
	}
	report.live.assign(live.begin(), live.end());
}

} // namespace Libs::Graphics::ShaderRecompiler::CallSplit
