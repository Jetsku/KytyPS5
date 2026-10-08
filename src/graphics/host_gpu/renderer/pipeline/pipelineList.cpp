#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <map>
#include <mutex>
#include <xxhash.h>
#include <zstd.h>

namespace Libs::Graphics {

namespace {

template <typename T>
void Put(std::vector<uint8_t>& out, T value) {
	const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
	out.insert(out.end(), bytes, bytes + sizeof(value));
}

void PutBytes(std::vector<uint8_t>& out, std::span<const uint8_t> bytes) {
	Put<uint32_t>(out, static_cast<uint32_t>(bytes.size()));
	out.insert(out.end(), bytes.begin(), bytes.end());
}

void PutText(std::vector<uint8_t>& out, const std::string& text) {
	PutBytes(out, std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}

template <typename Word>
void PutArray(std::vector<uint8_t>& out, std::span<const Word> words) {
	Put<uint32_t>(out, static_cast<uint32_t>(words.size()));
	const auto* bytes = reinterpret_cast<const uint8_t*>(words.data());
	out.insert(out.end(), bytes, bytes + words.size_bytes());
}

// Bounds-checked reader over one section: a count never allocates more than the rest of the section
// could hold.
class Reader {
public:
	explicit Reader(std::span<const uint8_t> bytes): m_bytes(bytes) {}

	template <typename T>
	T Get() {
		T value {};
		if (m_failed || sizeof(T) > m_bytes.size() - m_position) {
			m_failed = true;
			return value;
		}
		std::memcpy(&value, m_bytes.data() + m_position, sizeof(T));
		m_position += sizeof(T);
		return value;
	}
	// A count of items of at least `min_item_bytes` each.
	uint32_t Count(size_t min_item_bytes) {
		const auto count = Get<uint32_t>();
		if (m_failed || static_cast<uint64_t>(count) * std::max<size_t>(1, min_item_bytes) > m_bytes.size() - m_position) {
			m_failed = true;
			return 0;
		}
		return count;
	}
	template <typename Word>
	std::vector<Word> Array() {
		const auto        count = Count(sizeof(Word));
		std::vector<Word> out(count);
		if (count != 0) std::memcpy(out.data(), m_bytes.data() + m_position, count * sizeof(Word));
		m_position += static_cast<size_t>(count) * sizeof(Word);
		return out;
	}
	std::vector<uint8_t> Bytes() { return Array<uint8_t>(); }
	std::string          Text() {
		const auto bytes = Bytes();
		return {bytes.begin(), bytes.end()};
	}
	[[nodiscard]] bool Failed() const { return m_failed; }
	[[nodiscard]] bool AtEnd() const { return m_position == m_bytes.size(); }

private:
	std::span<const uint8_t> m_bytes;
	size_t                   m_position = 0;
	bool                     m_failed   = false;
};

bool Fail(std::string* error, const std::string& message) {
	if (error != nullptr) *error = message;
	return false;
}

// Section header: id, flags (bit 0: zstd), stored size, raw size, XXH3-64 of the raw bytes.
constexpr size_t   SectionHeaderBytes = 4 + 4 + 8 + 8 + 8;
constexpr uint32_t SectionZstd        = 1;
constexpr uint64_t MaxSectionBytes    = 256ull * 1024 * 1024;

std::vector<uint8_t> EncodeInfo(const PipelineList::Info& info) {
	std::vector<uint8_t> out;
	PutText(out, info.title_id);
	PutText(out, info.app_version);
	PutText(out, info.producer);
	for (const auto size: info.input_info_sizes) Put<uint32_t>(out, size);
	Put<uint64_t>(out, info.created_unix);
	return out;
}

std::vector<uint8_t> EncodeLevels(const std::vector<std::string>& levels) {
	std::vector<uint8_t> out;
	Put<uint32_t>(out, static_cast<uint32_t>(levels.size()));
	for (const auto& level: levels) PutText(out, level);
	return out;
}

std::vector<uint8_t> EncodeShaders(const std::vector<PipelineList::Shader>& shaders) {
	std::vector<uint8_t> out;
	Put<uint32_t>(out, static_cast<uint32_t>(shaders.size()));
	for (const auto& s: shaders) {
		Put<uint32_t>(out, static_cast<uint32_t>(s.file));
		Put<uint64_t>(out, s.offset);
		Put<uint32_t>(out, s.code_size);
		Put<uint64_t>(out, s.code_hash_low);
		Put<uint64_t>(out, s.code_hash_high);
		Put<uint64_t>(out, s.prefix_hash);
		Put<uint32_t>(out, s.stage);
		Put<uint8_t>(out, static_cast<uint8_t>(s.kind));
		Put<uint64_t>(out, s.hash);
		Put<uint32_t>(out, s.user_data_count);
		Put<uint32_t>(out, s.wave_size);
		Put<uint32_t>(out, s.user_data_base);
		Put<uint8_t>(out, s.plain_mip_stats_variant ? 1u : 0u);
		PutArray<uint32_t>(out, s.static_state);
		PutBytes(out, s.input_info);
	}
	return out;
}

std::vector<uint8_t> EncodePermutations(const std::vector<PipelineList::Permutation>& permutations) {
	std::vector<uint8_t> out;
	Put<uint32_t>(out, static_cast<uint32_t>(permutations.size()));
	for (const auto& p: permutations) {
		Put<uint32_t>(out, p.shader);
		Put<uint32_t>(out, p.push_data_cursor);
		PutBytes(out, p.specialization);
	}
	return out;
}

std::vector<uint8_t> EncodePipelines(const std::vector<PipelineList::Pipeline>& pipelines) {
	std::vector<uint8_t> out;
	Put<uint32_t>(out, static_cast<uint32_t>(pipelines.size()));
	for (const auto& p: pipelines) {
		Put<uint64_t>(out, p.key);
		PutArray<uint16_t>(out, p.levels);
		Put<uint32_t>(out, static_cast<uint32_t>(p.stages.size()));
		for (const auto& stage: p.stages) {
			Put<uint32_t>(out, stage.permutation);
			Put<uint8_t>(out, stage.plain ? 1u : 0u);
			Put<uint64_t>(out, stage.bindings);
		}
		Put<uint32_t>(out, p.snapshot_words);
		PutArray<uint32_t>(out, p.words);
	}
	return out;
}

bool DecodeInfo(std::span<const uint8_t> bytes, PipelineList::Info& info) {
	Reader r(bytes);
	info.title_id    = r.Text();
	info.app_version = r.Text();
	info.producer    = r.Text();
	for (auto& size: info.input_info_sizes) size = r.Get<uint32_t>();
	info.created_unix = r.Get<uint64_t>();
	return !r.Failed(); // later formats may append fields
}

bool DecodeLevels(std::span<const uint8_t> bytes, std::vector<std::string>& levels) {
	Reader     r(bytes);
	const auto count = r.Count(4);
	for (uint32_t i = 0; i < count && !r.Failed(); i++) levels.push_back(r.Text());
	return !r.Failed() && r.AtEnd() && levels.size() <= UINT16_MAX;
}

bool DecodeShaders(std::span<const uint8_t> bytes, std::vector<PipelineList::Shader>& shaders) {
	Reader     r(bytes);
	const auto count = r.Count(74);
	shaders.reserve(count);
	for (uint32_t i = 0; i < count && !r.Failed(); i++) {
		PipelineList::Shader s;
		const auto file           = r.Get<uint32_t>();
		s.offset                  = r.Get<uint64_t>();
		s.code_size               = r.Get<uint32_t>();
		s.code_hash_low           = r.Get<uint64_t>();
		s.code_hash_high          = r.Get<uint64_t>();
		s.prefix_hash             = r.Get<uint64_t>();
		s.stage                   = r.Get<uint32_t>();
		const auto kind           = r.Get<uint8_t>();
		s.hash                    = r.Get<uint64_t>();
		s.user_data_count         = r.Get<uint32_t>();
		s.wave_size               = r.Get<uint32_t>();
		s.user_data_base          = r.Get<uint32_t>();
		s.plain_mip_stats_variant = r.Get<uint8_t>() != 0;
		s.static_state            = r.Array<uint32_t>();
		s.input_info              = r.Bytes();
		if (file != static_cast<uint32_t>(PipelineList::CodeFile::Executable) ||
		    kind > static_cast<uint8_t>(ShaderJournal::Kind::Compute) || s.code_size == 0 ||
		    s.code_size > (64u << 20u) / 4u) {
			return false;
		}
		s.file = static_cast<PipelineList::CodeFile>(file);
		s.kind = static_cast<ShaderJournal::Kind>(kind);
		shaders.push_back(std::move(s));
	}
	return !r.Failed() && r.AtEnd();
}

bool DecodePermutations(std::span<const uint8_t> bytes, size_t shaders, std::vector<PipelineList::Permutation>& permutations) {
	Reader     r(bytes);
	const auto count = r.Count(12);
	permutations.reserve(count);
	for (uint32_t i = 0; i < count && !r.Failed(); i++) {
		PipelineList::Permutation p;
		p.shader           = r.Get<uint32_t>();
		p.push_data_cursor = r.Get<uint32_t>();
		p.specialization   = r.Bytes();
		if (p.shader >= shaders) return false;
		permutations.push_back(std::move(p));
	}
	return !r.Failed() && r.AtEnd();
}

bool DecodePipelines(std::span<const uint8_t> bytes, size_t permutations, size_t levels,
                     std::vector<PipelineList::Pipeline>& pipelines) {
	Reader     r(bytes);
	const auto count = r.Count(24);
	pipelines.reserve(count);
	for (uint32_t i = 0; i < count && !r.Failed(); i++) {
		PipelineList::Pipeline p;
		p.key    = r.Get<uint64_t>();
		p.levels = r.Array<uint16_t>();
		const auto stages = r.Count(13);
		if (stages == 0 || stages > 2) return false;
		for (uint32_t j = 0; j < stages; j++) {
			PipelineList::Stage stage;
			stage.permutation = r.Get<uint32_t>();
			stage.plain       = r.Get<uint8_t>() != 0;
			stage.bindings    = r.Get<uint64_t>();
			if (stage.permutation >= permutations) return false;
			p.stages.push_back(stage);
		}
		p.snapshot_words = r.Get<uint32_t>();
		p.words          = r.Array<uint32_t>();
		if (r.Failed()) return false;
		for (const auto level: p.levels) {
			if (level >= levels) return false;
		}
		// The words are the snapshot, then the layout signature with its length in front.
		if (p.snapshot_words >= p.words.size() || p.words[p.snapshot_words] != p.words.size() - p.snapshot_words - 1) {
			return false;
		}
		pipelines.push_back(std::move(p));
	}
	return !r.Failed() && r.AtEnd();
}

uint64_t ShaderDigest(const PipelineList::Shader& s) {
	std::vector<uint8_t> bytes;
	Put<uint32_t>(bytes, s.stage);
	Put<uint8_t>(bytes, static_cast<uint8_t>(s.kind));
	Put<uint64_t>(bytes, s.hash);
	Put<uint32_t>(bytes, s.user_data_count);
	Put<uint32_t>(bytes, s.code_size);
	Put<uint32_t>(bytes, s.wave_size);
	Put<uint32_t>(bytes, s.user_data_base);
	Put<uint8_t>(bytes, s.plain_mip_stats_variant ? 1u : 0u);
	PutArray<uint32_t>(bytes, s.static_state);
	return XXH3_64bits(bytes.data(), bytes.size());
}

bool SameShader(const PipelineList::Shader& a, const PipelineList::Shader& b) {
	return a.stage == b.stage && a.kind == b.kind && a.hash == b.hash && a.user_data_count == b.user_data_count &&
	       a.code_size == b.code_size && a.wave_size == b.wave_size && a.user_data_base == b.user_data_base &&
	       a.plain_mip_stats_variant == b.plain_mip_stats_variant && a.static_state == b.static_state;
}

uint64_t PermutationDigest(const PipelineList::Permutation& p) {
	return XXH3_64bits_withSeed(p.specialization.data(), p.specialization.size(),
	                            (static_cast<uint64_t>(p.shader) << 32u) ^ p.push_data_cursor);
}

uint64_t PipelineDigest(const PipelineList::Pipeline& p) {
	std::vector<uint32_t> words;
	for (const auto& stage: p.stages) {
		words.push_back(stage.permutation);
		words.push_back(stage.plain ? 1u : 0u);
		words.push_back(static_cast<uint32_t>(stage.bindings));
		words.push_back(static_cast<uint32_t>(stage.bindings >> 32u));
	}
	words.push_back(p.snapshot_words);
	words.insert(words.end(), p.words.begin(), p.words.end());
	return XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
}

bool SamePipeline(const PipelineList::Pipeline& a, const PipelineList::Pipeline& b) {
	if (a.stages.size() != b.stages.size() || a.snapshot_words != b.snapshot_words || a.words != b.words) return false;
	for (size_t i = 0; i < a.stages.size(); i++) {
		if (a.stages[i].permutation != b.stages[i].permutation || a.stages[i].plain != b.stages[i].plain ||
		    a.stages[i].bindings != b.stages[i].bindings) {
			return false;
		}
	}
	return true;
}

std::span<const uint8_t> AsBytes(std::span<const uint32_t> words) {
	return {reinterpret_cast<const uint8_t*>(words.data()), words.size_bytes()};
}

} // namespace

namespace {

std::mutex&            ExecutableMutex() {
	static std::mutex mutex;
	return mutex;
}
std::filesystem::path& ExecutablePath() {
	static std::filesystem::path path;
	return path;
}

} // namespace

void SetPipelineListExecutable(const std::filesystem::path& path) {
	std::scoped_lock lock(ExecutableMutex());
	ExecutablePath() = path;
}

std::filesystem::path PipelineListExecutable() {
	std::scoped_lock lock(ExecutableMutex());
	return ExecutablePath();
}

// Pipeline journal payload ---------------------------------------------------------------------

std::vector<uint8_t> EncodePipelineJournalPayload(const std::string& level, std::span<const uint32_t> words,
                                                  uint32_t snapshot_words,
                                                  std::span<const std::pair<uint64_t, uint64_t>> identities) {
	std::vector<uint32_t> out {PipelineJournalPayloadMagic, snapshot_words, static_cast<uint32_t>(level.size())};
	for (size_t i = 0; i < level.size(); i += 4) {
		uint32_t word = 0;
		std::memcpy(&word, level.data() + i, std::min<size_t>(4, level.size() - i));
		out.push_back(word);
	}
	out.push_back(static_cast<uint32_t>(identities.size()));
	for (const auto& [identity, bindings]: identities) {
		out.push_back(static_cast<uint32_t>(identity));
		out.push_back(static_cast<uint32_t>(identity >> 32u));
		out.push_back(static_cast<uint32_t>(bindings));
		out.push_back(static_cast<uint32_t>(bindings >> 32u));
	}
	out.insert(out.end(), words.begin(), words.end());
	const auto* bytes = reinterpret_cast<const uint8_t*>(out.data());
	return {bytes, bytes + out.size() * sizeof(uint32_t)};
}

bool DecodePipelineJournalPayload(std::span<const uint8_t> bytes, PipelineJournalPayload& out) {
	if (bytes.size() % 4 != 0 || bytes.size() < 12) return false;
	std::vector<uint32_t> words(bytes.size() / 4);
	std::memcpy(words.data(), bytes.data(), bytes.size());
	if (words[0] != PipelineJournalPayloadMagic || words[2] > 128) return false;
	const size_t level_words = (words[2] + 3) / 4;
	if (3 + level_words > words.size()) return false;
	out.level.resize(words[2]);
	if (words[2] != 0) std::memcpy(out.level.data(), words.data() + 3, words[2]);
	size_t position = 3 + level_words;
	if (position >= words.size() || words[position] > 2) return false;
	const auto stages = words[position++];
	if (position + stages * 4 > words.size()) return false;
	out.identities.clear();
	for (uint32_t i = 0; i < stages; i++, position += 4) {
		out.identities.emplace_back(words[position] | (static_cast<uint64_t>(words[position + 1]) << 32u),
		                            words[position + 2] | (static_cast<uint64_t>(words[position + 3]) << 32u));
	}
	out.words.assign(words.begin() + static_cast<std::ptrdiff_t>(position), words.end());
	out.snapshot_words = words[1];
	return out.snapshot_words < out.words.size() &&
	       out.words[out.snapshot_words] == out.words.size() - out.snapshot_words - 1;
}

uint64_t PermutationIdentity(uint32_t stage, uint64_t hash, uint32_t user_data_count, uint32_t code_size,
                             uint32_t wave_size, uint32_t user_data_base, bool plain_mip_stats_variant,
                             uint32_t push_data_cursor, std::span<const uint32_t> static_state,
                             std::span<const uint8_t> specialization) {
	std::vector<uint32_t> words {stage,
	                             static_cast<uint32_t>(hash),
	                             static_cast<uint32_t>(hash >> 32u),
	                             user_data_count,
	                             code_size,
	                             wave_size,
	                             user_data_base,
	                             plain_mip_stats_variant ? 1u : 0u,
	                             push_data_cursor,
	                             static_cast<uint32_t>(static_state.size())};
	words.insert(words.end(), static_state.begin(), static_state.end());
	return XXH3_64bits_withSeed(specialization.data(), specialization.size(),
	                            XXH3_64bits(words.data(), words.size() * sizeof(uint32_t)));
}

uint64_t PermutationIdentity(const ShaderJournal::Source& source, const ShaderJournal::Entry& entry) {
	return PermutationIdentity(source.stage, source.hash, source.user_data_count, source.code_size, source.wave_size,
	                           source.user_data_base, source.plain_mip_stats_variant, entry.push_data_cursor,
	                           source.static_state, entry.specialization);
}

namespace PipelineList {

// Encoding ---------------------------------------------------------------------------------------

std::vector<uint8_t> Encode(const File& file, int zstd_level) {
	const std::pair<SectionId, std::vector<uint8_t>> sections[] = {
	    {SectionId::Info, EncodeInfo(file.info)},
	    {SectionId::Levels, EncodeLevels(file.levels)},
	    {SectionId::Shaders, EncodeShaders(file.shaders)},
	    {SectionId::Permutations, EncodePermutations(file.permutations)},
	    {SectionId::Pipelines, EncodePipelines(file.pipelines)},
	};
	std::vector<uint8_t> out(FileMagic, FileMagic + sizeof(FileMagic));
	Put<uint32_t>(out, FormatVersion);
	Put<uint32_t>(out, static_cast<uint32_t>(std::size(sections)));
	for (const auto& [id, raw]: sections) {
		std::vector<uint8_t> stored(ZSTD_compressBound(raw.size()));
		const auto           size = ZSTD_compress(stored.data(), stored.size(), raw.data(), raw.size(), zstd_level);
		uint32_t             flags = SectionZstd;
		if (ZSTD_isError(size) != 0) {
			stored = raw;
			flags  = 0;
		} else {
			stored.resize(size);
		}
		Put<uint32_t>(out, static_cast<uint32_t>(id));
		Put<uint32_t>(out, flags);
		Put<uint64_t>(out, stored.size());
		Put<uint64_t>(out, raw.size());
		Put<uint64_t>(out, XXH3_64bits(raw.data(), raw.size()));
		out.insert(out.end(), stored.begin(), stored.end());
	}
	return out;
}

bool DecodeSections(std::span<const uint8_t> bytes, std::vector<std::pair<uint32_t, std::vector<uint8_t>>>& sections,
                    std::string* error) {
	sections.clear();
	if (bytes.size() < sizeof(FileMagic) + 8 || std::memcmp(bytes.data(), FileMagic, sizeof(FileMagic)) != 0) {
		return Fail(error, "not a KYPLST1 file");
	}
	uint32_t version = 0, count = 0;
	std::memcpy(&version, bytes.data() + 8, 4);
	std::memcpy(&count, bytes.data() + 12, 4);
	if (version != FormatVersion) return Fail(error, "format version " + std::to_string(version));
	size_t position = sizeof(FileMagic) + 8;
	for (uint32_t i = 0; i < count; i++) {
		if (bytes.size() - position < SectionHeaderBytes) return Fail(error, "truncated section header");
		uint32_t id = 0, flags = 0;
		uint64_t stored = 0, raw = 0, hash = 0;
		std::memcpy(&id, bytes.data() + position, 4);
		std::memcpy(&flags, bytes.data() + position + 4, 4);
		std::memcpy(&stored, bytes.data() + position + 8, 8);
		std::memcpy(&raw, bytes.data() + position + 16, 8);
		std::memcpy(&hash, bytes.data() + position + 24, 8);
		position += SectionHeaderBytes;
		if (stored > bytes.size() - position || raw > MaxSectionBytes) return Fail(error, "section size out of bounds");
		const auto           data = bytes.subspan(position, static_cast<size_t>(stored));
		std::vector<uint8_t> out;
		if ((flags & SectionZstd) != 0) {
			out.resize(static_cast<size_t>(raw));
			const auto size = ZSTD_decompress(out.data(), out.size(), data.data(), data.size());
			if (ZSTD_isError(size) != 0 || size != raw) return Fail(error, "section decompression failed");
		} else {
			if (stored != raw) return Fail(error, "section size mismatch");
			out.assign(data.begin(), data.end());
		}
		if (XXH3_64bits(out.data(), out.size()) != hash) return Fail(error, "section checksum mismatch");
		sections.emplace_back(id, std::move(out));
		position += static_cast<size_t>(stored);
	}
	return true;
}

bool Decode(std::span<const uint8_t> bytes, File& file, std::string* error) {
	file = {};
	std::vector<std::pair<uint32_t, std::vector<uint8_t>>> sections;
	if (!DecodeSections(bytes, sections, error)) return false;
	const auto find = [&](SectionId id) -> const std::vector<uint8_t>* {
		for (const auto& [section, raw]: sections) {
			if (section == static_cast<uint32_t>(id)) return &raw;
		}
		return nullptr;
	};
	const auto* info         = find(SectionId::Info);
	const auto* levels       = find(SectionId::Levels);
	const auto* shaders      = find(SectionId::Shaders);
	const auto* permutations = find(SectionId::Permutations);
	const auto* pipelines    = find(SectionId::Pipelines);
	if (info == nullptr || levels == nullptr || shaders == nullptr || permutations == nullptr || pipelines == nullptr) {
		return Fail(error, "a required section is missing");
	}
	if (!DecodeInfo(*info, file.info)) return Fail(error, "malformed info section");
	if (!DecodeLevels(*levels, file.levels)) return Fail(error, "malformed levels section");
	if (!DecodeShaders(*shaders, file.shaders)) return Fail(error, "malformed shaders section");
	if (!DecodePermutations(*permutations, file.shaders.size(), file.permutations)) {
		return Fail(error, "malformed permutations section");
	}
	if (!DecodePipelines(*pipelines, file.permutations.size(), file.levels.size(), file.pipelines)) {
		return Fail(error, "malformed pipelines section");
	}
	return true;
}

// Snapshot words (pipelineSnapshot.cpp, GraphicsPipelineSnapshot::Serialize): magic, mesh flag, view
// mask, depth and stencil formats, color format count and formats, stage count, then per stage its
// flags, the module hash (two words), the entry point (length and packed characters) and the
// required subgroup size.
bool RewriteSnapshotModules(std::span<uint32_t> words, const std::function<uint64_t(uint32_t)>& hash_for) {
	constexpr uint32_t SnapshotMagic = 0x324e5350u; // "PSN2"
	const size_t       size          = words.size();
	if (size < 7 || words[0] != SnapshotMagic) return false;
	size_t     position = 5; // magic, mesh flag, view mask, depth format, stencil format
	const auto colors   = words[position++];
	if (colors > 8 || size - position < colors + 1) return false;
	position += colors;
	const auto stages = words[position++];
	if (stages > 2) return false;
	for (uint32_t i = 0; i < stages; i++) {
		if (size - position < 4) return false; // flags, hash (two words), entry point length
		position++;
		const auto hash     = hash_for(i);
		words[position]     = static_cast<uint32_t>(hash);
		words[position + 1] = static_cast<uint32_t>(hash >> 32u);
		position += 2;
		const auto name = words[position++];
		if (name > 256 || size - position < (name + 3) / 4 + 1) return false;
		position += (name + 3) / 4 + 1; // entry point, required subgroup size
	}
	return true;
}

Hash128 CodeHash(std::span<const uint8_t> code) {
	const auto hash = XXH3_128bits(code.data(), code.size());
	return {hash.low64, hash.high64};
}

uint64_t CodePrefixHash(std::span<const uint8_t> code) {
	return XXH3_64bits(code.data(), std::min<size_t>(64, code.size()));
}

CodeQuery QueryFor(std::span<const uint8_t> code) {
	CodeQuery query;
	query.size_bytes  = static_cast<uint32_t>(code.size());
	query.first_known = code.size() >= 4;
	if (query.first_known) std::memcpy(&query.first_word, code.data(), 4);
	query.prefix_hash = CodePrefixHash(code);
	query.hash        = CodeHash(code);
	return query;
}

std::vector<std::optional<uint64_t>> FindCode(std::span<const uint8_t> image, std::span<const CodeQuery> queries) {
	std::vector<std::optional<uint64_t>> found(queries.size());
	size_t                               open = 0;
	// `prefix`: the 64-byte prefix hash at `offset`, computed once per offset for every candidate.
	const auto matches = [&](size_t q, size_t offset, std::optional<uint64_t>& prefix) {
		const auto& query = queries[q];
		if (found[q].has_value() || query.size_bytes > image.size() - offset) return;
		const auto code = image.subspan(offset, query.size_bytes);
		uint64_t   hash = 0;
		if (query.size_bytes >= 64) {
			if (!prefix.has_value()) prefix = XXH3_64bits(image.data() + offset, 64);
			hash = *prefix;
		} else {
			hash = CodePrefixHash(code);
		}
		if (hash != query.prefix_hash || !(CodeHash(code) == query.hash)) return;
		found[q] = offset;
		open--;
	};
	// Queries with a known first word: a bitmap filter on it, then the hashes.
	std::unordered_multimap<uint32_t, size_t> by_word;
	std::vector<uint64_t>                     bitmap(1u << 18u, 0); // 2^24 bits
	const auto                                bit = [](uint32_t word) { return (word * 0x9E3779B1u) >> 8u; };
	// The others by prefix length (the hash of the first min(64, size) bytes at every offset).
	std::map<uint32_t, std::unordered_multimap<uint64_t, size_t>> by_prefix;
	for (size_t q = 0; q < queries.size(); q++) {
		if (queries[q].size_bytes == 0 || queries[q].size_bytes > image.size()) continue;
		open++;
		if (queries[q].first_known) {
			by_word.emplace(queries[q].first_word, q);
			const auto b = bit(queries[q].first_word);
			bitmap[b >> 6u] |= 1ull << (b & 63u);
		} else {
			by_prefix[std::min<uint32_t>(64, queries[q].size_bytes)].emplace(queries[q].prefix_hash, q);
		}
	}
	if (!by_word.empty() && image.size() >= 4) {
		for (size_t offset = 0; offset + 4 <= image.size() && open != 0; offset++) {
			uint32_t word = 0;
			std::memcpy(&word, image.data() + offset, 4);
			const auto b = bit(word);
			if ((bitmap[b >> 6u] & (1ull << (b & 63u))) == 0) continue;
			const auto              range = by_word.equal_range(word);
			std::optional<uint64_t> prefix;
			for (auto it = range.first; it != range.second; ++it) matches(it->second, offset, prefix);
		}
	}
	for (const auto& [length, table]: by_prefix) {
		for (size_t offset = 0; offset + length <= image.size() && open != 0; offset++) {
			std::optional<uint64_t> prefix = XXH3_64bits(image.data() + offset, length);
			const auto              range  = table.equal_range(*prefix);
			if (length < 64) prefix.reset();
			for (auto it = range.first; it != range.second; ++it) matches(it->second, offset, prefix);
		}
	}
	return found;
}

// Builder ----------------------------------------------------------------------------------------

Builder::Builder(Info info) {
	m_file.info = std::move(info);
}

uint32_t Builder::AddShader(const Shader& shader, bool& added) {
	const auto digest = ShaderDigest(shader);
	const auto range  = m_shaders.equal_range(digest);
	for (auto it = range.first; it != range.second; ++it) {
		if (SameShader(m_file.shaders[it->second], shader)) {
			added = false;
			return it->second;
		}
	}
	const auto index = static_cast<uint32_t>(m_file.shaders.size());
	m_file.shaders.push_back(shader);
	m_shaders.emplace(digest, index);
	added = true;
	return index;
}

uint32_t Builder::AddPermutation(const Permutation& permutation, bool& added) {
	const auto digest = PermutationDigest(permutation);
	const auto range  = m_permutations.equal_range(digest);
	for (auto it = range.first; it != range.second; ++it) {
		const auto& p = m_file.permutations[it->second];
		if (p.shader == permutation.shader && p.push_data_cursor == permutation.push_data_cursor &&
		    p.specialization == permutation.specialization) {
			added = false;
			return it->second;
		}
	}
	const auto index = static_cast<uint32_t>(m_file.permutations.size());
	m_file.permutations.push_back(permutation);
	m_permutations.emplace(digest, index);
	added = true;
	return index;
}

uint16_t Builder::AddLevel(const std::string& level) {
	if (const auto found = m_levels.find(level); found != m_levels.end()) return found->second;
	const auto index = static_cast<uint16_t>(m_file.levels.size());
	m_file.levels.push_back(level);
	m_levels.emplace(level, index);
	return index;
}

void Builder::AddPipeline(Pipeline pipeline, ExportStats& stats) {
	std::sort(pipeline.levels.begin(), pipeline.levels.end());
	pipeline.levels.erase(std::unique(pipeline.levels.begin(), pipeline.levels.end()), pipeline.levels.end());
	const auto digest = PipelineDigest(pipeline);
	const auto range  = m_pipelines.equal_range(digest);
	for (auto it = range.first; it != range.second; ++it) {
		auto& kept = m_file.pipelines[it->second];
		if (SamePipeline(kept, pipeline)) {
			kept.levels.insert(kept.levels.end(), pipeline.levels.begin(), pipeline.levels.end());
			std::sort(kept.levels.begin(), kept.levels.end());
			kept.levels.erase(std::unique(kept.levels.begin(), kept.levels.end()), kept.levels.end());
			stats.pipelines_merged++;
			return;
		}
	}
	m_pipelines.emplace(digest, static_cast<uint32_t>(m_file.pipelines.size()));
	m_file.pipelines.push_back(std::move(pipeline));
	stats.pipelines_kept++;
}

void Builder::AddJournals(const std::vector<ShaderJournal::Source>& sources,
                          const std::vector<ShaderJournal::Entry>& entries,
                          const std::vector<PipelineJournal::Record>& pipelines, const LocateFn& locate,
                          ExportStats& stats) {
	// Sources -> list shaders (located code only).
	std::vector<uint32_t> shader_of(sources.size(), UINT32_MAX);
	for (size_t i = 0; i < sources.size(); i++) {
		const auto& source = sources[i];
		stats.sources++;
		if (source.code.empty() || source.code.size() != source.code_size) {
			stats.sources_unlocated++;
			continue;
		}
		const auto offset = locate(source.code);
		if (!offset.has_value()) {
			stats.sources_unlocated++;
			continue;
		}
		const auto bytes = AsBytes(source.code);
		Shader     shader;
		shader.file      = CodeFile::Executable;
		shader.offset    = *offset;
		shader.code_size = source.code_size;
		const auto hash  = CodeHash(bytes);
		shader.code_hash_low           = hash.low;
		shader.code_hash_high          = hash.high;
		shader.prefix_hash             = CodePrefixHash(bytes);
		shader.stage                   = source.stage;
		shader.kind                    = source.kind;
		shader.hash                    = source.hash;
		shader.user_data_count         = source.user_data_count;
		shader.wave_size               = source.wave_size;
		shader.user_data_base          = source.user_data_base;
		shader.plain_mip_stats_variant = source.plain_mip_stats_variant;
		shader.static_state            = source.static_state;
		shader.input_info              = source.input_info;
		bool added = false;
		shader_of[i] = AddShader(shader, added);
		if (added) {
			stats.sources_located++;
		} else {
			stats.sources_duplicate++;
		}
	}
	// Entries -> permutations; the recording run's permutation identities (and their plain-module
	// variants) -> list permutations.
	std::unordered_map<uint64_t, std::pair<uint32_t, bool>> by_identity;
	for (const auto& entry: entries) {
		stats.entries++;
		if (entry.source >= sources.size() || shader_of[entry.source] == UINT32_MAX) {
			stats.entries_dropped++;
			continue;
		}
		Permutation permutation;
		permutation.shader           = shader_of[entry.source];
		permutation.push_data_cursor = entry.push_data_cursor;
		permutation.specialization   = entry.specialization;
		bool       added = false;
		const auto index = AddPermutation(permutation, added);
		if (added) stats.entries_kept++;
		const auto identity = PermutationIdentity(sources[entry.source], entry);
		by_identity.try_emplace(identity, index, false);
		by_identity.try_emplace(identity ^ PlainModuleIdentitySalt, index, true);
	}
	for (const auto& record: pipelines) {
		stats.pipelines++;
		PipelineJournalPayload payload;
		if (!DecodePipelineJournalPayload(record.payload, payload) || payload.identities.empty()) {
			stats.pipelines_bad++;
			continue;
		}
		Pipeline pipeline;
		pipeline.key = record.key;
		bool resolved = true;
		for (const auto& [identity, bindings]: payload.identities) {
			const auto found = by_identity.find(identity);
			if (identity == 0 || found == by_identity.end()) {
				resolved = false;
				break;
			}
			pipeline.stages.push_back({found->second.first, found->second.second, bindings});
		}
		if (!resolved) {
			stats.pipelines_unresolved++;
			continue;
		}
		pipeline.words          = payload.words;
		pipeline.snapshot_words = payload.snapshot_words;
		if (!RewriteSnapshotModules(std::span(pipeline.words).first(pipeline.snapshot_words),
		                            [](uint32_t) { return uint64_t {0}; })) {
			stats.pipelines_bad++;
			continue;
		}
		if (!payload.level.empty()) pipeline.levels.push_back(AddLevel(payload.level));
		AddPipeline(std::move(pipeline), stats);
	}
}

void Builder::AddList(const File& other, ExportStats& stats) {
	std::vector<uint32_t> shader_of(other.shaders.size());
	for (size_t i = 0; i < other.shaders.size(); i++) {
		bool added   = false;
		shader_of[i] = AddShader(other.shaders[i], added);
		stats.sources++;
		if (added) {
			stats.sources_located++;
		} else {
			stats.sources_duplicate++;
		}
	}
	std::vector<uint32_t> permutation_of(other.permutations.size());
	for (size_t i = 0; i < other.permutations.size(); i++) {
		auto permutation   = other.permutations[i];
		permutation.shader = shader_of[permutation.shader];
		bool added         = false;
		permutation_of[i]  = AddPermutation(permutation, added);
		stats.entries++;
		if (added) stats.entries_kept++;
	}
	for (const auto& source: other.pipelines) {
		stats.pipelines++;
		auto pipeline = source;
		for (auto& stage: pipeline.stages) stage.permutation = permutation_of[stage.permutation];
		pipeline.levels.clear();
		for (const auto level: source.levels) pipeline.levels.push_back(AddLevel(other.levels[level]));
		AddPipeline(std::move(pipeline), stats);
	}
}

// Resolving ----------------------------------------------------------------------------------------

Resolved Resolve(const File& file, const std::array<uint32_t, 3>& input_info_sizes, const ReadCodeFn& read,
                 const ReadImageFn& read_image, ResolveStats& stats) {
	Resolved out;
	out.sources.resize(file.shaders.size());
	out.usable.assign(file.shaders.size(), false);
	std::vector<size_t>    relocate;
	std::vector<CodeQuery> queries;
	stats.shaders += file.shaders.size();
	const auto accept = [&](size_t i, std::vector<uint32_t> code) {
		const auto& s      = file.shaders[i];
		auto&       source = out.sources[i];
		source.stage                   = s.stage;
		source.kind                    = s.kind;
		source.hash                    = s.hash;
		source.user_data_count         = s.user_data_count;
		source.code_size               = s.code_size;
		source.wave_size               = s.wave_size;
		source.user_data_base          = s.user_data_base;
		source.plain_mip_stats_variant = s.plain_mip_stats_variant;
		source.static_state            = s.static_state;
		source.input_info              = s.input_info;
		source.code                    = std::move(code);
		out.usable[i]                  = true;
		stats.resolved++;
	};
	const auto matches = [](const Shader& s, std::span<const uint8_t> bytes) {
		return CodeHash(bytes) == Hash128 {s.code_hash_low, s.code_hash_high} &&
		       XXH3_64bits(bytes.data(), bytes.size()) == s.hash;
	};
	for (size_t i = 0; i < file.shaders.size(); i++) {
		const auto& s = file.shaders[i];
		if (s.input_info.size() != input_info_sizes[static_cast<size_t>(s.kind)]) {
			stats.other_input_layout++;
			continue;
		}
		std::vector<uint32_t> code(s.code_size);
		const std::span       bytes(reinterpret_cast<uint8_t*>(code.data()), code.size() * sizeof(uint32_t));
		if (!read(s.file, s.offset, bytes)) {
			stats.unreadable++;
		} else if (matches(s, bytes)) {
			accept(i, std::move(code));
			continue;
		}
		relocate.push_back(i);
		CodeQuery query;
		query.size_bytes  = s.code_size * 4;
		query.prefix_hash = s.prefix_hash;
		query.hash        = {s.code_hash_low, s.code_hash_high};
		queries.push_back(query);
	}
	if (!relocate.empty() && read_image) {
		const auto image = read_image(CodeFile::Executable);
		const auto found = image.empty() ? std::vector<std::optional<uint64_t>>(queries.size())
		                                 : FindCode(image, queries);
		for (size_t q = 0; q < relocate.size(); q++) {
			const auto  i = relocate[q];
			const auto& s = file.shaders[i];
			if (!found[q].has_value()) {
				stats.mismatched++;
				continue;
			}
			std::vector<uint32_t> code(s.code_size);
			std::memcpy(code.data(), image.data() + *found[q], code.size() * sizeof(uint32_t));
			stats.relocated++;
			accept(i, std::move(code));
		}
	} else {
		stats.mismatched += relocate.size();
	}
	out.permutation_entry.assign(file.permutations.size(), UINT32_MAX);
	stats.permutations += file.permutations.size();
	for (size_t i = 0; i < file.permutations.size(); i++) {
		const auto& p = file.permutations[i];
		if (!out.usable[p.shader]) continue;
		out.permutation_entry[i] = static_cast<uint32_t>(out.entries.size());
		out.entries.push_back({p.shader, p.push_data_cursor, p.specialization});
		out.entry_permutation.push_back(static_cast<uint32_t>(i));
		stats.permutations_kept++;
	}
	stats.pipelines += file.pipelines.size();
	for (const auto& pipeline: file.pipelines) {
		if (std::all_of(pipeline.stages.begin(), pipeline.stages.end(), [&](const Stage& stage) {
			    return out.permutation_entry[stage.permutation] != UINT32_MAX;
		    })) {
			stats.pipelines_kept++;
		}
	}
	return out;
}

// No-code verification ------------------------------------------------------------------------------

namespace {

bool LowEntropy(const uint8_t* window) {
	std::array<bool, 256> seen {};
	uint32_t              distinct = 0;
	for (size_t i = 0; i < CodeWindows::WindowBytes && distinct < 8; i++) {
		if (!seen[window[i]]) {
			seen[window[i]] = true;
			distinct++;
		}
	}
	return distinct < 8;
}

} // namespace

void CodeWindows::Add(std::span<const uint8_t> code) {
	// Every byte offset: a copied run of 64 bytes anywhere is found, whatever its alignment.
	for (size_t i = 0; i + WindowBytes <= code.size(); i++) {
		if (LowEntropy(code.data() + i)) {
			m_low_entropy++;
			continue;
		}
		m_windows.insert(XXH3_64bits(code.data() + i, WindowBytes));
	}
}

std::vector<uint64_t> CodeWindows::FindIn(std::span<const uint8_t> data, size_t max_hits) const {
	std::vector<uint64_t> hits;
	if (m_windows.empty()) return hits;
	for (size_t i = 0; i + WindowBytes <= data.size() && hits.size() < max_hits; i++) {
		if (m_windows.contains(XXH3_64bits(data.data() + i, WindowBytes))) hits.push_back(i);
	}
	return hits;
}

} // namespace PipelineList

} // namespace Libs::Graphics
