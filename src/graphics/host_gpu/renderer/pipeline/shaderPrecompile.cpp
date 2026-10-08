#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <system_error>
#include <utility>
#include <xxhash.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace Libs::Graphics {

namespace {

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

template <typename T>
void Put(std::vector<uint8_t>& out, T value) {
	const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
	out.insert(out.end(), bytes, bytes + sizeof(value));
}

void PutBytes(std::vector<uint8_t>& out, std::span<const uint8_t> bytes) {
	Put<uint32_t>(out, static_cast<uint32_t>(bytes.size()));
	out.insert(out.end(), bytes.begin(), bytes.end());
}

void PutWords(std::vector<uint8_t>& out, std::span<const uint32_t> words) {
	Put<uint32_t>(out, static_cast<uint32_t>(words.size()));
	const auto* bytes = reinterpret_cast<const uint8_t*>(words.data());
	out.insert(out.end(), bytes, bytes + words.size_bytes());
}

// Bounds-checked cursor over one record's payload.
class Cursor {
public:
	explicit Cursor(std::span<const uint8_t> bytes): m_bytes(bytes) {}

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
	std::vector<uint8_t> Bytes() {
		const auto size = Get<uint32_t>();
		std::vector<uint8_t> out;
		if (m_failed || size > m_bytes.size() - m_position) {
			m_failed = true;
			return out;
		}
		out.assign(m_bytes.begin() + static_cast<std::ptrdiff_t>(m_position),
		           m_bytes.begin() + static_cast<std::ptrdiff_t>(m_position + size));
		m_position += size;
		return out;
	}
	std::vector<uint32_t> Words() {
		const auto count = Get<uint32_t>();
		std::vector<uint32_t> out;
		if (m_failed || static_cast<uint64_t>(count) * sizeof(uint32_t) > m_bytes.size() - m_position) {
			m_failed = true;
			return out;
		}
		out.resize(count);
		if (count != 0) std::memcpy(out.data(), m_bytes.data() + m_position, count * sizeof(uint32_t));
		m_position += static_cast<size_t>(count) * sizeof(uint32_t);
		return out;
	}
	[[nodiscard]] bool Ok() const { return !m_failed && m_position == m_bytes.size(); }

private:
	std::span<const uint8_t> m_bytes;
	size_t                   m_position = 0;
	bool                     m_failed   = false;
};

constexpr size_t RecordHeaderBytes = 4 + 4 + 8 + 8;

void AppendRecord(std::vector<uint8_t>& out, uint32_t kind, const std::vector<uint8_t>& payload) {
	Put<uint32_t>(out, ShaderJournal::RecordMagic);
	Put<uint32_t>(out, kind);
	Put<uint64_t>(out, payload.size());
	Put<uint64_t>(out, XXH3_64bits(payload.data(), payload.size()));
	out.insert(out.end(), payload.begin(), payload.end());
}

std::vector<uint8_t> BuildHeader(const std::vector<uint8_t>& identity) {
	std::vector<uint8_t> header(ShaderJournal::FileMagic,
	                            ShaderJournal::FileMagic + sizeof(ShaderJournal::FileMagic));
	Put<uint32_t>(header, ShaderJournal::FormatVersion);
	PutBytes(header, identity);
	Put<uint64_t>(header, XXH3_64bits(header.data(), header.size()));
	return header;
}

std::vector<uint8_t> EncodeSource(const ShaderJournal::Source& source) {
	std::vector<uint8_t> payload;
	Put<uint32_t>(payload, source.stage);
	Put<uint8_t>(payload, static_cast<uint8_t>(source.kind));
	Put<uint64_t>(payload, source.hash);
	Put<uint32_t>(payload, source.user_data_count);
	Put<uint32_t>(payload, source.wave_size);
	Put<uint32_t>(payload, source.user_data_base);
	Put<uint8_t>(payload, source.plain_mip_stats_variant ? 1u : 0u);
	PutWords(payload, source.static_state);
	PutWords(payload, source.code);
	PutBytes(payload, source.input_info);
	if (!source.back_code.empty()) PutWords(payload, source.back_code);
	return payload;
}

std::vector<uint8_t> EncodeEntry(uint32_t source, uint32_t cursor,
                                 std::span<const uint8_t> specialization) {
	std::vector<uint8_t> payload;
	Put<uint32_t>(payload, source);
	Put<uint32_t>(payload, cursor);
	PutBytes(payload, specialization);
	return payload;
}

uint64_t EntryDigest(uint32_t source, uint32_t cursor, std::span<const uint8_t> specialization) {
	std::vector<uint8_t> bytes;
	Put<uint32_t>(bytes, source);
	Put<uint32_t>(bytes, cursor);
	bytes.insert(bytes.end(), specialization.begin(), specialization.end());
	return XXH3_64bits(bytes.data(), bytes.size());
}

ShaderJournal::Source IdentityOf(const ShaderJournal::Source& source) {
	ShaderJournal::Source out;
	out.stage                   = source.stage;
	out.kind                    = source.kind;
	out.hash                    = source.hash;
	out.user_data_count         = source.user_data_count;
	out.code_size               = source.code_size;
	out.wave_size               = source.wave_size;
	out.user_data_base          = source.user_data_base;
	out.plain_mip_stats_variant = source.plain_mip_stats_variant;
	out.static_state            = source.static_state;
	out.code.resize(0);
	return out;
}

} // namespace

// ShaderJournal ------------------------------------------------------------------------------

uint64_t ShaderJournal::SourceDigest(const Source& source) {
	std::vector<uint8_t> bytes;
	Put<uint32_t>(bytes, source.stage);
	Put<uint8_t>(bytes, static_cast<uint8_t>(source.kind));
	Put<uint64_t>(bytes, source.hash);
	Put<uint32_t>(bytes, source.user_data_count);
	Put<uint32_t>(bytes, source.code_size);
	Put<uint32_t>(bytes, source.wave_size);
	Put<uint32_t>(bytes, source.user_data_base);
	Put<uint8_t>(bytes, source.plain_mip_stats_variant ? 1u : 0u);
	PutWords(bytes, source.static_state);
	return XXH3_64bits(bytes.data(), bytes.size());
}

bool ShaderJournal::SameIdentity(const Source& a, const Source& b) {
	return a.stage == b.stage && a.kind == b.kind && a.hash == b.hash &&
	       a.user_data_count == b.user_data_count && a.code_size == b.code_size && a.wave_size == b.wave_size &&
	       a.user_data_base == b.user_data_base &&
	       a.plain_mip_stats_variant == b.plain_mip_stats_variant && a.static_state == b.static_state;
}

void ShaderJournal::Log(const std::string& message) const {
	if (m_settings.log) m_settings.log(message);
}

ShaderJournal::ShaderJournal(Settings settings): m_settings(std::move(settings)) {
	Load();
	if (m_settings.background_writer) {
		m_writer = std::thread([this] { RunWriter(); });
	}
}

ShaderJournal::~ShaderJournal() {
	{
		std::scoped_lock lock(m_writer_mutex);
		m_stop = true;
	}
	m_writer_cv.notify_all();
	if (m_writer.joinable()) m_writer.join();
	Flush();
}

void ShaderJournal::Load() {
	const auto begin = NowNs();
	std::vector<uint8_t> file;
	std::error_code      error;
	const auto           size = std::filesystem::file_size(m_settings.path, error);
	if (!error) {
		if (std::ifstream in(m_settings.path, std::ios::binary); in) {
			file.resize(static_cast<size_t>(size));
			if (!file.empty() &&
			    !in.read(reinterpret_cast<char*>(file.data()), static_cast<std::streamsize>(file.size()))) {
				file.clear();
			}
		}
		m_stats.file_found = !file.empty();
		m_stats.file_bytes = file.size();
	}
	if (!file.empty()) {
		const auto header = BuildHeader(m_settings.identity);
		if (file.size() < header.size() || std::memcmp(file.data(), header.data(), header.size()) != 0) {
			m_stats.header_rejected = true; // another identity, format or damaged: start over
		} else {
			size_t position = header.size();
			for (;;) {
				if (file.size() - position < RecordHeaderBytes) break;
				uint32_t magic = 0, kind = 0;
				uint64_t payload_size = 0, checksum = 0;
				std::memcpy(&magic, file.data() + position, 4);
				std::memcpy(&kind, file.data() + position + 4, 4);
				std::memcpy(&payload_size, file.data() + position + 8, 8);
				std::memcpy(&checksum, file.data() + position + 16, 8);
				if (magic != RecordMagic || payload_size > file.size() - position - RecordHeaderBytes) break;
				const std::span<const uint8_t> payload(file.data() + position + RecordHeaderBytes,
				                                       static_cast<size_t>(payload_size));
				if (XXH3_64bits(payload.data(), payload.size()) != checksum) break;
				Cursor cursor(payload);
				if (kind == RecordSource || kind == RecordMergedSource) {
					Source source;
					source.stage                   = cursor.Get<uint32_t>();
					const auto source_kind         = cursor.Get<uint8_t>();
					source.hash                    = cursor.Get<uint64_t>();
					source.user_data_count         = cursor.Get<uint32_t>();
					source.wave_size               = cursor.Get<uint32_t>();
					source.user_data_base          = cursor.Get<uint32_t>();
					source.plain_mip_stats_variant = cursor.Get<uint8_t>() != 0;
					source.static_state            = cursor.Words();
					source.code                    = cursor.Words();
					source.input_info              = cursor.Bytes();
					if (kind == RecordMergedSource) source.back_code = cursor.Words();
					source.code_size               = static_cast<uint32_t>(source.code.size());
					if (!cursor.Ok() || source_kind > static_cast<uint8_t>(Kind::Compute)) break;
					source.kind = static_cast<Kind>(source_kind);
					m_sources.push_back(std::move(source));
				} else if (kind == RecordEntry) {
					Entry entry;
					entry.source           = cursor.Get<uint32_t>();
					entry.push_data_cursor = cursor.Get<uint32_t>();
					entry.specialization   = cursor.Bytes();
					if (!cursor.Ok() || entry.source >= m_sources.size()) break;
					m_entries.push_back(std::move(entry));
				} else {
					break;
				}
				position += RecordHeaderBytes + payload.size();
			}
			m_file_valid_bytes  = position;
			m_stats.damaged_bytes = file.size() - position;
		}
	}
	// Index what was loaded (a source seen twice keeps its first index, later copies are dead
	// weight that replay skips by the entries' indices pointing at either).
	m_known.reserve(m_sources.size());
	for (uint32_t i = 0; i < m_sources.size(); i++) {
		m_known.push_back(IdentityOf(m_sources[i]));
		m_by_digest.emplace(SourceDigest(m_sources[i]), i);
	}
	for (const auto& entry: m_entries) {
		m_known_entries.insert(EntryDigest(entry.source, entry.push_data_cursor, entry.specialization));
	}
	m_next_source = static_cast<uint32_t>(m_sources.size());
	m_file_bytes  = m_file_valid_bytes;
	m_file_ready  = false;
	m_stats.load_ns = NowNs() - begin;
	if (m_stats.header_rejected) {
		Log("Shader journal: " + m_settings.path.string() +
		    " is from another device or format and is replaced");
	}
}

void ShaderJournal::ReleaseLoaded() {
	m_sources = {};
	m_entries = {};
}

ShaderJournal::Appended ShaderJournal::AppendReplayOnly(std::vector<Source> sources, const std::vector<Entry>& entries) {
	Appended appended;
	// Loaded sources by identity, and the loaded entries.
	std::unordered_multimap<uint64_t, uint32_t> loaded;
	for (uint32_t i = 0; i < m_sources.size(); i++) loaded.emplace(SourceDigest(m_sources[i]), i);
	std::unordered_set<uint64_t> loaded_entries;
	for (const auto& entry: m_entries) {
		loaded_entries.insert(EntryDigest(entry.source, entry.push_data_cursor, entry.specialization));
	}
	std::vector<uint32_t> index_of(sources.size(), UINT32_MAX);
	for (size_t i = 0; i < sources.size(); i++) {
		if (sources[i].code.empty() || sources[i].code.size() != sources[i].code_size) continue;
		const auto digest = SourceDigest(sources[i]);
		const auto range  = loaded.equal_range(digest);
		for (auto it = range.first; it != range.second; ++it) {
			if (SameIdentity(m_sources[it->second], sources[i])) {
				index_of[i] = it->second;
				break;
			}
		}
	}
	for (const auto& entry: entries) {
		if (entry.source >= sources.size()) continue;
		auto& index = index_of[entry.source];
		if (index == UINT32_MAX) {
			if (sources[entry.source].code.empty() || sources[entry.source].code.size() != sources[entry.source].code_size) {
				continue;
			}
			index = static_cast<uint32_t>(m_sources.size());
			loaded.emplace(SourceDigest(sources[entry.source]), index);
			m_sources.push_back(std::move(sources[entry.source]));
			appended.sources++;
		}
		if (!loaded_entries.insert(EntryDigest(index, entry.push_data_cursor, entry.specialization)).second) continue;
		m_entries.push_back({index, entry.push_data_cursor, entry.specialization});
		appended.entries++;
	}
	return appended;
}

namespace {

// The identity bytes of a journal header (magic, format version, identity, XXH3-64 of the rest).
bool ReadHeaderIdentity(const std::filesystem::path& path, const char (&magic)[8], uint32_t version,
                        std::vector<uint8_t>& identity) {
	std::ifstream in(path, std::ios::binary);
	char          file_magic[8] {};
	uint32_t      file_version = 0, size = 0;
	if (!in.read(file_magic, 8) || std::memcmp(file_magic, magic, 8) != 0 ||
	    !in.read(reinterpret_cast<char*>(&file_version), 4) || file_version != version ||
	    !in.read(reinterpret_cast<char*>(&size), 4) || size > 1024 * 1024) {
		return false;
	}
	identity.resize(size);
	return size == 0 || static_cast<bool>(in.read(reinterpret_cast<char*>(identity.data()), size));
}

} // namespace

bool ShaderJournal::ReadIdentity(const std::filesystem::path& path, std::vector<uint8_t>& identity) {
	return ReadHeaderIdentity(path, FileMagic, FormatVersion, identity);
}

bool ShaderJournal::HasSource(const Source& identity) {
	std::scoped_lock lock(m_mutex);
	const auto       range = m_by_digest.equal_range(SourceDigest(identity));
	for (auto it = range.first; it != range.second; ++it) {
		if (SameIdentity(m_known[it->second], identity)) return true;
	}
	return false;
}

void ShaderJournal::Record(Source source, uint32_t push_data_cursor,
                           std::span<const uint8_t> specialization) {
	std::scoped_lock lock(m_mutex);
	const auto       digest = SourceDigest(source);
	uint32_t         id     = UINT32_MAX;
	const auto       range  = m_by_digest.equal_range(digest);
	for (auto it = range.first; it != range.second; ++it) {
		if (SameIdentity(m_known[it->second], source)) {
			id = it->second;
			break;
		}
	}
	std::vector<uint8_t> records;
	if (id == UINT32_MAX) {
		if (source.code.empty() || source.input_info.empty() || source.code.size() != source.code_size) {
			return;
		}
		const auto encoded = EncodeSource(source);
		if (m_file_bytes + m_pending_bytes + encoded.size() + RecordHeaderBytes +
		        specialization.size() + 64 > m_settings.max_file_bytes) {
			m_stats.over_capacity = true;
			return;
		}
		id = m_next_source++;
		m_known.push_back(IdentityOf(source));
		m_by_digest.emplace(digest, id);
		AppendRecord(records, source.back_code.empty() ? RecordSource : RecordMergedSource, encoded);
		m_stats.recorded_sources++;
	}
	const auto entry_digest = EntryDigest(id, push_data_cursor, specialization);
	if (!m_known_entries.insert(entry_digest).second) {
		if (records.empty()) return;
	} else {
		const auto encoded = EncodeEntry(id, push_data_cursor, specialization);
		if (records.empty() && m_file_bytes + m_pending_bytes + encoded.size() + RecordHeaderBytes >
		                           m_settings.max_file_bytes) {
			m_stats.over_capacity = true;
			m_known_entries.erase(entry_digest);
			return;
		}
		AppendRecord(records, RecordEntry, encoded);
		m_stats.recorded_entries++;
	}
	m_pending_bytes += records.size();
	m_pending.push_back(std::move(records));
}

bool ShaderJournal::AppendPending(bool /*final_flush*/) {
	std::scoped_lock write_lock(m_write_mutex);
	std::vector<std::vector<uint8_t>> batch;
	uint64_t                          batch_bytes = 0;
	{
		std::scoped_lock lock(m_mutex);
		if (m_pending.empty()) return true;
		batch.swap(m_pending);
		batch_bytes     = m_pending_bytes;
		m_pending_bytes = 0;
	}
	bool ok = true;
	std::error_code error;
	if (!m_file_ready) {
		if (!m_settings.path.parent_path().empty()) {
			std::filesystem::create_directories(m_settings.path.parent_path(), error);
		}
		if (m_file_valid_bytes != 0) {
			std::filesystem::resize_file(m_settings.path, m_file_valid_bytes, error);
			ok = !error;
		} else {
			std::ofstream out(m_settings.path, std::ios::binary | std::ios::trunc);
			const auto    header = BuildHeader(m_settings.identity);
			out.write(reinterpret_cast<const char*>(header.data()),
			          static_cast<std::streamsize>(header.size()));
			out.flush();
			ok = static_cast<bool>(out);
			if (ok) {
				m_file_valid_bytes = header.size();
				std::scoped_lock lock(m_mutex);
				m_file_bytes = std::max<uint64_t>(m_file_bytes, header.size());
			}
		}
		m_file_ready = ok;
	}
	if (ok) {
		std::ofstream out(m_settings.path, std::ios::binary | std::ios::app);
		for (const auto& record: batch) {
			out.write(reinterpret_cast<const char*>(record.data()),
			          static_cast<std::streamsize>(record.size()));
		}
		out.flush();
		ok = static_cast<bool>(out);
	}
	std::scoped_lock lock(m_mutex);
	if (ok) {
		m_stats.written_bytes += batch_bytes;
		m_file_bytes += batch_bytes;
	} else {
		// Keep the records for the next attempt, ahead of newer ones.
		m_stats.write_failures++;
		m_pending.insert(m_pending.begin(), std::make_move_iterator(batch.begin()),
		                 std::make_move_iterator(batch.end()));
		m_pending_bytes += batch_bytes;
	}
	return ok;
}

bool ShaderJournal::Flush() {
	return AppendPending(true);
}

void ShaderJournal::RunWriter() {
	std::unique_lock lock(m_writer_mutex);
	while (!m_stop) {
		m_writer_cv.wait_for(lock, std::chrono::nanoseconds(m_settings.flush_interval_ns),
		                     [this] { return m_stop; });
		if (m_stop) break;
		lock.unlock();
		AppendPending(false);
		lock.lock();
	}
}

ShaderJournal::Stats ShaderJournal::GetStats() {
	std::scoped_lock lock(m_mutex);
	return m_stats;
}

// PipelineJournal ----------------------------------------------------------------------------

namespace {

// Record: magic, payload size, key, XXH3-64 of key and payload, payload.
constexpr size_t PipelineRecordHeaderBytes = 4 + 4 + 8 + 8;

uint64_t PipelineRecordChecksum(uint64_t key, std::span<const uint8_t> payload) {
	return XXH3_64bits_withSeed(payload.data(), payload.size(), key);
}

std::vector<uint8_t> BuildPipelineHeader(const std::vector<uint8_t>& identity) {
	std::vector<uint8_t> header(PipelineJournal::FileMagic,
	                            PipelineJournal::FileMagic + sizeof(PipelineJournal::FileMagic));
	Put<uint32_t>(header, PipelineJournal::FormatVersion);
	PutBytes(header, identity);
	Put<uint64_t>(header, XXH3_64bits(header.data(), header.size()));
	return header;
}

} // namespace

PipelineJournal::PipelineJournal(Settings settings): m_settings(std::move(settings)) {
	Load();
	if (m_settings.background_writer) {
		m_writer = std::thread([this] { RunWriter(); });
	}
}

PipelineJournal::~PipelineJournal() {
	{
		std::scoped_lock lock(m_writer_mutex);
		m_stop = true;
	}
	m_writer_cv.notify_all();
	if (m_writer.joinable()) m_writer.join();
	if (!m_settings.sealed_writes) Flush();
}

void PipelineJournal::Log(const std::string& message) const {
	if (m_settings.log) m_settings.log(message);
}

void PipelineJournal::Load() {
	if (m_settings.path.empty()) return; // in memory only
	const auto           begin = NowNs();
	std::vector<uint8_t> file;
	std::error_code      error;
	const auto           size = std::filesystem::file_size(m_settings.path, error);
	if (!error) {
		if (std::ifstream in(m_settings.path, std::ios::binary); in) {
			file.resize(static_cast<size_t>(size));
			if (!file.empty() &&
			    !in.read(reinterpret_cast<char*>(file.data()), static_cast<std::streamsize>(file.size()))) {
				file.clear();
			}
		}
		m_stats.file_found = !file.empty();
		m_stats.file_bytes = file.size();
	}
	if (!file.empty()) {
		const auto header = BuildPipelineHeader(m_settings.identity);
		if (file.size() < header.size() || std::memcmp(file.data(), header.data(), header.size()) != 0) {
			m_stats.header_rejected = true;
		} else {
			size_t position = header.size();
			while (file.size() - position >= PipelineRecordHeaderBytes) {
				uint32_t magic = 0, payload_size = 0;
				uint64_t key = 0, checksum = 0;
				std::memcpy(&magic, file.data() + position, 4);
				std::memcpy(&payload_size, file.data() + position + 4, 4);
				std::memcpy(&key, file.data() + position + 8, 8);
				std::memcpy(&checksum, file.data() + position + 16, 8);
				if (magic != RecordMagic ||
				    payload_size > file.size() - position - PipelineRecordHeaderBytes) {
					break;
				}
				const std::span<const uint8_t> payload(file.data() + position + PipelineRecordHeaderBytes,
				                                       payload_size);
				if (PipelineRecordChecksum(key, payload) != checksum) break;
				if (m_known.insert(key).second) {
					m_loaded.push_back({key, std::vector<uint8_t>(payload.begin(), payload.end())});
				}
				position += PipelineRecordHeaderBytes + payload_size;
			}
			m_file_valid_bytes    = position;
			m_stats.damaged_bytes = file.size() - position;
		}
	}
	m_stats.loaded  = m_loaded.size();
	m_file_bytes    = m_file_valid_bytes;
	m_stats.load_ns = NowNs() - begin;
	if (m_stats.header_rejected) {
		Log(m_settings.path.string() + " is from another device, driver, salt or format and is replaced");
	}
}

void PipelineJournal::ReleaseLoaded() {
	m_loaded = {};
}

bool PipelineJournal::ReadIdentity(const std::filesystem::path& path, std::vector<uint8_t>& identity) {
	return ReadHeaderIdentity(path, FileMagic, FormatVersion, identity);
}

bool PipelineJournal::Contains(uint64_t key) const {
	std::scoped_lock lock(m_mutex);
	return m_known.contains(key);
}

size_t PipelineJournal::KnownCount() const {
	std::scoped_lock lock(m_mutex);
	return m_known.size();
}

bool PipelineJournal::Add(uint64_t key, std::span<const uint8_t> payload) {
	std::scoped_lock lock(m_mutex);
	if (m_known.contains(key)) return false;
	const auto bytes = PipelineRecordHeaderBytes + payload.size();
	if (m_file_bytes + m_pending_bytes + bytes + 64 > m_settings.max_file_bytes) {
		m_stats.over_capacity = true;
		return false;
	}
	m_known.insert(key);
	std::vector<uint8_t> record;
	record.reserve(bytes);
	Put<uint32_t>(record, RecordMagic);
	Put<uint32_t>(record, static_cast<uint32_t>(payload.size()));
	Put<uint64_t>(record, key);
	Put<uint64_t>(record, PipelineRecordChecksum(key, payload));
	record.insert(record.end(), payload.begin(), payload.end());
	m_pending_bytes += record.size();
	m_pending.push_back(std::move(record));
	m_stats.recorded++;
	return true;
}

void PipelineJournal::Reset() {
	std::scoped_lock write_lock(m_write_mutex);
	std::scoped_lock lock(m_mutex);
	m_known.clear();
	m_pending.clear();
	m_pending_bytes = 0;
	m_sealed        = 0;
	m_stats.reset   = true;
	if (m_settings.path.empty()) return;
	// The stale keys leave the file now, not at the next write (there may be none).
	std::error_code error;
	if (!m_settings.path.parent_path().empty()) {
		std::filesystem::create_directories(m_settings.path.parent_path(), error);
	}
	std::ofstream out(m_settings.path, std::ios::binary | std::ios::trunc);
	const auto    header = BuildPipelineHeader(m_settings.identity);
	out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
	out.flush();
	m_file_ready       = static_cast<bool>(out);
	m_file_valid_bytes = m_file_ready ? header.size() : 0;
	m_file_bytes       = m_file_valid_bytes;
}

void PipelineJournal::Seal() {
	std::scoped_lock lock(m_mutex);
	m_sealed = m_pending.size();
}

bool PipelineJournal::AppendPending() {
	std::scoped_lock                  write_lock(m_write_mutex);
	std::vector<std::vector<uint8_t>> batch;
	uint64_t                          batch_bytes = 0;
	{
		std::scoped_lock lock(m_mutex);
		const auto count = m_settings.sealed_writes ? m_sealed : m_pending.size();
		if (count == 0) return true;
		if (m_settings.path.empty()) { // in memory only: nothing to write
			m_pending.erase(m_pending.begin(), m_pending.begin() + static_cast<std::ptrdiff_t>(count));
			m_pending_bytes = 0;
			for (const auto& record: m_pending) m_pending_bytes += record.size();
			m_sealed = 0;
			return true;
		}
		batch.assign(std::make_move_iterator(m_pending.begin()),
		             std::make_move_iterator(m_pending.begin() + static_cast<std::ptrdiff_t>(count)));
		m_pending.erase(m_pending.begin(), m_pending.begin() + static_cast<std::ptrdiff_t>(count));
		m_sealed = 0;
		for (const auto& record: batch) batch_bytes += record.size();
		m_pending_bytes -= batch_bytes;
	}
	bool            ok = true;
	std::error_code error;
	if (!m_file_ready) {
		if (!m_settings.path.parent_path().empty()) {
			std::filesystem::create_directories(m_settings.path.parent_path(), error);
		}
		if (m_file_valid_bytes != 0) {
			std::filesystem::resize_file(m_settings.path, m_file_valid_bytes, error);
			ok = !error;
		} else {
			std::ofstream out(m_settings.path, std::ios::binary | std::ios::trunc);
			const auto    header = BuildPipelineHeader(m_settings.identity);
			out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
			out.flush();
			ok = static_cast<bool>(out);
			if (ok) {
				m_file_valid_bytes = header.size();
				std::scoped_lock lock(m_mutex);
				m_file_bytes = std::max<uint64_t>(m_file_bytes, header.size());
			}
		}
		m_file_ready = ok;
	}
	if (ok) {
		std::ofstream out(m_settings.path, std::ios::binary | std::ios::app);
		for (const auto& record: batch) {
			out.write(reinterpret_cast<const char*>(record.data()), static_cast<std::streamsize>(record.size()));
		}
		out.flush();
		ok = static_cast<bool>(out);
	}
	std::scoped_lock lock(m_mutex);
	if (ok) {
		m_stats.written_bytes += batch_bytes;
		m_file_bytes += batch_bytes;
	} else {
		m_stats.write_failures++;
		const auto restored = batch.size();
		m_pending.insert(m_pending.begin(), std::make_move_iterator(batch.begin()),
		                 std::make_move_iterator(batch.end()));
		m_pending_bytes += batch_bytes;
		if (m_settings.sealed_writes) m_sealed += restored;
	}
	return ok;
}

bool PipelineJournal::Flush() {
	return AppendPending();
}

void PipelineJournal::RunWriter() {
	std::unique_lock lock(m_writer_mutex);
	while (!m_stop) {
		m_writer_cv.wait_for(lock, std::chrono::nanoseconds(m_settings.flush_interval_ns), [this] { return m_stop; });
		if (m_stop) break;
		lock.unlock();
		AppendPending();
		lock.lock();
	}
}

PipelineJournal::Stats PipelineJournal::GetStats() const {
	std::scoped_lock lock(m_mutex);
	return m_stats;
}

// ShaderPrecompiler --------------------------------------------------------------------------

void LowerCurrentThreadPriority() {
#if defined(_WIN32)
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
}

ShaderPrecompiler::ShaderPrecompiler(const ShaderJournal& journal, Settings settings, CompileFn compile)
    : m_journal(journal), m_settings(std::move(settings)), m_compile(std::move(compile)) {
	m_total = m_journal.Entries().size();
	if (m_settings.priority_source && m_total != 0) {
		const auto& sources = m_journal.Sources();
		const auto& entries = m_journal.Entries();
		std::vector<char> priority(sources.size(), 0);
		for (size_t i = 0; i < sources.size(); i++) priority[i] = m_settings.priority_source(sources[i]) ? 1 : 0;
		m_order.reserve(entries.size());
		for (uint32_t i = 0; i < entries.size(); i++) {
			if (entries[i].source < priority.size() && priority[entries[i].source] != 0) m_order.push_back(i);
		}
		const auto first = m_order.size();
		for (uint32_t i = 0; i < entries.size(); i++) {
			if (entries[i].source >= priority.size() || priority[entries[i].source] == 0) m_order.push_back(i);
		}
		if (first == 0) m_order.clear(); // nothing to move
	}
	if (m_settings.max_entries != 0) m_total = std::min<uint64_t>(m_total, m_settings.max_entries);
	m_begin_ns    = NowNs();
	m_reported_ns = m_begin_ns;
	if (m_total == 0 || m_settings.threads == 0) {
		m_finished = true;
		return;
	}
	const auto threads = static_cast<uint32_t>(std::min<uint64_t>(m_settings.threads, m_total));
	m_workers_running  = threads;
	if (m_settings.log) {
		m_settings.log("Shader precompile: replaying " + std::to_string(m_total) + " permutations of " +
		               std::to_string(m_journal.Sources().size()) + " sources on " +
		               std::to_string(threads) + " threads");
	}
	for (uint32_t i = 0; i < threads; i++) {
		m_threads.emplace_back([this, i] { Worker(i); });
	}
}

ShaderPrecompiler::~ShaderPrecompiler() {
	Stop();
	for (auto& thread: m_threads) {
		if (thread.joinable()) thread.join();
	}
}

void ShaderPrecompiler::Stop() {
	m_stop.store(true, std::memory_order_release);
}

void ShaderPrecompiler::Wait() {
	std::unique_lock lock(m_mutex);
	m_finished_cv.wait(lock, [this] { return m_finished; });
}

ShaderPrecompiler::Progress ShaderPrecompiler::GetProgress() const {
	Progress p;
	p.total    = m_total;
	p.done     = m_done.load(std::memory_order_relaxed);
	p.compiled = m_compiled.load(std::memory_order_relaxed);
	p.present  = m_present.load(std::memory_order_relaxed);
	p.skipped  = m_skipped.load(std::memory_order_relaxed);
	p.failed   = m_failed.load(std::memory_order_relaxed);
	std::scoped_lock lock(m_mutex);
	p.finished = m_finished;
	return p;
}

void ShaderPrecompiler::NoteDone(Outcome outcome) {
	switch (outcome) {
		case Outcome::Compiled: m_compiled.fetch_add(1, std::memory_order_relaxed); break;
		case Outcome::Present: m_present.fetch_add(1, std::memory_order_relaxed); break;
		case Outcome::Skipped: m_skipped.fetch_add(1, std::memory_order_relaxed); break;
		case Outcome::Failed: m_failed.fetch_add(1, std::memory_order_relaxed); break;
	}
	const auto done = m_done.fetch_add(1, std::memory_order_relaxed) + 1;
	if (!m_settings.log) return;
	const auto now = NowNs();
	std::scoped_lock lock(m_mutex);
	const auto step = std::max<uint64_t>(1, m_total * m_settings.progress_percent / 100);
	if (done - m_reported_done >= step || now - m_reported_ns >= m_settings.progress_ns) {
		if (done == m_total) return; // the final line follows
		m_reported_done = done;
		m_reported_ns   = now;
		char text[256];
		std::snprintf(text, sizeof(text),
		              "Shader precompile: %llu/%llu (%llu%%): %llu compiled, %llu present, %llu "
		              "skipped, %llu failed, %.1f s",
		              static_cast<unsigned long long>(done), static_cast<unsigned long long>(m_total),
		              static_cast<unsigned long long>(done * 100 / m_total),
		              static_cast<unsigned long long>(m_compiled.load()),
		              static_cast<unsigned long long>(m_present.load()),
		              static_cast<unsigned long long>(m_skipped.load()),
		              static_cast<unsigned long long>(m_failed.load()),
		              static_cast<double>(now - m_begin_ns) / 1.0e9);
		m_settings.log(text);
	}
}

void ShaderPrecompiler::Worker(uint32_t index) {
#if defined(_WIN32)
	if (m_settings.low_priority) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
	if (m_settings.thread_init) m_settings.thread_init(index);
	const auto& sources = m_journal.Sources();
	const auto& entries = m_journal.Entries();
	while (!m_stop.load(std::memory_order_acquire)) {
		const auto next = m_next.fetch_add(1, std::memory_order_relaxed);
		if (next >= m_total) break;
		const auto& entry = entries[m_order.empty() ? next : m_order[next]];
		Outcome     outcome = Outcome::Failed;
		try {
			outcome = m_compile(sources[entry.source], entry);
		} catch (const std::exception&) {
			outcome = Outcome::Failed;
		}
		NoteDone(outcome);
	}
	if (m_workers_running.fetch_sub(1, std::memory_order_acq_rel) == 1) {
		const auto done = m_done.load(std::memory_order_relaxed);
		if (m_settings.log) {
			char text[256];
			std::snprintf(text, sizeof(text),
			              "Shader precompile: %s %llu/%llu: %llu compiled, %llu present, %llu skipped, "
			              "%llu failed, %.1f s",
			              done == m_total ? "done," : "stopped,", static_cast<unsigned long long>(done),
			              static_cast<unsigned long long>(m_total),
			              static_cast<unsigned long long>(m_compiled.load()),
			              static_cast<unsigned long long>(m_present.load()),
			              static_cast<unsigned long long>(m_skipped.load()),
			              static_cast<unsigned long long>(m_failed.load()),
			              static_cast<double>(NowNs() - m_begin_ns) / 1.0e9);
			m_settings.log(text);
		}
		if (m_settings.on_finished) m_settings.on_finished();
		std::scoped_lock lock(m_mutex);
		m_finished = true;
		m_finished_cv.notify_all();
	}
}

} // namespace Libs::Graphics
