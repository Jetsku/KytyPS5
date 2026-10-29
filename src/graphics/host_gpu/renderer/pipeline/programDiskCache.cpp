#include "graphics/host_gpu/renderer/pipeline/programDiskCache.h"

#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <string_view>
#include <system_error>
#include <utility>
#include <xxhash.h>

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

// Bounds-checked cursor over one record.
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

	std::span<const uint8_t> Take(uint64_t size) {
		if (m_failed || size > m_bytes.size() - m_position) {
			m_failed = true;
			return {};
		}
		const auto result = m_bytes.subspan(m_position, static_cast<size_t>(size));
		m_position += static_cast<size_t>(size);
		return result;
	}

	std::span<const uint8_t> Sized() { return Take(Get<uint32_t>()); }
	std::span<const uint8_t> SizedWords() { return Take(uint64_t {Get<uint32_t>()} * 4u); }

	[[nodiscard]] bool   Failed() const { return m_failed; }
	[[nodiscard]] bool   AtEnd() const { return !m_failed && m_position == m_bytes.size(); }
	[[nodiscard]] size_t Position() const { return m_position; }

private:
	std::span<const uint8_t> m_bytes;
	size_t                   m_position = 0;
	bool                     m_failed   = false;
};

bool Equal(std::span<const uint8_t> a, std::span<const uint8_t> b) {
	return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()) == 0);
}

} // namespace

size_t ProgramDiskCache::DigestHash::operator()(const Digest& digest) const noexcept {
	size_t value = 0;
	std::memcpy(&value, digest.data(), sizeof(value));
	return value;
}

void ProgramDiskCache::BuildSourceKey(const SourceKeyInputs& inputs, SourceKey& key) {
	auto& bytes = key.bytes;
	bytes.clear();
	bytes.reserve(64 + inputs.static_state.size_bytes() + inputs.code.size_bytes() +
	              inputs.back_code.size_bytes());
	Put<uint32_t>(bytes, 1); // key layout
	Put<uint32_t>(bytes, inputs.stage);
	Put<uint64_t>(bytes, inputs.hash);
	Put<uint32_t>(bytes, inputs.user_data_count);
	Put<uint32_t>(bytes, inputs.code_size);
	PutWords(bytes, inputs.static_state);
	Put<uint32_t>(bytes, inputs.wave_size);
	Put<uint32_t>(bytes, inputs.user_data_base);
	Put<uint8_t>(bytes, inputs.plain_mip_stats_variant ? 1u : 0u);
	PutWords(bytes, inputs.code);
	PutWords(bytes, inputs.back_code);
	const auto digest = XXH3_128bits(bytes.data(), bytes.size());
	std::memcpy(key.digest.data(), &digest.low64, sizeof(digest.low64));
	std::memcpy(key.digest.data() + 8, &digest.high64, sizeof(digest.high64));
}

ProgramDiskCache::ProgramDiskCache(Settings settings): m_settings(std::move(settings)) {
	m_last_save_ns = NowNs();
	m_loader       = std::thread([this] { Load(); });
	if (m_settings.periodic_saves) {
		m_saver = std::thread([this] { RunSaver(); });
	}
}

ProgramDiskCache::~ProgramDiskCache() {
	{
		std::scoped_lock lock(m_saver_mutex);
		m_stop = true;
	}
	m_saver_cv.notify_all();
	if (m_saver.joinable()) m_saver.join();
	if (m_loader.joinable()) m_loader.join();
	Flush();
}

void ProgramDiskCache::WaitLoaded() {
	std::unique_lock lock(m_mutex);
	m_loaded_cv.wait(lock, [this] { return m_loaded; });
}

bool ProgramDiskCache::ParseRecord(std::span<const uint8_t> bytes, Record& record) const {
	Cursor header(bytes);
	const auto magic    = header.Get<uint32_t>();
	const auto kind     = header.Get<uint32_t>();
	const auto size     = header.Get<uint64_t>();
	const auto checksum = header.Get<uint64_t>();
	if (header.Failed() || magic != RecordMagic ||
	    (kind != RecordSource && kind != RecordPermutation) ||
	    size != bytes.size() - RecordHeaderBytes) {
		return false;
	}
	const auto payload = bytes.subspan(RecordHeaderBytes);
	if (XXH3_64bits(payload.data(), payload.size()) != checksum) {
		return false;
	}
	record       = {};
	record.bytes = bytes;
	record.kind  = kind;
	Cursor c(payload);
	const auto digest = c.Take(record.digest.size());
	if (!digest.empty()) std::memcpy(record.digest.data(), digest.data(), digest.size());
	if (kind == RecordSource) {
		record.key                = c.Sized();
		const auto skip           = c.Get<uint8_t>();
		record.skip_dispatch      = skip == 1u;
		record.plan               = c.Sized();
		if (skip > 1u) return false;
		// The digest must be the key's own (a record cannot be filed under another key).
		const auto own = XXH3_128bits(record.key.data(), record.key.size());
		Digest     expected {};
		std::memcpy(expected.data(), &own.low64, sizeof(own.low64));
		std::memcpy(expected.data() + 8, &own.high64, sizeof(own.high64));
		if (c.Failed() || expected != record.digest) return false;
	} else {
		record.cursor         = c.Get<uint32_t>();
		record.specialization = c.Sized();
		record.info           = c.Sized();
		record.spirv          = c.SizedWords();
		record.spirv_plain    = c.SizedWords();
		if (record.spirv.empty()) return false;
	}
	return c.AtEnd();
}

bool ProgramDiskCache::IndexRecord(uint32_t id) {
	auto& record = m_records[id];
	if (m_ambiguous.contains(record.digest)) return false;
	// Records dropped by verification no longer count (a fresh one may replace them).
	if (record.kind == RecordSource) {
		auto& candidates = m_sources[record.digest];
		std::erase_if(candidates, [&](uint32_t other) { return m_records[other].invalid; });
		for (const auto other: candidates) {
			if (Equal(m_records[other].key, record.key)) return false; // duplicate
		}
		if (!candidates.empty()) {
			// Two different keys with one digest: neither they nor their permutations are used.
			m_ambiguous.insert(record.digest);
			m_stats.ambiguous++;
			return false;
		}
		candidates.push_back(id);
		return true;
	}
	auto& candidates = m_permutations[record.digest];
	std::erase_if(candidates, [&](uint32_t other) { return m_records[other].invalid; });
	for (const auto other: candidates) {
		const auto& existing = m_records[other];
		if (existing.cursor == record.cursor &&
		    Equal(existing.specialization, record.specialization)) {
			return false; // duplicate
		}
	}
	candidates.push_back(id);
	return true;
}

void ProgramDiskCache::Load() {
	Profiler::SetThreadName("ProgramCacheLoad");
	const auto begin = NowNs();
	const auto path  = Common::PathToString(m_settings.path);
	std::vector<uint8_t> file;
	bool                 found = Common::File::IsFileExisting(m_settings.path);
	if (found) {
		Common::File input(m_settings.path, Common::File::Mode::Read);
		const auto   size = input.IsInvalid() ? 0 : input.Size();
		if (size != 0 && size <= m_settings.max_file_bytes + (64u << 20u)) {
			file.resize(static_cast<size_t>(size));
			uint32_t read = 0;
			input.Read(file.data(), static_cast<uint32_t>(file.size()), &read);
			if (read != file.size()) file.clear();
		}
		input.Close();
	}
	// Header: magic, format, identity, checksum of everything before it.
	size_t offset          = 0;
	bool   header_accepted = false;
	if (!file.empty()) {
		Cursor header(file);
		const auto magic   = header.Take(sizeof(FileMagic));
		const auto format  = header.Get<uint32_t>();
		const auto reserved = header.Get<uint32_t>();
		const auto identity = header.Take(header.Get<uint64_t>());
		const auto covered  = header.Position();
		const auto checksum = header.Get<uint64_t>();
		header_accepted = !header.Failed() && Equal(magic, {reinterpret_cast<const uint8_t*>(FileMagic), sizeof(FileMagic)}) &&
		                  format == FormatVersion && reserved == 0 &&
		                  XXH3_64bits(file.data(), covered) == checksum &&
		                  Equal(identity, m_settings.identity);
		offset = header.Position();
	}
	uint64_t rejected = 0;
	{
		std::scoped_lock lock(m_mutex);
		m_stats.file_found      = found;
		m_stats.file_bytes      = file.size();
		m_stats.header_rejected = found && !header_accepted;
		if (header_accepted) {
			m_file = std::move(file);
			// Records until the end, or until the first damaged one (its size field cannot be
			// trusted, so nothing after it is either).
			while (offset < m_file.size()) {
				const auto remaining = m_file.size() - offset;
				if (remaining < RecordHeaderBytes) {
					rejected++;
					break;
				}
				uint64_t size = 0;
				std::memcpy(&size, m_file.data() + offset + 8, sizeof(size));
				if (size > remaining - RecordHeaderBytes) {
					rejected++;
					break;
				}
				const auto bytes = std::span<const uint8_t>(m_file).subspan(
				    offset, static_cast<size_t>(RecordHeaderBytes + size));
				Record record;
				if (!ParseRecord(bytes, record)) {
					rejected++;
					break;
				}
				const auto id = static_cast<uint32_t>(m_records.size());
				m_records.push_back(record);
				if (IndexRecord(id)) {
					if (record.kind == RecordSource) m_stats.loaded_sources++;
					else m_stats.loaded_permutations++;
				} else {
					// Duplicate or ambiguous: kept out of the index and out of later saves.
					m_records[id].invalid = true;
				}
				offset += bytes.size();
			}
			m_bytes = offset;
		} else {
			m_bytes = 0;
		}
		m_loaded_count = static_cast<uint32_t>(m_records.size());
		// A damaged tail or dropped records are rewritten at the next save. A file with another
		// identity is left alone until this run has records of its own to save in its place.
		if (rejected != 0 || m_stats.ambiguous != 0) {
			m_changes++;
		}
		m_stats.rejected_records = rejected;
		m_stats.load_ns          = NowNs() - begin;
		m_loaded                 = true;
	}
	m_loaded_cv.notify_all();
	if (!m_settings.quiet) {
		if (!found) {
			Log::WriteToConsoleAndLog(fmt::format("Program cache: {} not found; starting empty\n", path));
		} else if (!header_accepted) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Program cache: {} was written by another codegen version, configuration or device "
			    "(or is damaged); not used, the next save replaces it\n",
			    path));
		} else {
			const auto stats = GetStats();
			Log::WriteToConsoleAndLog(fmt::format(
			    "Program cache: loaded {} sources and {} permutations ({:.1f} MiB) from {} in "
			    "{:.1f} ms{}\n",
			    stats.loaded_sources, stats.loaded_permutations,
			    static_cast<double>(stats.file_bytes) / (1024.0 * 1024.0), path,
			    static_cast<double>(stats.load_ns) / 1.0e6,
			    rejected != 0 ? "; dropped a damaged tail" : ""));
		}
	}
}

std::optional<ProgramDiskCache::SourceRecord> ProgramDiskCache::FindSource(const SourceKey& key, bool note_use) {
	WaitLoaded();
	std::scoped_lock lock(m_mutex);
	if (!m_ambiguous.contains(key.digest)) {
		if (const auto found = m_sources.find(key.digest); found != m_sources.end()) {
			for (const auto id: found->second) {
				const auto& record = m_records[id];
				if (!record.invalid && Equal(record.key, key.bytes)) {
					m_stats.source_hits++;
					if (note_use) NoteUsed(id);
					return SourceRecord {.id = id, .skip_dispatch = record.skip_dispatch,
					                     .plan = record.plan};
				}
			}
		}
	}
	m_stats.source_misses++;
	return std::nullopt;
}

std::optional<ProgramDiskCache::PermutationRecord>
ProgramDiskCache::FindPermutation(const Digest& source, uint32_t push_data_cursor,
                                  std::span<const uint8_t> specialization, bool note_use) {
	WaitLoaded();
	std::scoped_lock lock(m_mutex);
	if (!m_ambiguous.contains(source)) {
		if (const auto found = m_permutations.find(source); found != m_permutations.end()) {
			for (const auto id: found->second) {
				const auto& record = m_records[id];
				if (!record.invalid && record.cursor == push_data_cursor &&
				    Equal(record.specialization, specialization)) {
					m_stats.permutation_hits++;
					if (note_use) NoteUsed(id);
					return PermutationRecord {.id = id, .info = record.info, .spirv = record.spirv,
					                          .spirv_plain = record.spirv_plain};
				}
			}
		}
	}
	m_stats.permutation_misses++;
	return std::nullopt;
}

void ProgramDiskCache::AppendRecord(uint32_t kind, const std::vector<uint8_t>& payload,
                                    const Digest& digest) {
	// Requires m_mutex.
	const auto total = RecordHeaderBytes + payload.size();
	if (m_bytes + total > m_settings.max_file_bytes && !MakeRoom(total)) {
		if (!m_stats.over_capacity && !m_settings.quiet) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Program cache: {} would exceed {} MiB; new programs are no longer added\n",
			    Common::PathToString(m_settings.path), m_settings.max_file_bytes >> 20u));
		}
		m_stats.over_capacity = true;
		return;
	}
	auto& bytes = m_added.emplace_back();
	bytes.reserve(total);
	Put<uint32_t>(bytes, RecordMagic);
	Put<uint32_t>(bytes, kind);
	Put<uint64_t>(bytes, payload.size());
	Put<uint64_t>(bytes, XXH3_64bits(payload.data(), payload.size()));
	bytes.insert(bytes.end(), payload.begin(), payload.end());
	Record record;
	if (!ParseRecord(bytes, record) || record.digest != digest) {
		m_added.pop_back();
		return;
	}
	const auto id = static_cast<uint32_t>(m_records.size());
	m_records.push_back(record);
	if (!IndexRecord(id)) {
		m_records[id].invalid = true;
		return;
	}
	m_bytes += total;
	m_changes++;
	m_last_change_ns = NowNs();
	if (kind == RecordSource) m_stats.added_sources++;
	else m_stats.added_permutations++;
}

void ProgramDiskCache::AddSource(const SourceKey& key, bool skip_dispatch,
                                 std::span<const uint8_t> plan) {
	WaitLoaded();
	std::vector<uint8_t> payload;
	payload.reserve(key.digest.size() + key.bytes.size() + plan.size() + 16);
	payload.insert(payload.end(), key.digest.begin(), key.digest.end());
	PutBytes(payload, key.bytes);
	Put<uint8_t>(payload, skip_dispatch ? 1u : 0u);
	PutBytes(payload, plan);
	{
		std::scoped_lock lock(m_mutex);
		AppendRecord(RecordSource, payload, key.digest);
	}
	m_saver_cv.notify_all();
}

void ProgramDiskCache::AddPermutation(const Digest& source, uint32_t push_data_cursor,
                                      std::span<const uint8_t> specialization,
                                      std::span<const uint8_t> info,
                                      std::span<const uint32_t> spirv,
                                      std::span<const uint32_t> spirv_plain) {
	WaitLoaded();
	std::vector<uint8_t> payload;
	payload.reserve(source.size() + specialization.size() + info.size() + spirv.size_bytes() +
	                spirv_plain.size_bytes() + 32);
	payload.insert(payload.end(), source.begin(), source.end());
	Put<uint32_t>(payload, push_data_cursor);
	PutBytes(payload, specialization);
	PutBytes(payload, info);
	PutWords(payload, spirv);
	PutWords(payload, spirv_plain);
	{
		std::scoped_lock lock(m_mutex);
		AppendRecord(RecordPermutation, payload, source);
	}
	m_saver_cv.notify_all();
}

void ProgramDiskCache::Invalidate(uint32_t id) {
	std::scoped_lock lock(m_mutex);
	if (id >= m_records.size() || m_records[id].invalid) return;
	m_records[id].invalid = true;
	if (!m_records[id].evicted) m_bytes -= m_records[id].bytes.size();
	m_changes++;
	m_last_change_ns = NowNs();
	m_stats.invalidated++;
}

void ProgramDiskCache::NoteUsed(uint32_t id) {
	// Requires m_mutex.
	if (!m_settings.evict_unused || id >= m_records.size()) return;
	auto& record = m_records[id];
	if (!record.used && id < m_loaded_count) {
		m_order_dirty = true; // its place in the file moves to the used group (saved at exit)
	}
	record.used = true;
	// An evicted record used again is kept if it still fits (else a later run translates it).
	if (record.evicted && m_bytes + record.bytes.size() <= m_settings.max_file_bytes) {
		record.evicted = false;
		m_bytes += record.bytes.size();
		m_stats.restored++;
		m_changes++;
	}
}

bool ProgramDiskCache::MakeRoom(uint64_t bytes) {
	// Requires m_mutex.
	if (!m_settings.evict_unused || bytes > m_settings.max_file_bytes) return false;
	uint64_t available = 0;
	for (uint32_t id = 0; id < m_loaded_count && m_bytes + bytes > m_settings.max_file_bytes + available; id++) {
		const auto& record = m_records[id];
		if (!record.used && !record.evicted && !record.invalid) available += record.bytes.size();
	}
	if (m_bytes + bytes > m_settings.max_file_bytes + available) return false;
	uint64_t freed = 0;
	for (uint32_t id = 0; id < m_loaded_count && m_bytes + bytes > m_settings.max_file_bytes; id++) {
		auto& record = m_records[id];
		if (record.used || record.evicted || record.invalid) continue;
		record.evicted = true;
		m_bytes -= record.bytes.size();
		freed += record.bytes.size();
		m_stats.evicted++;
		m_stats.evicted_bytes += record.bytes.size();
	}
	if (freed != 0) {
		m_changes++;
		if (m_stats.evicted_bytes == freed && !m_settings.quiet) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Program cache: {} is at {} MiB; records this run has not used are dropped for new ones\n",
			    Common::PathToString(m_settings.path), m_settings.max_file_bytes >> 20u));
		}
	}
	return true;
}

bool ProgramDiskCache::Due(uint64_t now) const {
	// Requires m_mutex.
	const auto pending = m_changes - m_saved_changes;
	if (pending == 0) return false;
	const auto quiet      = now - std::min(now, m_last_change_ns);
	const auto since_save = now - std::min(now, m_last_save_ns);
	const auto interval   = m_settings.interval_ns;
	// A burst (level load) has ended; a trickle; a steady stream.
	return (pending >= m_settings.min_new && quiet >= m_settings.settle_ns &&
	        since_save >= interval / 3) ||
	       (quiet >= m_settings.settle_ns && since_save >= interval) ||
	       since_save >= 3 * interval;
}

void ProgramDiskCache::RunSaver() {
	Profiler::SetThreadName("ProgramCacheSaver");
	WaitLoaded();
	std::unique_lock lock(m_saver_mutex);
	while (!m_stop) {
		m_saver_cv.wait_for(lock, std::chrono::seconds(1));
		if (m_stop) break;
		bool due = false;
		{
			std::scoped_lock state(m_mutex);
			due = Due(NowNs());
		}
		if (due) {
			lock.unlock();
			Save(false);
			lock.lock();
		}
	}
}

bool ProgramDiskCache::Flush() {
	WaitLoaded();
	{
		std::scoped_lock lock(m_mutex);
		if (m_changes == m_saved_changes && !m_order_dirty) return true;
	}
	return Save(true);
}

bool ProgramDiskCache::Save(bool final_save) {
	std::scoped_lock save_lock(m_save_mutex);
	const auto begin = NowNs();
	// Snapshot: records are immutable, so their bytes can be written without the lock.
	std::vector<std::span<const uint8_t>> records;
	uint64_t                              changes = 0;
	{
		std::scoped_lock lock(m_mutex);
		if (m_changes == m_saved_changes && !(final_save && m_order_dirty)) return true;
		changes = m_changes;
		m_order_dirty = false;
		records.reserve(m_records.size());
		const auto keep = [&](const Record& record) {
			return !record.invalid && !record.evicted && !m_ambiguous.contains(record.digest);
		};
		if (m_settings.evict_unused) {
			// Order of last use: unused loaded records, used loaded ones, this run's.
			for (uint32_t pass = 0; pass < 2; pass++) {
				for (uint32_t id = 0; id < m_loaded_count; id++) {
					const auto& record = m_records[id];
					if (keep(record) && record.used == (pass == 1)) records.push_back(record.bytes);
				}
			}
			for (uint32_t id = m_loaded_count; id < m_records.size(); id++) {
				if (keep(m_records[id])) records.push_back(m_records[id].bytes);
			}
		} else {
			for (const auto& record: m_records) {
				if (keep(record)) records.push_back(record.bytes);
			}
		}
	}
	std::vector<uint8_t> header;
	header.insert(header.end(), std::begin(FileMagic), std::end(FileMagic));
	Put<uint32_t>(header, FormatVersion);
	Put<uint32_t>(header, 0);
	Put<uint64_t>(header, m_settings.identity.size());
	header.insert(header.end(), m_settings.identity.begin(), m_settings.identity.end());
	Put<uint64_t>(header, XXH3_64bits(header.data(), header.size()));

	const auto& path      = m_settings.path;
	auto        temp_path = path;
	temp_path += ".tmp";
	bool     ok      = true;
	uint64_t written = 0;
	if (path.has_parent_path() && !Common::File::IsDirectoryExisting(path.parent_path())) {
		ok = Common::File::CreateDirectories(path.parent_path());
	}
	Common::File file;
	// The previous temporary file may still be held open briefly (see the rename below).
	for (uint32_t attempt = 0; ok && attempt < 12 && !file.Create(temp_path); attempt++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5u << std::min(attempt, 5u)));
	}
	ok = ok && !file.IsInvalid();
	const auto write = [&](std::span<const uint8_t> bytes) {
		// Common::File sizes are 32-bit.
		while (ok && !bytes.empty()) {
			const auto chunk = static_cast<uint32_t>(
			    std::min<size_t>(bytes.size(), std::numeric_limits<uint32_t>::max() / 2));
			uint32_t done = 0;
			file.Write(bytes.data(), chunk, &done);
			ok = done == chunk;
			written += done;
			bytes = bytes.subspan(chunk);
		}
	};
	if (ok) {
		write(header);
		for (const auto& record: records) {
			write(record);
		}
		ok = ok && !file.IsInvalid() && file.Flush();
	}
	file.Close();
	std::error_code error;
	if (ok) {
		// One step: a kill at any point leaves the previous or the new complete file. A scanner or
		// a sync client that briefly holds the file open (a sharing violation on Windows) makes
		// the replace fail; it is retried for up to about a second.
		for (uint32_t attempt = 0; attempt < 12; attempt++) {
			error.clear();
			std::filesystem::rename(temp_path, path, error);
			if (!error) break;
			std::this_thread::sleep_for(std::chrono::milliseconds(5u << std::min(attempt, 5u)));
		}
		ok = !error;
	}
	const auto ns = NowNs() - begin;
	{
		std::scoped_lock lock(m_mutex);
		m_last_save_ns = NowNs();
		if (ok) {
			m_saved_changes = changes;
			m_stats.saves++;
			m_stats.saved_bytes = written;
			m_stats.save_ns += ns;
		} else {
			m_stats.save_failures++;
		}
	}
	if (!m_settings.quiet) {
		if (ok) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Program cache: saved {} records ({:.1f} MiB) to {} ({}, {:.1f} ms)\n",
			    records.size(), static_cast<double>(written) / (1024.0 * 1024.0),
			    Common::PathToString(path), final_save ? "exit" : "periodic",
			    static_cast<double>(ns) / 1.0e6));
		} else {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Program cache: failed to write {}{}\n", Common::PathToString(path),
			    error ? " (" + error.message() + ")" : std::string()));
		}
	}
	return ok;
}

ProgramDiskCache::Stats ProgramDiskCache::GetStats() {
	WaitLoaded();
	std::scoped_lock lock(m_mutex);
	return m_stats;
}

} // namespace Libs::Graphics
