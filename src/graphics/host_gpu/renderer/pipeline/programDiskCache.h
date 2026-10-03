#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMDISKCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMDISKCACHE_H_

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

// Persistent translated-program cache (KYTY_PROGRAM_CACHE; PipelineCache::ProgramCache uses it).
// It keeps what translation produced in earlier runs, so that a program seen before is bound
// without translating it again: per program source, the sealed resource plan (or the
// skip-dispatch verdict); per permutation, the SPIR-V (and its plain LOD-stats variant) and the
// compiled shader metadata. The encodings are ShaderRecompiler::IR::ProgramCodec's.
//
// Keys (exact, no hash is trusted):
// - A source record holds its full key (SourceKey: stage, guest hash, user-data count, code size,
//   stage static key, compile options, and every guest code word, merged back half included)
//   and matches only a byte-equal key. Its 128-bit digest only selects candidates; a digest that
//   two different keys share in one file is dropped (ambiguous).
// - A permutation record names its source by digest and holds the push-data cursor and the
//   specialization bytes it was compiled for; it matches only equal ones.
// - Everything else that can change a translation is the file's identity (header): the codegen
//   version (a build-time hash of the recompiler sources, recompiler/CodegenFingerprint.h), the
//   runtime codegen fingerprint (CodegenOptions, emitter host state, every KYTY_* codegen
//   switch) and the device. A file with another identity is not used, and the next save replaces
//   it with this run's records (so a build or switch change costs one cold run).
//
// File: header (magic, format, identity, checksum), then records (magic, kind, size, XXH3-64 of
// the payload, payload). Loading rejects the whole file on any header mismatch and keeps records
// up to the first damaged one (bad magic, size or checksum: a truncated or corrupt tail).
//
// Threads: the file is read once on a loader thread started by the constructor. Bounded batches
// are validated by up to Settings::load_threads workers, then indexed in file order; lookups
// wait for the complete load. Records are immutable once added. Saves (whole file to "<file>.tmp",
// flushed, then renamed over the file in one step) run on a saver thread, coalesced like the driver pipeline
// cache saves, and once more at shutdown (Flush); a draw never waits for file I/O.
class ProgramDiskCache {
public:
	using Digest = std::array<uint8_t, 16>;

	struct Settings {
		std::filesystem::path path;
		// Header identity: must be byte-equal for the file's records to be used.
		std::vector<uint8_t>  identity;
		// Background saves while running (false: only Flush saves).
		bool                  periodic_saves = true;
		// New records stop being added beyond this file size.
		uint64_t              max_file_bytes = 512ull * 1024 * 1024;
		// Coalescing (see Due()), like the driver pipeline cache saves: after a burst (at least
		// min_new new records, none for settle_ns, a third of an interval since the last save),
		// after a trickle (none for settle_ns, an interval since the last save), and after three
		// intervals regardless. Each save rewrites the whole file.
		uint64_t              min_new     = 16;
		uint64_t              settle_ns   = 3'000'000'000ull;
		uint64_t              interval_ns = 60'000'000'000ull;
		// Tests: log nothing.
		bool                  quiet = false;
		// Validation workers including the loader (1: no helpers). Set before construction;
		// loading starts immediately. Clamped to [1, 64] and to each batch's record count.
		uint32_t              load_threads = 1;
	};

	// Canonical source key bytes and their digest.
	struct SourceKeyInputs {
		uint32_t                  stage           = 0;
		uint64_t                  hash            = 0;
		uint32_t                  user_data_count = 0;
		uint32_t                  code_size       = 0;
		std::span<const uint32_t> static_state;
		uint32_t                  wave_size       = 0;
		uint32_t                  user_data_base  = 0;
		bool                      plain_mip_stats_variant = false;
		std::span<const uint32_t> code;
		std::span<const uint32_t> back_code;
	};
	struct SourceKey {
		std::vector<uint8_t> bytes;
		Digest               digest {};
	};
	static void BuildSourceKey(const SourceKeyInputs& inputs, SourceKey& key);

	struct SourceRecord {
		uint32_t                 id            = 0;
		bool                     skip_dispatch = false;
		std::span<const uint8_t> plan;
	};
	struct PermutationRecord {
		uint32_t                 id = 0;
		std::span<const uint8_t> info;
		std::span<const uint8_t> spirv;       // words, little-endian, not necessarily aligned
		std::span<const uint8_t> spirv_plain; // empty when the program has no plain variant
	};

	struct Stats {
		bool     file_found       = false;
		bool     header_rejected  = false;
		uint64_t file_bytes       = 0;
		uint64_t load_ns          = 0;
		uint64_t read_ns          = 0; // file read/allocation; excludes header validation
		uint64_t validate_ns      = 0; // framing, worker startup/join, checksum and decoding
		uint64_t index_ns         = 0; // ordered publication and indexing
		uint32_t load_workers     = 0; // maximum pool size including loader; 0: no framed records
		uint64_t loaded_sources   = 0;
		uint64_t loaded_permutations = 0;
		uint64_t rejected_records = 0; // damaged tail
		uint64_t ambiguous        = 0;
		uint64_t source_hits      = 0;
		uint64_t source_misses    = 0;
		uint64_t permutation_hits = 0;
		uint64_t permutation_misses = 0;
		uint64_t added_sources    = 0;
		uint64_t added_permutations = 0;
		uint64_t invalidated      = 0;
		uint64_t saves            = 0;
		uint64_t saved_bytes      = 0;
		uint64_t save_ns          = 0;
		uint64_t save_failures    = 0;
		bool     over_capacity    = false;
	};

	explicit ProgramDiskCache(Settings settings);
	// Flushes pending records.
	~ProgramDiskCache();
	ProgramDiskCache(const ProgramDiskCache&)            = delete;
	ProgramDiskCache& operator=(const ProgramDiskCache&) = delete;

	// Both wait for the loader. The returned spans stay valid for the cache's lifetime.
	[[nodiscard]] std::optional<SourceRecord> FindSource(const SourceKey& key);
	[[nodiscard]] std::optional<PermutationRecord>
	FindPermutation(const Digest& source, uint32_t push_data_cursor,
	                std::span<const uint8_t> specialization);

	// No-ops for keys already present, and beyond max_file_bytes.
	void AddSource(const SourceKey& key, bool skip_dispatch, std::span<const uint8_t> plan);
	void AddPermutation(const Digest& source, uint32_t push_data_cursor,
	                    std::span<const uint8_t> specialization, std::span<const uint8_t> info,
	                    std::span<const uint32_t> spirv, std::span<const uint32_t> spirv_plain);
	// A record that verification found different from a fresh translation: never returned again,
	// and left out of every later save.
	void Invalidate(uint32_t id);

	// Writes the file now if records were added or invalidated since the last save (the final
	// save; called by the destructor). Returns false when a write failed.
	bool Flush();

	// Waits for the loader.
	[[nodiscard]] Stats GetStats();
	[[nodiscard]] const std::filesystem::path& Path() const { return m_settings.path; }

	// File format constants (tests build damaged files from them).
	static constexpr char     FileMagic[8]     = {'K', 'Y', 'P', 'R', 'O', 'G', 'C', '1'};
	static constexpr uint32_t FormatVersion    = 1;
	static constexpr uint32_t RecordMagic      = 0x4352504bu; // "KPRC"
	static constexpr uint32_t RecordSource      = 1;
	static constexpr uint32_t RecordPermutation = 2;
	static constexpr size_t   RecordHeaderBytes = 4 + 4 + 8 + 8;

private:
	struct DigestHash {
		size_t operator()(const Digest& digest) const noexcept;
	};
	struct Record {
		std::span<const uint8_t> bytes; // whole record, header included
		uint32_t                 kind = 0;
		Digest                   digest {}; // sources: own digest; permutations: their source's
		// Sources.
		std::span<const uint8_t> key;
		bool                     skip_dispatch = false;
		std::span<const uint8_t> plan;
		// Permutations.
		uint32_t                 cursor = 0;
		std::span<const uint8_t> specialization;
		std::span<const uint8_t> info;
		std::span<const uint8_t> spirv;
		std::span<const uint8_t> spirv_plain;
		bool                     invalid = false;
	};

	void Load();
	void WaitLoaded();
	bool ParseRecord(std::span<const uint8_t> bytes, Record& record) const;
	// Requires m_mutex. Adds a parsed record to the index; false when it was a duplicate or made
	// its digest ambiguous (the record is then not indexed).
	bool IndexRecord(uint32_t id);
	void AppendRecord(uint32_t kind, const std::vector<uint8_t>& payload, const Digest& digest);
	bool Save(bool final_save);
	bool Due(uint64_t now) const;
	void RunSaver();

	Settings m_settings;

	std::mutex              m_mutex;
	std::condition_variable m_loaded_cv;
	bool                    m_loaded = false;
	std::vector<uint8_t>    m_file;           // loaded file contents (records point into it)
	std::deque<std::vector<uint8_t>> m_added; // records added by this run (stable storage)
	std::vector<Record>     m_records;
	std::unordered_map<Digest, std::vector<uint32_t>, DigestHash> m_sources;
	std::unordered_map<Digest, std::vector<uint32_t>, DigestHash> m_permutations;
	std::unordered_set<Digest, DigestHash>                        m_ambiguous;
	uint64_t m_bytes         = 0; // file size a save would write
	uint64_t m_changes       = 0; // additions and invalidations
	uint64_t m_saved_changes = 0;
	uint64_t m_last_change_ns = 0;
	uint64_t m_last_save_ns   = 0;
	Stats    m_stats;

	std::mutex              m_save_mutex; // one save at a time
	std::mutex              m_saver_mutex;
	std::condition_variable m_saver_cv;
	bool                    m_stop = false;
	std::thread             m_loader;
	std::thread             m_saver;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMDISKCACHE_H_
