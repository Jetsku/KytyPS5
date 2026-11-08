#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMDISKCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PROGRAMDISKCACHE_H_

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
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
// the payload, payload). Packed records (KYTY_PROGRAM_CACHE_COMPRESS) keep their lookup keys plain and the
// rest as one checksummed zstd frame: a packed permutation's info and SPIR-V are expanded on every
// hit into a buffer the returned record owns (nothing stays expanded in the cache); a packed
// source's plan is expanded on its first hit and kept (plans are referenced for the run). The
// size cap counts stored bytes, so about four times as many programs fit. Loading rejects the whole file on any header mismatch and keeps records
// up to the first damaged one (bad magic, size or checksum: a truncated or corrupt tail).
//
// Threads: the file is read and indexed on a loader thread started by the constructor; lookups
// wait for it. Records are immutable once added. Saves (whole file to "<file>.tmp", flushed, then
// renamed over the file in one step) run on a saver thread, coalesced like the driver pipeline
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
		// KYTY_PROGRAM_CACHE_EVICT: at the size limit, loaded records this run has not used are
		// dropped (oldest first in file order) to make room for new ones, and saves write the
		// unused loaded records first, then the used ones, then this run's: the file order is the
		// order of last use across runs, so the oldest-unused records go first.
		bool                  evict_unused = false;
		// KYTY_PROGRAM_CACHE_COMPRESS: zstd level for permutations added by this run (0: stored
		// plain). Loading reads plain and packed records either way.
		int                   compress_level = 0;
		// Coalescing (see Due()), like the driver pipeline cache saves: after a burst (at least
		// min_new new records, none for settle_ns, a third of an interval since the last save),
		// after a trickle (none for settle_ns, an interval since the last save), and after three
		// intervals regardless. Each save rewrites the whole file.
		uint64_t              min_new     = 16;
		uint64_t              settle_ns   = 3'000'000'000ull;
		uint64_t              interval_ns = 60'000'000'000ull;
		// Tests: log nothing.
		bool                  quiet = false;
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
		// A packed record's expanded body: the spans above point into it (null: into the cache).
		std::shared_ptr<const std::vector<uint8_t>> body;
	};

	struct Stats {
		bool     file_found       = false;
		bool     header_rejected  = false;
		uint64_t file_bytes       = 0;
		uint64_t load_ns          = 0;
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
		uint64_t evicted          = 0; // loaded records dropped for room (evict_unused)
		uint64_t evicted_bytes    = 0;
		uint64_t restored         = 0; // evicted records used again this run and kept
		uint64_t packed_added     = 0; // permutations this run stored packed
		uint64_t packed_raw_bytes = 0; // their payload bytes before / after packing
		uint64_t packed_bytes     = 0;
		uint64_t pack_ns          = 0;
		uint64_t expanded         = 0; // packed records expanded by a hit
		uint64_t expanded_bytes   = 0;
		uint64_t expand_ns        = 0;
	};

	explicit ProgramDiskCache(Settings settings);
	// Flushes pending records.
	~ProgramDiskCache();
	ProgramDiskCache(const ProgramDiskCache&)            = delete;
	ProgramDiskCache& operator=(const ProgramDiskCache&) = delete;

	// Both wait for the loader. The returned spans stay valid for the cache's lifetime (a packed
	// permutation's: while its record's `body` lives).
	// note_use=false (a precompile replay of the journal): the hit does not count as a use for
	// evict_unused, so the journal alone does not pin every record.
	[[nodiscard]] std::optional<SourceRecord> FindSource(const SourceKey& key, bool note_use = true);
	[[nodiscard]] std::optional<PermutationRecord>
	FindPermutation(const Digest& source, uint32_t push_data_cursor,
	                std::span<const uint8_t> specialization, bool note_use = true);

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
	static constexpr uint32_t RecordPermutationPacked = 3;
	static constexpr uint32_t RecordSourcePacked      = 4;
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
		// Packed records: the zstd frame (permutations: info + SPIR-V, never kept expanded;
		// sources: the plan, which ExpandSource fills in once).
		std::span<const uint8_t> packed;
		uint32_t                 raw_size = 0;
		bool                     expanded = true;
		bool                     invalid = false;
		// evict_unused: returned by a lookup this run; left out of saves.
		bool                     used    = false;
		bool                     evicted = false;
	};

	void Load();
	void WaitLoaded();
	bool ParseRecord(std::span<const uint8_t> bytes, Record& record) const;
	// Parses the expanded body of a packed permutation (info, SPIR-V, plain SPIR-V).
	static bool ParseBody(std::span<const uint8_t> body, Record& record);
	// Decompresses a packed record's frame (no lock needed: the stored bytes are immutable).
	static std::shared_ptr<std::vector<uint8_t>> Unpack(std::span<const uint8_t> packed,
	                                                    uint32_t raw_size);
	// Packs `body` at m_settings.compress_level; empty when packing fails or does not pay.
	std::vector<uint8_t> Pack(std::span<const uint8_t> body) const;
	// Takes and returns `lock` (on m_mutex): expands the packed source `id`'s plan, unlocked while
	// decompressing. False when the frame is damaged (the record is then invalidated).
	bool ExpandSource(uint32_t id, std::unique_lock<std::mutex>& lock);
	// Requires m_mutex.
	void InvalidateLocked(uint32_t id);
	[[nodiscard]] static bool IsSourceKind(uint32_t kind) {
		return kind == RecordSource || kind == RecordSourcePacked;
	}
	// Requires m_mutex. Adds a parsed record to the index; false when it was a duplicate or made
	// its digest ambiguous (the record is then not indexed).
	bool IndexRecord(uint32_t id);
	// Requires m_mutex. A lookup returned the record (evict_unused bookkeeping).
	void NoteUsed(uint32_t id);
	// Requires m_mutex. Evicts unused loaded records until `bytes` more fit; false when they do not.
	bool MakeRoom(uint64_t bytes);
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
	std::deque<std::vector<uint8_t>> m_expanded; // plans of expanded packed sources
	std::vector<Record>     m_records;
	uint32_t                m_loaded_count = 0; // records [0, m_loaded_count) came from the file
	bool                    m_order_dirty  = false; // evict_unused: use order changed since the last save
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
