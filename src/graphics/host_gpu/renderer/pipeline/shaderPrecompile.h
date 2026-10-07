#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERPRECOMPILE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERPRECOMPILE_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

// Shader precompile (KYTY_SHADER_PRECOMPILE=1, default off): programs the game compiled in earlier
// runs are compiled again on background threads at start-up, before the game asks for them, so
// that the first visit to an area finds its programs published instead of translating and
// emitting them on the command processor.
//
// What is predicted, and how: the game's shaders cannot be enumerated statically with the keys
// the program cache uses. A permutation is keyed by the guest code, the stage's static key (vertex
// input layout, interpolators, render-target export mapping, ...) and a resource specialization
// (descriptor formats, image dimensions and numeric classes read from guest memory at the draw).
// The journal therefore stores the inputs of every permutation a run compiled: per program source
// the code words, the compile options and the stage input info; per permutation the push-data
// cursor and the encoded specialization. It stores inputs, not outputs, so it stays usable when
// the translator changes (the persistent program cache, programDiskCache.h, is dropped then).
//
// ShaderJournal is the file: loaded by the constructor (a damaged tail keeps the intact prefix,
// another identity discards the file), appended to by a writer thread. ShaderPrecompiler replays
// the loaded entries in file order (the order the earlier run needed them in) on worker threads
// through a callback that compiles one permutation (PipelineCache's program cache).
class ShaderJournal {
public:
	enum class Kind : uint8_t { Vertex = 0, Pixel = 1, Compute = 2 };

	struct Settings {
		std::filesystem::path path;
		// Header identity: the file is used only when byte-equal (device, input layouts).
		std::vector<uint8_t>  identity;
		// New records stop being added beyond this file size.
		uint64_t              max_file_bytes = 256ull * 1024 * 1024;
		// Background writes (false: only Flush writes; tests).
		bool                  background_writer = true;
		uint64_t              flush_interval_ns = 2'000'000'000ull;
		// Messages (null: none).
		std::function<void(const std::string&)> log;
	};

	// One program source. Identity: stage, hash, user_data_count, code_size, wave_size,
	// user_data_base, plain_mip_stats_variant and static_state; code and input_info are the
	// translation inputs that identity stands for.
	struct Source {
		uint32_t              stage                   = 0; // ShaderType
		Kind                  kind                    = Kind::Vertex;
		uint64_t              hash                    = 0;
		uint32_t              user_data_count         = 0;
		uint32_t              code_size               = 0; // words; code.size() once filled
		uint32_t              wave_size               = 0;
		uint32_t              user_data_base          = 0;
		bool                  plain_mip_stats_variant = false;
		std::vector<uint32_t> static_state;
		std::vector<uint32_t> code;
		// The stage input info struct, bytes as in memory, with its runtime pointers cleared.
		std::vector<uint8_t>  input_info;
	};
	struct Entry {
		uint32_t             source = 0; // index into Sources()
		uint32_t             push_data_cursor = 0;
		std::vector<uint8_t> specialization;
	};

	struct Stats {
		bool     file_found      = false;
		bool     header_rejected = false;
		uint64_t file_bytes      = 0;
		uint64_t load_ns         = 0;
		uint64_t damaged_bytes   = 0; // ignored tail
		uint64_t recorded_sources = 0;
		uint64_t recorded_entries = 0;
		uint64_t written_bytes   = 0;
		uint64_t write_failures  = 0;
		bool     over_capacity   = false;
	};

	static constexpr char     FileMagic[8]  = {'K', 'Y', 'S', 'H', 'J', 'R', 'N', '1'};
	static constexpr uint32_t FormatVersion = 1;
	static constexpr uint32_t RecordMagic   = 0x524a484bu; // "KHJR"
	static constexpr uint32_t RecordSource  = 1;
	static constexpr uint32_t RecordEntry   = 2;

	explicit ShaderJournal(Settings settings);
	// Writes pending records.
	~ShaderJournal();
	ShaderJournal(const ShaderJournal&)            = delete;
	ShaderJournal& operator=(const ShaderJournal&) = delete;

	// What the file held (immutable until ReleaseLoaded).
	[[nodiscard]] const std::vector<Source>& Sources() const { return m_sources; }
	[[nodiscard]] const std::vector<Entry>&  Entries() const { return m_entries; }

	// Frees Sources() and Entries() (after the replay is done; nothing may use them then).
	void ReleaseLoaded();

	// Whether a source with this identity is journaled already (loaded or recorded this run); the
	// caller then needs not fill code and input_info.
	[[nodiscard]] bool HasSource(const Source& identity);
	// Journals one compiled permutation. A known permutation is ignored; a new source needs code
	// and input_info.
	void Record(Source source, uint32_t push_data_cursor, std::span<const uint8_t> specialization);

	// Writes the pending records now. False when a write failed.
	bool Flush();

	[[nodiscard]] Stats GetStats();
	[[nodiscard]] const std::filesystem::path& Path() const { return m_settings.path; }

private:
	static uint64_t SourceDigest(const Source& source);
	static bool     SameIdentity(const Source& a, const Source& b);
	void            Load();
	bool            AppendPending(bool final_flush);
	void            RunWriter();
	void            Log(const std::string& message) const;

	Settings m_settings;
	std::vector<Source> m_sources;
	std::vector<Entry>  m_entries;

	std::mutex m_mutex;
	// Identity digest -> indices into m_known (sources loaded or recorded).
	std::unordered_multimap<uint64_t, uint32_t> m_by_digest;
	// Identity of every known source (code and input_info omitted for recorded ones).
	std::vector<Source> m_known;
	std::unordered_set<uint64_t> m_known_entries; // digest of (source index, cursor, specialization)
	std::vector<std::vector<uint8_t>> m_pending;  // encoded records, in order
	uint64_t m_pending_bytes = 0;
	uint32_t m_next_source   = 0;
	uint64_t m_file_valid_bytes = 0; // length of the intact prefix (0: the file is to be recreated)
	uint64_t m_file_bytes    = 0;    // length after the pending records
	bool     m_file_ready    = false; // the header is in the file
	Stats    m_stats;

	std::mutex              m_write_mutex; // one write at a time
	std::mutex              m_writer_mutex;
	std::condition_variable m_writer_cv;
	bool                    m_stop = false;
	std::thread             m_writer;
};

// A journal of keyed records next to the shader journal (KYTY_PIPELINE_JOURNAL and
// KYTY_PIPELINE_KNOWN, pipelineCache.cpp): each record is a 64-bit key (a pipeline's content hash)
// with an opaque payload, kept once per key, in the order the keys were first recorded. The file
// has the shader journal's layout (a header with an identity, then checksummed records; a damaged
// tail keeps the intact prefix, another identity discards the file). Records are appended by a
// writer thread, or, with `sealed_writes`, only up to the last Seal() when Flush() runs: the
// known-pipeline list is written with the driver cache file, so it never names a pipeline the
// saved driver cache cannot hold.
class PipelineJournal {
public:
	struct Settings {
		std::filesystem::path path; // empty: in memory only (nothing is loaded or written)
		std::vector<uint8_t>  identity;
		uint64_t              max_file_bytes    = 64ull * 1024 * 1024;
		bool                  background_writer = true;
		uint64_t              flush_interval_ns = 2'000'000'000ull;
		// Flush writes only the records recorded before the last Seal().
		bool                  sealed_writes     = false;
		std::function<void(const std::string&)> log;
	};
	struct Record {
		uint64_t             key = 0;
		std::vector<uint8_t> payload;
	};
	struct Stats {
		bool     file_found      = false;
		bool     header_rejected = false;
		uint64_t file_bytes      = 0;
		uint64_t load_ns         = 0;
		uint64_t damaged_bytes   = 0;
		uint64_t loaded          = 0;
		uint64_t recorded        = 0;
		uint64_t written_bytes   = 0;
		uint64_t write_failures  = 0;
		bool     over_capacity   = false;
		bool     reset           = false;
	};

	static constexpr char     FileMagic[8]  = {'K', 'Y', 'P', 'L', 'J', 'R', 'N', '1'};
	static constexpr uint32_t FormatVersion = 1;
	static constexpr uint32_t RecordMagic   = 0x524a4c50u; // "PLJR"

	explicit PipelineJournal(Settings settings);
	~PipelineJournal();
	PipelineJournal(const PipelineJournal&)            = delete;
	PipelineJournal& operator=(const PipelineJournal&) = delete;

	// What the file held, in file order (immutable until ReleaseLoaded).
	[[nodiscard]] const std::vector<Record>& Loaded() const { return m_loaded; }
	void ReleaseLoaded();

	// Whether the key was loaded or recorded (and not dropped by Reset).
	[[nodiscard]] bool Contains(uint64_t key) const;
	// Adds a record for a new key. False for a known key or beyond the size cap.
	bool Add(uint64_t key, std::span<const uint8_t> payload = {});
	// Forgets every key, loaded or recorded; the next write recreates the file (the driver cache
	// it describes was rejected or started empty).
	void Reset();
	// Marks the records added so far as writable (sealed_writes).
	void Seal();
	// Writes the pending (sealed) records now. False when a write failed.
	bool Flush();

	[[nodiscard]] Stats GetStats() const;
	[[nodiscard]] size_t KnownCount() const;
	[[nodiscard]] const std::filesystem::path& Path() const { return m_settings.path; }

private:
	void Load();
	bool AppendPending();
	void RunWriter();
	void Log(const std::string& message) const;

	Settings            m_settings;
	std::vector<Record> m_loaded;

	mutable std::mutex                m_mutex;
	std::unordered_set<uint64_t>      m_known;
	std::vector<std::vector<uint8_t>> m_pending; // encoded records, in order
	size_t                            m_sealed        = 0; // m_pending[0, m_sealed) may be written
	uint64_t                          m_pending_bytes = 0;
	uint64_t                          m_file_valid_bytes = 0;
	uint64_t                          m_file_bytes       = 0;
	bool                              m_file_ready       = false;
	Stats                             m_stats;

	std::mutex              m_write_mutex;
	std::mutex              m_writer_mutex;
	std::condition_variable m_writer_cv;
	bool                    m_stop = false;
	std::thread             m_writer;
};

// Below-normal priority for the calling thread (background compile workers; Windows only).
void LowerCurrentThreadPriority();

// Replays journal entries on worker threads.
class ShaderPrecompiler {
public:
	enum class Outcome : uint8_t {
		Compiled, // translated, emitted or reloaded, and published
		Present,  // already published (the game or an earlier entry got there first)
		Skipped,  // not replayable (stale input, skip-dispatch program, declined)
		Failed,
	};
	using CompileFn = std::function<Outcome(const ShaderJournal::Source&, const ShaderJournal::Entry&)>;

	struct Settings {
		uint32_t threads     = 2;
		// Replays at most this many entries (0: all).
		uint64_t max_entries = 0;
		// Workers run below normal priority (the game's threads come first).
		bool     low_priority = true;
		// Entries of the sources this accepts are replayed first, in file order, then the rest in
		// file order (also under max_entries: the cap drops the last of the rest). Called once per
		// source, from the constructor. The software ray-tracing kernels use it: their pipelines
		// take seconds to build and the game needs them at its first lighting pass.
		std::function<bool(const ShaderJournal::Source&)> priority_source;
		// Called on each worker thread when it starts (name, priority).
		std::function<void(uint32_t worker)> thread_init;
		std::function<void(const std::string&)> log;
		// Called once on the last worker after the final line (nothing reads the journal's loaded
		// content any more).
		std::function<void()> on_finished;
		// A progress line after this many percent and at least this many nanoseconds.
		uint32_t progress_percent = 10;
		uint64_t progress_ns      = 5'000'000'000ull;
	};

	struct Progress {
		uint64_t total    = 0;
		uint64_t done     = 0;
		uint64_t compiled = 0;
		uint64_t present  = 0;
		uint64_t skipped  = 0;
		uint64_t failed   = 0;
		bool     finished = false;
	};

	// Starts at once; the journal must outlive this object.
	ShaderPrecompiler(const ShaderJournal& journal, Settings settings, CompileFn compile);
	~ShaderPrecompiler();
	ShaderPrecompiler(const ShaderPrecompiler&)            = delete;
	ShaderPrecompiler& operator=(const ShaderPrecompiler&) = delete;

	// Entries not started yet are dropped; entries in progress finish.
	void Stop();
	// Returns once every entry was replayed (or Stop took effect).
	void Wait();
	[[nodiscard]] Progress GetProgress() const;

private:
	void Worker(uint32_t index);
	void NoteDone(ShaderPrecompiler::Outcome outcome);

	const ShaderJournal& m_journal;
	Settings             m_settings;
	CompileFn            m_compile;
	uint64_t             m_total = 0;
	uint64_t             m_begin_ns = 0;

	// Replay order (indices into the journal's entries) when priority_source reordered them.
	std::vector<uint32_t> m_order;
	std::atomic<uint64_t> m_next {0};
	std::atomic<bool>     m_stop {false};
	std::atomic<uint64_t> m_done {0}, m_compiled {0}, m_present {0}, m_skipped {0}, m_failed {0};
	std::atomic<uint32_t> m_workers_running {0};

	mutable std::mutex      m_mutex; // progress reporting
	std::condition_variable m_finished_cv;
	uint64_t                m_reported_done = 0;
	uint64_t                m_reported_ns   = 0;
	bool                    m_finished      = false;
	std::vector<std::thread> m_threads;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERPRECOMPILE_H_
