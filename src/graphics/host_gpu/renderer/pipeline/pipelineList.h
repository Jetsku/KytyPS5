#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIST_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIST_H_

#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// Pipeline journal record payload (KYTY_PIPELINE_JOURNAL, pipelineCache.cpp), words: magic, snapshot
// word count, level marker (length and packed characters), stage count and per stage the module's
// permutation identity and bindings hash (two words each), then the snapshot words
// (GraphicsPipelineSnapshot::Serialize), the layout signature length and the signature.
struct PipelineJournalPayload {
	std::string                                level;
	std::vector<std::pair<uint64_t, uint64_t>> identities; // per stage: permutation identity, bindings hash
	std::vector<uint32_t>                      words;      // snapshot words, signature length, signature
	uint32_t                                   snapshot_words = 0;
};
inline constexpr uint32_t PipelineJournalPayloadMagic = 0x314a4c50u; // "PLJ1"

[[nodiscard]] std::vector<uint8_t> EncodePipelineJournalPayload(const std::string& level,
                                                                std::span<const uint32_t> words,
                                                                uint32_t snapshot_words,
                                                                std::span<const std::pair<uint64_t, uint64_t>> identities);
[[nodiscard]] bool DecodePipelineJournalPayload(std::span<const uint8_t> bytes, PipelineJournalPayload& out);

// The translator-independent identity of a program permutation, which its shader modules carry
// (ShaderModuleRegistry, pipelineCache.cpp): the shader journal's source identity, the push-data cursor
// and the encoded specialization. The mip-statistics "plain" module of the same permutation has
// identity ^ PlainModuleIdentitySalt.
[[nodiscard]] uint64_t PermutationIdentity(uint32_t stage, uint64_t hash, uint32_t user_data_count, uint32_t code_size,
                                           uint32_t wave_size, uint32_t user_data_base, bool plain_mip_stats_variant,
                                           uint32_t push_data_cursor, std::span<const uint32_t> static_state,
                                           std::span<const uint8_t> specialization);
[[nodiscard]] uint64_t PermutationIdentity(const ShaderJournal::Source& source, const ShaderJournal::Entry& entry);
// A program source's guest hash from its code: XXH3-64 of the code, or for a merged stage the XXH3-64
// of the two halves' XXH3-64 hashes (shader.cpp, the NGG GS front and back halves).
[[nodiscard]] uint64_t MergedSourceHash(std::span<const uint32_t> code, std::span<const uint32_t> back_code);
inline constexpr uint64_t PlainModuleIdentitySalt = 0x9E3779B97F4A7C15ull;

// KYPLST1: a shareable, keys-only list of the pipelines a game needs (KYTY_PIPELINE_LIST,
// pipelineCache.cpp; kyty_pipeline_list exports, merges and verifies it).
//
// It holds no game code and no translator output. A shader is named by where its code sits in the
// game's own files (the executable's byte offset and size) and by strong hashes of that code; the
// emulator reads the code from the player's copy and uses the shader only when the hashes match.
// Next to it: the program source identity the shader journal keeps (stage, user data, static state,
// the stage input info bytes), permutations (push-data cursor, encoded resource specialization) and
// graphics pipelines (the stages as permutation references, then the Vulkan state words of
// GraphicsPipelineSnapshot::Serialize with the module hashes cleared, and the layout signature), with
// the boot levels each pipeline was recorded in. Device-specific input fields (host subgroup size,
// mesh group splitting) are recomputed by the loader, so the list is not tied to a GPU.
//
// File: magic "KYPLST1\0", format version, section count, then per section a header (id, flags, stored
// and raw size, XXH3-64 of the raw bytes) and its zstd-compressed bytes. Unknown sections are skipped;
// every count and size is bounds-checked against the section it is in.
// The host path of the game executable the emulator loads (set at start-up, before the renderer
// exists: the pipeline list reads shader code from it).
void                                SetPipelineListExecutable(const std::filesystem::path& path);
[[nodiscard]] std::filesystem::path PipelineListExecutable();

namespace PipelineList {

inline constexpr char     FileMagic[8]  = {'K', 'Y', 'P', 'L', 'S', 'T', '1', '\0'};
inline constexpr uint32_t FormatVersion = 2;

enum class SectionId : uint32_t {
	Info         = 1,
	Levels       = 2,
	Shaders      = 3,
	Permutations = 4,
	Pipelines    = 5,
	Files        = 6, // the game data files shaders are read from (optional; CodeFile 1..n)
};

// The file a shader's code is read from.
// 0 is the game's executable (/app0/eboot.bin); n >= 1 is File::files[n - 1], a data file of the
// game (path relative to /app0, '/' separated): Insomniac titles keep most shader code in d/material.
enum class CodeFile : uint32_t {
	Executable = 0,
};
inline constexpr uint32_t MaxDataFiles = 4096;

struct Info {
	std::string title_id;    // TITLE_ID of param.json / param.sfo
	std::string app_version; // APP_VER (contentVersion), e.g. "01.018.000"
	std::string producer;    // the emulator build or tool that wrote the list
	// sizeof(ShaderVertexInputInfo), sizeof(ShaderPixelInputInfo), sizeof(ShaderComputeInputInfo) of the
	// producer: the stage input info bytes are those structs.
	std::array<uint32_t, 3> input_info_sizes {};
	uint64_t                created_unix = 0;
	// The renderer's pipeline state format the pipelines were recorded with (StateFormat of the
	// producer; 0: a list older than the field). A list of another format keeps its shaders (the
	// precompile is still exact) but its pipelines are not replayed: they would describe create infos
	// this build no longer makes.
	uint32_t state_format = 0;
};
// Bump when CreatePipelineInternal (pipeline/shaders.cpp) or GraphicsPipelineSnapshot changes the
// graphics pipeline create infos it makes for the same draw state (released lists then keep only
// their shaders until they are re-recorded).
inline constexpr uint32_t StateFormat = 1;

struct Shader {
	CodeFile file      = CodeFile::Executable;
	uint64_t offset    = 0;
	uint32_t code_size = 0; // words
	uint64_t code_hash_low = 0, code_hash_high = 0; // XXH3-128 of the code bytes
	uint64_t prefix_hash = 0; // XXH3-64 of the first min(64, size) code bytes (relocation search)
	// A merged stage's back half (Source::back_code), located the same way; size 0: none.
	uint64_t back_offset    = 0;
	uint32_t back_code_size = 0; // words
	uint64_t back_hash_low = 0, back_hash_high = 0, back_prefix_hash = 0;
	// The program source identity (ShaderJournal::Source; `hash` is the XXH3-64 of the code bytes).
	uint32_t              stage                   = 0;
	ShaderJournal::Kind   kind                    = ShaderJournal::Kind::Vertex;
	uint64_t              hash                    = 0;
	uint32_t              user_data_count         = 0;
	uint32_t              wave_size               = 0;
	uint32_t              user_data_base          = 0;
	bool                  plain_mip_stats_variant = false;
	std::vector<uint32_t> static_state;
	std::vector<uint8_t>  input_info;
};

struct Permutation {
	uint32_t             shader = 0;
	uint32_t             push_data_cursor = 0;
	std::vector<uint8_t> specialization;
};

struct Stage {
	uint32_t permutation = 0;
	bool     plain       = false; // the permutation's mip-statistics plain module
	uint64_t bindings    = 0;     // the producer's descriptor bindings hash (a different one: skipped)
};

struct Pipeline {
	uint64_t              key = 0; // the producer's content key (informational; runtime keys differ)
	std::vector<uint16_t> levels;  // indices into File::levels
	std::vector<Stage>    stages;  // snapshot order (vertex or mesh first)
	std::vector<uint32_t> words;   // snapshot words (module hashes 0), signature length, signature
	uint32_t              snapshot_words = 0;
};

struct File {
	Info                     info;
	std::vector<std::string> levels;
	std::vector<Shader>      shaders;
	std::vector<Permutation> permutations;
	std::vector<Pipeline>    pipelines;
	std::vector<std::string> files; // data files (CodeFile n >= 1 is files[n - 1])
};

[[nodiscard]] std::vector<uint8_t> Encode(const File& file, int zstd_level = 19);
// False (and `error`) for another format or a damaged file; `file` is then unspecified.
[[nodiscard]] bool Decode(std::span<const uint8_t> bytes, File& file, std::string* error = nullptr);
// The raw (decompressed) bytes of every section, for the verify tool.
[[nodiscard]] bool DecodeSections(std::span<const uint8_t> bytes, std::vector<std::pair<uint32_t, std::vector<uint8_t>>>& sections,
                                  std::string* error = nullptr);

// Snapshot words with every module hash replaced by `hash_for(stage index)` (0 clears them). False
// when the words are not a snapshot.
[[nodiscard]] bool RewriteSnapshotModules(std::span<uint32_t> words, const std::function<uint64_t(uint32_t)>& hash_for);

// The values of a stored snapshot that come from the recording device, not from the draw: the
// dynamic state VK_DYNAMIC_STATE_ATTACHMENT_FEEDBACK_LOOP_ENABLE_EXT (present when the device enabled
// attachment feedback loops, shaders.cpp) and each stage's required subgroup size
// (GraphicContext::GraphicsSubgroupSize). The run-time pipeline key holds the running device's own
// values, so a pipeline recorded on another device would be built and never found.
//
// Canonical stored form (what the exporter and the merge write): the feedback-loop state absent, every
// stage's required subgroup size 0. The loader applies this device's values to a list pipeline
// before it builds it. `subgroup_for` gets the stage's VkShaderStageFlags word and its index in the
// snapshot; null means 0 for every stage. `words` are a pipeline's words (snapshot words, then the
// layout signature length and the signature), `snapshot_words` the snapshot's part, which grows or
// shrinks by one word with the state. False (nothing changed) when the words are not a snapshot.
inline constexpr uint32_t DynamicStateAttachmentFeedbackLoop = 1000524000u; // VK_DYNAMIC_STATE_ATTACHMENT_FEEDBACK_LOOP_ENABLE_EXT
using SubgroupSizeFn = std::function<uint32_t(uint32_t stage_flags, uint32_t stage_index)>;
[[nodiscard]] bool ApplySnapshotDeviceState(std::vector<uint32_t>& words, uint32_t& snapshot_words, bool feedback_loop,
                                            const SubgroupSizeFn& subgroup_for);
// The canonical form (above) of `pipeline`.
[[nodiscard]] bool NormalizePipelineDeviceState(Pipeline& pipeline);

struct Hash128 {
	uint64_t low = 0, high = 0;
	bool     operator==(const Hash128&) const = default;
};
[[nodiscard]] Hash128  CodeHash(std::span<const uint8_t> code);
[[nodiscard]] uint64_t CodePrefixHash(std::span<const uint8_t> code);

// Finds code in a file image by its hashes alone, in one pass over every byte offset: the exporter
// locates journaled code in the executable, the loader relocates shaders whose recorded offset does not
// hold them in the player's file (another executable wrapper). Per query the first matching offset.
struct CodeQuery {
	uint32_t size_bytes  = 0;
	uint32_t first_word  = 0; // the code's first 4 bytes (a filter; the list does not store them)
	bool     first_known = false;
	uint64_t prefix_hash = 0; // CodePrefixHash
	Hash128  hash;            // CodeHash
};
[[nodiscard]] std::vector<std::optional<uint64_t>> FindCode(std::span<const uint8_t> image, std::span<const CodeQuery> queries);
[[nodiscard]] CodeQuery QueryFor(std::span<const uint8_t> code);

// Exporting: shader journals and pipeline journals (one or several recording runs) merged into one
// list. Shaders are kept only when their code is found in the executable image; their permutations
// and the pipelines using them follow. Duplicates (same source identity, permutation, or pipeline
// stages and state) are merged; a pipeline's levels are the union of its records' level markers.
struct ExportStats {
	uint64_t sources = 0, sources_located = 0, sources_unlocated = 0, sources_duplicate = 0;
	uint64_t entries = 0, entries_kept = 0, entries_dropped = 0;
	uint64_t pipelines = 0, pipelines_kept = 0, pipelines_merged = 0, pipelines_unresolved = 0, pipelines_bad = 0;
};

class Builder {
public:
	explicit Builder(Info info);

	// One recording run: its shader journal content and pipeline journal records. `locate` gives the
	// executable offset of a source's code (nullopt: not found, the source is dropped).
	struct Location {
		CodeFile file   = CodeFile::Executable;
		uint64_t offset = 0;
	};
	using LocateFn = std::function<std::optional<Location>(std::span<const uint32_t> code)>;
	void AddJournals(const std::vector<ShaderJournal::Source>& sources, const std::vector<ShaderJournal::Entry>& entries,
	                 const std::vector<PipelineJournal::Record>& pipelines, const LocateFn& locate, ExportStats& stats);
	// The data file named `path` (relative to /app0) as a CodeFile, added to the list's table once.
	CodeFile AddFile(const std::string& path);
	// Another list (merge).
	void AddList(const File& other, ExportStats& stats);

	[[nodiscard]] File Finish() const { return m_file; }

private:
	uint32_t AddShader(const Shader& shader, bool& added);
	uint32_t AddPermutation(const Permutation& permutation, bool& added);
	uint16_t AddLevel(const std::string& level);
	void     AddPipeline(Pipeline pipeline, ExportStats& stats);

	File                                          m_file;
	std::unordered_multimap<uint64_t, uint32_t>   m_shaders;      // identity digest -> index
	std::unordered_multimap<uint64_t, uint32_t>   m_permutations; // digest -> index
	std::unordered_multimap<uint64_t, uint32_t>   m_pipelines;    // digest -> index
	std::unordered_map<std::string, uint16_t>     m_levels;
};

// Loading: a list's shaders checked against the player's game files and turned into replayable
// shader journal sources and entries (code filled), and its pipelines into the permutations' runtime
// identities.
struct ResolveStats {
	uint64_t shaders = 0, resolved = 0, relocated = 0, mismatched = 0, unreadable = 0, other_input_layout = 0;
	uint64_t permutations = 0, permutations_kept = 0;
	uint64_t pipelines = 0, pipelines_kept = 0;
};

// Reads `out.size()` bytes at `offset` of `file`; false when it cannot.
using ReadCodeFn = std::function<bool(CodeFile file, uint64_t offset, std::span<uint8_t> out)>;
// The whole file (relocation search); empty when unavailable.
using ReadImageFn = std::function<std::vector<uint8_t>(CodeFile file)>;

struct Resolved {
	// Index-aligned with the list's shaders: code filled when the shader matched (else empty).
	std::vector<ShaderJournal::Source> sources;
	std::vector<bool>                  usable;
	// The usable permutations, as shader journal entries (source = list shader index), and the list
	// permutation index of each.
	std::vector<ShaderJournal::Entry> entries;
	std::vector<uint32_t>             entry_permutation;
	// Per list permutation: index into `entries`, or UINT32_MAX when unusable.
	std::vector<uint32_t> permutation_entry;
	// Per list shader, where its code (and a merged stage's back half) sits in the file read; usable only.
	std::vector<uint64_t> offsets, back_offsets;
};

[[nodiscard]] Resolved Resolve(const File& file, const std::array<uint32_t, 3>& input_info_sizes, const ReadCodeFn& read,
                               const ReadImageFn& read_image, ResolveStats& stats);

// A stored vertex input info layout of an earlier release and the fields this build added to it:
// `bytes` go in at byte `at` of the old layout (inserts in ascending order). A shader whose dword at
// `skip_if_nonzero_at` (old layout) is not zero keeps its old bytes, so Resolve counts it as another
// input layout (fields that cannot be filled in without the guest registers).
struct VertexLayoutUpgrade {
	struct Insert {
		uint32_t             at = 0;
		std::vector<uint8_t> bytes;
	};
	uint32_t            from_size = 0;
	std::vector<Insert> inserts;
	uint32_t            skip_if_nonzero_at = UINT32_MAX;
};
struct UpgradeStats {
	uint64_t upgraded = 0, skipped = 0;
};
// Rewrites the vertex input infos of `upgrade.from_size` bytes to `to_size` bytes; nothing when the
// inserts do not add up to `to_size`.
void UpgradeVertexInputInfos(File& file, const VertexLayoutUpgrade& upgrade, uint32_t to_size, UpgradeStats& stats);

// Retargeting: the list re-stamped for another copy of the game (another version or SKU) whose shaders
// `resolved` found. Shaders not found are dropped with their permutations and pipelines; the offsets are the
// new ones. The caller sets the title and version in `info`.
struct RetargetStats {
	uint64_t shaders = 0, same_offset = 0, moved = 0, missing = 0;
	uint64_t pipelines = 0, pipelines_kept = 0;
};
[[nodiscard]] File Retarget(const File& file, const Resolved& resolved, RetargetStats& stats);

// The "no game code" proof: every 64-byte window of known shader code (any byte offset), minus
// low-entropy windows (fewer than 8 distinct byte values, e.g. zero runs that also occur in plain
// data), is searched for at every byte offset of the data.
class CodeWindows {
public:
	static constexpr size_t WindowBytes = 64;
	void Add(std::span<const uint8_t> code);
	// Offsets in `data` where a known code window starts.
	[[nodiscard]] std::vector<uint64_t> FindIn(std::span<const uint8_t> data, size_t max_hits = 16) const;
	[[nodiscard]] size_t   Count() const { return m_windows.size(); }
	[[nodiscard]] uint64_t SkippedLowEntropy() const { return m_low_entropy; }

private:
	std::unordered_set<uint64_t> m_windows;
	uint64_t                     m_low_entropy = 0;
};

} // namespace PipelineList

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIST_H_
