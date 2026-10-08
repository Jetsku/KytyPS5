// The KYPLST1 pipeline list (graphics/host_gpu/renderer/pipeline/pipelineList.h, KYTY_PIPELINE_LIST):
// the file round-trips and refuses damaged input; journals export to a list without code, keyed on
// the guest shaders (executable offset and hashes) and the permutation identities; lists merge; a
// list resolves against an executable image, rejecting shaders whose bytes differ and finding moved
// ones; the no-code check finds copied code; the shader journal takes list content for the replay.
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <xxhash.h>

using namespace Libs::Graphics;

static int g_failed = 0;
#define CHECK(expr)                                                                \
	do {                                                                           \
		if (!(expr)) {                                                             \
			std::printf("PipelineListTests: FAILED: %s (line %d)\n", #expr, __LINE__); \
			++g_failed;                                                            \
		}                                                                          \
	} while (0)

namespace {

template <typename Handle>
Handle FakeHandle(uint64_t value) {
	return Handle(reinterpret_cast<typename Handle::CType>(value));
}

// A minimal create info shaped like CreatePipelineInternal's (pipeline/shaders.cpp).
struct Example {
	vk::Format                               colors[1] = {vk::Format::eR8G8B8A8Unorm};
	vk::PipelineRenderingCreateInfo          rendering {};
	vk::PipelineShaderStageCreateInfo        stages[2] {};
	vk::VertexInputAttributeDescription      attribute {0, 0, vk::Format::eR32G32B32Sfloat, 0};
	vk::VertexInputBindingDescription        binding {0, 12, vk::VertexInputRate::eVertex};
	vk::PipelineVertexInputStateCreateInfo   vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	vk::PipelineViewportStateCreateInfo      viewport {};
	vk::PipelineRasterizationStateCreateInfo rasterization {};
	vk::PipelineMultisampleStateCreateInfo   multisample {};
	vk::PipelineColorBlendAttachmentState    blend {};
	vk::PipelineColorBlendStateCreateInfo    color_blend {};
	vk::DynamicState                         dynamic_states[2] = {vk::DynamicState::eViewportWithCount,
	                                                              vk::DynamicState::eScissorWithCount};
	vk::PipelineDynamicStateCreateInfo       dynamic {};
	vk::GraphicsPipelineCreateInfo           info {};

	explicit Example(vk::PrimitiveTopology topology) {
		rendering.colorAttachmentCount    = 1;
		rendering.pColorAttachmentFormats = colors;
		stages[0].stage                   = vk::ShaderStageFlagBits::eVertex;
		stages[0].module                  = FakeHandle<vk::ShaderModule>(0x1000);
		stages[0].pName                   = "main";
		stages[1].stage                   = vk::ShaderStageFlagBits::eFragment;
		stages[1].module                  = FakeHandle<vk::ShaderModule>(0x2000);
		stages[1].pName                   = "main";
		vertex_input.vertexBindingDescriptionCount   = 1;
		vertex_input.pVertexBindingDescriptions      = &binding;
		vertex_input.vertexAttributeDescriptionCount = 1;
		vertex_input.pVertexAttributeDescriptions    = &attribute;
		input_assembly.topology                      = topology;
		viewport.viewportCount                       = 1;
		viewport.scissorCount                        = 1;
		rasterization.lineWidth                      = 1.0f;
		multisample.rasterizationSamples             = vk::SampleCountFlagBits::e1;
		blend.colorWriteMask                         = vk::ColorComponentFlagBits::eR;
		color_blend.attachmentCount                  = 1;
		color_blend.pAttachments                     = &blend;
		dynamic.dynamicStateCount                    = 2;
		dynamic.pDynamicStates                       = dynamic_states;
		info.pNext               = &rendering;
		info.stageCount          = 2;
		info.pStages             = stages;
		info.pVertexInputState   = &vertex_input;
		info.pInputAssemblyState = &input_assembly;
		info.pViewportState      = &viewport;
		info.pRasterizationState = &rasterization;
		info.pMultisampleState   = &multisample;
		info.pColorBlendState    = &color_blend;
		info.pDynamicState       = &dynamic;
		info.layout              = FakeHandle<vk::PipelineLayout>(0x77);
	}
};

// Snapshot words with SPIR-V-like module hashes, then a layout signature (pipeline/shaders.cpp layout).
std::vector<uint32_t> PipelineWords(vk::PrimitiveTopology topology, uint32_t& snapshot_words) {
	const Example example(topology);
	const auto    snapshot = GraphicsPipelineSnapshot::Capture(example.info);
	std::vector<uint32_t> words;
	if (snapshot == nullptr ||
	    !snapshot->Serialize(words, [](vk::ShaderModule module) {
		    return reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(module)) * 0x9E3779B97F4A7C15ull;
	    })) {
		return {};
	}
	snapshot_words                        = static_cast<uint32_t>(words.size());
	const std::vector<uint32_t> signature = {1, 0, 0x11, 64, 0, 7, 1, 0x10};
	words.push_back(static_cast<uint32_t>(signature.size()));
	words.insert(words.end(), signature.begin(), signature.end());
	return words;
}

// Fake guest code: distinct pseudo-random words (shader code is not low-entropy).
std::vector<uint32_t> Code(uint32_t seed, uint32_t words) {
	std::mt19937          random(seed);
	std::vector<uint32_t> code(words);
	for (auto& word: code) word = random();
	return code;
}

ShaderJournal::Source SourceOf(std::vector<uint32_t> code, ShaderJournal::Kind kind, uint32_t stage, size_t input_bytes) {
	ShaderJournal::Source source;
	source.stage           = stage;
	source.kind            = kind;
	source.hash            = XXH3_64bits(code.data(), code.size() * 4);
	source.user_data_count = 16;
	source.code_size       = static_cast<uint32_t>(code.size());
	source.wave_size       = 32;
	source.static_state    = {stage, 7, 9};
	source.code            = std::move(code);
	source.input_info.assign(input_bytes, 0);
	source.input_info[0] = static_cast<uint8_t>(stage);
	return source;
}

// An "executable": padding with the shaders' code at known offsets.
std::vector<uint8_t> Image(const std::vector<ShaderJournal::Source>& sources, std::vector<uint64_t>& offsets,
                           size_t lead = 0) {
	std::vector<uint8_t> image(lead + 4096, 0xCD);
	for (const auto& source: sources) {
		image.resize(image.size() + 256, 0);
		offsets.push_back(image.size());
		const auto* bytes = reinterpret_cast<const uint8_t*>(source.code.data());
		image.insert(image.end(), bytes, bytes + source.code.size() * 4);
	}
	image.resize(image.size() + 1024, 0xEE);
	return image;
}

constexpr std::array<uint32_t, 3> InputSizes = {96, 64, 48};

PipelineList::Info TestInfo() {
	PipelineList::Info info;
	info.title_id         = "PPSA00000";
	info.app_version      = "01.000.000";
	info.producer         = "PipelineListTests";
	info.input_info_sizes = InputSizes;
	info.created_unix     = 1234;
	return info;
}

struct Recording {
	std::vector<ShaderJournal::Source>   sources;
	std::vector<ShaderJournal::Entry>    entries;
	std::vector<PipelineJournal::Record> pipelines;
};

// Two shaders (VS, PS), two VS permutations, one PS permutation, and pipelines over them.
Recording MakeRecording(const std::string& level, vk::PrimitiveTopology topology, uint32_t pixel_seed = 2) {
	Recording r;
	r.sources.push_back(SourceOf(Code(1, 300), ShaderJournal::Kind::Vertex, 1, InputSizes[0]));
	r.sources.push_back(SourceOf(Code(pixel_seed, 200), ShaderJournal::Kind::Pixel, 2, InputSizes[1]));
	r.entries.push_back({0, 4, {1, 2, 3}});
	r.entries.push_back({0, 8, {1, 2, 3}});
	r.entries.push_back({1, 0, {9}});
	uint32_t   snapshot_words = 0;
	const auto words          = PipelineWords(topology, snapshot_words);
	for (const auto vs: {0, 1}) {
		const std::vector<std::pair<uint64_t, uint64_t>> identities = {
		    {PermutationIdentity(r.sources[0], r.entries[vs]), 0x51},
		    // The pixel stage's mip-statistics plain module.
		    {PermutationIdentity(r.sources[1], r.entries[2]) ^ PlainModuleIdentitySalt, 0x52}};
		r.pipelines.push_back({0x1000u + static_cast<uint64_t>(vs) + static_cast<uint64_t>(topology) * 16,
		                       EncodePipelineJournalPayload(level, words, snapshot_words, identities)});
	}
	return r;
}

PipelineList::Builder::LocateFn LocatorFor(const std::vector<uint8_t>& image) {
	return [&image](std::span<const uint32_t> code) {
		const std::span bytes(reinterpret_cast<const uint8_t*>(code.data()), code.size_bytes());
		const auto      query = PipelineList::QueryFor(bytes);
		return PipelineList::FindCode(image, std::span(&query, 1))[0];
	};
}

PipelineList::ReadCodeFn ReaderFor(const std::vector<uint8_t>& image) {
	return [&image](PipelineList::CodeFile, uint64_t offset, std::span<uint8_t> out) {
		if (offset > image.size() || out.size() > image.size() - offset) return false;
		std::memcpy(out.data(), image.data() + offset, out.size());
		return true;
	};
}

void TestExportAndRoundTrip() {
	auto                  recording = MakeRecording("level_a", vk::PrimitiveTopology::eTriangleList);
	std::vector<uint64_t> offsets;
	const auto            image = Image(recording.sources, offsets);

	PipelineList::Builder     builder(TestInfo());
	PipelineList::ExportStats stats;
	builder.AddJournals(recording.sources, recording.entries, recording.pipelines, LocatorFor(image), stats);
	const auto file = builder.Finish();
	CHECK(stats.sources_located == 2 && stats.sources_unlocated == 0);
	CHECK(stats.entries_kept == 3 && stats.pipelines_kept == 2 && stats.pipelines_unresolved == 0);
	CHECK(file.shaders.size() == 2 && file.permutations.size() == 3 && file.pipelines.size() == 2);
	CHECK(file.shaders[0].offset == offsets[0] && file.shaders[1].offset == offsets[1]);
	CHECK(file.levels.size() == 1 && file.levels[0] == "level_a");
	CHECK(file.pipelines[0].stages.size() == 2 && !file.pipelines[0].stages[0].plain &&
	      file.pipelines[0].stages[1].plain && file.pipelines[0].stages[1].bindings == 0x52);
	CHECK(file.pipelines[0].stages[0].permutation == 0 && file.pipelines[1].stages[0].permutation == 1);
	// No translator output: the module hashes are cleared.
	const auto modules = GraphicsPipelineSnapshot::SerializedModules(
	    std::span(file.pipelines[0].words).first(file.pipelines[0].snapshot_words));
	CHECK(modules.size() == 2 && modules[0] == 0 && modules[1] == 0);

	// Round trip.
	const auto         bytes = PipelineList::Encode(file);
	PipelineList::File decoded;
	std::string        error;
	CHECK(PipelineList::Decode(bytes, decoded, &error));
	CHECK(decoded.info.title_id == "PPSA00000" && decoded.info.app_version == "01.000.000" &&
	      decoded.info.input_info_sizes == InputSizes && decoded.info.created_unix == 1234);
	CHECK(decoded.levels == file.levels && decoded.shaders.size() == 2 && decoded.permutations.size() == 3);
	CHECK(decoded.shaders[1].static_state == file.shaders[1].static_state &&
	      decoded.shaders[1].input_info == file.shaders[1].input_info &&
	      decoded.shaders[1].code_hash_high == file.shaders[1].code_hash_high &&
	      decoded.shaders[1].kind == ShaderJournal::Kind::Pixel);
	CHECK(decoded.permutations[2].specialization == std::vector<uint8_t>({9}));
	CHECK(decoded.pipelines.size() == 2 && decoded.pipelines[1].words == file.pipelines[1].words &&
	      decoded.pipelines[1].levels == file.pipelines[1].levels && decoded.pipelines[1].key == file.pipelines[1].key);

	// Damage anywhere is refused: a flipped byte in each section's data, a truncated file.
	for (size_t at: {size_t {40}, bytes.size() / 2, bytes.size() - 3}) {
		auto damaged = bytes;
		damaged[at] ^= 0x40;
		PipelineList::File out;
		CHECK(!PipelineList::Decode(damaged, out));
	}
	{
		PipelineList::File out;
		CHECK(!PipelineList::Decode(std::span(bytes).first(bytes.size() - 1), out));
		CHECK(!PipelineList::Decode(std::span(bytes).first(20), out));
	}
	// An unknown section is skipped.
	{
		auto       extended = bytes;
		uint32_t   count    = 0;
		std::memcpy(&count, extended.data() + 12, 4);
		count++;
		std::memcpy(extended.data() + 12, &count, 4);
		const std::vector<uint8_t> raw = {1, 2, 3};
		const auto                 put = [&](auto value) {
            const auto* p = reinterpret_cast<const uint8_t*>(&value);
            extended.insert(extended.end(), p, p + sizeof(value));
		};
		put(uint32_t {99});
		put(uint32_t {0});
		put(uint64_t {raw.size()});
		put(uint64_t {raw.size()});
		put(XXH3_64bits(raw.data(), raw.size()));
		extended.insert(extended.end(), raw.begin(), raw.end());
		PipelineList::File out;
		CHECK(PipelineList::Decode(extended, out) && out.pipelines.size() == 2);
	}

	// No code bytes in the file: no 64-byte window of either shader's code in it or its sections.
	PipelineList::CodeWindows windows;
	for (const auto& source: recording.sources) {
		windows.Add({reinterpret_cast<const uint8_t*>(source.code.data()), source.code.size() * 4});
	}
	CHECK(windows.Count() > 900);
	CHECK(windows.FindIn(bytes).empty());
	std::vector<std::pair<uint32_t, std::vector<uint8_t>>> sections;
	CHECK(PipelineList::DecodeSections(bytes, sections) && sections.size() == 5);
	for (const auto& [id, raw]: sections) CHECK(windows.FindIn(raw).empty());
	// ... while a list that carried code (here in an input info) is caught, at any alignment.
	auto leaky = file;
	leaky.shaders[0].input_info.assign(3, 0);
	const auto* code = reinterpret_cast<const uint8_t*>(recording.sources[1].code.data());
	leaky.shaders[0].input_info.insert(leaky.shaders[0].input_info.end(), code + 5, code + 5 + 70);
	std::vector<std::pair<uint32_t, std::vector<uint8_t>>> leaky_sections;
	CHECK(PipelineList::DecodeSections(PipelineList::Encode(leaky), leaky_sections));
	size_t hits = 0;
	for (const auto& [id, raw]: leaky_sections) hits += windows.FindIn(raw).size();
	CHECK(hits != 0);
}

void TestResolve() {
	auto                  recording = MakeRecording("level_a", vk::PrimitiveTopology::eTriangleList);
	std::vector<uint64_t> offsets;
	const auto            image = Image(recording.sources, offsets);
	PipelineList::Builder     builder(TestInfo());
	PipelineList::ExportStats export_stats;
	builder.AddJournals(recording.sources, recording.entries, recording.pipelines, LocatorFor(image), export_stats);
	const auto file = builder.Finish();

	// The same executable: every shader resolves with the recorded code.
	{
		PipelineList::ResolveStats stats;
		const auto resolved = PipelineList::Resolve(file, InputSizes, ReaderFor(image), {}, stats);
		CHECK(stats.resolved == 2 && stats.mismatched == 0 && stats.relocated == 0);
		CHECK(resolved.usable[0] && resolved.usable[1]);
		CHECK(resolved.sources[0].code == recording.sources[0].code && resolved.sources[1].hash == recording.sources[1].hash);
		CHECK(resolved.entries.size() == 3 && stats.pipelines_kept == 2);
		// The runtime identity of a resolved permutation is the recording's.
		CHECK(PermutationIdentity(resolved.sources[0], resolved.entries[1]) ==
		      PermutationIdentity(recording.sources[0], recording.entries[1]));
	}
	// Another game version (one changed code byte): that shader and everything using it is dropped.
	{
		auto changed = image;
		changed[offsets[1] + 17] ^= 1;
		PipelineList::ResolveStats stats;
		const auto resolved = PipelineList::Resolve(file, InputSizes, ReaderFor(changed),
		                                            [&](PipelineList::CodeFile) { return changed; }, stats);
		CHECK(stats.resolved == 1 && stats.mismatched == 1);
		CHECK(resolved.usable[0] && !resolved.usable[1] && resolved.sources[1].code.empty());
		CHECK(resolved.entries.size() == 2 && resolved.permutation_entry[2] == UINT32_MAX);
		CHECK(stats.pipelines_kept == 0);
	}
	// Another executable wrapper (the code 3 bytes further on): found by the hashes.
	{
		std::vector<uint64_t> moved_offsets;
		const auto            moved = Image(recording.sources, moved_offsets, 3);
		PipelineList::ResolveStats stats;
		const auto resolved = PipelineList::Resolve(file, InputSizes, ReaderFor(moved),
		                                            [&](PipelineList::CodeFile) { return moved; }, stats);
		CHECK(stats.resolved == 2 && stats.relocated == 2 && stats.mismatched == 0);
		CHECK(resolved.sources[1].code == recording.sources[1].code);
	}
	// Another emulator's input structs: those shaders are not used.
	{
		PipelineList::ResolveStats stats;
		const auto resolved = PipelineList::Resolve(file, {96, 72, 48}, ReaderFor(image), {}, stats);
		CHECK(stats.other_input_layout == 1 && resolved.usable[0] && !resolved.usable[1]);
	}
	// Unlocated code (not in the executable) does not enter the list.
	{
		auto extra = recording;
		extra.sources.push_back(SourceOf(Code(77, 64), ShaderJournal::Kind::Compute, 4, InputSizes[2]));
		extra.entries.push_back({2, 0, {}});
		PipelineList::Builder     b(TestInfo());
		PipelineList::ExportStats s;
		b.AddJournals(extra.sources, extra.entries, extra.pipelines, LocatorFor(image), s);
		CHECK(s.sources_unlocated == 1 && s.entries_dropped == 1 && b.Finish().shaders.size() == 2);
	}
}

// A merged stage (NGG mesh program: GS front half and back half): both halves are located,
// resolved and checked, and the shader journal keeps the back half.
void TestMergedStage() {
	auto source      = SourceOf(Code(11, 160), ShaderJournal::Kind::Vertex, 5, InputSizes[0]);
	source.back_code = Code(12, 96);
	source.hash      = MergedSourceHash(source.code, source.back_code);
	ShaderJournal::Source back_only;
	back_only.code = source.back_code;
	std::vector<uint64_t> offsets;
	const auto            image = Image({source, back_only}, offsets);
	const std::vector<ShaderJournal::Entry> entries = {{0, 0, {4}}};

	PipelineList::Builder     builder(TestInfo());
	PipelineList::ExportStats stats;
	builder.AddJournals({source}, entries, {}, LocatorFor(image), stats);
	const auto file = builder.Finish();
	CHECK(file.shaders.size() == 1 && file.shaders[0].offset == offsets[0] && file.shaders[0].back_offset == offsets[1] &&
	      file.shaders[0].back_code_size == 96);
	PipelineList::File decoded;
	CHECK(PipelineList::Decode(PipelineList::Encode(file), decoded) && decoded.shaders.size() == 1 &&
	      decoded.shaders[0].back_hash_high == file.shaders[0].back_hash_high);
	{
		PipelineList::ResolveStats rs;
		const auto resolved = PipelineList::Resolve(decoded, InputSizes, ReaderFor(image), {}, rs);
		CHECK(rs.resolved == 1 && resolved.sources[0].back_code == source.back_code && resolved.sources[0].code == source.code);
		CHECK(!resolved.entries.empty() &&
		      PermutationIdentity(resolved.sources[0], resolved.entries[0]) == PermutationIdentity(source, entries[0]));
	}
	{
		auto changed = image;
		changed[offsets[1] + 8] ^= 2; // the back half differs
		PipelineList::ResolveStats rs;
		const auto resolved = PipelineList::Resolve(decoded, InputSizes, ReaderFor(changed),
		                                            [&](PipelineList::CodeFile) { return changed; }, rs);
		CHECK(rs.resolved == 0 && rs.mismatched == 1 && !resolved.usable[0] && resolved.entries.empty());
	}
	// Without the back half in the executable the source is not exported.
	{
		std::vector<uint64_t>     front_offsets;
		const auto                front_only = Image({source}, front_offsets);
		PipelineList::Builder     b(TestInfo());
		PipelineList::ExportStats s;
		b.AddJournals({source}, entries, {}, LocatorFor(front_only), s);
		CHECK(s.sources_unlocated == 1 && b.Finish().shaders.empty());
	}
	// The shader journal file keeps the back half.
	const auto path = std::filesystem::temp_directory_path() / "pipeline-list-tests.shaders.journal";
	std::filesystem::remove(path);
	{
		ShaderJournal::Settings settings;
		settings.path              = path;
		settings.identity          = {1, 2, 3};
		settings.background_writer = false;
		ShaderJournal journal(std::move(settings));
		journal.Record(source, 0, entries[0].specialization);
		CHECK(journal.Flush());
	}
	{
		ShaderJournal::Settings settings;
		settings.path              = path;
		settings.identity          = {1, 2, 3};
		settings.background_writer = false;
		ShaderJournal journal(std::move(settings));
		CHECK(journal.Sources().size() == 1 && journal.Sources()[0].back_code == source.back_code &&
		      journal.Sources()[0].code == source.code && journal.Entries().size() == 1);
	}
	std::filesystem::remove(path);
}

void TestMerge() {
	// Two recording runs: the same shaders, one pipeline state shared, other levels.
	auto                  a = MakeRecording("level_a", vk::PrimitiveTopology::eTriangleList);
	auto                  b = MakeRecording("level_b", vk::PrimitiveTopology::eTriangleList);
	auto                  c = MakeRecording("level_c", vk::PrimitiveTopology::eTriangleStrip, 3);
	std::vector<uint64_t> offsets;
	auto                  all = a.sources;
	all.push_back(c.sources[1]);
	const auto image = Image(all, offsets);

	PipelineList::Builder     first(TestInfo());
	PipelineList::ExportStats stats;
	first.AddJournals(a.sources, a.entries, a.pipelines, LocatorFor(image), stats);
	first.AddJournals(b.sources, b.entries, b.pipelines, LocatorFor(image), stats);
	const auto ab = first.Finish();
	CHECK(ab.shaders.size() == 2 && ab.permutations.size() == 3 && ab.pipelines.size() == 2);
	CHECK(stats.pipelines_merged == 2 && ab.levels.size() == 2);
	CHECK(ab.pipelines[0].levels == std::vector<uint16_t>({0, 1}));

	PipelineList::Builder     second(TestInfo());
	PipelineList::ExportStats stats2;
	second.AddJournals(c.sources, c.entries, c.pipelines, LocatorFor(image), stats2);
	const auto list_c = second.Finish();

	// Lists merge into a union: shaders and permutations by identity, pipelines by state and stages.
	PipelineList::Builder     merged(TestInfo());
	PipelineList::ExportStats stats3;
	merged.AddList(list_c, stats3);
	merged.AddList(ab, stats3);
	merged.AddList(ab, stats3); // twice: nothing new
	const auto out = merged.Finish();
	CHECK(out.shaders.size() == 3);          // VS shared, two pixel shaders
	CHECK(out.permutations.size() == 4);     // two VS permutations, a PS permutation of each pixel shader
	CHECK(out.pipelines.size() == 4);        // strip pipelines (level_c) and list pipelines (level_a, level_b)
	CHECK(out.levels.size() == 3 && out.levels[0] == "level_c");
	size_t with_two_levels = 0;
	for (const auto& pipeline: out.pipelines) with_two_levels += pipeline.levels.size() == 2 ? 1 : 0;
	CHECK(with_two_levels == 2);
	PipelineList::File decoded;
	CHECK(PipelineList::Decode(PipelineList::Encode(out), decoded) && decoded.pipelines.size() == 4);
}

void TestReplayInput() {
	// List content joins the replay input after a journal's, without duplicates.
	ShaderJournal::Settings settings;
	settings.background_writer = false;
	ShaderJournal journal(std::move(settings));
	auto          recording = MakeRecording("level_a", vk::PrimitiveTopology::eTriangleList);
	auto          first     = journal.AppendReplayOnly(recording.sources, recording.entries);
	CHECK(first.sources == 2 && first.entries == 3);
	CHECK(journal.Sources().size() == 2 && journal.Entries().size() == 3);
	auto again = journal.AppendReplayOnly(recording.sources, recording.entries);
	CHECK(again.sources == 0 && again.entries == 0 && journal.Entries().size() == 3);
	// A source without code (unresolved) and its entries are left out.
	auto missing = recording.sources;
	missing[0].code.clear();
	std::vector<ShaderJournal::Entry> entries = {{0, 12, {}}, {1, 4, {5}}};
	const auto partial = journal.AppendReplayOnly(missing, entries);
	CHECK(partial.sources == 0 && partial.entries == 1 && journal.Entries().size() == 4 &&
	      journal.Entries()[3].source == 1);
}

void TestRewrite() {
	uint32_t snapshot_words = 0;
	auto     words          = PipelineWords(vk::PrimitiveTopology::ePointList, snapshot_words);
	CHECK(!words.empty());
	auto snapshot = std::span(words).first(snapshot_words);
	CHECK(PipelineList::RewriteSnapshotModules(snapshot, [](uint32_t stage) { return uint64_t {0xABC0} + stage; }));
	const auto modules = GraphicsPipelineSnapshot::SerializedModules(snapshot);
	CHECK(modules.size() == 2 && modules[0] == 0xABC0 && modules[1] == 0xABC1);
	// The words still deserialize, with the modules looked up by the new hashes.
	const auto copy = GraphicsPipelineSnapshot::Deserialize(
	    snapshot, [](uint64_t hash) { return FakeHandle<vk::ShaderModule>(hash); }, FakeHandle<vk::PipelineLayout>(1));
	CHECK(copy != nullptr && copy->Info().pInputAssemblyState->topology == vk::PrimitiveTopology::ePointList);
	std::vector<uint32_t> junk = {1, 2, 3};
	CHECK(!PipelineList::RewriteSnapshotModules(junk, [](uint32_t) { return uint64_t {0}; }));
	auto truncated = std::vector<uint32_t>(snapshot.begin(), snapshot.begin() + 12);
	CHECK(!PipelineList::RewriteSnapshotModules(truncated, [](uint32_t) { return uint64_t {0}; }));
}

} // namespace

int main() {
	TestExportAndRoundTrip();
	TestResolve();
	TestMergedStage();
	TestMerge();
	TestReplayInput();
	TestRewrite();
	if (g_failed != 0) {
		std::printf("PipelineListTests: %d check(s) failed\n", g_failed);
		return 1;
	}
	std::printf("PipelineListTests: all passed\n");
	return 0;
}
