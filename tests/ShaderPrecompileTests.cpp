// Shader precompile journal and replay driver (graphics/host_gpu/renderer/pipeline/shaderPrecompile.h):
// the journal file round-trips, deduplicates, survives a damaged tail, is replaced for another
// identity and respects its size cap; the replay driver visits every entry once, in order on one
// thread, reports progress and stops.
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

void Expect(bool condition, const char* what) {
	if (!condition) {
		std::printf("ShaderPrecompileTests: failed: %s\n", what);
		g_failures++;
	}
}

std::filesystem::path TestPath(const char* name) {
	return std::filesystem::current_path() / (std::string("shader-precompile-test-") + name + ".bin");
}

std::vector<uint8_t> Identity(const char* text) {
	return std::vector<uint8_t>(text, text + std::char_traits<char>::length(text));
}

ShaderJournal::Settings MakeSettings(const std::filesystem::path& path, const char* identity = "device A") {
	ShaderJournal::Settings settings;
	settings.path              = path;
	settings.identity          = Identity(identity);
	settings.background_writer = false;
	return settings;
}

ShaderJournal::Source MakeSource(uint64_t hash, uint32_t words, ShaderJournal::Kind kind = ShaderJournal::Kind::Pixel) {
	ShaderJournal::Source source;
	source.stage           = kind == ShaderJournal::Kind::Compute ? 4u : kind == ShaderJournal::Kind::Pixel ? 2u : 1u;
	source.kind            = kind;
	source.hash            = hash;
	source.user_data_count = 12;
	source.code_size       = words;
	source.wave_size       = 64;
	source.user_data_base  = 0;
	source.static_state    = {1, 2, static_cast<uint32_t>(hash)};
	source.code.resize(words);
	for (uint32_t i = 0; i < words; i++) source.code[i] = static_cast<uint32_t>(hash) + i;
	source.input_info.assign(40, static_cast<uint8_t>(hash));
	return source;
}

std::vector<uint8_t> Spec(uint8_t value, size_t size = 9) {
	return std::vector<uint8_t>(size, value);
}

void Remove(const std::filesystem::path& path) {
	std::error_code error;
	std::filesystem::remove(path, error);
}

std::vector<uint8_t> ReadAll(const std::filesystem::path& path) {
	std::ifstream in(path, std::ios::binary);
	return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void WriteAll(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void TestRoundTripAndDedup() {
	const auto path = TestPath("roundtrip");
	Remove(path);
	{
		ShaderJournal journal(MakeSettings(path));
		Expect(journal.Sources().empty() && journal.Entries().empty(), "a missing file loads empty");
		Expect(!journal.GetStats().file_found, "a missing file is not found");
		auto a = MakeSource(0x11, 8);
		Expect(!journal.HasSource(a), "an unknown source is not known");
		journal.Record(a, 0, Spec(1));
		Expect(journal.HasSource(a), "a recorded source is known");
		journal.Record(a, 0, Spec(1)); // duplicate permutation
		journal.Record(a, 0, Spec(2)); // other specialization
		journal.Record(a, 3, Spec(1)); // other cursor
		auto identity_only = MakeSource(0x11, 8);
		identity_only.code.clear(); // a known source needs no code
		identity_only.input_info.clear();
		journal.Record(identity_only, 5, Spec(3));
		journal.Record(MakeSource(0x22, 4, ShaderJournal::Kind::Vertex), 0, Spec(4));
		auto no_code = MakeSource(0x33, 4);
		no_code.code.clear();
		journal.Record(no_code, 0, Spec(5)); // a new source without code is dropped
		Expect(!journal.HasSource(no_code), "a source without code is not recorded");
		auto bad_size = MakeSource(0x44, 4);
		bad_size.code_size = 5;
		journal.Record(bad_size, 0, Spec(5));
		Expect(!journal.HasSource(bad_size), "a source whose code size disagrees is not recorded");
		Expect(journal.Flush(), "flush succeeds");
		const auto stats = journal.GetStats();
		Expect(stats.recorded_sources == 2 && stats.recorded_entries == 5, "recorded counts");
	}
	{
		ShaderJournal journal(MakeSettings(path));
		Expect(journal.GetStats().file_found && !journal.GetStats().header_rejected, "file loads");
		Expect(journal.Sources().size() == 2 && journal.Entries().size() == 5, "loaded counts");
		const auto& s = journal.Sources()[0];
		const auto expected = MakeSource(0x11, 8);
		Expect(s.hash == 0x11 && s.stage == expected.stage && s.kind == expected.kind &&
		           s.code == expected.code && s.input_info == expected.input_info &&
		           s.static_state == expected.static_state && s.code_size == 8 &&
		           s.user_data_count == 12 && s.wave_size == 64,
		       "source fields round-trip");
		Expect(journal.Sources()[1].kind == ShaderJournal::Kind::Vertex, "kind round-trips");
		const auto& e = journal.Entries();
		Expect(e[0].source == 0 && e[0].push_data_cursor == 0 && e[0].specialization == Spec(1), "entry 0");
		Expect(e[1].specialization == Spec(2) && e[2].push_data_cursor == 3 && e[3].push_data_cursor == 5 &&
		           e[4].source == 1,
		       "entries keep their order");
		// Loaded content is known: nothing is journaled twice across runs.
		Expect(journal.HasSource(expected), "a loaded source is known");
		journal.Record(expected, 0, Spec(1));
		journal.Record(expected, 0, Spec(7)); // new
		journal.Flush();
		Expect(journal.GetStats().recorded_entries == 1 && journal.GetStats().recorded_sources == 0,
		       "only the new permutation is recorded");
	}
	{
		ShaderJournal journal(MakeSettings(path));
		Expect(journal.Entries().size() == 6 && journal.Entries()[5].specialization == Spec(7),
		       "a later run appends");
	}
	Remove(path);
}

void TestDamagedFile() {
	const auto path = TestPath("damaged");
	Remove(path);
	{
		ShaderJournal journal(MakeSettings(path));
		for (uint32_t i = 0; i < 4; i++) {
			journal.Record(MakeSource(0x100 + i, 6), 0, Spec(static_cast<uint8_t>(i)));
		}
		journal.Flush();
	}
	const auto intact = ReadAll(path);
	// Truncated tail: the intact prefix loads, and a later record continues after it.
	for (const size_t cut: {size_t {1}, size_t {30}, size_t {80}}) {
		auto bytes = intact;
		bytes.resize(bytes.size() - cut);
		WriteAll(path, bytes);
		{
			ShaderJournal journal(MakeSettings(path));
			const auto stats = journal.GetStats();
			Expect(stats.damaged_bytes != 0, "a truncated tail is reported");
			Expect(journal.Entries().size() < 4 && journal.Entries().size() >= 2, "the intact prefix loads");
			for (const auto& entry: journal.Entries()) {
				Expect(entry.source < journal.Sources().size(), "entries refer to loaded sources");
			}
			journal.Record(MakeSource(0x900, 3), 1, Spec(9));
			journal.Flush();
		}
		ShaderJournal again(MakeSettings(path));
		Expect(again.GetStats().damaged_bytes == 0, "the damaged tail was cut off by the next write");
		Expect(!again.Entries().empty() && again.Entries().back().specialization == Spec(9),
		       "the record after the truncation is kept");
	}
	// A flipped byte in a payload: records before it load, it and everything after do not.
	{
		auto bytes = intact;
		bytes[bytes.size() / 2] ^= 0xff;
		WriteAll(path, bytes);
		ShaderJournal journal(MakeSettings(path));
		Expect(journal.GetStats().damaged_bytes != 0 && journal.Entries().size() < 4, "corruption is detected");
	}
	// Garbage and an empty file load as nothing, without a crash.
	WriteAll(path, std::vector<uint8_t>(100, 0xab));
	{
		ShaderJournal journal(MakeSettings(path));
		Expect(journal.Entries().empty() && journal.GetStats().header_rejected, "garbage is rejected");
	}
	WriteAll(path, {});
	{
		ShaderJournal journal(MakeSettings(path));
		Expect(journal.Entries().empty(), "an empty file loads empty");
	}
	Remove(path);
}

void TestIdentity() {
	const auto path = TestPath("identity");
	Remove(path);
	{
		ShaderJournal journal(MakeSettings(path, "device A"));
		journal.Record(MakeSource(0x1, 4), 0, Spec(1));
		journal.Flush();
	}
	{
		ShaderJournal journal(MakeSettings(path, "device B"));
		Expect(journal.Entries().empty() && journal.GetStats().header_rejected, "another identity is rejected");
		Expect(!journal.HasSource(MakeSource(0x1, 4)), "a rejected file's sources are not known");
		journal.Record(MakeSource(0x2, 4), 0, Spec(2));
		journal.Flush();
	}
	{
		ShaderJournal journal(MakeSettings(path, "device B"));
		Expect(journal.Entries().size() == 1 && journal.Sources()[0].hash == 0x2,
		       "the file was replaced by the new identity's records");
	}
	{
		ShaderJournal journal(MakeSettings(path, "device A"));
		Expect(journal.Entries().empty(), "the old identity's records are gone");
	}
	Remove(path);
}

void TestCapacity() {
	const auto path = TestPath("capacity");
	Remove(path);
	auto settings = MakeSettings(path);
	settings.max_file_bytes = 600;
	{
		ShaderJournal journal(settings);
		for (uint32_t i = 0; i < 20; i++) journal.Record(MakeSource(0x500 + i, 16), 0, Spec(1));
		journal.Flush();
		Expect(journal.GetStats().over_capacity, "the cap is reached");
		Expect(std::filesystem::file_size(path) <= 600 + 200, "the file stays near the cap");
	}
	{
		ShaderJournal journal(settings);
		Expect(!journal.Entries().empty() && journal.Entries().size() < 20, "a capped file loads what fit");
	}
	Remove(path);
}

void TestBackgroundWriter() {
	const auto path = TestPath("writer");
	Remove(path);
	auto settings = MakeSettings(path);
	settings.background_writer = true;
	settings.flush_interval_ns = 20'000'000ull;
	{
		ShaderJournal journal(settings);
		journal.Record(MakeSource(0x7, 4), 0, Spec(1));
		for (int i = 0; i < 200 && !std::filesystem::exists(path); i++) {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		Expect(std::filesystem::exists(path), "the writer thread wrote the file");
		journal.Record(MakeSource(0x8, 4), 0, Spec(2)); // written by the destructor at the latest
	}
	ShaderJournal journal(MakeSettings(path));
	Expect(journal.Entries().size() == 2, "records written by the thread and the destructor");
	Remove(path);
}

ShaderJournal MakeJournalWith(const std::filesystem::path& path, uint32_t entries) {
	Remove(path);
	{
		ShaderJournal journal(MakeSettings(path));
		for (uint32_t i = 0; i < entries; i++) {
			journal.Record(MakeSource(0x1000 + i % 7, 4 + i % 7), i, Spec(static_cast<uint8_t>(i)));
		}
		journal.Flush();
	}
	return ShaderJournal(MakeSettings(path));
}

void TestReplayDriver() {
	const auto path = TestPath("replay");
	auto       journal = MakeJournalWith(path, 100);
	Expect(journal.Entries().size() == 100, "journal for replay");

	// One thread: every entry once, in file order.
	{
		std::vector<uint32_t> order;
		std::vector<std::string> lines;
		ShaderPrecompiler::Settings settings;
		settings.threads = 1;
		settings.log     = [&](const std::string& line) { lines.push_back(line); };
		settings.progress_percent = 25;
		std::atomic<int> finished {0};
		settings.on_finished = [&] { finished++; };
		ShaderPrecompiler precompiler(journal, settings, [&](const auto& source, const auto& entry) {
			order.push_back(entry.push_data_cursor);
			Expect(source.code.size() == source.code_size, "replay gets the entry's source");
			return entry.push_data_cursor % 4 == 0   ? ShaderPrecompiler::Outcome::Compiled
			       : entry.push_data_cursor % 4 == 1 ? ShaderPrecompiler::Outcome::Present
			       : entry.push_data_cursor % 4 == 2 ? ShaderPrecompiler::Outcome::Skipped
			                                         : ShaderPrecompiler::Outcome::Failed;
		});
		precompiler.Wait();
		const auto progress = precompiler.GetProgress();
		Expect(progress.finished && progress.total == 100 && progress.done == 100, "progress totals");
		Expect(progress.compiled == 25 && progress.present == 25 && progress.skipped == 25 &&
		           progress.failed == 25,
		       "outcome counts");
		bool in_order = order.size() == 100;
		for (uint32_t i = 0; i < order.size(); i++) in_order = in_order && order[i] == i;
		Expect(in_order, "one thread replays in file order");
		Expect(finished == 1, "on_finished runs once");
		Expect(lines.size() >= 3 && lines.back().find("done") != std::string::npos &&
		           lines.back().find("100/100") != std::string::npos,
		       "progress lines and a final line");
		Expect(lines.size() >= 2 && lines[1].find("%)") != std::string::npos, "a progress line has a percentage");
	}
	// Several threads: every entry exactly once.
	{
		std::mutex mutex;
		std::multiset<uint32_t> seen;
		ShaderPrecompiler::Settings settings;
		settings.threads = 4;
		ShaderPrecompiler precompiler(journal, settings, [&](const auto&, const auto& entry) {
			std::scoped_lock lock(mutex);
			seen.insert(entry.push_data_cursor);
			return ShaderPrecompiler::Outcome::Compiled;
		});
		precompiler.Wait();
		bool each_once = seen.size() == 100;
		for (uint32_t i = 0; i < 100; i++) each_once = each_once && seen.count(i) == 1;
		Expect(each_once, "four threads replay every entry exactly once");
	}
	// max_entries, a throwing callback.
	{
		std::atomic<int> calls {0};
		ShaderPrecompiler::Settings settings;
		settings.threads     = 2;
		settings.max_entries = 10;
		ShaderPrecompiler precompiler(journal, settings, [&](const auto&, const auto&) -> ShaderPrecompiler::Outcome {
			calls++;
			throw std::runtime_error("compile failed");
		});
		precompiler.Wait();
		Expect(calls == 10 && precompiler.GetProgress().failed == 10 && precompiler.GetProgress().total == 10,
		       "max_entries limits the replay; an exception counts as a failure");
	}
	// Stop: entries not started are dropped, the destructor returns.
	{
		std::atomic<int> calls {0};
		std::atomic<bool> release {false};
		std::vector<std::string> lines;
		ShaderPrecompiler::Settings settings;
		settings.threads = 1;
		settings.log     = [&](const std::string& line) { lines.push_back(line); };
		ShaderPrecompiler precompiler(journal, settings, [&](const auto&, const auto&) {
			calls++;
			while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
			return ShaderPrecompiler::Outcome::Compiled;
		});
		while (calls == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
		precompiler.Stop();
		release = true;
		precompiler.Wait();
		Expect(calls == 1 && precompiler.GetProgress().done == 1, "Stop drops entries not started");
		Expect(!lines.empty() && lines.back().find("stopped") != std::string::npos, "a stopped replay says so");
	}
	// An empty journal finishes at once.
	{
		const auto empty_path = TestPath("replay-empty");
		Remove(empty_path);
		ShaderJournal empty(MakeSettings(empty_path));
		ShaderPrecompiler precompiler(empty, {}, [](const auto&, const auto&) {
			return ShaderPrecompiler::Outcome::Compiled;
		});
		precompiler.Wait();
		Expect(precompiler.GetProgress().finished && precompiler.GetProgress().total == 0, "nothing to replay");
	}
	Remove(path);
}

PipelineJournal::Settings MakePipelineSettings(const std::filesystem::path& path,
                                               const char* identity = "device A", bool sealed = false) {
	PipelineJournal::Settings settings;
	settings.path              = path;
	settings.identity          = Identity(identity);
	settings.background_writer = false;
	settings.sealed_writes     = sealed;
	return settings;
}

// Pipeline journal: keyed records round-trip in first-recorded order, a key is kept once (also
// across runs), a damaged tail keeps the intact prefix, another identity starts over, the cap holds.
void TestPipelineJournal() {
	const auto path = TestPath("pipelines");
	Remove(path);
	{
		PipelineJournal journal(MakePipelineSettings(path));
		Expect(journal.Loaded().empty() && !journal.GetStats().file_found, "pipeline journal: missing file loads empty");
		Expect(journal.Add(0x10, Spec(1, 40)), "pipeline journal: a new key is recorded");
		Expect(!journal.Add(0x10, Spec(2, 40)), "pipeline journal: a known key is not recorded again");
		Expect(journal.Add(0x20), "pipeline journal: an empty payload is fine");
		Expect(journal.Add(0x30, Spec(3, 7)), "pipeline journal: third key");
		Expect(journal.Contains(0x20) && !journal.Contains(0x40), "pipeline journal: contains");
		Expect(journal.Flush(), "pipeline journal: flush");
	}
	{
		PipelineJournal journal(MakePipelineSettings(path));
		const auto& loaded = journal.Loaded();
		Expect(loaded.size() == 3 && loaded[0].key == 0x10 && loaded[0].payload == Spec(1, 40) &&
		           loaded[1].key == 0x20 && loaded[1].payload.empty() && loaded[2].payload == Spec(3, 7),
		       "pipeline journal: records round-trip in order");
		Expect(!journal.Add(0x10, Spec(9, 3)), "pipeline journal: loaded keys are known");
		Expect(journal.Add(0x40, Spec(4, 5)), "pipeline journal: a later run appends");
		journal.ReleaseLoaded();
		Expect(journal.Loaded().empty() && journal.Contains(0x10), "pipeline journal: released content stays known");
	}
	// Damage: a flipped payload byte in the last record drops it and keeps the rest.
	auto bytes = ReadAll(path);
	bytes[bytes.size() - 2] ^= 0xffu;
	WriteAll(path, bytes);
	{
		PipelineJournal journal(MakePipelineSettings(path));
		Expect(journal.Loaded().size() == 3 && journal.GetStats().damaged_bytes != 0,
		       "pipeline journal: damaged tail ignored");
		Expect(journal.Add(0x50, Spec(5, 2)) && journal.Flush(), "pipeline journal: append after damage");
	}
	{
		PipelineJournal journal(MakePipelineSettings(path));
		Expect(journal.Loaded().size() == 4 && journal.Loaded()[3].key == 0x50,
		       "pipeline journal: the damaged tail was cut before appending");
	}
	{
		PipelineJournal journal(MakePipelineSettings(path, "device B"));
		Expect(journal.Loaded().empty() && journal.GetStats().header_rejected,
		       "pipeline journal: another identity is rejected");
		journal.Add(0x60);
		journal.Flush();
	}
	{
		PipelineJournal journal(MakePipelineSettings(path, "device B"));
		Expect(journal.Loaded().size() == 1 && journal.Loaded()[0].key == 0x60,
		       "pipeline journal: the replaced file holds only the new identity's records");
	}
	// Capacity.
	Remove(path);
	{
		auto settings           = MakePipelineSettings(path);
		settings.max_file_bytes = 400;
		PipelineJournal journal(std::move(settings));
		uint64_t accepted = 0;
		for (uint64_t key = 1; key <= 20; key++) accepted += journal.Add(key, Spec(1, 32)) ? 1 : 0;
		Expect(accepted > 0 && accepted < 20 && journal.GetStats().over_capacity, "pipeline journal: size cap");
		Expect(!journal.Contains(20), "pipeline journal: a refused key is not known");
		journal.Flush();
	}
	Expect(std::filesystem::file_size(path) <= 400, "pipeline journal: the file stays under the cap");
	Remove(path);
}

// Known-pipeline list mode: Flush writes only what was recorded before Seal (the keys the saved
// driver cache can hold); Reset empties the file at once.
void TestPipelineJournalSealed() {
	const auto path = TestPath("known");
	Remove(path);
	{
		PipelineJournal journal(MakePipelineSettings(path, "driver 1", true));
		journal.Add(1);
		journal.Add(2);
		journal.Seal();
		journal.Add(3); // created while the driver cache was serialized: not in the saved file
		Expect(journal.Flush(), "known list: flush");
		journal.Seal();
	} // no write at destruction in sealed mode
	{
		PipelineJournal journal(MakePipelineSettings(path, "driver 1", true));
		Expect(journal.Loaded().size() == 2 && journal.Contains(1) && journal.Contains(2) && !journal.Contains(3),
		       "known list: only sealed keys were written");
		journal.Add(3);
		journal.Seal();
		journal.Flush();
	}
	{
		PipelineJournal journal(MakePipelineSettings(path, "driver 1", true));
		Expect(journal.Loaded().size() == 3, "known list: a later save appends");
		journal.Reset();
		Expect(!journal.Contains(1) && journal.KnownCount() == 0, "known list: reset forgets every key");
	}
	{
		PipelineJournal journal(MakePipelineSettings(path, "driver 1", true));
		Expect(journal.Loaded().empty() && !journal.GetStats().header_rejected,
		       "known list: reset leaves an empty file at once");
	}
	{
		PipelineJournal journal(MakePipelineSettings(path, "driver 2", true));
		Expect(journal.GetStats().header_rejected, "known list: another driver is rejected");
	}
	Remove(path);
	// In memory only (an empty path): keys are kept, nothing is read or written.
	{
		PipelineJournal journal(MakePipelineSettings({}, "driver 1", true));
		Expect(journal.Add(7) && journal.Contains(7) && !journal.Add(7), "memory list: keys are kept");
		journal.Seal();
		Expect(journal.Flush() && journal.GetStats().written_bytes == 0, "memory list: nothing is written");
		journal.Reset();
		Expect(!journal.Contains(7), "memory list: reset");
	}
}

} // namespace

int main() {
	TestRoundTripAndDedup();
	TestDamagedFile();
	TestIdentity();
	TestCapacity();
	TestBackgroundWriter();
	TestReplayDriver();
	TestPipelineJournal();
	TestPipelineJournalSealed();
	if (g_failures != 0) {
		std::printf("ShaderPrecompileTests: %d checks failed\n", g_failures);
		return 1;
	}
	std::printf("ShaderPrecompileTests: all checks passed\n");
	return 0;
}
