#include "graphics/host_gpu/renderer/pipeline/programDiskCache.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string_view>
#include <thread>
#include <xxhash.h>

using Libs::Graphics::ProgramDiskCache;
using Bytes = std::vector<uint8_t>;

static void Require(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", message);
		std::exit(1);
	}
}

template <typename T>
static void Put(Bytes& bytes, T value) {
	const auto* start = reinterpret_cast<const uint8_t*>(&value);
	bytes.insert(bytes.end(), start, start + sizeof(value));
}

static void Sized(Bytes& bytes, std::span<const uint8_t> value) {
	Put<uint32_t>(bytes, static_cast<uint32_t>(value.size()));
	bytes.insert(bytes.end(), value.begin(), value.end());
}

template <typename T>
static void Overwrite(Bytes& bytes, size_t offset, T value) {
	Require(offset + sizeof(value) <= bytes.size(), "fixture overwrite bounds");
	std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

static ProgramDiskCache::SourceKey Key(uint32_t index) {
	const uint32_t code[] = {index, 0x12345678};
	ProgramDiskCache::SourceKey key;
	ProgramDiskCache::BuildSourceKey({.stage = 1, .hash = index, .code = code}, key);
	return key;
}

static Bytes Header(std::span<const uint8_t> identity) {
	Bytes bytes(std::begin(ProgramDiskCache::FileMagic), std::end(ProgramDiskCache::FileMagic));
	Put<uint32_t>(bytes, ProgramDiskCache::FormatVersion);
	Put<uint32_t>(bytes, 0);
	Put<uint64_t>(bytes, identity.size());
	bytes.insert(bytes.end(), identity.begin(), identity.end());
	Put<uint64_t>(bytes, XXH3_64bits(bytes.data(), bytes.size()));
	return bytes;
}

static Bytes Record(uint32_t kind, const Bytes& payload) {
	Bytes bytes;
	Put<uint32_t>(bytes, ProgramDiskCache::RecordMagic);
	Put<uint32_t>(bytes, kind);
	Put<uint64_t>(bytes, payload.size());
	Put<uint64_t>(bytes, XXH3_64bits(payload.data(), payload.size()));
	bytes.insert(bytes.end(), payload.begin(), payload.end());
	return bytes;
}

static Bytes Source(uint32_t index, uint8_t plan = 7, uint8_t skip = 0) {
	const auto key = Key(index);
	Bytes payload(key.digest.begin(), key.digest.end());
	Sized(payload, key.bytes);
	Put<uint8_t>(payload, skip);
	const uint8_t plan_bytes[] = {plan, 8, 9};
	Sized(payload, plan_bytes);
	return Record(ProgramDiskCache::RecordSource, payload);
}

static Bytes Permutation(uint32_t index, uint8_t info = 11) {
	const auto key = Key(index);
	Bytes payload(key.digest.begin(), key.digest.end());
	Put<uint32_t>(payload, 17);
	const uint8_t specialization[] = {1, 3, 5};
	const uint8_t info_bytes[] = {info, 12};
	Sized(payload, specialization);
	Sized(payload, info_bytes);
	Put<uint32_t>(payload, 2); // word count
	Put<uint32_t>(payload, 0x07230203);
	Put<uint32_t>(payload, index);
	Put<uint32_t>(payload, 1);
	Put<uint32_t>(payload, 0x01020304);
	return Record(ProgramDiskCache::RecordPermutation, payload);
}

static void Append(Bytes& bytes, const Bytes& record) {
	bytes.insert(bytes.end(), record.begin(), record.end());
}

static void Write(const std::filesystem::path& path, const Bytes& bytes) {
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	Require(out.good(), "write fixture");
}

// These fallbacks let the first run demonstrate the missing capability at runtime.
template <typename Settings>
static void LoadThreads(Settings& settings, uint32_t count) {
	if constexpr (requires { settings.load_threads; }) settings.load_threads = count;
}

template <typename Stats>
static uint32_t LoadWorkers(const Stats& stats) {
	if constexpr (requires { stats.load_workers; }) return stats.load_workers;
	return 1;
}

static void ValidAndDuplicates(ProgramDiskCache::Settings settings, uint32_t threads) {
	LoadThreads(settings, threads);
	auto bytes = Header(settings.identity);
	// Permutations may precede their source, and the first duplicate wins.
	for (uint32_t i = 0; i < 40; ++i) {
		Append(bytes, Permutation(i));
		Append(bytes, Source(i));
		Append(bytes, Source(i, 99, 1));
		Append(bytes, Permutation(i, 99));
	}
	Write(settings.path, bytes);
	ProgramDiskCache cache(settings);
	const auto stats = cache.GetStats();
	Require(stats.loaded_sources == 40 && stats.loaded_permutations == 40,
	        "duplicates do not change loaded counts");
	Require(stats.rejected_records == 0 && !stats.header_rejected, "valid file accepted");
	const auto expected_workers = std::clamp(threads, 1u, 64u);
	Require(LoadWorkers(stats) == expected_workers, "preload uses requested workers and clamps count");
	const uint8_t specialization[] = {1, 3, 5};
	for (uint32_t i = 0; i < 40; ++i) {
		const auto source = cache.FindSource(Key(i));
		const auto permutation = cache.FindPermutation(Key(i).digest, 17, specialization);
		Require(source && !source->skip_dispatch && source->plan[0] == 7,
		        "first source duplicate wins in file order");
		Require(permutation && permutation->info[0] == 11 && permutation->spirv.size() == 8 &&
		            permutation->spirv_plain.size() == 4, "first permutation duplicate wins");
		Require(source->id == 4 * i + 1 && permutation->id == 4 * i,
		        "record IDs retain file order including unindexed duplicates");
	}
	const auto held_source = *cache.FindSource(Key(0));
	const auto held_permutation = *cache.FindPermutation(Key(0).digest, 17, specialization);
	for (uint32_t i = 40; i < 240; ++i) cache.AddSource(Key(i), true, {});
	Require(cache.Flush(), "flush loaded and newly added records");
	Require(held_source.plan[0] == 7 && held_permutation.info[0] == 11,
	        "returned spans remain stable after additions and save");
	ProgramDiskCache reloaded(settings);
	Require(reloaded.GetStats().loaded_sources == 240 &&
	            reloaded.GetStats().loaded_permutations == 40, "save drops duplicate records");
}

enum class Damage { Checksum, Size, Magic, Kind, Digest, Skip, Truncated };

static void CorruptMiddle(ProgramDiskCache::Settings settings, uint32_t threads, Damage damage) {
	LoadThreads(settings, threads);
	auto bytes = Header(settings.identity);
	for (uint32_t i = 0; i < 10; ++i) Append(bytes, Source(i));
	auto broken = Source(10);
	switch (damage) {
	case Damage::Checksum: broken.back() ^= 1; break;
	case Damage::Size: Overwrite<uint64_t>(broken, 8, std::numeric_limits<uint64_t>::max()); break;
	case Damage::Magic: broken[0] ^= 1; break;
	case Damage::Kind: Overwrite<uint32_t>(broken, 4, 99); break;
	case Damage::Digest:
		broken[ProgramDiskCache::RecordHeaderBytes] ^= 1;
		Overwrite<uint64_t>(broken, 16, XXH3_64bits(broken.data() + ProgramDiskCache::RecordHeaderBytes,
		                                        broken.size() - ProgramDiskCache::RecordHeaderBytes));
		break;
	case Damage::Skip:
		broken = Source(10, 7, 2); // good checksum but invalid semantic encoding
		break;
	case Damage::Truncated: broken.resize(7); break;
	}
	Append(bytes, broken);
	for (uint32_t i = 11; i < 40; ++i) Append(bytes, Source(i));
	Write(settings.path, bytes);
	{
		ProgramDiskCache cache(settings);
		const auto stats = cache.GetStats();
		Require(stats.loaded_sources == 10 && stats.rejected_records == 1,
		        "first damaged record discards every later record");
		Require(cache.FindSource(Key(9)).has_value() && !cache.FindSource(Key(10)) &&
		            !cache.FindSource(Key(39)), "only intact prefix is visible");
		Require(cache.Flush(), "repair damaged tail");
	}
	ProgramDiskCache repaired(settings);
	Require(repaired.GetStats().loaded_sources == 10 && repaired.GetStats().rejected_records == 0,
	        "repaired file contains exactly the intact prefix");
}

static void HeaderAndSmallFiles(ProgramDiskCache::Settings settings) {
	LoadThreads(settings, 8);
	auto bytes = Header(settings.identity);
	Write(settings.path, bytes);
	{
		ProgramDiskCache empty(settings);
		Require(empty.GetStats().loaded_sources == 0 && LoadWorkers(empty.GetStats()) == 0,
		        "empty accepted file launches no validation workers");
	}
	Append(bytes, Source(0));
	Write(settings.path, bytes);
	{
		ProgramDiskCache one(settings);
		Require(one.GetStats().loaded_sources == 1 && LoadWorkers(one.GetStats()) == 1,
		        "single record validates on loader without helper threads");
	}
	for (int variant = 0; variant < 5; ++variant) {
		auto invalid = bytes;
		auto other_settings = settings;
		if (variant == 0) other_settings.identity.push_back(99);
		if (variant == 1) invalid[0] ^= 1;
		if (variant == 2) invalid[8] ^= 1;
		if (variant == 3) invalid[12] = 1;
		if (variant == 4) invalid.resize(12);
		Write(settings.path, invalid);
		ProgramDiskCache rejected(other_settings);
		Require(rejected.GetStats().header_rejected && rejected.GetStats().loaded_sources == 0 &&
		            LoadWorkers(rejected.GetStats()) == 0, "header mismatch rejects before workers");
	}
}

static void ConcurrentLookups(ProgramDiskCache::Settings settings) {
	LoadThreads(settings, 8);
	auto bytes = Header(settings.identity);
	for (uint32_t i = 0; i < 400; ++i) Append(bytes, Source(i));
	Write(settings.path, bytes);
	ProgramDiskCache cache(settings);
	std::vector<std::thread> readers;
	for (uint32_t thread = 0; thread < 8; ++thread) {
		readers.emplace_back([&] {
			for (uint32_t i = 0; i < 400; ++i)
				Require(cache.FindSource(Key(i)).has_value(), "lookups wait for complete publication");
		});
	}
	for (auto& reader: readers) reader.join();
}

static void BatchBoundaryAndBadTail(ProgramDiskCache::Settings settings) {
	LoadThreads(settings, 8);
	auto bytes = Header(settings.identity);
	constexpr uint32_t prefix = 8200;
	for (uint32_t i = 0; i < prefix; ++i) Append(bytes, Source(i));
	auto bad = Source(prefix);
	bad.back() ^= 1;
	Append(bytes, bad);
	// These 24-byte headers must not cause an unbounded vector<Record> allocation or be used
	// after the first failed checksum. The loader needs only one bounded batch of scratch slots.
	const Bytes tiny(ProgramDiskCache::RecordHeaderBytes, 0);
	for (uint32_t i = 0; i < 100'000; ++i) Append(bytes, tiny);
	Append(bytes, Source(prefix + 1));
	Write(settings.path, bytes);
	{
		ProgramDiskCache cache(settings);
		const auto stats = cache.GetStats();
		Require(stats.loaded_sources == prefix && stats.rejected_records == 1,
		        "multiple batches retain intact prefix before malformed tiny tail");
		for (const uint32_t i: {0u, 4095u, 4096u, 8191u, 8192u, prefix - 1}) {
			const auto found = cache.FindSource(Key(i));
			Require(found && found->id == i, "IDs and spans survive batch boundaries");
		}
		Require(!cache.FindSource(Key(prefix + 1)), "good record after bad tail is absent");
	}
	bytes = Header(settings.identity);
	Append(bytes, bad);
	Append(bytes, Source(0));
	Write(settings.path, bytes);
	ProgramDiskCache first_bad(settings);
	Require(first_bad.GetStats().loaded_sources == 0 && first_bad.GetStats().rejected_records == 1,
	        "bad first record rejects all later records");
}

// The benchmark copies its input before opening it: even a damaged real file is never rewritten.
static void Benchmark(const std::filesystem::path& input, ProgramDiskCache::Settings settings) {
	std::ifstream in(input, std::ios::binary);
	Bytes bytes((std::istreambuf_iterator<char>(in)), {});
	Require(bytes.size() >= 32, "benchmark input has a complete fixed header");
	uint64_t identity_size = 0;
	std::memcpy(&identity_size, bytes.data() + 16, sizeof(identity_size));
	Require(identity_size <= bytes.size() - 32, "benchmark identity bounds");
	settings.identity.assign(bytes.begin() + 24, bytes.begin() + 24 + identity_size);
	std::puts("threads,run,file_bytes,sources,permutations,rejected,workers,load_ms,read_ms,validate_ms,index_ms");
	constexpr uint32_t counts[] = {1u, 2u, 4u, 8u};
	// Rotate the order to spread page-cache, allocator and clock variation across
	// all thread counts rather than assigning every early sample to one setting.
	for (uint32_t run = 0; run < 5; ++run) {
		for (uint32_t order = 0; order < 4; ++order) {
			const auto count = counts[(run + order) % 4];
			LoadThreads(settings, count);
			Write(settings.path, bytes);
			ProgramDiskCache cache(settings);
			const auto s = cache.GetStats();
			Require(!s.header_rejected, "benchmark header accepted for extracted identity");
			std::printf("%u,%u,%llu,%llu,%llu,%llu,%u,%.3f", count, run,
			            static_cast<unsigned long long>(s.file_bytes),
			            static_cast<unsigned long long>(s.loaded_sources),
			            static_cast<unsigned long long>(s.loaded_permutations),
			            static_cast<unsigned long long>(s.rejected_records), LoadWorkers(s), s.load_ns / 1.0e6);
			// Optional metrics are absent in the initial RED run.
			const auto timings = []<typename Stats>(const Stats& stats) {
				if constexpr (requires { stats.read_ns; stats.validate_ns; stats.index_ns; }) {
					std::printf(",%.3f,%.3f,%.3f", stats.read_ns / 1.0e6,
					            stats.validate_ns / 1.0e6, stats.index_ns / 1.0e6);
				}
			};
			timings(s);
			std::putchar('\n');
		}
	}
}

int main(int argc, char** argv) {
	const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir = std::filesystem::temp_directory_path() /
	                 ("kyty_cache_preload_" + std::to_string(stamp));
	std::filesystem::create_directory(dir);
	ProgramDiskCache::Settings settings;
	settings.path = dir / "programs.bin";
	settings.identity = {1, 2, 3};
	settings.periodic_saves = false;
	settings.quiet = true;
	if (argc == 3 && std::string_view(argv[1]) == "--benchmark") {
		Benchmark(argv[2], settings);
	} else {
		for (const uint32_t count: {0u, 1u, 2u, 8u, 1000u}) {
			ValidAndDuplicates(settings, count);
			for (const auto damage: {Damage::Checksum, Damage::Size, Damage::Magic, Damage::Kind,
			                         Damage::Digest, Damage::Skip, Damage::Truncated})
				CorruptMiddle(settings, count, damage);
		}
		HeaderAndSmallFiles(settings);
		ConcurrentLookups(settings);
		BatchBoundaryAndBadTail(settings);
		std::puts("Program cache preload tests passed");
	}
	std::filesystem::remove(settings.path);
	std::filesystem::remove(dir);
}
