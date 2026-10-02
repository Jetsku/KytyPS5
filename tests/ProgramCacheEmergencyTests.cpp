#include "common/emergencySave.h"
#include "graphics/host_gpu/renderer/pipeline/programDiskCache.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>

using Libs::Graphics::ProgramDiskCache;
using namespace std::chrono_literals;

static void Require(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", message);
		std::exit(1);
	}
}

static ProgramDiskCache::SourceKey Key(uint32_t index) {
	const uint32_t code[] = {index, 0x12345678};
	ProgramDiskCache::SourceKey key;
	ProgramDiskCache::BuildSourceKey({.stage = 1, .hash = index, .code = code}, key);
	return key;
}

int main() {
	const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir = std::filesystem::temp_directory_path() /
	                 ("kyty_cache_emergency_" + std::to_string(stamp));
	std::filesystem::create_directory(dir);
	ProgramDiskCache::Settings settings;
	settings.path = dir / "programs.bin";
	settings.identity = {1, 2, 3};
	settings.periodic_saves = false;
	settings.quiet = true;
	constexpr uint32_t records = 128;
	{
		ProgramDiskCache cache(settings);
		cache.AddSource(Key(0), true, {});
		std::atomic<bool> first_written {false};
		Common::EmergencySave first([&] { first_written = cache.Flush(); });
		Require(first.Request(2s) && first_written, "emergency persists first dirty record");
		{
			// No normal Flush/destructor has run: only the emergency worker could write this.
			ProgramDiskCache loaded(settings);
			Require(loaded.FindSource(Key(0)).has_value(), "emergency-only file reloads");
		}
		std::atomic<bool> begin {false};
		std::thread producer([&] {
			while (!begin.load()) std::this_thread::yield();
			for (uint32_t i = 1; i < records; ++i) cache.AddSource(Key(i), true, {});
		});
		std::thread saver([&] {
			while (!begin.load()) std::this_thread::yield();
			for (int i = 0; i < 6; ++i) Require(cache.Flush(), "concurrent normal flush");
		});
		std::atomic<bool> written {false};
		Common::EmergencySave emergency([&] { written = cache.Flush(); });
		begin = true;
		Require(emergency.Request(2s), "emergency completes beside active producer/saver");
		Require(written, "emergency disk write succeeded");
		producer.join();
		saver.join();
		Require(cache.Flush(), "persist records added after emergency snapshot");
		// Reopen before the original cache is destroyed: no destructor save can mask failure.
		{
			ProgramDiskCache loaded(settings);
			for (uint32_t i = 0; i < records; ++i)
				Require(loaded.FindSource(Key(i)).has_value(), "all records survive concurrent saves");
		}
		// Simulate a failed new save. An existing complete cache must remain readable.
		auto temp = settings.path;
		temp += ".tmp";
		std::filesystem::create_directory(temp);
		cache.AddSource(Key(records), true, {});
		Require(!cache.Flush(), "unwritable temporary path reports failure");
		{
			ProgramDiskCache previous(settings);
			Require(previous.FindSource(Key(records - 1)).has_value(), "previous file retained");
			Require(!previous.FindSource(Key(records)).has_value(), "failed save not published");
		}
		std::filesystem::remove(temp);
	}
	std::filesystem::remove(settings.path);
	std::filesystem::remove(dir);
	std::puts("Program cache emergency tests passed");
}
