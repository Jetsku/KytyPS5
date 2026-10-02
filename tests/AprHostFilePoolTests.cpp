// Tests for the APR host file handle pool (libs/aprHostFilePool.h).
//
//   apr_host_file_pool_tests            bookkeeping + content checks (what CTest runs)
//   apr_host_file_pool_tests --bench    also prints the cost of "open, size, seek, read, close" per
//                                       read against a pooled handle, on a temporary file
#include "common/file.h"
#include "libs/aprHostFilePool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Libs::AprHostFiles::HostFilePool;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "AprHostFilePoolTests: FAILED: %s\n", message);
		std::exit(1);
	}
}

// Counts live handles, to check that the pool closes what it should.
struct CountingFile {
	static inline std::atomic<int> alive {0};
	CountingFile() { ++alive; }
	~CountingFile() { --alive; }
	CountingFile(const CountingFile&)            = delete;
	CountingFile& operator=(const CountingFile&) = delete;
};

const auto OpenAlways = [](CountingFile&) { return true; };
const auto OpenNever  = [](CountingFile&) { return false; };

void TestReuseAndConcurrentHandles() {
	{
		HostFilePool<CountingFile> pool(4, 2);
		{
			auto lease = pool.Acquire("a", OpenAlways);
			Check(static_cast<bool>(lease), "first acquire opens a handle");
		}
		{
			auto lease = pool.Acquire("a", OpenAlways);
			Check(static_cast<bool>(lease), "second acquire succeeds");
		}
		auto stats = pool.GetStats();
		Check(stats.opens == 1 && stats.reuses == 1, "sequential reads share one handle");
		Check(CountingFile::alive == 1 && pool.IdleCount("a") == 1, "one idle handle is kept");

		// Two leases held at once need two handles; both come back.
		{
			auto first  = pool.Acquire("a", OpenAlways);
			auto second = pool.Acquire("a", OpenAlways);
			Check(first && second, "concurrent leases both succeed");
			Check(pool.GetStats().opens == 2, "a second handle is opened while the first is leased");
			Check(CountingFile::alive == 2, "two handles are alive");
		}
		Check(pool.IdleCount("a") == 2 && CountingFile::alive == 2, "both handles return to the pool");
	}
	Check(CountingFile::alive == 0, "destroying the pool closes the idle handles");
}

void TestIdleCap() {
	{
		HostFilePool<CountingFile> pool(4, 2);
		{
			auto l1 = pool.Acquire("a", OpenAlways);
			auto l2 = pool.Acquire("a", OpenAlways);
			auto l3 = pool.Acquire("a", OpenAlways);
			auto l4 = pool.Acquire("a", OpenAlways);
			Check(CountingFile::alive == 4, "four concurrent leases open four handles");
		}
		Check(pool.IdleCount("a") == 2, "only max_idle_per_path handles stay idle");
		Check(pool.GetStats().closed_on_full == 2 && CountingFile::alive == 2,
		      "handles beyond the cap are closed on release");
	}
	Check(CountingFile::alive == 0, "no handle leaks after the cap test");
}

void TestDiscardAndFailure() {
	{
		HostFilePool<CountingFile> pool(4, 2);
		{
			auto lease = pool.Acquire("a", OpenAlways);
			lease.Discard();
			Check(!lease, "a discarded lease is empty");
			Check(CountingFile::alive == 0, "Discard closes the handle");
		}
		Check(pool.IdleCount("a") == 0 && pool.PathCount() == 0, "a discarded handle is never pooled");

		auto failed = pool.Acquire("missing", OpenNever);
		Check(!failed, "a failed open gives an empty lease");
		Check(pool.GetStats().open_failures == 1 && pool.PathCount() == 0,
		      "a failed open is counted and leaves no entry");
		Check(CountingFile::alive == 0, "a failed open leaks no handle");

		// A lease moved into another keeps exactly one return.
		auto original = pool.Acquire("a", OpenAlways);
		auto moved    = std::move(original);
		Check(!original && moved, "move transfers the lease");
		moved = HostFilePool<CountingFile>::Lease();
		Check(pool.IdleCount("a") == 1, "assigning an empty lease returns the handle once");
	}
	Check(CountingFile::alive == 0, "no handle leaks after discard/failure/move");
}

void TestLeastRecentlyUsedPathIsEvicted() {
	{
		HostFilePool<CountingFile> pool(2, 2);
		{ auto l = pool.Acquire("a", OpenAlways); }
		{ auto l = pool.Acquire("b", OpenAlways); }
		{ auto l = pool.Acquire("a", OpenAlways); } // "a" is now newer than "b"
		{ auto l = pool.Acquire("c", OpenAlways); }
		Check(pool.PathCount() == 2, "the pool keeps at most max_paths paths");
		Check(pool.IdleCount("b") == 0, "the least recently used path is evicted");
		Check(pool.IdleCount("a") == 1 && pool.IdleCount("c") == 1, "recent paths keep their handle");
		Check(pool.GetStats().evicted_paths == 1 && CountingFile::alive == 2,
		      "the evicted path's handle is closed");
	}
	Check(CountingFile::alive == 0, "no handle leaks after eviction");
}

// ---- real files ---------------------------------------------------------------------------

uint8_t PatternAt(uint64_t index, uint32_t salt) {
	return static_cast<uint8_t>(((index + salt) * 2654435761ull) >> 13);
}

void MakeFile(const std::filesystem::path& path, uint64_t size, uint32_t salt) {
	Common::File file;
	Check(file.Create(path), "create temporary file");
	std::vector<uint8_t> chunk(1u << 20);
	for (uint64_t done = 0; done < size;) {
		const auto count = static_cast<uint32_t>(std::min<uint64_t>(chunk.size(), size - done));
		for (uint32_t i = 0; i < count; i++) chunk[i] = PatternAt(done + i, salt);
		file.Write(chunk.data(), count);
		done += count;
	}
	file.Close();
}

// The same steps ReadHostFileToGuest takes: size, seek, read. `got` is the number of bytes read.
bool PooledRead(HostFilePool<Common::File>& pool, const std::filesystem::path& path, uint64_t offset,
                uint32_t size, std::vector<uint8_t>& out, uint32_t* got) {
	auto lease = pool.Acquire(path.string(), [&path](Common::File& f) {
		return f.Open(path, Common::File::Mode::Read);
	});
	if (!lease) return false;
	auto&      file      = lease.File();
	const auto file_size = file.Size();
	*got                 = 0;
	if (offset >= file_size) return true;
	if (!file.Seek(offset)) {
		lease.Discard();
		return false;
	}
	out.resize(size);
	file.Read(out.data(), size, got);
	return true;
}

bool MatchesPattern(const std::vector<uint8_t>& data, uint32_t count, uint64_t offset, uint32_t salt) {
	for (uint32_t i = 0; i < count; i++) {
		if (data[i] != PatternAt(offset + i, salt)) return false;
	}
	return true;
}

void TestRealFilesContentAndEof(const std::filesystem::path& dir) {
	constexpr uint64_t SIZE = 3u * 1024 * 1024 + 123;
	const auto         path = dir / "pak0.bin";
	MakeFile(path, SIZE, 1);

	HostFilePool<Common::File> pool(4, 4);
	std::vector<uint8_t>       data;
	uint32_t                   got = 0;

	Check(PooledRead(pool, path, 0, 222, data, &got) && got == 222 && MatchesPattern(data, got, 0, 1),
	      "read at offset 0");
	Check(PooledRead(pool, path, 1000000, 4096, data, &got) && got == 4096 &&
	          MatchesPattern(data, got, 1000000, 1),
	      "read in the middle");
	Check(PooledRead(pool, path, SIZE - 100, 500, data, &got) && got == 100 &&
	          MatchesPattern(data, got, SIZE - 100, 1),
	      "a read crossing the end returns only what exists");
	Check(PooledRead(pool, path, SIZE, 16, data, &got) && got == 0, "a read at the end returns 0");
	Check(PooledRead(pool, path, SIZE + 4096, 16, data, &got) && got == 0,
	      "a read past the end returns 0");
	// The same handle served all of these one after another: the cursor is repositioned each time.
	Check(pool.GetStats().opens == 1 && pool.GetStats().reuses == 4,
	      "five sequential reads of one file use a single open");

	// Reading the same offset twice gives the same bytes even after reads elsewhere.
	Check(PooledRead(pool, path, 17, 64, data, &got) && MatchesPattern(data, got, 17, 1),
	      "the cursor does not leak between reads");

	std::vector<uint8_t> none;
	uint32_t             none_got = 0;
	Check(!PooledRead(pool, dir / "does-not-exist.bin", 0, 16, none, &none_got),
	      "a missing file reports failure");
}

void TestRealFilesConcurrent(const std::filesystem::path& dir) {
	constexpr uint64_t SIZE    = 2u * 1024 * 1024;
	constexpr int      THREADS = 8;
	constexpr int      READS   = 3000;
	std::filesystem::path paths[3] = {dir / "c0.bin", dir / "c1.bin", dir / "c2.bin"};
	for (uint32_t i = 0; i < 3; i++) MakeFile(paths[i], SIZE, 10 + i);

	// max_paths = 2 with three files forces evictions while threads hold leases.
	HostFilePool<Common::File> pool(2, 2);
	std::atomic<int>           failures {0};
	std::vector<std::thread>   threads;
	for (int t = 0; t < THREADS; t++) {
		threads.emplace_back([&, t] {
			std::mt19937_64      rng(1234 + t);
			std::vector<uint8_t> data;
			for (int n = 0; n < READS; n++) {
				const auto     which  = static_cast<uint32_t>(rng() % 3);
				const uint64_t offset = rng() % SIZE;
				const auto     size   = static_cast<uint32_t>(1 + rng() % 5000);
				uint32_t       got    = 0;
				if (!PooledRead(pool, paths[which], offset, size, data, &got)) {
					failures++;
					continue;
				}
				const auto expected = static_cast<uint32_t>(std::min<uint64_t>(size, SIZE - offset));
				if (got != expected || !MatchesPattern(data, got, offset, 10 + which)) failures++;
			}
		});
	}
	for (auto& thread: threads) thread.join();
	Check(failures == 0, "every concurrent read returns the right bytes");
	const auto stats = pool.GetStats();
	Check(stats.opens + stats.reuses == static_cast<uint64_t>(THREADS) * READS,
	      "every acquire is either an open or a reuse");
	Check(stats.open_failures == 0, "no open failed under concurrency");
	Check(stats.reuses > stats.opens, "handles are mostly reused");
}

// ---- timing -------------------------------------------------------------------------------

void Bench(const std::filesystem::path& dir) {
	constexpr uint64_t SIZE  = 64ull * 1024 * 1024;
	constexpr int      READS = 20000;
	constexpr uint32_t BYTES = 222; // the typical APR pak read in Crash Bandicoot 4 / Astro Bot
	const auto         path  = dir / "bench.bin";
	MakeFile(path, SIZE, 7);

	std::mt19937_64 rng(99);
	std::vector<uint64_t> offsets(READS);
	for (auto& offset: offsets) offset = rng() % (SIZE - BYTES);
	std::vector<uint8_t> buffer(BYTES);

	// Warm the OS cache so both variants measure call overhead, not disk latency.
	{
		Common::File file;
		file.Open(path, Common::File::Mode::Read);
		std::vector<uint8_t> sink(1u << 20);
		for (uint64_t done = 0; done < SIZE; done += sink.size()) file.Read(sink.data(), static_cast<uint32_t>(sink.size()));
	}

	const auto time_us = [](auto&& body) {
		const auto begin = std::chrono::steady_clock::now();
		body();
		return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
	};

	uint64_t sink_bytes = 0;
	const double open_each = time_us([&] {
		for (const auto offset: offsets) {
			Common::File file;
			file.Open(path, Common::File::Mode::Read);
			const auto file_size = file.Size();
			if (offset >= file_size) continue;
			file.Seek(offset);
			uint32_t got = 0;
			file.Read(buffer.data(), BYTES, &got);
			sink_bytes += got;
			file.Close();
		}
	});

	HostFilePool<Common::File> pool(4, 4);
	const double pooled = time_us([&] {
		for (const auto offset: offsets) {
			auto lease = pool.Acquire(path.string(), [&path](Common::File& f) {
				return f.Open(path, Common::File::Mode::Read);
			});
			auto&      file      = lease.File();
			const auto file_size = file.Size();
			if (offset >= file_size) continue;
			file.Seek(offset);
			uint32_t got = 0;
			file.Read(buffer.data(), BYTES, &got);
			sink_bytes += got;
		}
	});

	std::printf("AprHostFilePoolTests bench: %d reads of %u bytes at random offsets (warm cache)\n", READS, BYTES);
	std::printf("  open + size + seek + read + close : %7.2f us/read\n", open_each / READS);
	std::printf("  pooled handle (size + seek + read): %7.2f us/read   (%.1fx faster)\n", pooled / READS, open_each / pooled);
	std::printf("  (checksum %llu)\n", static_cast<unsigned long long>(sink_bytes));
}

} // namespace

int main(int argc, char** argv) {
	bool bench = false;
	for (int i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "--bench") == 0) bench = true;
	}

	TestReuseAndConcurrentHandles();
	TestIdleCap();
	TestDiscardAndFailure();
	TestLeastRecentlyUsedPathIsEvicted();

	const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto dir   = std::filesystem::temp_directory_path() / ("kyty_apr_pool_tests_" + std::to_string(stamp));
	std::filesystem::create_directories(dir);
	TestRealFilesContentAndEof(dir);
	TestRealFilesConcurrent(dir);
	if (bench) Bench(dir);
	std::error_code ec;
	std::filesystem::remove_all(dir, ec);

	std::puts("AprHostFilePoolTests: passed");
	return 0;
}
