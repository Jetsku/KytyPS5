#ifndef KYTY_TESTS_FILE_TIME_TEST_UTIL_H_
#define KYTY_TESTS_FILE_TIME_TEST_UTIL_H_

#include <chrono>
#include <cstdint>
#include <filesystem>

// Sets a host file's write time to `unix_ns` nanoseconds since the Unix epoch.
inline void SetUnixWriteTime(const std::filesystem::path& path, int64_t unix_ns) {
#ifdef _MSVC_STL_VERSION
	// MSVC's file_clock counts 100 ns units from 1601-01-01 (the FILETIME epoch, as
	// sysTimer.h's conversion). clock_cast goes through utc_clock and its leap-second table,
	// which produced a time SetFileTime rejects ("The parameter is incorrect") on this host.
	constexpr int64_t UNIX_EPOCH_IN_FILETIME = 116444736000000000;
	const std::filesystem::file_time_type file_time {
	    std::filesystem::file_time_type::duration {unix_ns / 100 + UNIX_EPOCH_IN_FILETIME}};
	std::filesystem::last_write_time(path, file_time);
#else
	const std::chrono::sys_time<std::chrono::nanoseconds> time {std::chrono::nanoseconds(unix_ns)};
	const auto file_time = std::chrono::file_clock::from_sys(time);
	std::filesystem::last_write_time(
	    path, std::chrono::floor<std::filesystem::file_time_type::duration>(file_time));
#endif
}

#endif // KYTY_TESTS_FILE_TIME_TEST_UTIL_H_
