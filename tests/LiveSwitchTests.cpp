// Live switches (common/liveSwitch.h, KYTY_LIVE_FILE).
// 1. Parsing and application at the flip: values, unset (KEY=), several switches of one name, the
//    on-change hook, unknown keys, unreadable lines, comments, CRLF, UTF-8 and UTF-16LE byte-order
//    marks, the sequence number and "no change" in the log line.
// 2. Registration: a switch registered after a live change of its name takes the live value, one
//    registered before any takes the startup environment.
// 3. KYTY_LIVE_TEST_CP_SPIN_US: the flip waits that long, and stops waiting when set back to 0.
// 4. The live-control thread: KYTY_LIVE_FILE in a temporary directory, the file replaced the way
//    Run-AstroAB.ps1 does it (write a temporary file, then rename over the live one).

#include "common/liveSwitch.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "LiveSwitchTests: failed: %s\n", text);
		std::fflush(stderr);
		std::abort();
	}
}

void SetEnv(const char* name, const char* value) {
#if defined(_WIN32)
	(void)_putenv_s(name, value);
#else
	(void)setenv(name, value, 1);
#endif
}

std::mutex               g_log_mutex;
std::vector<std::string> g_log;

void Sink(const std::string& line) {
	std::lock_guard lock(g_log_mutex);
	g_log.push_back(line);
}

std::vector<std::string> TakeLog() {
	std::lock_guard lock(g_log_mutex);
	auto lines = std::move(g_log);
	g_log.clear();
	return lines;
}

bool Contains(const std::vector<std::string>& lines, const char* text) {
	for (const auto& line: lines) {
		if (line.find(text) != std::string::npos) {
			return true;
		}
	}
	return false;
}

void Dump(const std::vector<std::string>& lines) {
	for (const auto& line: lines) {
		std::printf("    %s", line.c_str());
	}
}

int64_t ParseCount(const char* value) {
	return value != nullptr ? static_cast<int64_t>(std::strtoul(value, nullptr, 10)) : 8;
}

int g_hook_calls = 0;
int64_t g_hook_previous = -1;
int64_t g_hook_value    = -1;

void Hook(int64_t previous, int64_t value) {
	g_hook_calls++;
	g_hook_previous = previous;
	g_hook_value    = value;
}

// Registered before main, with none of these variables set.
Live::Switch g_count("KYTY_LIVE_TEST_COUNT", ParseCount);
Live::Switch g_count_twin("KYTY_LIVE_TEST_COUNT", ParseCount);
Live::Switch g_on("KYTY_LIVE_TEST_ON", Live::ParseDefaultOn, Hook);
Live::Switch g_off("KYTY_LIVE_TEST_OFF", Live::ParseDefaultOff);

void TestApply() {
	Check(g_count.Get() == 8 && g_count_twin.Get() == 8, "startup default");
	Check(g_on.Get() == 1 && g_off.Get() == 0, "default on / default off");
	Check(!Live::Testing::HasStaged(), "nothing staged at start");

	const auto flips0 = Live::Testing::CpFlips();
	Live::OnCpFlip();
	Check(Live::Testing::CpFlips() == flips0 + 1, "flip counted");
	Check(TakeLog().empty(), "no log without a staged change");

	Live::Testing::StageText("# A/B side B\r\nKYTY_LIVE_SEQ=1\r\nKYTY_LIVE_TEST_COUNT=0\r\n  KYTY_LIVE_TEST_ON = 0 \r\n");
	Check(Live::Testing::HasStaged(), "staged");
	Check(g_count.Get() == 8, "not applied before the flip");
	Live::OnCpFlip();
	Check(!Live::Testing::HasStaged(), "consumed at the flip");
	Check(g_count.Get() == 0 && g_count_twin.Get() == 0, "both switches of one name changed");
	Check(g_on.Get() == 0, "default-on switch off");
	Check(g_hook_calls == 1 && g_hook_previous == 1 && g_hook_value == 0, "on-change hook");
	auto log = TakeLog();
	Dump(log);
	Check(log.size() == 1, "one log line");
	Check(log[0].rfind("Live: seq 1 applied at CP flip ", 0) == 0, "log line start");
	Check(log[0].find("(t_ms ") != std::string::npos, "log line time");
	Check(log[0].find("KYTY_LIVE_TEST_COUNT=0 (8 -> 0)") != std::string::npos, "count change listed once");
	Check(log[0].find("KYTY_LIVE_TEST_COUNT=0 (8 -> 0), KYTY_LIVE_TEST_COUNT") == std::string::npos,
	      "twin not listed twice");
	Check(log[0].find("KYTY_LIVE_TEST_ON=0 (1 -> 0)") != std::string::npos, "on change listed");

	// The same desired state again: no change, the sequence number still logged.
	Live::Testing::StageText("KYTY_LIVE_SEQ=2\nKYTY_LIVE_TEST_COUNT=0\n");
	Live::OnCpFlip();
	log = TakeLog();
	Dump(log);
	Check(log.size() == 1 && Contains(log, "Live: seq 2 applied") && Contains(log, "): no change"),
	      "no change");
	Check(g_hook_calls == 1, "hook only on a change");

	// KEY= is unset: the flag's default. Keys the file does not mention keep their value.
	Live::Testing::StageText("KYTY_LIVE_SEQ=3\nKYTY_LIVE_TEST_COUNT=\n");
	Live::OnCpFlip();
	log = TakeLog();
	Dump(log);
	Check(g_count.Get() == 8 && g_count_twin.Get() == 8, "unset restores the default");
	Check(g_on.Get() == 0, "unmentioned key kept");
	Check(Contains(log, "KYTY_LIVE_TEST_COUNT= (0 -> 8)"), "unset listed");

	// A key listed twice keeps its last value; unknown keys and unreadable lines are logged.
	Live::Testing::StageText(
	    "KYTY_LIVE_SEQ=4\nkyty_live_test_off=1\nKYTY_LIVE_TEST_COUNT=5\nKYTY_LIVE_TEST_COUNT=6\n"
	    "KYTY_NOT_A_LIVE_SWITCH=1\ngarbage line\nKYTY-BAD=1\nKYTY_LIVE_SEQ=x\n; comment\n");
	Live::OnCpFlip();
	log = TakeLog();
	Dump(log);
	Check(g_off.Get() == 1, "lower-case key accepted");
	Check(g_count.Get() == 6, "last value wins");
	Check(Contains(log, "Live: seq 4: KYTY_NOT_A_LIVE_SWITCH is not a live switch"), "unknown key");
	Check(Contains(log, "cannot read the line 'garbage line'"), "line without =");
	Check(Contains(log, "cannot read the line 'KYTY-BAD=1'"), "bad key");
	Check(Contains(log, "cannot read the line 'KYTY_LIVE_SEQ=x'"), "bad sequence number");

	// A file without a sequence number.
	Live::Testing::StageText("KYTY_LIVE_TEST_COUNT=7");
	Live::OnCpFlip();
	log = TakeLog();
	Dump(log);
	Check(g_count.Get() == 7 && Contains(log, "Live: seq - applied"), "no sequence number");

	// A newer version staged before the flip replaces the older one.
	Live::Testing::StageText("KYTY_LIVE_SEQ=10\nKYTY_LIVE_TEST_COUNT=1\n");
	Live::Testing::StageText("KYTY_LIVE_SEQ=11\nKYTY_LIVE_TEST_COUNT=2\n");
	Live::OnCpFlip();
	log = TakeLog();
	Dump(log);
	Check(g_count.Get() == 2 && log.size() == 1 && Contains(log, "seq 11"), "latest version wins");

	// Byte-order marks: UTF-8, and UTF-16LE (Windows PowerShell 5.1's ">").
	Live::Testing::StageText("\xEF\xBB\xBFKYTY_LIVE_SEQ=12\nKYTY_LIVE_TEST_COUNT=3\n");
	Live::OnCpFlip();
	Check(g_count.Get() == 3 && Contains(TakeLog(), "seq 12"), "UTF-8 BOM");
	const char16_t wide[] = u"KYTY_LIVE_SEQ=13\r\nKYTY_LIVE_TEST_COUNT=4\r\n";
	std::string    utf16("\xFF\xFE", 2);
	utf16.append(reinterpret_cast<const char*>(wide), sizeof(wide) - sizeof(char16_t));
	Live::Testing::StageText(utf16);
	Live::OnCpFlip();
	Check(g_count.Get() == 4 && Contains(TakeLog(), "seq 13"), "UTF-16LE BOM");
	std::printf("apply: ok\n");
}

void TestRegistration() {
	// Registered after a live change of its name: the live value, not the environment.
	SetEnv("KYTY_LIVE_TEST_LAZY", "1");
	Live::Testing::StageText("KYTY_LIVE_SEQ=20\nKYTY_LIVE_TEST_LAZY=5\n");
	Live::OnCpFlip();
	auto log = TakeLog();
	Dump(log);
	Check(Contains(log, "KYTY_LIVE_TEST_LAZY is not a live switch"), "not registered yet");
	static Live::Switch lazy("KYTY_LIVE_TEST_LAZY", ParseCount);
	Check(lazy.Get() == 5, "late registration takes the live value");

	// Registered without a live change: the startup environment.
	SetEnv("KYTY_LIVE_TEST_ENV", "9");
	static Live::Switch from_env("KYTY_LIVE_TEST_ENV", ParseCount);
	Check(from_env.Get() == 9, "environment value");
	Live::Testing::StageText("KYTY_LIVE_SEQ=21\nKYTY_LIVE_TEST_ENV=\n");
	Live::OnCpFlip();
	Check(from_env.Get() == 8, "unset after an environment value");
	TakeLog();
	std::printf("registration: ok\n");
}

double FlipMs() {
	const auto start = std::chrono::steady_clock::now();
	Live::OnCpFlip();
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// The fastest of a few flips (a loaded machine may preempt one).
double MinFlipMs() {
	double best = 1e9;
	for (int i = 0; i < 5; i++) {
		best = std::min(best, FlipMs());
	}
	return best;
}

void TestSpin() {
	Check(MinFlipMs() < 1.0, "no spin by default");
	Live::Testing::StageText("KYTY_LIVE_SEQ=30\nKYTY_LIVE_TEST_CP_SPIN_US=3000\n");
	const auto first  = FlipMs();
	const auto second = MinFlipMs();
	std::printf("  spin 3000 us: flips took %.3f and %.3f ms\n", first, second);
	Check(first >= 3.0 && second >= 3.0, "the flip waits the spin");
	Check(second < 50.0, "the spin is bounded");
	Check(Contains(TakeLog(), "KYTY_LIVE_TEST_CP_SPIN_US=3000 (0 -> 3000)"), "spin change logged");
	Live::Testing::StageText("KYTY_LIVE_SEQ=31\nKYTY_LIVE_TEST_CP_SPIN_US=0\n");
	(void)FlipMs();
	Check(MinFlipMs() < 1.0, "spin off again");
	TakeLog();
	std::printf("spin: ok\n");
}

bool WaitFor(bool (*predicate)(), double seconds) {
	const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
	while (std::chrono::steady_clock::now() < end) {
		if (predicate()) {
			return true;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	return predicate();
}

void WriteReplacing(const std::filesystem::path& path, const std::string& text) {
	const auto temp = path.string() + ".tmp";
	{
		std::ofstream out(temp, std::ios::binary | std::ios::trunc);
		out << text;
	}
	std::error_code error;
	std::filesystem::rename(temp, path, error);
	Check(!error, "rename over the live file");
}

void TestThread() {
	const auto dir = std::filesystem::temp_directory_path() /
	                 ("kyty-live-switch-tests-" + std::to_string(std::chrono::steady_clock::now()
	                                                                  .time_since_epoch()
	                                                                  .count()));
	std::filesystem::create_directories(dir);
	const auto path = dir / "live.txt";
	WriteReplacing(path, "KYTY_LIVE_SEQ=40\nKYTY_LIVE_TEST_ON=1\n");
	SetEnv("KYTY_LIVE_FILE", path.string().c_str());
	Check(!Live::Enabled(), "not enabled before Start");
	Live::Start();
	Check(Live::Enabled(), "enabled after Start");
	Live::Start(); // once
	Check(WaitFor([] { return Live::Testing::HasStaged(); }, 3.0), "initial file staged");
	Live::OnCpFlip();
	auto log = TakeLog();
	Dump(log);
	Check(Contains(log, "Live: watching KYTY_LIVE_FILE="), "startup line");
	Check(Contains(log, "KYTY_LIVE_TEST_COUNT") && Contains(log, "KYTY_LIVE_TEST_CP_SPIN_US"),
	      "startup line lists the switches");
	Check(Contains(log, "Live: seq 40 applied") && g_on.Get() == 1, "initial file applied");

	// Replaced by a newer version: picked up within a few polls.
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	WriteReplacing(path, "KYTY_LIVE_SEQ=41\nKYTY_LIVE_TEST_ON=0\n");
	Check(WaitFor([] { return Live::Testing::HasStaged(); }, 3.0), "new version staged");
	Live::OnCpFlip();
	log = TakeLog();
	Dump(log);
	Check(Contains(log, "Live: seq 41 applied") && g_on.Get() == 0, "new version applied");

	// The same size again (only the digit differs): the file time still tells.
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	WriteReplacing(path, "KYTY_LIVE_SEQ=42\nKYTY_LIVE_TEST_ON=1\n");
	Check(WaitFor([] { return Live::Testing::HasStaged(); }, 3.0), "same-size version staged");
	Live::OnCpFlip();
	log = TakeLog();
	Dump(log);
	Check(Contains(log, "Live: seq 42 applied") && g_on.Get() == 1, "same-size version applied");

	// Unchanged file: nothing staged again.
	std::this_thread::sleep_for(std::chrono::milliseconds(700));
	Check(!Live::Testing::HasStaged(), "an unchanged file is not staged again");

	// CP busy time (cp_busy_us): slices so far plus the running one, which an application at a
	// flip is always inside of.
	auto busy_us = [](const std::vector<std::string>& lines) -> int64_t {
		for (const auto& line: lines) {
			const auto at = line.find("cp_busy_us ");
			if (at != std::string::npos) {
				return std::strtoll(line.c_str() + at + 11, nullptr, 10);
			}
		}
		return -1;
	};
	Live::Testing::StageText("KYTY_LIVE_SEQ=43\n");
	Live::OnCpFlip();
	const auto busy0 = busy_us(TakeLog());
	Live::CpSliceBegin();
	std::this_thread::sleep_for(std::chrono::milliseconds(30));
	Live::Testing::StageText("KYTY_LIVE_SEQ=44\n");
	Live::OnCpFlip();
	const auto busy1 = busy_us(TakeLog());
	std::this_thread::sleep_for(std::chrono::milliseconds(30));
	Live::CpSliceEnd();
	std::this_thread::sleep_for(std::chrono::milliseconds(200)); // outside a slice: not counted
	Live::Testing::StageText("KYTY_LIVE_SEQ=45\n");
	Live::OnCpFlip();
	const auto busy2 = busy_us(TakeLog());
	std::printf("  cp_busy_us %lld -> %lld -> %lld\n", static_cast<long long>(busy0),
	            static_cast<long long>(busy1), static_cast<long long>(busy2));
	Check(busy0 >= 0 && busy1 - busy0 >= 30000 && busy1 - busy0 < 1000000, "running slice counted");
	Check(busy2 - busy1 >= 30000 && busy2 - busy1 < 150000, "slice end counted, idle time not");

	std::error_code error;
	std::filesystem::remove_all(dir, error);
	std::printf("thread: ok\n");
}

} // namespace

int main() {
	Live::Testing::SetLogSink(Sink);
	TestApply();
	TestRegistration();
	TestSpin();
	TestThread();
	std::printf("LiveSwitchTests: all passed\n");
	return 0;
}
