#include "common/hangWatchdog.h"
#include "common/hostException.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
void Check(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr, "HangWatchdogTests: failed: %s\n", message);
    std::abort();
  }
}
void Decisions() {
  HangWatchdog::StallDetector d;
  Check(!d.Poll(0, 0, 0, 5000), "idle startup is unarmed");
  Check(!d.Poll(20000, 1, 0, 5000), "flips alone do not arm");
  Check(!d.Poll(30000, 1, 1, 5000), "first activity arms without firing");
  Check(!d.Poll(34999, 1, 1, 5000), "threshold exclusive below five seconds");
  Check(!d.Poll(35000, 1, 2, 5000), "CP progress cancels stall");
  Check(!d.Poll(39999, 1, 2, 5000), "CP reset retained");
  Check(!d.Poll(40000, 2, 2, 5000), "flip progress cancels stall");
  Check(d.Poll(45000, 2, 2, 5000), "exactly five seconds fires");
  Check(!d.Poll(99999, 2, 2, 5000), "only once");
}
void ConcurrentSnapshot() {
  std::atomic<bool> ready{false}, release{false};
  std::thread worker([&] {
    HangWatchdog::SetThreadName("blocked-test-worker");
    HangWatchdog::SetCpContext(33, 77);
    HangWatchdog::Scope outer("program-compile", 0x305afd0aa0f66b9aULL, 6);
    HangWatchdog::Scope inner("master-gpu", 0x1234, 71, 70);
    ready.store(true);
    ready.notify_one();
    release.wait(false);
  });
  ready.wait(false);
  HangWatchdog::NoteQueue(33, 77, "blocked");
  HangWatchdog::NoteQueueWait(33, "wait-reg-mem", 0x56ddaffa0ULL, 8, 7,
                              0xffffffff, 3, 4);
  for (unsigned i = 0; i < 40; ++i)
    HangWatchdog::NotePacket(33, 77, 0x100 + i, 0x3c, i);
  HangWatchdog::NoteEvent(0x222, "test-equeue", 0x44, -11, false, 0, 0x555);
  const HangWatchdog::SemaphoreValue waited{0x777, 123, 0x1000},
      signalled{0x888, 456, 0};
  HangWatchdog::NoteNativeSubmit(0x999, 456, 0xaaa, std::span(&waited, 1),
                                 std::span(&signalled, 1));
  HangWatchdog::RegisterGuestCode(0x900000000ULL, 0x10000000, "test-guest.bin");
  const auto snapshot = HangWatchdog::SnapshotForTest();
  Check(snapshot.find("native-wait semaphore=0x777 value=123 stages=0x1000") !=
            std::string::npos,
        "native dependency operands copied");
  Check(snapshot.find("native-signal semaphore=0x888 value=456") !=
            std::string::npos,
        "native producer copied");
  Check(snapshot.find("guest-module name='test-guest.bin'") !=
            std::string::npos,
        "guest module offsets available without tracing");
  Check(snapshot.find("blocked-test-worker") != std::string::npos,
        "thread name");
  Check(snapshot.find("program-compile address=0x305afd0aa0f66b9a") !=
            std::string::npos,
        "in-flight hash");
  Check(snapshot.find(
            "kind=master-gpu address=0x1234 expected=0x47 observed=0x46") !=
            std::string::npos,
        "nested native wait");
  Check(snapshot.find("queue=33 guest_queue=64") != std::string::npos,
        "internal to guest queue mapping");
  Check(snapshot.find("blocked=wait-reg-mem address=0x56ddaffa0 expected=0x8 "
                      "observed=0x7") != std::string::npos,
        "guest wait operands");
  Check(snapshot.find("packet=40 ") != std::string::npos &&
            snapshot.find("packet=8 ") == std::string::npos,
        "bounded packet history wraps");
  Check(snapshot.find("name='test-equeue'") != std::string::npos,
        "queue registration copied");
  release.store(true);
  release.notify_one();
  worker.join();
  Check(HangWatchdog::SnapshotForTest().find("kind=master-gpu") ==
            std::string::npos,
        "completed waits disappear");
  HangWatchdog::ClearQueueWait(33);
  HangWatchdog::NoteEvent(0x222, "test-equeue", 0x44, -11, false, 0, 0, true);
  Check(HangWatchdog::SnapshotForTest().find("name='test-equeue'") ==
            std::string::npos,
        "deleted registrations disappear");
}
void PublicationStress() {
  std::atomic<bool> finish{false};
  std::thread writer([&] {
    for (unsigned n = 0; n < 100000; ++n) {
      HangWatchdog::Scope scope("publication-stress", n, n, n, n, n);
    }
    finish.store(true);
  });
  while (!finish.load()) {
    const auto text = HangWatchdog::SnapshotForTest();
    const auto pos = text.find("kind=publication-stress ");
    if (pos == std::string::npos)
      continue;
    unsigned long long a, b, c, d, e;
    Check(std::sscanf(text.c_str() + pos,
                      "kind=publication-stress address=0x%llx expected=0x%llx "
                      "observed=0x%llx mask=0x%llx aux=0x%llx",
                      &a, &b, &c, &d, &e) == 5,
          "published fields parse");
    Check(a == b && b == c && c == d && d == e,
          "snapshot cannot mix publication generations");
  }
  writer.join();
}
void ConcurrentPackets() {
  std::atomic<unsigned> finished{0};
  std::atomic<bool> start{false};
  auto publish = [&](uint64_t worker) {
    start.wait(false);
    for (uint64_t n = 1; n <= 50000; ++n) {
      const auto marker = (worker << 48) | n;
      HangWatchdog::NotePacket(0, marker, marker, 0x3c, marker, marker, marker,
                               marker, "concurrent-packet");
    }
    finished.fetch_add(1);
  };
  std::thread parser(publish, 1), resolver(publish, 2);
  unsigned checked = 0;
  start.store(true);
  start.notify_all();
  do {
    const auto text = HangWatchdog::SnapshotForTest();
    size_t pos = 0;
    while ((pos = text.find("type=concurrent-packet ", pos)) !=
           std::string::npos) {
      unsigned long long submission, address, opcode, a, b, c, d;
      Check(std::sscanf(text.c_str() + pos,
                        "type=concurrent-packet submission=%llu address=0x%llx "
                        "opcode=0x%llx args=0x%llx,0x%llx,0x%llx,0x%llx",
                        &submission, &address, &opcode, &a, &b, &c, &d) == 7,
            "concurrent packets parse");
      Check(submission == address && address == a && a == b && b == c &&
                c == d && opcode == 0x3c,
            "parser and resolver cannot mix a packet publication");
      ++checked;
      ++pos;
    }
  } while (finished.load() != 2);
  parser.join();
  resolver.join();
  Check(checked != 0, "concurrent packet snapshots exercised");
}
void AutoDetection() {
  using HangWatchdog::IsNvidiaBlackwell;
  Check(IsNvidiaBlackwell(0x10de, 0x2b85, "NVIDIA GeForce RTX 5090"), "RTX 5090");
  Check(IsNvidiaBlackwell(0x10de, 0x2d04, "NVIDIA GeForce RTX 5060 Ti"), "RTX 5060 Ti");
  Check(IsNvidiaBlackwell(0x10de, 0x2c05, "NVIDIA GeForce RTX 5070 Ti"), "RTX 5070 Ti");
  Check(IsNvidiaBlackwell(0x10de, 0x3001, "NVIDIA GeForce RTX 5080 SUPER"),
        "later device id, RTX 50 name");
  Check(!IsNvidiaBlackwell(0x10de, 0x2204, "NVIDIA GeForce RTX 3090"), "RTX 3090");
  Check(!IsNvidiaBlackwell(0x10de, 0x2684, "NVIDIA GeForce RTX 4090"), "RTX 4090");
  Check(!IsNvidiaBlackwell(0x10de, 0x1eb0, "Quadro RTX 5000"), "Turing Quadro RTX 5000");
  Check(!IsNvidiaBlackwell(0x1002, 0x744c, "AMD Radeon RX 7900 XTX"), "AMD");
  Check(!IsNvidiaBlackwell(0x8086, 0x2c05, "RTX 5090 lookalike"), "vendor first");
}
#ifdef _WIN32
// The emulator's host fault handler ends the process for a fault it cannot resolve. A fault inside
// EnterProbe/LeaveProbe (the watchdog's stack walk through a corrupt guest frame) must reach the
// walker's own __except instead.
std::atomic<int> g_terminal_faults{0};
bool TerminalHandler(const Common::HostException::ExceptionInfo &) {
  g_terminal_faults++;
  return false;
}
// Called, not inlined: clang-cl's __try catches faults raised in callees (as RtlVirtualUnwind's
// are), not a load inside the guarded block itself.
__declspec(noinline) uint64_t Read(const volatile uint64_t *address) { return *address; }
bool ReadFaults(const volatile uint64_t *address) {
  __try {
    (void)Read(address);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return true;
  }
  return false;
}
void ProbeFaults() {
  Check(Common::HostException::InstallHandler(TerminalHandler), "install host fault handler");
  const auto *bad = reinterpret_cast<const volatile uint64_t *>(0x6c5f3262);
  Common::HostException::EnterProbe();
  Common::HostException::EnterProbe();
  Check(ReadFaults(bad), "nested probe fault reaches __except");
  Common::HostException::LeaveProbe();
  Check(ReadFaults(bad), "probe fault reaches __except");
  Common::HostException::LeaveProbe();
  Check(g_terminal_faults.load() == 0, "probe faults skip the terminating handler");
  Check(ReadFaults(bad), "unprobed fault still reaches __except after the handler declines");
  Check(g_terminal_faults.load() == 1, "unprobed fault reaches the terminating handler");
}
#endif
} // namespace
int main(int argc, char **argv) {
  if (argc > 1 && std::string_view(argv[1]) == "--fatal-shutdown") {
    const auto dir =
        std::filesystem::temp_directory_path() /
        ("kyty-watchdog-fatal-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    HangWatchdog::Initialize(dir.string());
    HangWatchdog::Scope scope("test-fatal-operation", 0x12345678, 99);
    HangWatchdog::NoteFatal("mock terminal failure", "renderer/master.cpp", 53);
    HangWatchdog::Shutdown();
    Check(std::filesystem::is_regular_file(dir / "watchdog.txt"),
          "fatal shutdown saves a report before stopping the monitor");
    std::ifstream file(dir / "watchdog.txt");
    const std::string text((std::istreambuf_iterator<char>(file)), {});
    Check(text.find("trigger=terminal-error") != std::string::npos &&
              text.find("mock terminal failure") != std::string::npos &&
              text.find("test-fatal-operation") != std::string::npos,
          "fatal report is explicit and retains the failure and active scope");
#ifdef _WIN32
    Check(text.find("rip=0x") != std::string::npos,
          "fatal shutdown also preserves native contexts");
#endif
    file.close();
    std::filesystem::remove(dir / "watchdog.txt");
    std::filesystem::remove(dir);
    std::puts("HangWatchdogTests: fatal shutdown passed");
    return 0;
  }
  if (argc > 1 && std::string_view(argv[1]) == "--fire") {
    const auto dir =
        std::filesystem::temp_directory_path() /
        ("kyty-watchdog-fire-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
#ifdef _WIN32
    _putenv_s("KYTY_HANG_WATCHDOG_MS", "1000");
#else
    setenv("KYTY_HANG_WATCHDOG_MS", "1000", 1);
#endif
    HangWatchdog::Initialize(dir.string());
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    Check(!std::filesystem::exists(dir), "idle monitor performs no file I/O");
    {
      HangWatchdog::Scope scope("test-stalled-cp", 0x12345678, 99);
      HangWatchdog::NoteFlip();
      HangWatchdog::NoteSubmission();
      std::this_thread::sleep_for(std::chrono::milliseconds(3300));
      HangWatchdog::Shutdown();
      Check(std::filesystem::is_regular_file(dir / "watchdog.txt"),
            "armed watchdog fires without hang tracing");
      std::ifstream file(dir / "watchdog.txt");
      const std::string text((std::istreambuf_iterator<char>(file)), {});
      Check(text.find("test-stalled-cp") != std::string::npos,
            "live active wait reaches file");
#ifdef _WIN32
      Check(text.find("native tid=") != std::string::npos,
            "Windows thread contexts reach file");
      Check(text.find("rip=0x") != std::string::npos,
            "Windows registers retained");
      Check(text.find("native-module name=") != std::string::npos,
            "Windows module inventory");
#endif
      if (argc > 2)
        std::filesystem::copy_file(
            dir / "watchdog.txt", std::filesystem::u8path(argv[2]),
            std::filesystem::copy_options::overwrite_existing);
    }
    std::filesystem::remove(dir / "watchdog.txt");
    std::filesystem::remove(dir);
    std::puts("HangWatchdogTests: fire passed");
    return 0;
  }

  Decisions();
  ConcurrentSnapshot();
  PublicationStress();
  ConcurrentPackets();
  AutoDetection();
#ifdef _WIN32
  ProbeFaults();
#endif
  HangWatchdog::NoteFatal("first mock fatal", "source/renderer.cpp", 53);
  HangWatchdog::NoteFatal("later cleanup failure", "source/cleanup.cpp", 99);
  const auto fatal = HangWatchdog::SnapshotForTest();
  Check(fatal.find("file='renderer.cpp' line=53 message='first mock fatal'") !=
                std::string::npos &&
            fatal.find("later cleanup failure") == std::string::npos,
        "original failure survives later teardown errors");
  const auto dir =
      std::filesystem::temp_directory_path() /
      ("kyty-watchdog-test-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  Check(HangWatchdog::WriteSnapshotForTest(dir.string()),
        "snapshot writes without tracing enabled");
  Check(std::filesystem::file_size(dir / "watchdog.txt") > 0,
        "persistent report");
  std::filesystem::remove(dir / "watchdog.txt");
  std::filesystem::remove(dir);
  const auto unicode_dir = dir.string() + "-utf8-\xc3\xa4";
  Check(HangWatchdog::WriteSnapshotForTest(unicode_dir),
        "UTF-8 report directory");
  const auto unicode_path = std::filesystem::u8path(unicode_dir);
  Check(std::filesystem::is_regular_file(unicode_path / "watchdog.txt"),
        "UTF-8 file on disk");
  std::filesystem::remove(unicode_path / "watchdog.txt");
  std::filesystem::remove(unicode_path);
  HangWatchdog::g_enabled.store(false);
  {
    HangWatchdog::Scope off("disabled-wait");
    Check(HangWatchdog::SnapshotForTest().find("disabled-wait") ==
              std::string::npos,
          "disabled metadata path");
  }
  std::puts("HangWatchdogTests: passed");
}
