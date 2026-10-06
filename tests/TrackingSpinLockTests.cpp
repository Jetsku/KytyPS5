#include "common/virtualMemory.h"
#include "graphics/host_gpu/memoryTracker.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <sched.h>
#include <time.h>

using Libs::Graphics::TrackingSpinLock;
namespace Libs::LibKernel::Memory {
// This harness never changes page protection; fail if that seam is reached.
bool ProtectGuestHostMemory(uint64_t, uint64_t, Common::VirtualMemory::Mode) {
    std::abort();
}
}
static void Check(bool ok, const char* text) {
    if (!ok) { std::fprintf(stderr, "TrackingSpinLockTests: FAIL: %s\n", text); std::exit(1); }
}
static double Seconds(clockid_t clock) {
    timespec ts{}; Check(clock_gettime(clock, &ts) == 0, "clock_gettime");
    return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}
// All threads on one available CPU: twelve waiters oversubscribe it while the
// owner deliberately sleeps. Per-thread CPU clocks exclude the owner/harness.
static void CpuBudget(bool expect_parking) {
    cpu_set_t allowed; CPU_ZERO(&allowed);
    Check(sched_getaffinity(0, sizeof(allowed), &allowed) == 0, "get affinity");
    int cpu = 0; while (!CPU_ISSET(cpu, &allowed)) ++cpu;
    cpu_set_t one; CPU_ZERO(&one); CPU_SET(cpu, &one);
    Check(sched_setaffinity(0, sizeof(one), &one) == 0, "pin oversubscription");
    TrackingSpinLock lock;
    lock.lock();
    std::atomic<unsigned> ready{0};
    double costs[12]{};
    std::vector<std::thread> waiters;
    for (unsigned i = 0; i != 12; ++i) waiters.emplace_back([&, i] {
        const double start = Seconds(CLOCK_THREAD_CPUTIME_ID);
        ready.fetch_add(1, std::memory_order_release);
        lock.lock(); lock.unlock();
        costs[i] = Seconds(CLOCK_THREAD_CPUTIME_ID) - start;
    });
    while (ready.load(std::memory_order_acquire) != 12) std::this_thread::yield();
    const double start = Seconds(CLOCK_MONOTONIC);
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    lock.unlock();
    for (auto& t : waiters) t.join();
    double cpu_seconds = 0; for (auto cost : costs) cpu_seconds += cost;
    const double wall = Seconds(CLOCK_MONOTONIC) - start;
    std::printf("oversubscription: waiters=12 cpu=%.6f wall=%.6f ratio=%.6f expected=%s\n", cpu_seconds, wall, cpu_seconds/wall, expect_parking ? "parking" : "spin");
    Check(sched_setaffinity(0, sizeof(allowed), &allowed) == 0, "restore affinity");
    Check(expect_parking ? cpu_seconds < wall * .15 : cpu_seconds > wall * .50,
          expect_parking ? "contended waiters burn CPU instead of parking" : "original spin mode no longer burns CPU");
}
// Protocol stress uses the real lock, ordinary non-atomic protected data, and
// forced sleeping waves. This is not a fairness guarantee or a timing benchmark.
static void ExclusionAndWakeChain() {
    TrackingSpinLock lock;
    uint64_t sequence = 0, complement = ~uint64_t{0};
    constexpr unsigned waves = 24, threads = 48, iterations = 1000;
    for (unsigned wave = 0; wave != waves; ++wave) {
        lock.lock();
        std::atomic<unsigned> ready{0};
        std::vector<std::thread> waiters;
        for (unsigned n = 0; n != threads; ++n) waiters.emplace_back([&] {
            ready.fetch_add(1, std::memory_order_release);
            for (unsigned i = 0; i != iterations; ++i) {
                errno = E2BIG;
                lock.lock();
                Check(errno == E2BIG, "acquisition preserves errno");
                Check(complement == ~sequence, "exclusion/acquire-release ordering");
                ++sequence; complement = ~sequence;
                if ((i % 251) == 0) std::this_thread::yield();
                lock.unlock();
                Check(errno == E2BIG, "wake preserves errno");
            }
        });
        while (ready.load(std::memory_order_acquire) != threads) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        lock.unlock();
        for (auto& t : waiters) t.join();
    }
    Check(sequence == uint64_t(waves) * threads * iterations, "all waiters make progress");
    std::printf("exclusion/order/wake-chain: waves=%u waiters=%u acquisitions=%llu PASS\n",
                waves, threads, static_cast<unsigned long long>(sequence));
}

#include <csignal>
#include <fstream>
#include <string>
#include <type_traits>
#include <sys/wait.h>
#include <pthread.h>
static constinit std::atomic<unsigned> signals{0}, handler_entered{0};
static TrackingSpinLock* signal_lock = nullptr;
static bool handler_locks = false;
static void Signal(int) {
    const int saved = errno;
    handler_entered.fetch_add(1, std::memory_order_relaxed);
    if (handler_locks) { signal_lock->lock(); signal_lock->unlock(); }
    signals.fetch_add(1, std::memory_order_relaxed);
    errno = saved;
}
static void AwaitFutex(uint32_t tid) {
    for (unsigned i = 0; i != 2000; ++i) {
        std::ifstream file("/proc/self/task/" + std::to_string(tid) + "/wchan");
        std::string text; file >> text;
        if (text.find("futex") != std::string::npos) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(false, "waiter actually sleeping in futex");
}
static void SignalsAndSpuriousWakes() {
    std::atomic<uint32_t> mismatch{0};
    errno = ERANGE;
    Libs::Graphics::TrackerParking::Wait(mismatch, Libs::Graphics::TrackerParking::WAITERS);
    Check(errno == ERANGE, "expected-value mismatch (EAGAIN) preserves errno");
    Libs::Graphics::TrackerParking::Wake(mismatch);
    Check(errno == ERANGE, "wake with no sleepers preserves errno");
    struct sigaction action{}; action.sa_handler = Signal;
    sigemptyset(&action.sa_mask); action.sa_flags = 0; // Deliberately not SA_RESTART.
    Check(sigaction(SIGUSR1, &action, nullptr) == 0, "install signal");
    TrackingSpinLock lock;
    lock.lock();
    std::atomic<uint32_t> tid{0};
    std::atomic<bool> acquired{false};
    std::thread waiter([&] {
        tid.store(Libs::Graphics::TrackerParking::ThreadId(), std::memory_order_release);
        errno = ENOTTY;
        lock.lock();
        Check(errno == ENOTTY, "errno preserved after EINTR/spurious wait");
        acquired.store(true, std::memory_order_release);
        lock.unlock();
        Check(errno == ENOTTY, "errno preserved after wake");
    });
    while (tid.load(std::memory_order_acquire) == 0) std::this_thread::yield();
    AwaitFutex(tid.load());
    // White-box injection only: on Linux x86-64 the first member is the futex
    // word. Standard-layout pointer interconvertibility makes this address
    // well-defined; no mock waits and no production test hooks.
    static_assert(std::is_standard_layout_v<TrackingSpinLock>);
    for (unsigned i = 0; i != 100; ++i) {
        Check(pthread_kill(waiter.native_handle(), SIGUSR1) == 0, "interrupt sleeping waiter");
        Libs::Graphics::TrackerParking::RawSyscall(SYS_futex, reinterpret_cast<long>(&lock), FUTEX_WAKE_PRIVATE, 1);
        AwaitFutex(tid.load());
        Check(!acquired.load(std::memory_order_acquire), "spurious wake must not grant held lock");
    }
    lock.unlock(); waiter.join();
    Check(signals.load() != 0, "signals actually delivered");
    // Interrupt a blocked waiter with a handler that itself needs the lock.
    // The owner is a different thread, so releasing it must let BOTH the nested
    // handler and the interrupted waiter progress.
    lock.lock(); tid.store(0); handler_entered.store(0);
    signal_lock = &lock; handler_locks = true;
    std::thread nested([&] {
        tid.store(Libs::Graphics::TrackerParking::ThreadId(), std::memory_order_release);
        errno = EDOM; lock.lock(); lock.unlock();
        Check(errno == EDOM, "nested signal acquisition preserves errno");
    });
    while (tid.load(std::memory_order_acquire) == 0) std::this_thread::yield();
    AwaitFutex(tid.load());
    Check(pthread_kill(nested.native_handle(), SIGUSR1) == 0, "nested handler delivery");
    while (handler_entered.load() == 0) std::this_thread::yield();
    lock.unlock(); nested.join(); handler_locks = false; signal_lock = nullptr;
    std::printf("EINTR/spurious wakes/signals/nested contended handler: signals=%u PASS\n", signals.load());
}
static void FailureDiagnostics(const char* mode, const char* expected) {
    int pipefd[2]; Check(pipe(pipefd) == 0, "diagnostic pipe");
    fflush(nullptr);
    const pid_t child = fork(); Check(child >= 0, "fork");
    if (child == 0) {
        close(pipefd[0]); dup2(pipefd[1], STDERR_FILENO); dup2(pipefd[1], STDOUT_FILENO); close(pipefd[1]);
        TrackingSpinLock lock;
        if (std::strcmp(mode, "recursive") == 0) { lock.lock(); lock.lock(); }
        else if (std::strcmp(mode, "signal-recursive") == 0) {
            lock.lock(); signal_lock = &lock; handler_locks = true; raise(SIGUSR1);
        } else {
            lock.lock(); std::thread wrong([&] { lock.unlock(); }); wrong.join();
        }
        _exit(99);
    }
    close(pipefd[1]); std::string text; char buf[256]; ssize_t n;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) text.append(buf, n);
    close(pipefd[0]); int status = 0; Check(waitpid(child, &status, 0) == child, "wait child");
    Check(WIFEXITED(status) && WEXITSTATUS(status) == (321 & 255), "owner violation fails promptly with legacy exit status");
    Check(text.find(expected) != std::string::npos, "owner violation diagnostic text");
    std::printf("diagnostic %s: status=%d PASS\n", mode, WEXITSTATUS(status));
}
#include <sys/ptrace.h>
#include <sys/mman.h>
// Deliver a signal at the FIRST instruction boundary after ownership is
// claimed, before any separate owner-publication store can run. ptrace makes
// the otherwise tiny recursive-signal deadlock window deterministic.
static void AcquisitionPublicationSignal() {
    void* memory = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    Check(memory != MAP_FAILED, "shared lock mapping");
    auto* lock = new(memory) TrackingSpinLock;
    auto* state = reinterpret_cast<std::atomic<uint32_t>*>(lock);
    struct sigaction action{}; action.sa_handler = Signal; sigemptyset(&action.sa_mask);
    Check(sigaction(SIGUSR1, &action, nullptr) == 0, "publication signal handler");
    fflush(nullptr);
    pid_t child = fork(); Check(child >= 0, "publication fork");
    if (child == 0) {
        signal_lock = lock; handler_locks = true;
        if (ptrace(PTRACE_TRACEME, 0, nullptr, nullptr) != 0) _exit(98);
        raise(SIGSTOP);
        lock->lock(); _exit(99);
    }
    int status = 0; Check(waitpid(child, &status, 0) == child && WIFSTOPPED(status), "trace initial stop");
    unsigned steps = 0;
    while (state->load(std::memory_order_relaxed) == 0 && steps != 10000) {
        Check(ptrace(PTRACE_SINGLESTEP, child, nullptr, nullptr) == 0, "step lock acquisition");
        Check(waitpid(child, &status, 0) == child && WIFSTOPPED(status), "trace instruction stop");
        ++steps;
    }
    Check(state->load() != 0, "observed atomic lock claim");
    Check(ptrace(PTRACE_CONT, child, nullptr, reinterpret_cast<void*>(SIGUSR1)) == 0, "deliver publication-window signal");
    bool exited = false;
    for (unsigned i = 0; i != 1000; ++i) {
        if (waitpid(child, &status, WNOHANG) == child) { exited = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!exited) { kill(child, SIGKILL); waitpid(child, &status, 0); }
    munmap(memory, 4096);
    Check(exited && WIFEXITED(status) && WEXITSTATUS(status) == (321 & 255),
          "signal after atomic ownership claim must diagnose recursion, not deadlock");
    std::printf("atomic ownership-publication signal: steps=%u status=65 PASS\n", steps);
}
int main(int argc, char** argv) {
    Check(!Libs::Graphics::TrackerParking::Enabled(), "uninitialized policy defaults OFF");
    const char* env = std::getenv("KYTY_TRACKER_LOCK_PARK");
    const bool requested = env != nullptr && std::strcmp(env, "1") == 0;
    Libs::Graphics::PageManager pages;
    Libs::Graphics::MemoryTracker tracker(pages); // Startup env parse, before threads/signals.
    Check(Libs::Graphics::TrackerParking::Enabled() == requested, "only exact 1 enables parking");
    setenv("KYTY_TRACKER_LOCK_PARK", requested ? "0" : "1", 1);
    Libs::Graphics::MemoryTracker second_tracker(pages);
    Check(Libs::Graphics::TrackerParking::Enabled() == requested, "startup policy cannot change while live");
    if (argc > 1 && std::strcmp(argv[1], "--policy-only") == 0) {
        std::printf("exact opt-in/frozen startup policy: PASS\n"); return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--signals-only") == 0) {
        SignalsAndSpuriousWakes(); return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--cpu-only") == 0) {
        CpuBudget(requested); return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--publication-signal") == 0) {
        AcquisitionPublicationSignal(); return 0;
    }
    const bool parking = argc == 1 || std::strcmp(argv[1], "--spin") != 0;
    CpuBudget(parking);
    ExclusionAndWakeChain();
    if (parking) {
        SignalsAndSpuriousWakes();
        AcquisitionPublicationSignal();
    }
    FailureDiagnostics("recursive", "recursive region tracking lock");
    FailureDiagnostics("non-owner", "region tracking lock released by non-owner");
    if (parking) FailureDiagnostics("signal-recursive", "recursive region tracking lock");
    return 0;
}
