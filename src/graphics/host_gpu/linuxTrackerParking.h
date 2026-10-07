#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_LINUXTRACKERPARKING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_LINUXTRACKERPARKING_H_

// Deliberately local to Linux x86-64 TrackingSpinLock, not a general mutex.
#if defined(__linux__) && defined(__x86_64__)
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <linux/futex.h>
#include <sys/syscall.h>

namespace Libs::Graphics::TrackerParking {
static_assert(std::atomic<uint32_t>::is_always_lock_free);
static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));
static_assert(alignof(std::atomic<uint32_t>) >= 4);
static_assert(sizeof(uint32_t) == 4 && sizeof(long) == 8);

// 0 not configured (off), 1 configured off, 2 configured on. No lazy guard,
// TLS, allocation, environment or clock access in lock/unlock. First startup
// configuration wins; changing the environment after publication has no effect.
inline constinit std::atomic<uint32_t> policy {0};
inline void InitializeBeforeGuestPublication() noexcept {
    const char* value = std::getenv("KYTY_TRACKER_LOCK_PARK");
    const uint32_t setting = value != nullptr && value[0] == '1' && value[1] == '\0' ? 2 : 1;
    uint32_t expected = 0;
    policy.compare_exchange_strong(expected, setting, std::memory_order_relaxed);
}
inline bool Enabled() noexcept { return policy.load(std::memory_order_relaxed) == 2; }

// Linux syscall ABI directly: returns negative error numbers without touching
// libc errno, a PLT resolver, TLS initialization or cancellation machinery.
inline long RawSyscall(long number, long a1 = 0, long a2 = 0, long a3 = 0,
                       long a4 = 0, long a5 = 0, long a6 = 0) noexcept {
    register long r10 __asm__("r10") = a4;
    register long r8 __asm__("r8") = a5;
    register long r9 __asm__("r9") = a6;
    long result;
    __asm__ volatile("syscall" : "=a"(result)
                     : "a"(number), "D"(a1), "S"(a2), "d"(a3),
                       "r"(r10), "r"(r8), "r"(r9)
                     : "rcx", "r11", "memory", "cc");
    return result;
}
// Keep the existing diagnostics but do not run fmt/logging/exit hooks from a
// signal handler. Same status as EXIT_HALT (321, observed as 65 by waitpid).
template<size_t N> [[noreturn]] inline void Fail(const char (&text)[N]) noexcept {
    long written = 0;
    while (written < static_cast<long>(N - 1)) {
        long n = RawSyscall(SYS_write, 2, reinterpret_cast<long>(text + written), N - 1 - written);
        if (n == -EINTR) continue;
        if (n <= 0) break;
        written += n;
    }
    RawSyscall(SYS_exit_group, 321);
    __builtin_trap();
}
inline constexpr uint32_t WAITERS = 0x80000000u;
inline uint32_t ThreadId() noexcept {
    const auto thread = static_cast<uint32_t>(RawSyscall(SYS_gettid));
    if (thread == 0 || (thread & WAITERS) != 0) Fail("region tracking thread id is out of range\n");
    return thread;
}
inline void Wait(std::atomic<uint32_t>& state, uint32_t expected) noexcept {
    const long result = RawSyscall(SYS_futex, reinterpret_cast<long>(&state), FUTEX_WAIT_PRIVATE, expected);
    if (result != 0 && result != -EAGAIN && result != -EINTR)
        Fail("region tracking futex wait failed\n");
}
inline void Wake(std::atomic<uint32_t>& state) noexcept {
    long result;
    do { result = RawSyscall(SYS_futex, reinterpret_cast<long>(&state), FUTEX_WAKE_PRIVATE, 1); }
    while (result == -EINTR);
    if (result < 0) Fail("region tracking futex wake failed\n");
}
} // namespace Libs::Graphics::TrackerParking
#endif
#endif
