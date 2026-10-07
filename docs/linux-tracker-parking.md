# Draft: opt-in Linux x86-64 tracker futex parking

## Scope and status

Source-only draft against Jetsku/KytyPS5 `58a7f743c445714050e6e1cd3288bfc2cb38c032`. This proposal contains only the tracker-lock parking implementation, its integration and regressions. Native EXTRQ rewriting, the readback diagnostic, unrelated Linux compilation fixes, binaries, guest code, game patches, private logs, saves and user configuration are excluded.

The individual gameplay FPS benefit of this change has not been established. Earlier Sky Garden measurements enabled this lock change together with a separate native EXTRQ experiment; those measurements must not be attributed to this proposal alone. Subsequent gameplay crashes and an owner-reported whole-system freeze remain unresolved. This draft does not claim to fix those failures or certify whole-emulator stability.

## Startup policy and compatibility

Only exact `KYTY_TRACKER_LOCK_PARK=1` before the first MemoryTracker construction enables the additional Linux x86-64 path. The first constructor freezes a constant-initialized atomic policy before guest publication; later environment changes do not change that policy. Lock and unlock do not parse the environment or initialize a lazy guard on the enabled path.

    KYTY_TRACKER_LOCK_PARK=1 <normal emulator command>

Unset, empty, `0`, `01`, `true` and other values keep the new path off. Disabled TrackingSpinLock explicitly uses int15's spin implementation, including existing renderer-batch read-only/pause and diagnostic branches. This means the old generic default-on lazy parking is no longer used by TrackingSpinLock when the opt-in is absent. Disabled mode is not instruction-for-instruction identical: it adds a policy check, parameter check and storage.

PageManager's separate generic ParkingSpinLock and unrelated callers retain their existing semantics. The new defaulted `allow_parking` parameter is false only for the disabled TrackingSpinLock caller; other callers retain its true default. This does not make the entire memory-fault path async-signal-safe. Non-Linux/non-x86-64 code retains the existing paths.

## Protocol

The new path combines owner TID and a waiter bit in one 32-bit always-lock-free atomic word. Acquisition publishes ownership with a single compare/exchange rather than a later separate owner store. It checks recursion on every contention loop, performs 256 fixed PAUSE iterations, then uses expected-value FUTEX_WAIT_PRIVATE. Release publishes zero with release ordering and wakes at most one sleeping waiter. Contended winners retain the waiter bit to preserve the wake chain.

Linux x86-64 raw gettid/futex/write/exit_group syscalls avoid libc errno modification, TLS/PLT initialization, cancellation, allocation, clocks, profiler and watchdog work in the enabled lock path. The prior watchdog/MemoryStats/Profiler lock-contention seams do not observe this enabled path; do not interpret missing counters as absence of waiting.

Unsupported cases and limitations include owner death, destruction with waiters, process-shared use, cancellation/nonlocal exit while owning, and incompatible nested lock ordering. Acquisition has no hard latency bound if an owner does not release. Wake retries EINTR; fatal diagnostic writes can block on a blocked stderr. Disabled mode retains its prior separate owner-publication gap and is not universally signal-safe.

## Regression evidence

The implementation and test source files are byte-identical to the previously built int15 performance candidate. Local Linux x86-64 evidence on Fedora 44 / Intel Core i5-14600K includes:

- Three selected integrated CTests passed: tracking_spin_lock_parking, tracking_spin_lock_original and tracking_spin_lock_publication.
- Earlier isolated tracking_spin_lock_tests, memory_tracker_tests and page_manager_tests ran with enabled and disabled parking: six selected executable/mode runs passed.
- Exact-policy/CPU cases covered unset, empty, 0, 1, 01, true and 2, with renderer-batch both off and on: 14 selected cases passed.
- Interrupted acquisition-publication regression passed ten repetitions.

The full lock suite exercises exclusion, ordering, wake chains, kernel futex sleep, EINTR/spurious wakes, nested contended signal handlers, errno preservation, recursive/non-owner diagnostics and the atomic ownership-publication signal window. The publication test controls only a test-created child using ptrace; PTRACE_TRACEME denial is a failure, not a silent pass.

This is selected regression evidence, not the complete emulator suite, Windows/macOS validation, TSAN validation or standalone gameplay-performance proof. Historical local logs are not included in this source-only contribution.

## Reproduce

Build the test target explicitly; it is EXCLUDE_FROM_ALL:

    cmake --build _Build/linux --target tracking_spin_lock_tests
    ctest --test-dir _Build/linux --output-on-failure -R '^tracking_spin_lock_(parking|original|publication)$'

CTest sets the exact environment required by each full-suite mode. Directly invoking the full-suite executable without the corresponding environment is not supported by its argv-selected CPU-budget expectation; use the registered entries above.

For lightweight exact startup-policy checks, set or unset the environment in a fresh process and use:

    KYTY_TRACKER_LOCK_PARK=1 _Build/linux/tracking_spin_lock_tests --policy-only
    KYTY_TRACKER_LOCK_PARK=0 _Build/linux/tracking_spin_lock_tests --policy-only

The environment must be established before construction, not changed while guest code is active. Maintainer review, CI and independent gameplay controls are still required before this draft should become merge-ready.
