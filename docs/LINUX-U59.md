# U59 + Demon's Souls: Linux x86-64 build

> First written for the 2026-09-30 release (`u59-windows-20260930-demons`).

From now on the GitHub workflow `.github/workflows/u59-linux.yml` attaches
`KytyPS5-U59-Linux-x86_64.tar.gz` (built with Clang on Ubuntu 24.04, not tested in a game) to each release;
extract it, keep the folder together and run `./kyty_launcher` from that folder, so it finds `u59-preset.json`
(without the preset, lighting can render black). No game files, compatibility cheat files, firmware, saves
or caches are distributed. This is an unofficial build, not an upstream Linux release. The general Linux
build steps and dependencies are in the main README.

## Build

Follow the Linux dependencies in the main README. On Fedora, install:

```sh
sudo dnf install clang lld cmake ninja-build glslang qt6-qtbase-devel \
  mesa-libGL-devel libX11-devel libXcursor-devel libXext-devel \
  libXfixes-devel libXi-devel libXrandr-devel libXScrnSaver-devel \
  libXtst-devel libxkbcommon-devel alsa-lib-devel pulseaudio-libs-devel \
  systemd-devel dbus-devel wayland-devel wayland-protocols-devel
```

From this branch:

```sh
git submodule update --init --recursive
cmake -S . -B _Build/linux -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DKYTY_EMULATOR_IPO=ON -DKYTY_BUILD_ORIGIN=Fork \
  -DKYTY_BUILD_REPOSITORY=Jetsku/KytyPS5 \
  -DKYTY_RELEASE_TAG=u59-linux-local
cmake --build _Build/linux --target launcher --parallel 8
cmake --install _Build/linux --prefix _Build/linux/install
cp tools/u59-preset.json _Build/linux/install/
./_Build/linux/install/kyty_launcher
```

The launcher reads the adjacent preset automatically. Direct `kyty_emulator`
CLI calls do not read it automatically. Add your legally obtained game folders
in launcher settings and configure display/per-game settings there.

## Guest write tracking (current main)

Kyty notices the game's writes to memory it shares with the GPU by protecting
those pages and catching the write fault. On Linux each fault is a signal plus
`mprotect` calls, and every `mprotect` takes the process-wide memory-map lock,
so with many game threads writing at once the faults queue behind each other.
A Linux user measured about 83 us per fault (Windows: a few us) and about 8 fps
at the Astro Bot Sky Garden. Current main opens a larger window around each
write fault, scaled automatically to the fault cost the emulator measures on
the PC (`KYTY_FAULT_AHEAD_ADAPT`, on by default; Linux with `mprotect` starts at
1 MiB windows). With a slow-fault simulation on Windows: about 9 -> 27 fps; in a
WSL2 benchmark the writer threads' stalls fell from 35-44 ms to about 1 ms per
frame. The log shows the measured costs in a `Kyty platform:` line and
`Kyty fault cost:` lines (at startup and every 60 s).

1. Kernel 6.4 or newer is recommended (`uname -r`).
2. Raise the memory-map limit (protected pages split the guest mappings; at the
   default limit of 65530, `mprotect` fails and Kyty stops):
   `sudo sysctl -w vm.max_map_count=1048576` (until reboot; to keep it, put
   `vm.max_map_count=1048576` in `/etc/sysctl.d/99-kyty.conf`).
3. Optional: `"KYTY_UFFD_WP": "1"` in `u59-preset.json` tracks the writes with
   userfaultfd write-protection instead of `mprotect` (off by default; untested
   in a game on Linux so far). It needs kernel 5.19 or newer; 6.4 adds it for
   all guest memory. In a WSL2 benchmark with 15 writing threads a fault cost
   about 11 us instead of about 230 us; with the larger windows above, plain
   `mprotect` was as fast in that benchmark. The log then says `guest write
   tracking with userfaultfd write-protection (KYTY_UFFD_WP=1): on`. If it says
   `unavailable`, the kernel is too old or userfaultfd is blocked, and
   everything runs as without the flag.

## Thread placement (KYTY_CPU_RESERVE, KYTY_VCACHE_PLACEMENT)

The command-processor thread hands every draw to Vulkan in order and limits the frame rate in
draw-heavy views. `KYTY_CPU_RESERVE` (a core of its own for it) and `KYTY_VCACHE_PLACEMENT` were
Windows-only (CPU sets); on Linux the log said `no allowed processor in group 0` and every thread
ran anywhere. Linux now has the same placement: the topology comes from sysfs (SMT siblings, L3
groups, the amd_pstate prefcore or ACPI CPPC ranking) and threads get hard affinities. Linux has no
process default CPU sets, so threads without a role get the general processors: the main thread's
affinity before the other threads start, and a monitor pass once a second for threads started
later (a thread inherits its creator's affinity). A thread found with an affinity this module did
not set is left alone. The startup affinity (`taskset`) is the process mask.

`KYTY_VCACHE_PLACEMENT=ccd` (new, Windows and Linux) applies the V-cache layout to CPUs with two L3
groups of similar size (Ryzen 7900X/7950X/9900X/9950X): the CCD with the preferred core gets the
emulator's threads, the other CCD the game's threads (`KYTY_VCACHE_GUEST` defaults to `cold` in
this mode). Measured on a Ryzen 9 9900X + RX 7900 XT (CachyOS, kernel 7.3, non-PGO build), Astro
Bot, a draw-heavy forward view in the Sky Garden, frames per second averaged over 15-20 s:

| Placement | fps |
| --- | --- |
| none (before) | 42.7, 44.2, 42.8 |
| CP, sequencer and recorder on cores of their own (`KYTY_CPU_RESERVE=stages`) | 47.9, 48.8, 49.8 |
| emulator CCD / game CCD only | 50.4 |
| both (`KYTY_CPU_RESERVE=stages` + `KYTY_VCACHE_PLACEMENT=ccd`) | 58.9, 56.1, 57.0, 55.7 |

The CP was busy 97% of every second in all rows but the last (the GPU about 67%, waiting for work).
A 382 s session with both switches averaged 59.1 fps (360 s at 55 fps or more).

## Portability changes

- Enable exceptions for `src/common/profiler.cpp` on Linux only: loading
  diagnostics use `try`/`catch`, while the global compiler flags disable exceptions.
- Specify the `uint64_t` return type of the `WriteFaultWindow` lambda. On Linux,
  `uint64_t` and the result of `strtoull` are distinct types. Its logic is unchanged.

AI assistance was used to diagnose these errors, apply the minimal changes,
build and test the binaries, prepare documentation, and publish this fork.
No upstream pull request or claim of human code review is made.

## Verification and limitations

The initial local build was tested on Fedora 44 x86-64 with Clang 22.1.8 and
Qt 6.11.2. Release mode and IPO are enabled. Fourteen selected CTest tests
passed, including emulator CLI validation, profiler counters, page manager,
draw preparation, game patch filtering and input pulse helpers.
Installed CLI help and launcher startup with the normal desktop backend,
Wayland and offscreen were checked. Forced X11 platform initialization failed
in the build session; the default desktop launch succeeded.

The full regression suite and game/Vulkan rendering were not tested.
Demon's Souls correctness and FPS are not verified or promised.
Qt libraries/plugins are included in the local archive, but system libraries
remain dependencies. Compatibility with other Linux distributions is untested.
Keep the whole extracted runtime directory together.

See the archive's `LINUX-README.txt`, `test-results.txt`, and
`linux-portability.patch` for build-specific details. Preserve `LICENSE` and
third-party attribution when redistributing.
