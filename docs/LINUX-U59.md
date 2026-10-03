# U59 + Demon's Souls: Linux x86-64 build

> Written for the 2026-09-30 release (`u59-windows-20260930-demons`); main has moved on since then.

This branch builds the source of `u59-windows-20260930-demons`
(commit `3ea4c7562ee5c6cdc8009367d34770fc50eb1d6e`) with two Linux portability fixes.
It does not add U60 or RT work, game files, compatibility cheat files, firmware,
saves or caches. This is an unofficial local build, not an upstream Linux release.

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
./_Build/linux/install/launcher
```

The launcher reads the adjacent preset automatically. Direct `kyty_emulator`
CLI calls do not read it automatically. Add your legally obtained game folders
in launcher settings and configure display/per-game settings there.

## Testing branch `test/fault-robust` (slow guest write tracking on Linux)

Kyty notices the game's writes to memory it shares with the GPU by protecting
those pages and catching the write fault. On Linux each fault is a signal plus
`mprotect` calls, and every `mprotect` takes the process-wide memory-map lock,
so with many game threads writing at once the faults queue behind each other.
A Linux user measured about 83 us per fault (Windows: a few us) and about 8 fps
at the Astro Bot Sky Garden. This branch:

- opens a larger window around each write fault, scaled automatically to the
  fault cost the emulator measures on the PC (`KYTY_FAULT_AHEAD_ADAPT`, on by
  default; nothing to set). With a slow-fault simulation on Windows: 9 -> 21 fps;
- adds optional userfaultfd write-protection for guest write tracking
  (`KYTY_UFFD_WP=1`, off by default; untested in a game on Linux so far). It
  avoids `mprotect` for the per-frame protection changes: in a WSL2 benchmark
  with 15 writing threads a fault cost about 11 us instead of about 230 us;
- logs the measured costs: a `Kyty platform:` line and `Kyty fault cost:` lines
  (at startup and every 60 s) with the kernel, the memory-map limit, whether
  userfaultfd is in use, and the per-fault and per-call costs.

To test it:

1. Kernel 6.4 or newer is recommended (`uname -r`). `KYTY_UFFD_WP` needs at
   least 5.19; 6.4 adds it for all guest memory.
2. Raise the memory-map limit (protected pages split the guest mappings; at the
   default limit of 65530, `mprotect` fails and Kyty stops):
   `sudo sysctl -w vm.max_map_count=1048576` (until reboot; to keep it, put
   `vm.max_map_count=1048576` in `/etc/sysctl.d/99-kyty.conf`).
3. Build this branch as above, then run the Astro Bot Sky Garden twice:
   - A: with the preset as it is;
   - B: with `"KYTY_UFFD_WP": "1"` added to `u59-preset.json`. The log should
     then say `guest write tracking with userfaultfd write-protection
     (KYTY_UFFD_WP=1): on`. If it says `unavailable`, the kernel is too old or
     userfaultfd is blocked, and everything runs as in A.
4. Report the fps of each run with the `Kyty platform:` and `Kyty fault cost:`
   lines from the terminal or log.

The launcher's "AMD CPU patch" emulates the `VRSQRTPS` instruction with a trap
on every execution. On one Zen 3 CPU it raised the Sky Garden from about 8 to
17 fps; on a Zen 4 CPU it lowered it from 34 to 18 fps. Try both settings and
report which is faster.

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
