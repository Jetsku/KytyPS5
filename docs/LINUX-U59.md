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
