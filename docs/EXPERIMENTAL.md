# U59 integration release (Windows x64)

This release builds on the U59 renderer and the Demon's Souls changes of the previous U59 release.

## New in this update

- Lower VRAM use, on in the bundled `u59-preset.json`:
  - `KYTY_FUNCTION_ARRAY_SHRINK`: the shader recompiler emulates LDS in vertex and pixel shaders with an array of
    8,192 dwords (32 KiB) per shader invocation. The NVIDIA driver reserves local memory for such an array for every
    thread the GPU can keep resident and keeps it until the game exits: about 3.9 GiB on an RTX 3090 for the eight
    Astro Bot pixel shaders that use it. These arrays are now shrunk to the elements the shader can reach before the
    driver sees the shader (32-224 dwords in Astro Bot; Demon's Souls has three such shaders). The shader translation
    and the program cache are unchanged.
  - `KYTY_VRAM_GC_BUDGET`: textures and buffers are collected against the VRAM budget as it is while the game runs,
    and textures are aged by frames, instead of by thresholds fixed at startup.
  - Measured on an RTX 3090 (24 GB) at the Astro Bot Sky Garden start view (see "Measured" below): dedicated VRAM
    6.3 GB instead of 9.4 GB, and a peak since the start of 7.5 GB instead of 12.4 GB. With both flags set to `"0"`,
    the same build used as much VRAM as the previous release. A first launch with empty caches peaked at about 7.5 GB
    instead of 12.4 GB with the flags off. The frame rate did not change measurably.
  - Smaller cards, imitated on the RTX 3090 by another process that holds 12 GiB or 16 GiB of its 24 GiB from boot
    (about what a 12 GB card leaves, and less than an 8 GB card leaves): Astro Bot now reaches the Sky Garden and runs
    at 32-33 fps there. With the previous release's preset it got stuck in the galaxy map in both cases. With 16 GiB
    held, about 0.3 GB spilled to shared memory. Tested on NVIDIA only.
- Release builds compile from the PGO profile's source path. clang-cl names functions in anonymous namespaces after a hash of the source
  path, so earlier GitHub builds missed the profile for all of them (2,499 of the 17,426 functions in the previous
  release's profile). The release now compiles from the path the profile was recorded at (`C:\kyty-src`). This did
  not change the Astro Bot frame rate measurably.
- `kyty_emulator.exe` is built with a new PGO profile, recorded in Astro Bot with this source (`tools/pgo/`).

## Earlier in U59 integration

- Draw-run batching (`KYTY_DRAW_RUN`, `KYTY_DRAW_RUN_ACQUIRE` and `KYTY_DRAW_RUN_PUSH`, on in the preset): a draw that
  continues the previous draw's structure (targets, programs, textures, samplers) is committed as a delta
  (command-processor time -2.4%, frame rate +1.9% in a same-process A/B launch at the Astro Bot Sky Garden start view).
- Possible fixes for device-lost (masterSemaphore) GPU hangs, the `VK_ERROR_DEVICE_LOST` crashes reported on RTX 40
  and 50 series cards, ported from Senaxx's fork. None of them is confirmed to fix those crashes.
  - `S_MEMREALTIME` reads the GPU clock (`VK_KHR_shader_clock`) instead of returning a fixed placeholder.
    `KYTY_REALTIME_CLOCK=0` restores the placeholder.
  - A DPP lane read from a lane that EXEC disables keeps the destination value, and DPP row scans read by
    `v_readlane` become native subgroup reductions. `KYTY_DPP_SKIP_INACTIVE=0` and `KYTY_LANE_REDUCTIONS=0` turn these
    off.
  - A shader dispatcher loop ends after at most 4096 block transitions. `KYTY_DISPATCHER_CAP=<n>` sets the limit
    (`0`: no limit); if a game draws something wrong with the limit, `0` removes it.
- Optional device-fault diagnostics for GPU crashes; see "If the GPU crashes" below.
- Upstream KytyPS5 changes (second sync): DualSense speaker and haptic audio over Bluetooth; in-game keys (by
  default 1, 2 and 3) that cycle the controller's speaker volume, vibration and trigger-effect intensity; a fix for
  audio popping and time-stretching; Hades II fixes; a `NetResolverAbort` stub; and more shader opcodes.
- Text that games draw with the system font uses the bundled Roboto font (`3rdparty/tracy/profiler/src/font/` in the
  package).
- Flags that are off by default and left off by the preset: `KYTY_CP_CPU_ONLY_QUERY`,
  `KYTY_CP_BINDING_MEMO_PREFETCH`, `KYTY_CP_BINDING_HOT_MEMO`, `KYTY_BUFFER_REFRESH_FUSION`,
  `KYTY_CPU_COPY_PAGE_SKIP` and `KYTY_REGISTERED_SHADER_CODE`.
- Command-processor work, behind flags that the bundled `u59-preset.json` turns on:
  - a fix for draw-preparation workers that stopped waking (`KYTY_DRAW_PREP_COLD_TOKEN`);
  - cheaper per-draw commits (`KYTY_CP_COMMIT=all`);
  - descriptor sets written on the recorder thread, push-descriptor and metadata-clear memos, and fewer GPU progress
    queries (`KYTY_RECORDER_DESCRIPTOR_SETS`, `KYTY_PUSH_SHADOW_FRESH_SKIP`, `KYTY_META_CLEAR_MEMO`,
    `KYTY_PENDING_REFRESH_US`).
- Lower VRAM use, also behind preset flags:
  - sparse residency for partially resident textures and for the BDA page table;
  - idle limits for the native image pool and the tiler scratch pool;
  - images unused for 600 frames are freed.
- Upstream KytyPS5 changes up to the first sync: controller, audio and compatibility fixes, and the layered VideoOut
  presenter.

## Measured

Astro Bot (PPSA21567), Sky Garden start view; RTX 3090, Ryzen 9 7950X3D; 1920x1080 output at a 120 Hz vblank.
Six timed runs of each build on one PC, alternating, each with its bundled preset:

| Build | Frame rate: mean (range) | Dedicated VRAM: mean | Peak since start: mean (range) |
|---|---|---|---|
| Previous release (`u59-windows-20261002-int3`) | 33.6 fps (32.2-35.8) | 9.4 GB | 12.4 GB (9.8-13.1) |
| This release (a local build of the same source) | 33.0 fps (31.7-33.8) | 6.3 GB | 7.5 GB (7.3-7.8) |

- The frame-rate difference is within the spread between runs. Three runs of the same build with
  `KYTY_FUNCTION_ARRAY_SHRINK` and `KYTY_VRAM_GC_BUDGET` set to `"0"` read 33.1 fps and 9.4 GB.
- The dedicated VRAM mean is over the timed minute; the peak is the highest two-second sample since the start. The
  previous release reaches its peak for a moment while the galaxy map streams.
- With 12 GiB of the card held by another process from boot, this build reached the Sky Garden at 33.1 fps, with a
  peak of 7.4 GB and nothing spilled to shared memory.
- Demon's Souls boots to its menu with the preset. Its frame rate with this build was not measured.

## Launching

Extract the archive and open `launcher.exe` directly. When `u59-preset.json` is beside the executable, the launcher
applies its environment and clears inherited KYTY/TRACY variables. The archive contains no `Kyty.ini`: the launcher uses
the shared settings file `C:\ProgramData\Kyty\Kyty.ini`, so existing game directories and per-game settings stay
available. If an older archive left a `Kyty.ini` beside the launcher, move it aside. No game files, saves, caches or
patches are distributed.

Optional: `"KYTY_PRESENT_BOX_DOWNSCALE": "1"` in `u59-preset.json` presents the 4K frame with a two-texel box filter
when the window is between half and full size (for example 2560x1440), which removes a fine one-pixel stipple
the default blit leaves. It is off by default; other window sizes are unaffected.

## If the GPU crashes

Add `"KYTY_DEVICE_FAULT_DIAGNOSTICS": "1"` to `u59-preset.json` and run the game until the crash happens again. With
it, the emulator enables the driver's fault reporting (`VK_EXT_device_fault`, and on NVIDIA the diagnostic checkpoints
and resource tracking). When the device is lost, the console prints a block that starts with `--- Device loss` and
contains the fault addresses, the driver's fault description and the last GPU checkpoints with the vertex, pixel and
compute shader hashes of the draws in flight. When the driver returns binary fault data, the emulator also writes it
to `_device_fault.nv-gpudmp` in its working folder. Send that console text (from the `--- Device loss` line on), the
`.nv-gpudmp` file if there is one, the GPU model and the driver version. The diagnostics can cost speed, so remove the
line again afterwards.

## Caveats

- Program caches are reused only from a build with the same shader translation. Otherwise the first launch of each
  game compiles its shaders again, so the first load is slow and the game stutters until the cache fills. With
  `KYTY_FUNCTION_ARRAY_SHRINK`, the driver also compiles the pipelines of the shaders whose arrays shrink once more.
- `KYTY_FUNCTION_ARRAY_SHRINK` and `KYTY_VRAM_GC_BUDGET` were tested on NVIDIA only. `"0"` in `u59-preset.json` turns
  either off.
- The PGO profile comes from Astro Bot only. Other games run with code laid out for Astro Bot.
- The upstream controller and audio changes were not tested by hand.
- The preset also sets `KYTY_SRT_VARIANT_READS=1`, which Demon's Souls needs.

## Building from source

Follow the Windows requirements in [README](../README.md#build-requirements-windows) and clone recursively. Configure
Release with Ninja in an x64 Visual Studio developer shell, as in `.github/workflows/u59-windows-release.yml`:

- Use clang-cl, lld-link and llvm-lib from LLVM 22.1.3: the profile needs the compiler version that recorded it.
  Standard-library code only matches it with the same MSVC headers (14.51, Visual Studio 2026 18.10).
- Configure from `C:\kyty-src`, a directory junction to the checkout (`mklink /J C:\kyty-src <checkout>`, then
  `cmake -S C:\kyty-src -B <new build directory>`). clang-cl names functions in anonymous namespaces after a hash
  of the source path, so a build from any other path loses their part of the profile.
- Add `-DKYTY_EMULATOR_IPO=ON` and `-DKYTY_PGO_USE=C:/kyty-src/tools/pgo/u59-int4-sg-1.profdata`. Without
  `KYTY_PGO_USE` the build works, but without the profile's speedup.
- Build `launcher` and `kyty_emulator`, install to `_Build/windows/install`, and copy `tools/u59-preset.json` beside
  `launcher.exe`.

The Windows release is produced by a tagged GitHub Actions build. Original licenses and credits remain intact.
Personal handoffs, local editor configuration, captures, saves and caches are excluded. Historical results and their
limitations are described in [CHANGES-U59.md](CHANGES-U59.md).
