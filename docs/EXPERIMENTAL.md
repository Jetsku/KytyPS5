# U59 integration release (Windows x64)

This release builds on the U59 renderer and the Demon's Souls changes of the previous U59 release.

## New in this update

- Draw-run batching is on by default (`KYTY_DRAW_RUN`, `KYTY_DRAW_RUN_ACQUIRE` and `KYTY_DRAW_RUN_PUSH` in the
  bundled `u59-preset.json`): a draw that continues the previous draw's structure (targets, programs, textures,
  samplers) is committed as a delta. Measured at the Astro Bot Sky Garden start view in one same-process A/B
  launch: command-processor time -2.4%, frame rate +1.9%. Its verify mode found no differences.
- Possible fixes for device-lost (masterSemaphore) GPU hangs, the `VK_ERROR_DEVICE_LOST` crashes reported on RTX 40
  and 50 series cards. None of them is confirmed to fix those crashes. On the test PC (RTX 3090), turning all of them
  off did not change the Astro Bot frame rate.
  - `S_MEMREALTIME` reads the GPU clock (`VK_KHR_shader_clock`) instead of returning a fixed placeholder, so guest
    waits timed by it end. On a GPU without the device clock, the subgroup clock or the placeholder is used, with a
    console warning. `KYTY_REALTIME_CLOCK=0` restores the placeholder.
  - A DPP lane read from a lane that EXEC disables keeps the destination value, as on the PS5, and DPP row scans
    read by `v_readlane` become native subgroup reductions. `KYTY_DPP_SKIP_INACTIVE=0` and `KYTY_LANE_REDUCTIONS=0`
    turn these off.
  - A shader whose control flow runs as a dispatcher loop (the fallback for control flow that cannot be structured)
    leaves that loop after at most 4096 block transitions. `KYTY_DISPATCHER_CAP=<n>` sets the limit (`0`: no limit);
    if a game draws something wrong with the limit, `0` removes it.
  - These three ports come from Senaxx's fork.
- Optional device-fault diagnostics for GPU crashes; see "If the GPU crashes" below.
- Upstream KytyPS5 changes (second sync): DualSense speaker and haptic audio over Bluetooth; in-game keys (by
  default 1, 2 and 3) that cycle the controller's speaker volume, vibration and trigger-effect intensity; a fix for
  audio popping and time-stretching; Hades II startup and colour-clear fixes; a `NetResolverAbort` stub; and more
  shader opcodes (64-bit ALU and atomics, FP64 equality comparisons, 32-bit image compare-swap atomics, BVH
  intersections).
- Text that games draw with the system font (for example Astro Bot's galaxy-map prompts) uses the bundled Roboto
  font again (`3rdparty/tracy/profiler/src/font/` in the package) instead of a blocky 8x16 fallback.
- `kyty_emulator.exe` is built with a new PGO profile, recorded in Astro Bot (`tools/pgo/`).
- New flags, off by default and left off by the preset: `KYTY_CP_CPU_ONLY_QUERY`, `KYTY_CP_BINDING_MEMO_PREFETCH`,
  `KYTY_CP_BINDING_HOT_MEMO`, `KYTY_BUFFER_REFRESH_FUSION` and `KYTY_CPU_COPY_PAGE_SKIP` (no measurable effect in
  same-process A/B launches), and `KYTY_REGISTERED_SHADER_CODE` (not measured).

## Earlier in U59 integration

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
Three timed runs of each build on one PC, alternating, each with its bundled preset:

| Build | Frame rate: mean (range) | Dedicated VRAM |
|---|---|---|
| Previous release (`u59-windows-20261002-int2`) | 34.4 fps (33.3-36.2) | 9.6 GB |
| This release (a local build of the same source) | 35.7 fps (34.9-37.3) | 9.4 GB |

- The frame-rate difference is within the spread between runs.
- The VRAM figure is the mean over the timed minute. A spike to about 13 GB can occur for a moment while the galaxy
  map streams.
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

- The first launch of each game compiles its shaders again. Program caches from older builds are not reused, so the
  first load is slow and the game stutters until the cache fills.
- The PGO profile comes from Astro Bot only. Other games run with code laid out for Astro Bot.
- The upstream controller and audio changes were not tested by hand.
- The preset also sets `KYTY_SRT_VARIANT_READS=1`, which Demon's Souls needs.

## Building from source

Follow the Windows requirements in [README](../README.md#build-requirements-windows) and clone recursively. Configure
Release with Ninja in an x64 Visual Studio developer shell, as in `.github/workflows/u59-windows-release.yml`:

- Use clang-cl, lld-link and llvm-lib from LLVM 22.1.3: the profile needs the compiler version that recorded it.
- Add `-DKYTY_EMULATOR_IPO=ON` and `-DKYTY_PGO_USE=<checkout>/tools/pgo/u59-int3-sg-1.profdata`. Without
  `KYTY_PGO_USE` the build works, but without the profile's speedup.
- Build `launcher` and `kyty_emulator`, install to `_Build/windows/install`, and copy `tools/u59-preset.json` beside
  `launcher.exe`.

The Windows release is produced by a tagged GitHub Actions build. Original licenses and credits remain intact.
Personal handoffs, local editor configuration, captures, saves and caches are excluded. Historical results and their
limitations are described in [CHANGES-U59.md](CHANGES-U59.md).
