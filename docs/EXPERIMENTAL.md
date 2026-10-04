# U59 integration release (Windows x64)

This release builds on the U59 renderer and the Demon's Souls changes of the previous U59 release.

## New in this update

- Unused occlusion queries are reset in batches (`KYTY_OCCLUSION_RESET_BATCH=1` in the preset), up to 64 per command
  instead of one each, with every query boundary, result copy and reduction unchanged. Measured with the switch
  toggled in the same process: Go-Go Archipelago +1.5% to +1.9% fps, Bathhouse Battle +2.6%; Crash Site and the Sky
  Garden unchanged.
- New shader pipelines are prepared ahead on worker threads (`KYTY_PIPELINE_PREFETCH=1` and
  `KYTY_PIPELINE_PREFETCH_PROGRAMS=1` in the preset). Draws still wait for their exact pipeline, so nothing is drawn
  differently; the work just starts earlier. With an empty shader cache the command processor stalled 68-70 s
  instead of 87-89 s along the Sky Garden route and 57-59 s instead of 76 s along the Creamy Canyon route, and the
  slowest 1% of Creamy Canyon frames took about 100-130 ms instead of 230-245 ms. With a warm cache fps is unchanged.
- Small uploads use their own ring buffer (`KYTY_RAM_SMALL_UPLOAD_RING=1` in the preset): about 430-540 MB less RAM
  (private bytes), about 385 MB less shared GPU memory and 60-150 MB less VRAM in the Sky Garden and Go-Go
  Archipelago, with fps differences within run-to-run noise.
- Shader storage writes are tagged again at command emission when binding preparation submitted their original
  recording (`KYTY_SHADER_WRITE_RETICK=1` in the preset). This prevents a side readback from publishing old
  contents before the actual GPU write. A deterministic Vulkan test reproduces the race; whether it fixes the
  reported RTX 40/50 hangs remains unproven. `KYTY_SHADER_WRITE_RETICK=0` restores the previous behavior.
  - The logs of RTX 50 PCs that freeze at the title screen -> galaxy map step all end in this readback path:
    the game's render threads re-read small flags a shader writes every frame. If your RTX 40/50 PC freezes
    there, try this release with its bundled preset.
- The hang watchdog is on only for NVIDIA RTX 50 GPUs (`KYTY_HANG_WATCHDOG=auto` in the preset; `=1` turns it on for any
  GPU, `=0` off). When the picture stops for 5 seconds it writes `_HangTrace\watchdog-*\watchdog.txt` beside the
  emulator: what every thread is doing and waiting for. If your RTX 50 PC still freezes, please send that file.
  On other GPUs its hot paths stay off; see `docs/HANG-WATCHDOG.md`.
- The storage-write fix costs nothing measurable on one PC (RTX 3090, Ryzen 9 7950X3D): switched off and on in the
  same process, Sky Garden -0.14% +/- 0.50%, Creamy Canyon -0.13% +/- 0.66%, identical VRAM.
- kyty_emulator is built with a new PGO profile recorded in Astro Bot with this source. The shader translation is
  unchanged, so int5's program caches stay valid.

From int5 (`u59-windows-20261003-int5`):

- Batched occlusion queries, on in the bundled `u59-preset.json` (`KYTY_OCCLUSION_BATCH=1`,
  `KYTY_OCCLUSION_SLOTS=16384`):
  - Astro Bot's Creamy Canyon (the snow level in the Gorilla Nebula) draws about 4,700 small depth-only boxes per
    frame, each inside its own occlusion query, to find out what is hidden. The emulator reduced the result of every
    query with its own GPU dispatches between full barriers: about 80 of the level's 98 ms of GPU time per frame,
    which held it at about 8 fps on an RTX 3090. The results of a command buffer are now reduced together, in one
    dispatch right before the command buffer is submitted. The game still reads every result at the same point.
  - Measured at the Creamy Canyon start view (see "Measured" below): 22.0 fps instead of 8.3 fps with the previous
    release. The Sky Garden start view, with about 4 such boxes per frame, runs at the same frame rate as before.
  - `KYTY_OCCLUSION_BATCH=verify` computes every result both ways and compares them. It found no difference in
    12.1 million results on the Creamy Canyon route and in more than 100,000 on the Sky Garden route.
    `KYTY_OCCLUSION_SLOTS` is the number of 256-byte result slots (default 1,024); more slots let the command
    processor run further ahead of the GPU.
- Swapchain fixes from upstream pull request #1001 by Ekt0re (not yet merged upstream):
  - A minimized or zero-sized window no longer stops the emulator; presentation pauses until the window is restored.
  - An image that the driver reports as suboptimal is presented before the swapchain is recreated, so its semaphore
    is not reused while it is still signaled.
  - In a test that moved, resized, maximized, minimized and restored the Astro Bot window at the Sky Garden, this
    build kept running and rendered at the same frame rate afterwards. The previous release also survived that test
    on the RTX 3090 test PC, so the crashes these fixes address were not reproduced there.
- Larger fault-ahead windows (`KYTY_FAULT_AHEAD_ADAPT`, on by default, no preset entry needed):
  - The emulator notices the game's writes to memory it shares with the GPU by write-protecting those pages and
    catching the first write fault. That fault now makes a 256 KiB window around it writable instead of 32 KiB, and
    512 KiB or 1 MiB on PCs where the emulator measures slow protection calls. Linux with `mprotect` starts at 1 MiB.
  - On the RTX 3090 / Ryzen 9 7950X3D PC (Windows 11), Astro Bot's Sky Garden took about 1,700 such faults per frame
    with 32 KiB windows and about 300 with 256 KiB. Frame rate +1.6% (95% interval +0.7% to +2.5%) in a launch that
    switched between both settings every few seconds.
  - The gain is large where faults are slow. A Linux user measured about 83 us per fault (this PC: a few us) and
    about 8 fps at the Sky Garden. With each fault and protection call slowed down on this PC to resemble that log,
    the Sky Garden ran at about 9 fps with 32 KiB windows and 27 fps with this build's default.
  - `"KYTY_FAULT_AHEAD_ADAPT": "0"` in `u59-preset.json` restores the 32 KiB windows.
- New log lines about this write tracking, for performance reports:
  - at startup, `Kyty platform:` (Windows version, hypervisor, memory integrity, whether `ntdll` entry points are
    hooked, and modules loaded from outside Windows and the emulator folder; on Linux the kernel,
    `vm.max_map_count` and userfaultfd support) and `Kyty fault cost:` (a benchmark of about 1 ms: the cost of one
    write fault and of one protection call);
  - while the game runs, every 60 s, `Kyty fault cost:` with the faults per frame and their average cost, and
    `Kyty BDA passes:`. `KYTY_FAULT_COST_LOG=<seconds>` changes the period; `0` turns these lines off.
- Linux: optional userfaultfd write-protection for the same tracking (`KYTY_UFFD_WP=1`, off by default and not yet
  tested in a game), and recommended system settings; see `docs/LINUX-U59.md` in the source.
- `kyty_emulator.exe` is built with a new PGO profile, recorded in Astro Bot with this source (`tools/pgo/`).

## Earlier in U59 integration

- Lower VRAM use, on in the bundled `u59-preset.json`:
  - `KYTY_FUNCTION_ARRAY_SHRINK`: the shader recompiler emulates LDS in vertex and pixel shaders with an array of
    8,192 dwords (32 KiB) per shader invocation. The NVIDIA driver reserves local memory for such an array for every
    thread the GPU can keep resident and keeps it until the game exits: about 3.9 GiB on an RTX 3090 for the eight
    Astro Bot pixel shaders that use it. These arrays are now shrunk to the elements the shader can reach before the
    driver sees the shader (32-224 dwords in Astro Bot; Demon's Souls has three such shaders).
  - `KYTY_VRAM_GC_BUDGET`: textures and buffers are collected against the VRAM budget as it is while the game runs,
    and textures are aged by frames, instead of by thresholds fixed at startup.
  - Measured on an RTX 3090 (24 GB) at the Astro Bot Sky Garden start view: dedicated VRAM 6.3 GB instead of 9.4 GB,
    and a peak since the start of 7.5 GB instead of 12.4 GB, with no measurable change in frame rate. With another
    process holding 12 GiB or 16 GiB of the card from boot, Astro Bot reaches the Sky Garden and runs at 32-33 fps
    there. Tested on NVIDIA only.
- Release builds compile from the PGO profile's source path (`C:\kyty-src`). clang-cl names functions in anonymous
  namespaces after a hash of the source path, so earlier GitHub builds missed the profile for all of them.
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

Astro Bot (PPSA21567); RTX 3090, Ryzen 9 7950X3D; 1920x1080 output at a 120 Hz vblank. Timed runs on one PC,
alternating between the two builds, each with its bundled preset:

| Start view | Previous release (`u59-windows-20261003-int4b`) | This release (a local build of the same source) |
|---|---|---|
| Creamy Canyon, two runs each | 8.3 fps (8.29-8.34) | 22.0 fps (21.94-22.01) |
| Sky Garden, three runs each | 33.5 fps (32.8-33.9) | 33.9 fps (33.1-34.5) |

- Creamy Canyon: command-processor time per frame fell from 102 ms to 42 ms, GPU time from 96 ms to 23 ms.
- Sky Garden: the frame-rate difference is within the spread between runs. A second block, taken while the remote
  session to the PC was disconnected, read 36.9 fps (two runs) and 36.0 fps (three runs). GPU time per frame was
  about 0.9 ms higher with this release in both blocks; at the Sky Garden the GPU waits for work 7-9 ms per
  frame, so the frame rate does not follow it. Dedicated VRAM stayed at 6.2-6.5 GB with both builds.
- Demon's Souls boots to its menu with the preset (Language Select at 60 fps). Its frame rate with this build was
  not measured.

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
- `KYTY_OCCLUSION_BATCH`, the larger fault-ahead windows and the swapchain fixes were tested on one PC (RTX 3090,
  Windows 11). The limits at which the windows grow come from that PC, a simulation of slow faults on it and one
  Linux user's log. `"KYTY_OCCLUSION_BATCH": "0"` and `"KYTY_FAULT_AHEAD_ADAPT": "0"` in `u59-preset.json` restore
  the previous behaviour.
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
- Add `-DKYTY_EMULATOR_IPO=ON` and `-DKYTY_PGO_USE=C:/kyty-src/tools/pgo/u59-int7-sg-1.profdata`. Without
  `KYTY_PGO_USE` the build works, but without the profile's speedup.
- Build `launcher` and `kyty_emulator`, install to `_Build/windows/install`, and copy `tools/u59-preset.json` beside
  `launcher.exe`.

The Windows release is produced by a tagged GitHub Actions build. Original licenses and credits remain intact.
Personal handoffs, local editor configuration, captures, saves and caches are excluded. Historical results and their
limitations are described in [CHANGES-U59.md](CHANGES-U59.md).
