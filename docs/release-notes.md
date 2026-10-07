# KytyPS5 U59 int17 pre-release — faster heavy levels, performance-mode occlusion, many fixes

> **Pre-release.** This is int16.1 plus the newest speed work and fixes (draw batching in the clock tower and Ape
> Escape, GPU-built indirect mesh draws, background ray tracing pipelines, and ports from chenxiao07's fork). If
> anything works worse than in int16.1, please report it (with your GPU and CPU model) and use int16.1.

## What's new in int17

- **Galaxy map fixed:** the red trails, pillars and blocky red areas around the volcano (the hidden world) and the
  red band at the bottom of the screen are gone. The game fast-clears one of its 64-bit render targets every frame and
  Kyty decoded that clear with a 32-bit decoder, so the clear was dropped and old frames piled up. The same fix covers
  R16F, RG16F, R16, RG16, R8, RG8 and B10G11R11 register clears (`KYTY_CLEAR_REGISTER_WIDE=0` restores the old
  behaviour). This bug was also in upstream Kyty.

- **Performance mode by default: occlusion queries off.** The game asks the GPU, thousands of times per frame,
  whether objects are hidden behind others. Answering those queries accurately costs about half the frame rate in
  the heaviest levels. int17 now answers "visible" to all of them by default (`KYTY_GPU_OCCLUSION=0`, as other
  Kyty forks do), which roughly doubles the frame rate in the snow level, the clock tower and Ape Escape.
  On the test PC (RTX 3090, Ryzen 9 7950X3D; a test build without the release build's profile optimization, which
  adds about 15%): snow 45 fps, clock tower 35, Ape Escape 38, Sky Garden 31, volcano 59.
  Possible side effects: the game may draw objects or effects that it would normally hide behind walls (rarely
  visible, and it can cost GPU time where a lot is hidden).
- **Accurate occlusion toggle:** Global settings > Graphics & display > "Accurate occlusion queries (slower; fixes
  rare visibility issues)" turns the accurate mode back on (it passes `--gpu-occlusion on`, which overrides the
  preset). Use it if you see objects through walls or other visibility problems, and please report where. The
  accurate mode is also faster than in int16: render passes are no longer split at every query, draws keep running
  in batches across the game's occlusion proxies, and it includes the volcano lava fix below.
  Same PC and build, accurate mode: snow 25 fps, clock tower 22, Ape Escape 19 (the int16 pre-release built the same
  way: about 17.5, 13 and 13).
- **Faster indirect draws in both modes:** indirect draws are prepared by the worker threads and continue draw
  batches, and indirect mesh draws are built on the GPU instead of waiting for the GPU to hand their arguments back.
- **Volcano lava fixed (accurate occlusion mode):** with the camera tilted down in the volcano level, the lava lake
  disappeared (flat orange fog). The emulator paired the game's occlusion queries by a wrong address rule; it now pairs them by address, which
  also keeps the occlusion gate on in Sky Garden.
- **Hang watchdog no longer kills the game:** on RTX 50 cards (where the watchdog is on by default) a 5 s stall could
  end the emulator with "Unhandled host exception ... RtlVirtualUnwind". The watchdog's stack dump is now guarded.
- **Ray tracing shaders build in the background:** the large software ray tracing pipelines no longer freeze the game
  on first use; their effect starts a moment later instead. If a build fails, ray tracing turns itself off for the
  session instead of stopping the emulator.
- **Out of video memory no longer ends the emulator:** an image that does not fit goes to system memory (slower, the
  console prints "Vulkan: video memory full ..."), and a buffer that does not fit is retried.
- **Galaxy map / file fix:** Astro Bot file ids are assigned one per file; six pairs of game files shared an id and one
  of each pair could be read in place of the other (one of them a galaxy-map icon).
- **Audio:** sounds the game sends to the DualSense speaker now play at 30% on the PC speakers (they were up to 19 dB
  louder than music and effects). Four new volume sliders (master, game sound, music, controller speaker on PC) are in
  the launcher's global Audio settings. AudioPropagation (3D sound) is implemented as an empty scene.
- **Ported from chenxiao07's fork:** protection kept on remapped memory views (fixes a "frozen model" class of bugs),
  small fibers on their own stacks (Demon's Souls job-thread crashes), indirect draws' start instance (Demon's Souls
  missing pillars and rock faces), faster guest memory search while streaming, a clean process exit after a fatal
  error, and faster release shader translation.
- **"AMD CPU patch" renamed** to "Intel CPU compatibility (AMD instruction patch)" in the launcher's Compatibility
  settings. It is the same setting (saved configurations and `--amd-cpu` keep working): on Intel CPUs it emulates the
  PS5's AMD-only instructions (EXTRQ/INSERTQ) and AMD's `VRSQRTPS` results. It is not needed on AMD CPUs. The console
  lines now read "AMD instruction patch for Intel CPUs: ...".

### New defaults and how to turn each off

Set these in the launcher's environment / preset (`tools/u59-preset.json` style, `NAME=value`). Use the off value if
you suspect a change, and please report what it fixed.

| Switch | Default | Off / old behaviour | What it does |
| --- | --- | --- | --- |
| `KYTY_GPU_OCCLUSION` (preset) | `0` (performance mode) | `1`, or the launcher's "Accurate occlusion queries" (`--gpu-occlusion on`) | Occlusion queries answered "visible" instead of counted on the GPU |
| `KYTY_OCCLUSION_SPLIT` | on | `0` | An occlusion query inside a render pass ends only the query, not the pass |
| `KYTY_DRAW_RUN_QUIET_OPS` | on | `0` | Occlusion dumps and CPU-timeline writes no longer end a draw batch |
| `KYTY_DRAW_PREP_INDIRECT` | on | `0` | Worker threads prepare single indirect draws |
| `KYTY_DRAW_RUN_INDIRECT` | on | `0` | Prepared indirect draws continue draw batches |
| `KYTY_DRAW_RUN_QUIET_SYNTHETIC` | on | `0` | In performance mode, the always-visible query results no longer end draw batches (about +1.5%) |
| `KYTY_DEPTH_FEEDBACK_LAZY` | on | `0` | Depth feedback handling only for draws that sample the depth image |
| `KYTY_NATIVE_INDIRECT_MESH` | on | `0` | Indirect mesh draws built on the GPU (no CPU readback wait) |
| `KYTY_RT_PIPELINE_ASYNC` | on | `0` | Ray tracing pipelines built in the background (`KYTY_RT_PIPELINE_WAIT_MS`, default 250) |
| `KYTY_RT_SOFTWARE` | `auto` (preset) | `0` | Software ray tracing (`0` skips the RT shaders; only with the patches on) |
| `KYTY_RT_FUNCTION` | `1` | `0` | Ray tracing node test as one shared shader function (smaller shaders) |
| `KYTY_IMAGE_SYSMEM_FALLBACK` | on | `0` | Images that do not fit in video memory go to system memory |
| `KYTY_REMAP_TRACKER_PROTECTION` | on | `0` | Restore write tracking on remapped memory views |
| `KYTY_FIBER_GUEST_STACK` | small fibers | `0` (`all` = every fiber) | Fibers under 256 KiB run on their own guest stacks |
| `KYTY_START_INSTANCE_SGPR` | on | `0` | Indirect draws pass their start instance to the shader |
| `KYTY_APR_HASHED_IDS` | off | `1` restores the old hashed ids | One file id per file path |
| `KYTY_FATAL_EXIT_PROCESS` | off | `1` restores the old exit path | Fatal errors end the process with TerminateProcess |
| `KYTY_IR_VALIDATE` | off in release | `1` runs it | Shader IR validation |
| `KYTY_AUDIO_DIAG` | on | `0` | Audio port open/close lines in the console (256 lines max) |
| `KYTY_AUDIO_PROPAGATION_RENDER` | direct sound | `silent` | AudioPropagation sources play their direct sound |
| Audio volumes | master/main/music 100%, pad speaker on PC 30% | launcher sliders, or `KYTY_AUDIO_MASTER_VOLUME`, `KYTY_AUDIO_MAIN_VOLUME`, `KYTY_AUDIO_MUSIC_VOLUME`, `KYTY_AUDIO_PAD_SPEAKER_ON_MAIN_VOLUME` (percent, 0-200) | |

Optional (off by default): `KYTY_WIDE_FILL_CLEAR=1`, `KYTY_GUEST_STORAGE_REPEAT=1`, `KYTY_READBACK_MERGE_GAP_KB=64`,
and the test tool `KYTY_VRAM_LIMIT_MB=<MiB>` (imitates a
smaller GPU). The accurate-mode switches above (`KYTY_OCCLUSION_SPLIT`, `KYTY_DRAW_RUN_QUIET_OPS`) only matter with accurate
occlusion on.

## From int16.1

- **Level title text fixed:** the level name shown when Astro Bot enters a level is drawn as in int14 again.

## From int16

- **Astro Bot without the "non RT patch":** the game's own tiled lighting, GI probes and ray traced shadows now run,
  so the two patches are no longer needed. Without them the robots used to be black and the lighting missing, because
  the lighting shaders contain ray-tracing instructions and were skipped. Those instructions now run in software on
  any GPU (NVIDIA, AMD and Intel).
- **How to play without the patches:** open the launcher, select Astro Bot, open the cheats window and untick
  "Select the existing non-tiled deferred-lighting renderer" and "Disable GI probes and lighting shaders". If you
  see problems, tick them again; this build also runs fine with them.
- **Speed:** on the test PC (RTX 3090, Ryzen 9 7950X3D) the full graphics cost about 3% in Sky Garden (about 33.6 fps
  against 34.8 fps with the patches) and practically nothing at the snow level start (21.1 against 21.2 fps).
- **First launch:** the first start after an update builds the large lighting shaders, which can freeze the picture
  for up to about 20 seconds once.

## Also in int15 and int15.1

- No more "unsupported sampled depth image" stops on fast GPUs, no crash when leaving extra levels in Sky Garden,
  about 20 upstream shader fixes, a Demon's Souls fix for an unknown image format, and thread priority changes.
- From int14: shader precompile after updates, faster first pipelines, a better automatic GPU choice, fewer loading
  crashes, Demon's Souls character creation, and the adaptive trigger fixes.

## Checked

- All automated tests pass (int17 as well).
- Astro Bot 1.018 on one PC (RTX 3090, Ryzen 9 7950X3D), patches off, both occlusion modes: Sky Garden, snow, clock
  tower, Ape Escape, volcano (lava with the camera tilted down in accurate mode), crash-site hub and the galaxy map,
  with no console errors.
- Not tested by us on AMD or Intel GPUs or on version 1.007; player reports from RTX 50
  and AMD cards are listed under known issues.

## Installing

1. Download `KytyPS5-U59-Windows-x64.zip` and extract it to a new folder.
2. Open `launcher.exe`. Your existing game list and settings are picked up automatically.
3. Game patches go in a `_Patches` folder next to the launcher; saves are kept per folder in `_SaveData`.

## Known issues

- **RTX 50 series:** the first launch after installing or updating could crash once while the ray tracing shaders
  were being built. The hang watchdog fix and the background shader builds should cure it; if it still happens,
  please send the console text.
- **AMD graphics cards:** with the two patches turned off, the GPU can stop responding ("device lost"), and the water
  in Go-Go Archipelago can make the frame rate drop sharply. On AMD, keep the patches on, or use int15.1.
- **Second (red) galaxy:** one player (RTX 5080, Intel CPU) crashed while arriving at Go-Go Archipelago for the first
  time (the loader read through a bad pointer). We could not reproduce it on our PC. A save from just before the red
  galaxy would help.
- One crash during startup was seen once in testing and could not be reproduced. If the console shows a crash, please
  send it; it now names the code path.
- The software ray tracing changes behaviour for other games too (for example Demon's Souls); please report any new
  problem there.
- Microsoft Defender may flag `launcher.exe` (`Trojan:Win32/Bearfoos.A!ml`, a machine-learning verdict on the
  unsigned launcher). It is built from this repository's source by the GitHub workflow.

## Testers wanted

Please report (GPU, CPU, driver, what you did, the console text) especially from:
- **RTX 50 series** cards (first launch after the update, ray tracing shaders, hang watchdog);
- **AMD Radeon** cards (device lost with the patches off, water in Go-Go Archipelago);
- **Intel CPUs** (with "Intel CPU compatibility" on and off; the red galaxy's first arrival);
- **8-12 GB graphics cards** (video memory full messages, stutter, frame rate with the defaults).

Switches for testing: `KYTY_RT_SOFTWARE=0 KYTY_RT_STUB=1` keeps the lighting but lets every ray miss (no ray-traced
shadows); `KYTY_RT_SOFTWARE=0` alone skips those shaders again (only with the patches on). The bundled preset's
`KYTY_SRT_VARIANT_READS=1` is required. See `U59-README.md` in the download for the full list of changes and switches.
