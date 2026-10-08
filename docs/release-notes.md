# KytyPS5 U59 int16.1 AMD test 2 — the fix for the "device lost" after the intro video on AMD Radeon cards

> **This is a test build for AMD Radeon owners whose game stops with "device lost" / ErrorDeviceLost, usually right
> after the intro video, at the title screen.** Everyone else: please keep using int16.1.

## What this build is

AMD test 1 showed which draw the GPU was stuck in: the title screen's particle draw (a mesh shader). That shader
waits at a barrier for part of its workgroup that never arrives, because of an extra memory barrier the emulator
adds after shared-memory (LDS) writes. This build has the fix, plus a second fix found while checking it:

- **LDS barrier fix (on by default):** the barrier added at `S_WAITCNT lgkmcnt(0)` is now limited to the wave's own
  lanes (subgroup scope) instead of the whole workgroup. `KYTY_LDS_WAITCNT_BARRIER=workgroup` restores int16.1,
  `KYTY_LDS_WAITCNT_BARRIER=0` removes the barrier (the workaround from the GitHub issue, which shows pink particle
  clouds on the title screen).
- **Wave32 fix (on by default, AMD only):** AMD runs vertex, mesh and pixel shaders 64 lanes wide, so two 32-lane
  game waves share one. Each wave now keeps to its own 32 lanes (`KYTY_WAVE32_CLUSTERS=0` turns this off).
- On NVIDIA both fixes leave the image and the speed unchanged.
- Everything else is the same as AMD test 1 (int16.1 plus the GPU breadcrumbs and the helper-lane fix).

## How to test (about 15 minutes, hands off)

1. Copy your working int16.1 folder (with its `_Patches`, `_PipelineCache` and `_SaveData`) to a new folder, then
   unzip this build into the new folder and let it overwrite the files.
2. Start `launcher.exe` from the new folder once, check that Astro Bot and your usual patches are listed, then close
   the launcher again. Do not start the game from it.
3. Double-click `AMD-Test-Kit.cmd` and leave the PC alone. The game opens and closes by itself 6 times. Two of the
   runs bring back the old setting on purpose, so the screen may go black for a few seconds when the GPU resets;
   that is expected.
4. When it says "Send this file: ...AMD-Test-Results-<date>.zip", send us that zip (and, if you can, a note on whether
   the title screen showed pink clouds in the runs that kept going).

You can also just play: this build should get past the title screen on its own. The kit changes nothing outside its
results folder. The zip contains the emulator's console logs, your OS, GPU, driver, CPU and RAM, and Windows' records
of display-driver resets during the test. `AMD-TEST-KIT-README.txt` has the details and options.

---

# KytyPS5 U59 int16.1 AMD test — for AMD Radeon cards that crash ("device lost") after the intro video

> **This is a test build for AMD Radeon owners whose game stops with "device lost" / ErrorDeviceLost, usually right
> after the intro video.** Everyone else: please keep using int16.1.

## What this build is

int16.1 plus crash diagnostics for AMD graphics cards and an automatic test kit. We have no AMD card ourselves, so
one test session on your PC tells us which part of the emulator makes the GPU stop responding.

- **GPU breadcrumbs:** after a device loss the console now names what the GPU was running (queue, operation and
  shader). They switch on with `KYTY_DEVICE_FAULT_DIAGNOSTICS=1`, which the test kit sets for you.
- **One new fix on by default:** pixel-shader helper lanes skip the compare-exchange loops that emulate some stores
  and atomics (`KYTY_PS_HELPER_ATOMICS_SKIP=0` restores the old code). It is safe on NVIDIA.
- **New option** `KYTY_PIPELINE_FAST_FIRST=nvidia`: quick unoptimized pipeline builds only on NVIDIA.
- Everything else is the same as int16.1.

## How to test (about 35 minutes, hands off)

1. Copy your working int16.1 folder (with its `_Patches`, `_PipelineCache` and `_SaveData`) to a new folder, then
   unzip this build into the new folder and let it overwrite the files.
2. Start `launcher.exe` from the new folder once, check that Astro Bot and your usual patches are listed, then close
   the launcher again. Do not start the game from it.
3. Double-click `AMD-Test-Kit.cmd` and leave the PC alone. The game opens and closes by itself about 19 times, each
   time with one setting changed. The screen may go black for a few seconds when the GPU resets; that is expected.
4. When it says "Send this file: ...AMD-Test-Results-<date>.zip", send us that zip.

The kit changes nothing outside its results folder. The zip contains the emulator's console logs, your OS, GPU,
driver, CPU and RAM, and Windows' records of display-driver resets during the test. `AMD-TEST-KIT-README.txt` has
the details and options.

---

# int16.1 release notes

> int16.1 is the int16 pre-release plus low-risk fixes and a much faster default for heavy levels. If anything works
> worse than in the int16 pre-release or int15, please report it (with your GPU and CPU model).

## What's new in int16.1

- **Galaxy map fixed:** the red trails, pillars and blocky red areas around the volcano (the hidden world) and the
  red band at the bottom of the screen are gone. The game fast-clears one of its 64-bit render targets every frame and
  Kyty decoded that clear with a 32-bit decoder, so the clear was dropped and old frames piled up. The same fix covers
  R16F, RG16F, R16, RG16, R8, RG8 and B10G11R11 register clears (`KYTY_CLEAR_REGISTER_WIDE=0` restores the old
  behaviour). This bug was also in upstream Kyty.

- **Performance mode by default: occlusion queries off.** The game asks the GPU, thousands of times per frame,
  whether objects are hidden behind others. Answering those queries accurately costs about half the frame rate in
  the heaviest levels. int16.1 now answers "visible" to all of them by default (`KYTY_GPU_OCCLUSION=0`, as other
  Kyty forks do), which roughly doubles the frame rate in the snow level, the clock tower and Ape Escape.
  On the test PC (RTX 3090, Ryzen 9 7950X3D; a test build without the release build's profile optimization, which
  adds about 15%): snow 41 fps, clock tower 31, Ape Escape 31, Sky Garden 30, volcano 59 (accurate mode: snow 20).
  Possible side effects: the game may draw objects or effects that it would normally hide behind walls (rarely
  visible, and it can cost GPU time where a lot is hidden).
- **Accurate occlusion toggle:** Global settings > Graphics & display > "Accurate occlusion queries (slower; fixes
  rare visibility issues)" turns the accurate mode back on (it passes `--gpu-occlusion on`, which overrides the
  preset). Use it if you see objects through walls or other visibility problems, and please report where. The
  accurate mode includes the volcano lava fix below.
- **Volcano lava fixed (accurate occlusion mode):** with the camera tilted down in the volcano level, the lava lake
  disappeared (flat orange fog). The emulator paired the game's occlusion queries by a wrong address rule; it now
  pairs them by address, which also keeps the occlusion gate on in Sky Garden.
- **Hang watchdog no longer kills the game:** on RTX 50 cards (where the watchdog is on by default) a 5 s stall could
  end the emulator with "Unhandled host exception ... RtlVirtualUnwind". The watchdog's stack dump is now guarded.
- **Out of video memory no longer ends the emulator:** an image that does not fit goes to system memory (slower; the
  console prints "Vulkan: video memory full ..."), and a buffer that does not fit is retried.
- **File id fix:** Astro Bot file ids are assigned one per file; six pairs of game files shared an id and one of each
  pair could be read in place of the other (one of them a galaxy-map icon).
- **Audio:** sounds the game sends to the DualSense speaker now play at 30% on the PC speakers (they were up to 19 dB
  louder than music and effects). Four new volume sliders (master, game sound, music, controller speaker on PC) are in
  the launcher's global Audio settings. The console also prints audio port diagnostics (`AudioDiag` lines).
- **"AMD CPU patch" renamed** to "Intel CPU compatibility (AMD instruction patch)" in the launcher's Compatibility
  settings. It is the same setting (saved configurations and `--amd-cpu` keep working): on Intel CPUs it emulates the
  PS5's AMD-only instructions (EXTRQ/INSERTQ) and AMD's `VRSQRTPS` results. It is not needed on AMD CPUs.
- **Smaller fixes** ported from chenxiao07's fork: a clean process exit after a fatal error (no more unkillable
  emulator processes), a faster guest memory search while streaming, and faster release shader translation.
- **Level title text fixed** (fixed after the int16 pre-release): the level name is drawn as in int14 again.

### New defaults and how to turn each off

| Switch | Default | Off / old behaviour | What it does |
| --- | --- | --- | --- |
| `KYTY_GPU_OCCLUSION` (preset) | `0` (performance mode) | `1`, or the launcher's "Accurate occlusion queries" (`--gpu-occlusion on`) | Occlusion queries answered "visible" instead of counted on the GPU |
| `KYTY_IMAGE_SYSMEM_FALLBACK` | on | `0` | Images that do not fit in video memory go to system memory |
| `KYTY_APR_HASHED_IDS` | off | `1` restores the old hashed ids | One file id per file path |
| `KYTY_FATAL_EXIT_PROCESS` | off | `1` restores the old exit path | Fatal errors end the process with TerminateProcess |
| `KYTY_IR_VALIDATE` | off in release | `1` runs it | Shader IR validation |
| `KYTY_AUDIO_DIAG` | on | `0` | Audio port open/close lines in the console (256 lines max) |
| Audio volumes | master/main/music 100%, pad speaker on PC 30% | launcher sliders, or `KYTY_AUDIO_MASTER_VOLUME`, `KYTY_AUDIO_MAIN_VOLUME`, `KYTY_AUDIO_MUSIC_VOLUME`, `KYTY_AUDIO_PAD_SPEAKER_ON_MAIN_VOLUME` (percent, 0-200) | |

Optional (off by default): `KYTY_WIDE_FILL_CLEAR=1`, `KYTY_GUEST_STORAGE_REPEAT=1`, `KYTY_READBACK_MERGE_GAP_KB=64`,
and the test tool `KYTY_VRAM_LIMIT_MB=<MiB>` (imitates a smaller GPU).

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

## Also in int15

- No more "unsupported sampled depth image" stops on fast GPUs, no crash when leaving extra levels in Sky Garden,
  about 20 upstream shader fixes, a Demon's Souls fix for an unknown image format, and thread priority changes.
- From int14: shader precompile after updates, faster first pipelines, a better automatic GPU choice, fewer loading
  crashes, Demon's Souls character creation, and the adaptive trigger fixes.

## Checked

- All automated tests pass.
- Astro Bot 1.018 on one PC (RTX 3090, Ryzen 9 7950X3D), patches off: Sky Garden, snow, clock tower, Ape Escape and
  the volcano with the default (performance) occlusion mode, and snow plus the volcano lava with the camera tilted
  down in accurate mode, with no console errors.
- Not tested by us on AMD or Intel GPUs or on version 1.007; player reports from RTX 50
  and AMD cards are listed under known issues.

## Installing

1. Download `KytyPS5-U59-Windows-x64.zip` and extract it to a new folder.
2. Open `launcher.exe`. Your existing game list and settings are picked up automatically.
3. Game patches go in a `_Patches` folder next to the launcher; saves are kept per folder in `_SaveData`.

## Known issues

- **Grass flicker:** small grass/moss clumps on sand (e.g. the crash-site hub) can flicker between two looks from
  frame to frame. Also in earlier builds; being looked at.
- **RTX 50 series:** the first launch after installing or updating could crash once while the ray tracing shaders
  were being built. The hang watchdog fix should cure the crash we know of; if it still happens, start the game again
  and please send the console text.
- **AMD graphics cards:** with the two patches turned off, the GPU can stop responding ("device lost"), and the water
  in Go-Go Archipelago can make the frame rate drop sharply. On AMD, keep the patches on, or use int15.
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
