# KytyPS5 U59 release notes, older versions

## int21

int21 is int20 plus faster draw validation, fixes for GPU device losses and a silent exit, the latest upstream KytyPS5 changes, ports from chenxiao07's fork and a new profile-guided optimisation profile; the older notes are in docs/release-notes-history.md. The first launch rebuilds the shader and program caches once (the pipeline key changed), so the first minutes can stutter.

- **Faster:** the command processor skips the full validity check of a prepared draw when the GPU has never touched the memory it reads (`KYTY_CLEAN_UNTOUCHED=1`, on in the preset; exact, verified over tens of millions of draws), plus a new profile-guided optimisation profile. Measured with the play folder's three game patches on (non-tiled renderer, GI off, fixed 1080p): pirate island about 58 fps against about 53 in int20, and Sky Garden at its 60 fps pacing cap where int20 ran about 54. Without game patches Sky Garden gained about 1.4% over the last test build.
- **Fixed (device loss):** a small upload's copy from guest memory could run after later writes the command processor or the game made to the same memory, so earlier draws read future data and the GPU could hang. Writes now wait for or take over the queued copies (`KYTY_UPLOAD_DMA_WRITE_DRAIN=1`, on in the preset; `=labels` is a stricter mode for anyone who still sees losses). Also fixed: the texture cache could miss guest writes on pages whose protection was still pending, and draws could keep using a buffer that a later merge had retired (`KYTY_REBIND_AFTER_JOIN=1`).
- **Fixed (silent exit):** on Windows set to force Control Flow Guard on every program, Astro Bot closed without a message (ntdll 0xC0000409) right after shader loading (`KYTY_GUEST_TEB_STACK_RANGE=1`, on in the preset; issue #66).
- **Fixed:** exits on colour targets with an invalid format and on a stencil plane shared with another render target (`KYTY_STENCIL_TARGETS_APART=1`), guest file writes from GPU-protected pages that were silently cut short, and pad presses lost during frame hitches (`KYTY_PAD_STATE_COALESCE=1`).
- **Crash reports:** `KYTY_DEVICE_FAULT_EXT=1` is on in the preset and now writes a complete `_device_fault.nv-gpudmp` file next to `_kyty.txt` when the GPU is lost. Please attach it to device-loss reports.
- **Correction to int20:** int20's notes said the Astro Bot "too many objects/batches" crash was fixed; it was only rarer. The cause was that the game's draw threads overflow their command buffers when they fall a frame behind, mostly on CPUs with fewer cores. int21 doubles those buffers for Astro Bot 01.018 (`KYTY_ASTRO_CB_SCALE=2`) and spreads the game's threads over both CCDs on X3D CPUs (`KYTY_VCACHE_GUEST=all`): 0 crashes in about 20 boots of speedrun_inflator, the level that crashed on every boot before. Players with 8-core CPUs: a report from speedrun_inflator would help.
- **Upstream KytyPS5:** about 150 upstream changes, among them user clip planes, f16 interpolation, alpha-to-mask (`KYTY_ALPHA_TO_MASK`), start on GTX 10xx/RDNA1 cards without fragment-shader barycentrics, a keyboard player option, a performance overlay and fixes for other games behind switches that are off by default. Details are in the commit history.
- **Thanks:** chenxiao07 for the fork changes ported here (colour-target and stencil fixes, the Control Flow Guard fix, pad state, file writes, APR file reads and more), and the upstream KytyPS5 contributors.
- **Known issues:** rare GPU device losses can still happen, seen in sponge_dirty_bath without game patches. Please report them with `_kyty.txt` and `_device_fault.nv-gpudmp`; workaround: `KYTY_UPLOAD_DMA_WRITE_DRAIN=labels` or `KYTY_DEFER_PROTECTS=0` in `u59-preset.json` (about 3% slower).
- **Upgrading:** extract into a new folder, or replace `u59-preset.json` too; an old preset leaves the new fixes off.

## int20

int20 is int19.4 plus a fix for Astro Bot's "too many objects" crash, smoother frame pacing, a new profile-guided optimisation profile and fixes contributed through pull requests; the older notes are in docs/release-notes-history.md. The first launch rebuilds the shader and program caches once, so the first minutes can stutter.

- **Fixed (Astro Bot "too many objects/batches in this scene"):** the game no longer stops with its own `command_buffer_writer.ppr.cpp:31` error while loading or playing. Kyty showed the game's CPU threads that GPU work had finished as soon as it was recorded; the game now sees end-of-pipe labels only when the GPU work has really finished, while the command processor keeps its own waits at record time, so the fix costs no measurable fps (`KYTY_LABEL_MODE=split`, the new preset default; `record` restores the old behaviour). In testing: 0 crashes in 11 Sky Garden boots, against 3 in 7 with int19.4's settings.
- **Smoother:** while a new pipeline is still being built Kyty skips that draw instead of stalling (`KYTY_PIPELINE_SKIP`), and the command processor journals its self-label waits (`KYTY_CP_WAIT_JOURNAL`), both on in the preset. Partially resident textures are on in automatic mode: on cards below 16 GB of VRAM they lower VRAM use in large scenes (16 GB and larger cards keep them off; `KYTY_TEXTURE_SPARSE_RESIDENCY=1` forces them on).
- **Faster:** a new profile-guided optimisation profile recorded over six levels in the new label mode, and faster range-set bookkeeping.
- **Fixed:** a crash on NVIDIA 580 drivers in Astro's Playroom (late boolean phi values in SPIR-V), and a rare shader-compiler crash from references to instructions that specialization removed.
- **Fixed:** Chinese text is no longer drawn as "?" (Kyty falls back to a CJK font from Windows).
- **New, opt-in:** `KYTY_CUDA_EARLY_INIT=1` initialises CUDA before the guest address space so NVIDIA Smooth Motion can start, and `KYTY_PRESENT_ABS_VBLANK=1` paces presents on an absolute vblank timeline (a fix for spiky frame times on some systems).
- **Thanks to contributors:** pernahlebihtampan (#49, the NVIDIA 580 crash in Astro's Playroom), Majed-allababedi (#63, fewer RangeSet reallocations), ccxp (#45, the CJK font fallback), luizhtss (#58, early CUDA init for Smooth Motion), sniperspy69 (#65, the present timer for spiky frame times) and ali-commits (#50 and #51, test and Qt packaging fixes).
- **Known issues:** if you get a GPU device loss, reproduce it once with `KYTY_DEVICE_FAULT_DIAGNOSTICS=1` and Printf output set to File, then attach `_kyty.txt` and the `_kyty` folder to an issue; workaround: `KYTY_DEFER_PROTECTS=0` in `u59-preset.json`. If a level shows wrong graphics or freezes with the new label mode, try `KYTY_LABEL_MODE=record` and report it.
- **Upgrading:** extract into a new folder, or replace `u59-preset.json` too; an old preset keeps the old label mode and the crash.
- **AMD, Intel and Linux users:** their fixes are in the int19 notes.

## int19.4

int19.4 is int19.1 plus the unreleased int19.2 and int19.3 work, more performance work and the latest upstream KytyPS5 changes; the older notes are in docs/release-notes-history.md. The first launch rebuilds the shader and program caches once, so the first minutes can stutter.

- **Faster:** a few percent faster in scenes limited by the command processor. GPU compute dispatches the game does not order now overlap, shader code generation is tighter, more draws reuse the previous draw's state, empty indirect dispatches are skipped, and the release build has a new profile-guided optimisation profile recorded over several levels.
- **New (automatic V-cache placement on dual-CCD X3D CPUs):** on a Ryzen 9 7900X3D, 7950X3D, 9900X3D or 9950X3D without Process Lasso, Kyty keeps the game, command-processor and render threads on the CCD with the 3D V-cache and moves background threads to the other CCD. It does nothing when Process Lasso (or another tool) sets CPU sets or affinity for Kyty, and nothing on other CPUs. `KYTY_VCACHE_PLACEMENT=0` turns it off.
- **Fixed:** with the game's "[Performance]" patches on, the emulator no longer closes ("buffer ADD_TID is only valid for compute shaders") or freezes at some spots in Sky Garden (from int19.2).
- **Fixed:** a GPU device loss (`ErrorDeviceLost`) when a partially resident texture got more mip levels, and NaN handling of two pack instructions (from int19.3).
- **From upstream KytyPS5:** shader and texture fixes for other games (among them Demon's Souls), DLC and add-on support, input mapping tabs in the launcher with an emulated touchpad and rumble, and more system modules.
- **Launcher renamed:** the launcher is now `kyty_launcher.exe`. Extracting over an old folder leaves the old `launcher.exe` behind; delete it and start `kyty_launcher.exe`.
- **Known issues:** in some levels the game can stop with its own "too many objects/batches in this scene" error (`command_buffer_writer.ppr.cpp:31`); this is under investigation. If you get a GPU device loss, reproduce it once with `KYTY_DEVICE_FAULT_DIAGNOSTICS=1` and Printf output set to File, then attach `_kyty.txt` and the `_kyty` folder to an issue; workaround: `KYTY_DEFER_PROTECTS=0` in `u59-preset.json`.
- **AMD, Intel and Linux users:** their fixes are in the int19 notes.

## int19.3 (not released separately; included in int19.4)

int19.3 was int19.2 plus the fifth sync with upstream KytyPS5, a device-loss fix, a shader accuracy fix and some opt-in switches.

- **Fixed (GPU device loss with sparse textures):** when a partially resident texture got more mip levels, Kyty bound the new memory on a second Vulkan queue while work already sent to the GPU could still use that texture, which the Vulkan specification does not allow. The bind is now ordered on the GPU after the work in flight, and later work waits for it. This fixed the `ErrorDeviceLost` in a galaxy-map case; confirmation on an RTX 4090 and an RTX 5070 is still pending.
- **Fixed (shader accuracy):** `V_CVT_PKNORM_I16_F32` and `V_CVT_PKNORM_U16_F32` now turn NaN into 0, as the RDNA2 instruction set specifies (the signed form gave 0x7fff).
- **New (automatic V-cache placement on dual-CCD X3D CPUs):** on a Ryzen 9 7900X3D, 7950X3D, 9900X3D or 9950X3D without Process Lasso, Kyty now keeps the game, command-processor and render threads on the CCD with the 3D V-cache and moves background threads (shader compiles, file I/O, watchdogs) to the other CCD. On a 7950X3D with an RTX 3090 and no Process Lasso, Sky Garden went from 34.1 to 45.8 fps; the world map did not change (it is limited by the GPU there). It does nothing when Process Lasso (or another tool) sets CPU sets or affinity for Kyty, and nothing on single-CCD or non-X3D CPUs. `KYTY_VCACHE_PLACEMENT=0` turns it off.
- **Upstream sync 5** (143 upstream commits). Kyty's own defaults and the `u59-preset.json` behaviour stay as in int19.2. Taken from upstream, among others:
  - Block-compressed (BC) textures in the PS5 R64 tiling mode are now accepted by the tiler (upstream b7b335ac6, `tile.cpp`).
  - Pixel shaders: in ordinary draws, flat and smooth reads of the same vertex output now share one location and read the raw vertex values; rectangle lists keep their old handling. `KYTY_PS_PER_VERTEX=0` also turns this off.
  - More GPU instructions and fixes (16-bit buffer loads, 64-bit compares, 64-bit LDS max and compare-exchange, signed 32-bit image atomics, R8 SNORM sampling, CMASK and RG32 clear decoding), uploads of images larger than the upload ring, a lost-completion fix for batched asynchronous file reads (APR), HDR presentation, the splash screen and a reorganised launcher (Kyty's audio sliders, occlusion and Intel CPU options are kept).
  - Controllers: a game can now open only the standard pad port of the configured user, and only once. The first pad opens are printed in the console log, so a game that opens another port shows up there.
- **New opt-in switches** (all off unless set):
  - `KYTY_GPU_TIMESTAMP_SCALE=<100-200>`: stretches the GPU time the game measures within each frame by that percentage, so Astro Bot's dynamic resolution keeps a lower internal resolution.
  - `KYTY_AFTERMATH=1`, for developers: NVIDIA Nsight Aftermath crash dumps on a device loss. Release builds do not include it; it works only in a build configured with the Aftermath SDK (CMake `KYTY_AFTERMATH_SDK`).
  - `KYTY_AGC_CB_WATERMARK=1` and `KYTY_GUEST_PROBE`: diagnostics for Astro Bot's "too many objects/batches" stop (see docs/EXPERIMENTAL.md).
  - `KYTY_READ_FAULT_DIAG` and `KYTY_READ_FAULT_PUBLISHED`: a diagnostic and an experiment for game threads that read GPU-written memory.

## int19.2 (not released separately; included in int19.4)

- **Fixed:** with the game's "[Performance]" patches on (no grass, SSAO/DoF/SSR off, ...), the emulator no longer closes ("SPIR-V validation failed ... buffer ADD_TID is only valid for compute shaders") and no longer drops to a few fps and freezes at some spots in Sky Garden. Leftover descriptors in slots the game does not use no longer compile a new shader version for every value.

## int19.1

- **Fixed:** the emulator no longer closes (`occlusion.cpp:405`) with "Accurate occlusion queries" on, seen in rocket_hourglass.
- **Faster:** the release build's profile-guided optimisation profile was retrained on the int19 code, a few percent more fps in Sky Garden on an RTX 3090 / Ryzen 9 7950X3D.

## int19

> **int19 is mainly for AMD Radeon owners whose game crashed (`amdvlk64.dll+0x22240fc`) or showed broken
> graphics with int18, and for anyone who heard a buzzing sound after some minutes.** NVIDIA users benefit too: Sky
> Garden runs about 12% faster than with int18. The first launch rebuilds the shader and program caches once, so the
> first minutes can stutter.

## Everyone

- **Faster command processing:** measured on an RTX 3090 (Ryzen 9 7950X3D), Sky Garden went from 32.8 fps with int18
  to 35.1 with the draw-run changes below (+7%), and the sequencer prefetch adds another 5-8% (launches in A B B A
  order: Sky Garden 34.9 -> 36.5 fps, pirate island 29.3 -> 31.6, clock tower 38.4 -> 40.9).
  - `KYTY_CP_SEQ_PREFETCH=1` (new default): while the emulator waits for the GPU at a "wait for idle", it prepares
    the draws that follow. `KYTY_CP_SEQ_PREFETCH=0` turns it off.
  - Draw runs through draws that write buffers (`KYTY_DRAW_RUN_WRITES`), fewer repeated checks of images and pages
    (`KYTY_DRAW_RUN_IMAGE_EPOCH`, `KYTY_DRAW_PREP_CLEAN_SINCE`), larger descriptor pools.
- **Two draw-path shortcuts:** Sky Garden command processing takes 0.6 ms less per frame (+2% fps). The statistics
  counters are now owned by the GPU thread (`KYTY_CP_OWNED_COUNTERS=1`), and a draw run's eligibility is carried
  across continuations (`KYTY_DRAW_RUN_CONT_ELIGIBLE=1`). Both are on by default; set either to 0 to turn it off.
- **Batched page re-protection after read uploads:** the write protection of pages after a read upload is now applied
  in a batch on the upload worker instead of one call each on the command-processor thread. Sky Garden command
  processing takes 0.5 ms less per frame (+2% fps). `KYTY_DEFER_PROTECTS=0` turns it off. The idea comes from the
  peepostare fork; this is a re-implementation, not their code. A race between a deferred write watch and a concurrent protection update (which left a page writable while tracked as clean, and caused Vulkan device loss on NVIDIA within minutes) was found and fixed before release, with a unit test; `KYTY_DEFER_PROTECTS_STATS` now reports `race_pages`/`lost_pages`.
- **Less waiting between draw preparation and the command processor (publication snapshot):** a lock shared by the
  draw-prep workers and the command processor (the pending-publication check) is now read lock-free
  (`KYTY_PUBLICATION_SNAPSHOT=1`, the default; `=0` restores the old scan). Measured +3-4.6% fps in Sky Garden.
- **SRT trace replay:** the descriptor walks that refresh a draw's resources are recorded once per program and
  replayed with guards (`KYTY_SRT_TRACE=1`, the default; `=0` turns it off, `=verify` checks every replay against
  the full walk). Measured +5.4-5.9% fps in Sky Garden on top of the lock fix.
- **Direct reads in draw preparation:** `KYTY_DRAW_PREP_DIRECT_READS=1` (default) lets the draw-prep worker read
  guest pages directly into coalesced records instead of going through the slower path (worker side only).
  `=0` restores the old reads.
- **Diagnostic:** `KYTY_DRAW_PREP_PHASES=1` prints the time a draw-prep slot spends in each phase of its preparation.
  Off by default.
- **Fixes from chenxiao07's fork:** an image write-back no longer overwrites bytes the CPU wrote since (`KYTY_IMAGE_WRITEBACK_GUARD`),
  a CPU write keeps the GPU's other bytes of an image (`KYTY_IMAGE_GPU_KEEP`), invalid guest data is skipped instead
  of closing the emulator (`KYTY_SKIP_INVALID`), and the present thread no longer waits for the renderer's lock.

## AMD GPUs

Majed-allababedi found where the AMD driver crash comes from and sent three fixes (pull requests #25, #26 and #27).
Thank you!

- **Driver crash at `amdvlk64.dll+0x22240fc` (#25):** the crash was not in the game's shaders but in one of Kyty's
  own texture-tiling compute shaders. AMD's compiler crashed on the way it picked per-size values. The shader now
  picks them one by one; the result is the same bytes, the driver no longer crashes building it.
- **Depth-compare sampling of a missing texture (#26):** when a shader samples a depth texture that is not there,
  Kyty now binds a depth-format placeholder, as Vulkan requires. Before, it was a color placeholder, which AMD may
  turn into garbage.
- **Pixel-shader inputs (#27):** when a pixel shader reads a flat and a smooth value from the same vertex output,
  Kyty moved one of them to a spare slot that the vertex shader never wrote. NVIDIA tends to read zero there, AMD
  reads garbage (a likely cause of the pink and particle glitches). The vertex shader now writes that slot too, and
  pixel-shader inputs that are never read are no longer requested.
- **Sequencer prefetch:** see `KYTY_CP_SEQ_PREFETCH` under Everyone.

AMD test 4's workaround (building some pipelines without optimization) is not in this build: if #25 is the cause,
it is no longer needed. If you still crash, please say so and send your `_kyty.txt`.

## Intel CPUs

- **"Intel CPU compatibility" is now on automatically.** With it off, the audio engine thread could crash
  early (`Mutex.cpp:363 m_initialized`, or a crash in `memset`): the game's AMD-only `EXTRQ` instructions then went
  through a slow trap whose Windows exception frame can overwrite data the game keeps just below its stack pointer.
  The setting is now partly checked by default (automatic: on unless the CPU is an AMD); checked or unchecked forces
  it on or off (`--amd-cpu`, `--no-amd-cpu`, `KYTY_AMD_CPU=1|0|auto`). AMD CPUs are unchanged.

## Linux

On Linux, a file read into guest memory that the emulator had write-protected (GPU write tracking) stopped at
the first protected page and reported the short count as success: a worldmap level file read 72,704 of 271,864
bytes (the game's XML assert, crash in the loader thread) and a model file 768 bytes (crash in the model streamer
thread after skipping the first cutscene). The Windows build already finished such reads through a host buffer;
the Linux build now does the same. Windows is unchanged apart from this note.

## Audio

- **Buzzing / saw-wave sound after some minutes on certain levels:** this was caused by the emulator's audio-port
  table being limited to 256 ports. Astro Bot opens one port per 3D sound and holds up to several hundred in busy
  scenes, so new sounds were refused and the game's mixer produced a steady buzz. Ports now grow on demand up to
  4096 (`KYTY_AUDIO_OUT2_MAX_PORTS`; setting it to 256 restores the old behaviour). Confirmed fixed by a tester who
  could reproduce it within a minute. `KYTY_AUDIO_TRACE` now also logs port create/destroy counts and a BUZZ
  detector: it watches only ports that play to the main device (not the pad speaker, aux, vibration or 3D object
  ports) and reports an episode of at least one second of repeated blocks once, when it ends, with its duration.
- **Sound breaking up after some minutes (secondary robustness improvement, not the cause of the buzz above):**
  on some PCs the sound could turn crackly or cut out a few minutes into play.
  The sound card can run slightly faster than the emulator's own clock, so the audio buffer
  slowly drains until it runs dry. Two fixes, both on by default:
  - `KYTY_AUDIO_QUEUE_CATCHUP=1` lets the sound card set the pace when the buffer runs low, instead of the
    emulator's clock.
  - `KYTY_AUDIO_PAD_PRIME=1` fills the controller-speaker port with a little silence when it opens late, so it
    starts with a cushion instead of an empty buffer.
- **Buffer size:** `KYTY_AUDIO_LATENCY_MS` (10-500, default 40) sets how much audio is kept queued.
- **Reporting a sound problem:** start with `KYTY_AUDIO_TRACE=1` and the emulator prints audio statistics every
  10 seconds. They reach `_kyty.txt` only when `printf_direction=File` is set in `Kyty.ini`
  (`C:\ProgramData\Kyty\Kyty.ini` on Windows); with `Silent` or `Console` they go to the console only.
  Please send that log with your report.
- **Going back:** `KYTY_AUDIO_QUEUE_CATCHUP=0 KYTY_AUDIO_PAD_PRIME=0` restores the old pacing behaviour.
- **Honest status of the pacing fixes:** they have been verified only against a simulated fast sound card, not yet on
  a PC that actually shows that problem. Reports are welcome.

## How to test

Copy your working folder, unzip this build over it and start the game as usual. Play the first levels for a while.

- **AMD:** does it still crash, and do the graphics look right (no pink areas, no colorful particles)? If anything
  goes wrong, send `_kyty.txt` from the emulator folder and say which level.
- **Astro's Playroom on AMD:** do the clouds still break into blocks?
- **Everyone:** is it faster or slower than int18 for you?
- **Developers:** `KYTY_CP_COMMIT_STATS=1` now also prints a `StartPhases` line (draws split by start reason) and
  `KYTY_FAULT_MAP=1` splits write-fault handler time by cause; both are off by default.
  `KYTY_DEFER_PROTECTS_STATS=1` prints the batched re-protection counters.
  `KYTY_CP_COMMIT_STATS=steps` prints a per-step split of every draw's commit and gap.

---

# KytyPS5 U59 int18 — AMD fixes, upstream sync, rumble for other controllers

> int18 is int17 plus the fixes from the three AMD test builds, a sync with upstream Kyty, and fixes for
> missing-preset and controller problems. The first launch after the update rebuilds the shader and program caches
> once (the shader code generator changed), so the first minutes can stutter.

## What's new in int18

### AMD graphics cards

- **Title-screen hang fixed:** on AMD cards the GPU could stop responding ("device lost") at the same title-screen
  draw. A particle mesh shader waits for its own LDS writes inside branches that only some waves take; Kyty turned
  each wait into a workgroup-wide memory barrier, which AMD's driver handles like a workgroup barrier that the other
  waves never reach. The wait now orders only the wave's own memory (subgroup scope). `KYTY_LDS_WAITCNT_BARRIER`
  = `subgroup` (default), `workgroup` (the old behaviour) or `0` (no barrier).
- **Water and particle corruption fixed:** AMD runs vertex, mesh and pixel shaders with 64 lanes per subgroup, so one
  host subgroup held two of the PS5's 32-lane waves, and shaders read the other wave's lanes and masks (stretched
  water, pink particle clouds, loops that never end). Each wave now stays in its own 32 lanes
  (`KYTY_WAVE32_CLUSTERS=auto`, the default; `0` turns it off, `force` applies it everywhere). The log names each
  shader this applies to. NVIDIA is unchanged.
- **Large compute shared memory:** some Astro Bot compute shaders (GPU skinning) ask for 48 KiB of shared memory;
  AMD's driver allows 32 KiB, so Kyty cut it to 32 KiB and meshes with more than 682 bones could stretch. Such a
  shader now keeps its shared memory in a video memory buffer (from upstream Kyty, with Kyty's own buffer handling).
  `KYTY_LDS_DEVICE_BUFFER` = `auto` (default: only above the device limit), `0` (the old cut) or `force`;
  `KYTY_LDS_DEVICE_BUFFER_MB` (default 1024) caps the buffer. NVIDIA is unchanged (48 KiB fits).
- **Helper lanes no longer spin:** in pixel shaders, helper lanes skip the compare-exchange loops of emulated atomics,
  which could loop forever on AMD (`KYTY_PS_HELPER_ATOMICS_SKIP=0` restores the old code).
- **Less video memory without the preset:** `KYTY_FUNCTION_ARRAY_SHRINK` is now on by default (it was only set by
  `u59-preset.json`). Vertex and pixel shaders emulate shared memory with a large per-lane array that AMD backs with
  memory for every lane (about 2 MiB per wave); the shrink cuts it to what the shader can reach (at most 352 dwords in
  every Astro Bot 1.018 shader; about 3.9 GiB less video memory on an RTX 3090). `=0` turns it off and says so in the
  log. Two RX 6900 XT players whose launcher had not applied the preset crashed inside the driver without it.
- **Crash reports name the module and pipeline:** a host crash now writes its fault report into the log file
  (`_kyty.txt`), not only the console, with the module and offset of the crash (for example a driver DLL), each
  call-chain frame's module, and what the thread was doing (a graphics or compute pipeline build, with the guest
  shader hashes). Graphics pipeline builds are always traced in the log.
- **Diagnostics for testers** (off or unchanged by default): `KYTY_DEVICE_FAULT_DIAGNOSTICS=1` prints a device
  report and enables GPU breadcrumbs (`KYTY_GPU_BREADCRUMBS`), which name the GPU operations in flight after a device
  loss; `KYTY_COMPUTE_WAVE64=0`, `KYTY_PS_PER_VERTEX=0`, `KYTY_CLIP_GUARD=0`, `KYTY_VOLATILE_LOADS=1` and
  `KYTY_LOOP_GUARD_SHADERS=all` (with `KYTY_LOOP_GUARD=<n>`) switch off or add single shader features to narrow
  down a driver problem.

### Everyone

- **Rumble for other controllers:** Xbox and other non-DualSense pads now rumble from Astro Bot's haptics (the game
  plays its vibration as audio for the DualSense; Kyty turns it into rumble with the per-channel gains). Thanks to
  Linky362 (PR #20).
- **Black lighting without the preset fixed:** Astro Bot without the lighting patches no longer renders black when
  `u59-preset.json` is not applied (launcher without the file, or `kyty_emulator` started directly):
  `KYTY_SRT_VARIANT_READS` is now on by default (`0` turns it off). The launcher warns when `u59-preset.json` is
  missing or invalid (it looks next to `launcher.exe` and one folder up), and the emulator prints a warning at startup
  when the preset's settings are absent. Keep using the preset: without it the game runs much slower (about 20
  instead of 32 fps on an RTX 3090 in the Sky Garden). A failed ray tracing pipeline build, which turns lighting off
  until restart, is now shown as a WARNING in the console.
- **Upstream Kyty sync:** the 151 upstream commits since the last sync were reviewed; 62 were taken and 20 partly
  (new shader instructions and formats, native 64-bit integers and 8/16-bit storage, store-completion waits, GLC/DLC
  memory coherence, D16 and formatted buffer access, kernel memory fixes, the main-thread window title). Upstream
  changes that conflict with Kyty U59's own designs (resource tracking, MAD rounding, readback sizes) were left out.
  No frame rate change was measured (Sky Garden, casino and pirate island A/B within noise). 8/16-bit storage is now
  a required GPU feature, so very old drivers without it cannot start.
- **Graphics fixes from upstream:** colour exports routed through `CB_SHADER_MASK` (`KYTY_CB_SHADER_MASK_EXPORTS`,
  on through the preset), linear render-target mip chains, rendered contents kept across 1D and 2D texture aliases,
  and streamed texture mips uploaded from partly committed sparse reservations (fixes the exit "failed to read mapped
  guest image backing").
- **GPU stores to untracked memory:** a shader store through a buffer descriptor computed on the GPU, to memory Kyty
  had not cached on the GPU yet, was dropped. Such stores are now logged and copied back right after the dispatch
  (`KYTY_BDA_STORE_LOG`, default on; `=0` restores the old drop).
- **Smaller fixes:** a depth image alias is found again for a smaller layer prefix (no second copy), trophy unlocks
  are copied once from the old `_SaveData` location (issue #18), and unknown socket options return the error codes the
  game expects on Windows.

### New defaults in int18

| Switch | Default | Off / old behaviour | What it does |
| --- | --- | --- | --- |
| `KYTY_LDS_WAITCNT_BARRIER` | `subgroup` | `workgroup` (int17), `0` | Scope of the barrier after a shader waits for its LDS writes |
| `KYTY_WAVE32_CLUSTERS` | `auto` | `0` | 32-lane waves stay in their own lanes on 64-lane host subgroups (AMD) |
| `KYTY_LDS_DEVICE_BUFFER` | `auto` | `0` | Compute shared memory above the device limit in a video memory buffer |
| `KYTY_PS_HELPER_ATOMICS_SKIP` | on | `0` | Pixel-shader helper lanes skip emulated atomic loops |
| `KYTY_FUNCTION_ARRAY_SHRINK` | on | `0` | Per-lane shader arrays cut to their reachable size |
| `KYTY_SRT_VARIANT_READS` | on | `0` | Lighting shaders run without the preset |
| `KYTY_BDA_STORE_LOG` | on | `0` | GPU stores to untracked memory are copied back |
| `KYTY_CB_SHADER_MASK_EXPORTS` (preset) | `1` | `0` | Colour exports routed by `CB_SHADER_MASK` |

### Checked

- All automated tests pass, including the new AMD-path tests (32-lane clusters and the shared-memory buffer, run on
  an RTX 3090 with the AMD limits forced).
- Astro Bot 1.018 on the test PC (RTX 3090, Ryzen 9 7950X3D): normal boot to the title screen and galaxy map with a
  cold shader cache, no console errors.
- The AMD fixes were tested by players on Radeon RX 6000 cards with the AMD test builds; please report how int18
  runs on your AMD card.

### Known issues

- **AMD graphics cards:** the int17 note (device lost with the patches off, slow water in Go-Go Archipelago) should be
  fixed; if you still see it, please send the console text and `_kyty.txt` (they now name the shader or driver
  module).
- The int17 known issues below still apply otherwise.

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
- **3D object audio now plays:** a large part of Astro Bot's sound goes to the PS5's 3D audio object ports (the
  ambisonics scene, a stereo pair and positioned per-sound effects), which Kyty used to drop. They are now mixed
  into the game sound. Launcher: global Audio settings > "3D audio objects (game's positional sounds)" switches
  them off (`--audio-objects off`, `KYTY_AUDIO_OBJECTS=0`), and a "3D audio objects" volume slider
  (`--audio-objects-volume`, `KYTY_AUDIO_OBJECTS_VOLUME`, 0-200%) sets their level.
- **Louder stereo output:** on a stereo device Kyty now downmixes the game's 7.1 output itself (front at full level)
  instead of SDL, which played it about 13 dB quieter. `KYTY_AUDIO_STEREO_DOWNMIX=0` restores SDL's downmix.
- **Ported from chenxiao07's fork:** protection kept on remapped memory views (fixes a "frozen model" class of bugs),
  small fibers on their own stacks (Demon's Souls job-thread crashes), indirect draws' start instance (Demon's Souls
  missing pillars and rock faces), faster guest memory search while streaming, a clean process exit after a fatal
  error, and faster release shader translation.
- **"AMD CPU patch" renamed** to "Intel CPU compatibility (AMD instruction patch)" in the launcher's Compatibility
  settings. It is the same setting (saved configurations and `--amd-cpu` keep working): on Intel CPUs it emulates the
  PS5's AMD-only instructions (EXTRQ/INSERTQ) and AMD's `VRSQRTPS` results. It is not needed on AMD CPUs. The console
  lines now read "AMD instruction patch for Intel CPUs: ...".
- **Read-only shader buffers (AMD graphics cards):** shaders that never write their storage buffers now declare them
  read-only, so AMD's driver can load them once per wave instead of per thread (from serbru20066666's fork, which
  measured a lighting shader in Astro's Playroom at 3.45 -> 2.76 ms on an RX 6800 XT). No change on NVIDIA. Not
  tested by us on AMD: if you see new graphics problems on an AMD card, try `KYTY_READONLY_BUFFERS=0` and please
  report the result. The first launch rebuilds the shader cache.

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
| `KYTY_READONLY_BUFFERS` | on | `0` | Storage buffers a shader never writes are declared read-only (faster on AMD) |
| `KYTY_APR_HASHED_IDS` | off | `1` restores the old hashed ids | One file id per file path |
| `KYTY_FATAL_EXIT_PROCESS` | off | `1` restores the old exit path | Fatal errors end the process with TerminateProcess |
| `KYTY_IR_VALIDATE` | off in release | `1` runs it | Shader IR validation |
| `KYTY_AUDIO_DIAG` | on | `0` | Audio port open/close lines in the console (256 lines max) |
| `KYTY_AUDIO_PROPAGATION_RENDER` | direct sound | `silent` | AudioPropagation sources play their direct sound |
| Audio volumes | master/main/music 100%, pad speaker on PC 30% | launcher sliders, or `KYTY_AUDIO_MASTER_VOLUME`, `KYTY_AUDIO_MAIN_VOLUME`, `KYTY_AUDIO_MUSIC_VOLUME`, `KYTY_AUDIO_PAD_SPEAKER_ON_MAIN_VOLUME` (percent, 0-200) | |
| `KYTY_AUDIO_OBJECTS` | `1` | `0`, or untick "3D audio objects" (`--audio-objects off`) | AudioOut2 3D object ports mixed into the game sound |
| `KYTY_AUDIO_STEREO_DOWNMIX` | on | `0` | 7.1 output downmixed to stereo by Kyty, not SDL |

Optional (off by default): `KYTY_WIDE_FILL_CLEAR=1`, `KYTY_GUEST_STORAGE_REPEAT=1`, `KYTY_READBACK_MERGE_GAP_KB=64`,
and the test tool `KYTY_VRAM_LIMIT_MB=<MiB>` (imitates a
smaller GPU). The accurate-mode switches above (`KYTY_OCCLUSION_SPLIT`, `KYTY_DRAW_RUN_QUIET_OPS`) only matter with accurate
occlusion on.

## From int16.1

- **Level title text fixed:** the level name shown when Astro Bot enters a level is drawn as in int14 again.

## What's new in int16.1 (main release)

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

### int16.1 defaults and how to turn each off

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

- **Grass flicker:** small grass/moss clumps on sand (e.g. the crash-site hub) can flicker between two looks from
  frame to frame. Also in earlier builds; being looked at.
- **RTX 50 series:** the first launch after installing or updating could crash once while the ray tracing shaders
  were being built. The hang watchdog fix and the background shader builds should cure it; if it still happens,
  please send the console text.
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
- **AMD Radeon** cards (device lost with the patches off, water in Go-Go Archipelago, frame rate and any new
  graphics problems with `KYTY_READONLY_BUFFERS=0` against the default);
- **Intel CPUs** (with "Intel CPU compatibility" on and off; the red galaxy's first arrival);
- **8-12 GB graphics cards** (video memory full messages, stutter, frame rate with the defaults).

Switches for testing: `KYTY_RT_SOFTWARE=0 KYTY_RT_STUB=1` keeps the lighting but lets every ray miss (no ray-traced
shadows); `KYTY_RT_SOFTWARE=0` alone skips those shaders again (only with the patches on). The bundled preset's
`KYTY_SRT_VARIANT_READS=1` is required. See `U59-README.md` in the download for the full list of changes and switches.
