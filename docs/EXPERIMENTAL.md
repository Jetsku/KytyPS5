# U59 integration release (Windows x64)

This release builds on the U59 renderer of the previous U59 releases.

## New in this update

Pre-release int16.1: int16 plus one fix.

- Reverted the upstream libFont commits "font: fix glyph metrics and raster sizes" (32e9fa30) and "font: add scaled
  kerning" (f860c9c5), which broke Astro Bot's level title text (wrong glyph sizes and spacing). The libFont code is
  back to its int14 state; the fix was checked on screen. The same reverts are in int15.1.
- Known issues: on RTX 50 GPUs the first launch can crash once while the RT kernels' pipelines are built; a second
  start works. On AMD GPUs with the two patch mods off, the GPU can be lost (VK_ERROR_DEVICE_LOST) and Go-Go
  Archipelago's water can spike frame times; AMD players should keep the patch mods on or use int15.1.

Pre-release int16 (`u59-windows-20261005-int16-pre`): the int15 release plus Astro Bot's own lighting without the
"non RT patch". If something works worse than in int15, please report it and use int15, or turn the two patch mods
back on.

- Compute shaders that contain IMAGE_BVH_INTERSECT_RAY run instead of being skipped: the instruction is emulated in
  software (`KYTY_RT_SOFTWARE`, default 1), with a per-ray node budget against runaway traversals
  (`KYTY_RT_NODE_BUDGET`, default 8192; `KYTY_RT_NODE_STATS` reports traversal statistics). Astro Bot's tiled
  lighting kernel 0x78af8e269b528b5c (and 11 related kernels) contain the instruction, so without the patch the whole
  lighting pass was dropped (black robots, unlit title). `KYTY_RT_SOFTWARE=0 KYTY_RT_STUB=1` lets every ray miss;
  `KYTY_RT_SOFTWARE=0` restores the skip.
- Stores and atomics through V#s computed at run time write through BDA (`KYTY_BDA_WRITES`, default 1).
- SRT walks: a flat read of an unmapped address reads 0; a null SRT pointer reads as a zero descriptor once the read
  fails; a program whose SRT reads follow a loop-carried pointer is skipped instead of exiting; the CFG structuriser
  gives a loop's break-region conditional a selection merge.
- The program disk cache keeps the new "uses BVH" and "writes through BDA" flags, and the codegen fingerprint covers the
  RT and BDA-write options.
- Crash reports print the faulting host call chain.
- Checked on one PC (RTX 3090, Ryzen 9 7950X3D, Astro Bot 1.018, PGO build): Sky Garden 32.0 and 35.3 fps without the
  patches against 34.8 with them; snow level 21.1 against 21.2. The first launch after an update builds the 13 RT
  kernels' pipelines (1-2.9 s each) on the command-processor thread.

int15 (main release, `u59-windows-20261005-int15`):

- Images read as colour over memory the texture cache holds as a plain D32 depth image (Astro Bot reuses depth memory
  as an RG16F DCC target at its 3328x1872 and 3840x2160 dynamic-resolution tiers) get a colour alias instead of
  stopping with "unsupported sampled depth image"; anything still unsupported binds a null texture and is reported
  once.
- `TextureCache::ClearImage` validates its range before any state change and skips a rejected clear with one
  `TextureCache: ClearImage skipped (<fault>): site=...` line per signature instead of exiting (leaving an extra
  level in Sky Garden stopped the emulator). DCC and CMASK fast-clear metadata is no longer applied to depth images
  (`TextureCache: DCC|CMASK metadata skipped for a depth image`).
- Upstream KytyPS5 (through 72e4989b1, reviewed and partly hand-ported): shader opcode and precision fixes (DS float
  min/max, DS_PERMUTE, DS masked OR, FP64 min/max/rounding, 64-bit image atomics, SDWA, V_CMPX_NE_U16, FLAT D16
  loads, ALIGNBYTE), no RTE rounding mode for FP64 shaders (an NVIDIA pipeline compile hang), image formats that name
  nothing bound empty (#1019), null SRT pointers read as zero (#987), guest thread priorities (#1050), the BDA page
  table cleared before first use (#1065), polygon draws as triangle fans, partially resident depth flags, libFont
  kerning and metrics (reverted in int15.1 and int16.1), trophy notifications. Not taken: the FP32 MAD rounding
  change (it conflicts with our position-invariant MAD mode) and the readback-window removal.
- Release notes correction for int14: the shader precompile mainly helps after an emulator update, when the program
  cache is rebuilt; with an unchanged emulator the program cache already covers revisits.
- Checked on one PC (RTX 3090, Ryzen 9 7950X3D): Sky Garden 34.8 fps (int14 34.1), snow level 21.2 fps (int14 21.7).

From the int14 pre-release (`u59-windows-20261005-int14-pre`):

- Shader precompile (`KYTY_SHADER_PRECOMPILE=1` in the preset): every shader variant the emulator translates is
  recorded in `_PipelineCache\<title>.shaders.journal`, and the next launch translates the recorded ones again on two
  low-priority background threads before the game asks for them. Sky Garden on a cold program cache: 437 shaders
  (12.3 s of translation) were built during play on the first run and 8 (0.3 s) on the second, after 433 were
  rebuilt in 6.4 s at startup. The first launch gains nothing; places never visited still compile on first use. The
  journal is tied to the GPU model; delete it after a driver update if in doubt. `KYTY_SHADER_PRECOMPILE=0` turns it
  off.
- Fast first pipelines (`KYTY_PIPELINE_FAST_FIRST=1` in the preset): a new pipeline is first built without driver
  optimization, so the draw does not wait for a full compile, and the optimized pipeline is built on two background
  threads and swapped in. Sky Garden on an empty pipeline cache: draws waited 0.31 s for pipelines instead of 1.38 s.
  `KYTY_PIPELINE_FAST_FIRST=0` turns it off; `KYTY_PIPELINE_FAST_FIRST_PROBE=1` asks the driver cache first (on NVIDIA
  the driver's own disk cache answers it, so pipelines are then built optimized as before).
- A pixel shader that samples one of several textures chosen at run time, with texture-LOD feedback, produced invalid
  SPIR-V (an `OpPhi` naming the wrong block); Demon's Souls stopped at character creation.
- With the launcher's Vulkan validation option on, validation errors are written to `_kyty_vulkan_validation.log`
  and no longer stop the game (Astro Bot stopped at boot on a known vertex/pixel interface message).
  `KYTY_VULKAN_VALIDATION_MODE=exit` restores stopping at the first error. Validation makes games much slower.
- Checked on one PC (RTX 3090, Ryzen 9 7950X3D): Sky Garden 33.0 fps with both new switches on against 33.4 fps for
  int13 built the same way (two alternating runs each; the difference is within run-to-run noise).

From the int13 pre-release (`u59-windows-20261005-int13-pre`):

- Adaptive triggers (also in int13): a vibration trigger reports "firing" only while it is pressed; in Astro's
  Playroom the gun fired by itself.
- The launcher's "AMD CPU patch" (`--amd-cpu`) now runs 11,026 of the game's 11,069 `VRSQRTPS` instructions as
  native code (43 still trap, 1,088 before): Sky Garden 34.5 fps with the option on (31.6 in int9 and int10, about
  the same as without the option now). The improved code analysis also applies to the red-zone protection.
- Game file reads into memory the emulator protects (Windows errors 998 and 1784) are retried through a temporary
  buffer instead of looking like an empty file to the game.
- When VRAM runs out while a texture is converted, idle scratch buffers are freed and the allocation retried, instead
  of stopping.
- A compute shader whose texture-LOD query result is unused no longer stops the shader translator (any GPU).
- Intel GPUs: mesh and pixel shaders require the subgroup width the guest wave needs (NVIDIA and AMD unchanged).
- Checked on one PC (RTX 3090, Ryzen 9 7950X3D): all 342 tests pass; Sky Garden 33-35 fps and Creamy Canyon 21.8 fps,
  as int11. Program caches stay valid.

From int11 (`u59-windows-20261005-int11`):

- More adaptive trigger fixes, tested in Astro's Playroom: the gacha capsules break on R2 again. A feedback trigger
  now reports "pushing" only while it is pressed (it did so untouched before, so the game never saw the press); the
  state follows every effect the game sets; and an L2/R2 that arrives as a button only (input remapping, some pads
  and tools) counts as a full press. Nothing else changed since int10; program caches stay valid.

From int10 (`u59-windows-20261004-int10`):

- Adaptive triggers: the game now sees the state of the trigger effects it sets (`scePadGetTriggerEffectState`
  always answered 0 before). Astro Bot levels in which L2/R2 actions, such as punches, did nothing now work. The
  state follows from the game's effect and how far the trigger is pressed, so it also works with other pads and the
  keyboard (a keyboard L2/R2 counts as fully pressed). Everything else is unchanged from int9; program caches built
  by int9 stay valid.
- Confirmed by players: int9 fixed the freeze on an RTX 5090 PC (Sky Garden at 38 fps).

From int9 (`u59-windows-20261004-int9`), about compatibility: the freeze on RTX 50 PCs, AMD and Intel GPUs and
CPUs, and older drivers.

- A fix for the freeze on NVIDIA RTX 50 PCs. A hang watchdog report from an RTX 5070 Ti showed the cause: the GPU
  waited for a texture copy that a CPU thread was about to finish, and that thread's copy-done signal and the frame
  presentation then blocked each other inside Windows (hardware-accelerated GPU scheduling). The emulator now waits
  for such copies on the CPU before it hands the GPU work that reads them to the driver, so the GPU never waits for a
  signal that only the CPU can give later; the upload DMA transfers follow the same rule. In the Sky Garden this came
  up about twice per frame. The fix is not yet confirmed on an RTX 50 PC.
  - With int7, one RTX 50 player found that the Fifo present mode (launcher: Configuration -> Present mode) avoids the
    freeze; with this fix the default Mailbox mode should work too.
  - If your RTX 50 PC still freezes, please send `_HangTrace\watchdog-*\watchdog.txt` from beside the emulator (the
    watchdog stays on only for RTX 50 GPUs). `KYTY_SUBMIT_WAIT_BEFORE_SIGNAL=1` restores the old behaviour, for
    comparisons only.
- GPUs other than recent NVIDIA ones. Several places where the emulator stopped because a GPU or driver lacked an
  optional feature now continue:
  - `VK_EXT_color_write_enable` and `VK_EXT_depth_clip_enable` are no longer required (older drivers). Without them
    the emulator uses static colour-write masks, and depth clamping only where the game turns depth clipping off.
  - Images the driver refuses are created without the optional usages and flags they can do without, instead of
    stopping with "image format does not support required usage". The log names each format once.
  - Compute shaders that compute a texture LOD enable `VK_KHR_compute_shader_derivatives` (or the NV extension). The
    shaders used it before without enabling it, which drivers other than NVIDIA's may reject.
  - Intel GPUs: compute shaders require a 32-wide subgroup, so the driver cannot split one guest wave over several
    narrower subgroups.
  - AMD GPUs: pixel and mesh shaders require the guest's wave size where the driver allows it; block-compressed volume
    textures drop optional flags until the driver accepts them; depth feedback loops without the host extension warn
    instead of stopping.
  - An image descriptor whose format names nothing is bound empty instead of aborting; an oversized image readback is
    staged in a private buffer instead of stopping; a GPU memory report below the caches' own size no longer stops
    the emulator.
  - GPUs without mesh shaders (GTX 10 series, RX 5000 series, Intel Iris Xe): a warning at startup, and a game that
    draws with mesh shaders stops at its first such draw with a message that names `VK_EXT_mesh_shader`, instead of a
    generic error. There is no fallback for those draws.
  - Windows: every emulator function the game calls realigns the stack first; game code that called in with a
    misaligned stack could crash the emulator.
  - Sources: Senaxx's fork (depth feedback loops, block-compressed volumes), upstream pull requests #845 by Matthew
    Hurley and #977 by tototomate123 (readback, memory accounting), and new fixes for this release.
  - On an NVIDIA RTX card each of these paths is the same as before. They were not tested on AMD or Intel hardware
    for this release.
- AMD and Intel CPUs (from upstream KytyPS5):
  - The launcher's "AMD CPU patch" option (`--amd-cpu`) emulates the game's `VRSQRTPS` instructions. Most of them are
    now patched to native code instead of trapping on every execution: at the Sky Garden start view 31.6 fps instead
    of 18.9 fps with the option on.
  - On CPUs without SSE4a (Intel), the game's `EXTRQ` instructions run through native code instead of a trap each, and
    `INSERTQ`, and `RDPID` and `CLWB` on CPUs without them, are emulated (int7 could not run them).
- Upstream KytyPS5 changes (third sync, through db1758527): rendering fixes (sampler snapshots shared across shader
  variants, image contents kept across unrelated buffer writes, BC5 textures, 64-bit image atomic max, explicit gather
  LOD, compute LDS clamped to the device limit), unopenable gamepads ignored, the cursor hidden over the game window
  after two idle seconds, DualSense vibration and volume settings in the launcher, periodic kernel timers, batched
  AIO, and network receives with `MSG_PEEK` and `MSG_WAITALL` on Windows.
- The first launch of each game rebuilds its shader program cache: the shader translation changed, so earlier
  caches are not reused. The first visit to each area is slow (Astro Bot's galaxy map ran at about 1 fps on an empty
  cache) and the game stutters until the cache fills.
- Opt-in switches, off by default and not in the preset: `KYTY_IR_LINEAR_USES`, `KYTY_FOLD_LANE_MASKS`,
  `KYTY_PRECISE_COND_WAITS` and `KYTY_BDA_SYNC_PER_SUBMISSION`.
- kyty_emulator is built with a new PGO profile, recorded in Astro Bot with this source.

From int7 (`u59-windows-20261004-int7`):

- New shader pipelines are prepared ahead on worker threads (`KYTY_PIPELINE_PREFETCH=1` and
  `KYTY_PIPELINE_PREFETCH_PROGRAMS=1` in the preset). With an empty shader cache the command processor stalled 68-70 s
  instead of 87-89 s along the Sky Garden route, and the slowest 1% of Creamy Canyon frames took about 100-130 ms
  instead of 230-245 ms. With a warm cache fps is unchanged.
- Small uploads use their own ring buffer (`KYTY_RAM_SMALL_UPLOAD_RING=1` in the preset): about 430-540 MB less RAM,
  about 385 MB less shared GPU memory and 60-150 MB less VRAM.
- Unused occlusion queries are reset in batches (`KYTY_OCCLUSION_RESET_BATCH=1` in the preset): Go-Go Archipelago
  +1.5% to +1.9% fps, Bathhouse Battle +2.6%.
- Shader storage writes are tagged again at command emission when binding preparation submitted their original
  recording (`KYTY_SHADER_WRITE_RETICK=1` in the preset), so a side readback cannot publish old contents before the
  actual GPU write. `KYTY_SHADER_WRITE_RETICK=0` restores the previous behavior.
- The hang watchdog is on only for NVIDIA RTX 50 GPUs (`KYTY_HANG_WATCHDOG=auto` in the preset; `=1` turns it on for any
  GPU, `=0` off). When the picture stops for 5 seconds it writes `_HangTrace\watchdog-*\watchdog.txt` beside the
  emulator: what every thread is doing and waiting for. See `docs/HANG-WATCHDOG.md`.

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

Astro Bot (PPSA21567); RTX 3090, Ryzen 9 7950X3D; 1920x1080 output at a 120 Hz vblank; warm program caches; the
bundled preset. This release was checked with single timed runs, not with an alternating comparison:

- Sky Garden start view: 35.2 fps with a local build of this release (int7: 33-35 fps in earlier runs).
- Creamy Canyon start view: 21.3 fps with a local build before the last compatibility merge (int7: 21.7-22.2 fps).
  The RTX 50 fix costs about 2% there (21.7 vs 22.2 fps when it was compared alone against int7, alternating
  launches); that scene is limited by the command processor.
- With the AMD CPU patch (`--amd-cpu`), Sky Garden: 31.6 fps (int7 with the patch: 18.9 fps).
- Demon's Souls boots to its menu with the preset (Language Select at 60 fps).
- With int5, batched occlusion queries took the Creamy Canyon start view from 8.3 to 22.0 fps.

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
- Microsoft Defender flagged the int7 `launcher.exe` as `Trojan:Win32/Bearfoos.A!ml` on the test PC. The `!ml` marks a
  machine-learning verdict, not the signature of known malware. The launcher is built by the GitHub Actions workflow
  `.github/workflows/u59-windows-release.yml` from `src/launcher` in this repository.
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
- Add `-DKYTY_EMULATOR_IPO=ON` and `-DKYTY_PGO_USE=C:/kyty-src/tools/pgo/u59-int9-sg-1.profdata`. Without
  `KYTY_PGO_USE` the build works, but without the profile's speedup.
- Build `launcher` and `kyty_emulator`, install to `_Build/windows/install`, and copy `tools/u59-preset.json` beside
  `launcher.exe`.

The Windows release is produced by a tagged GitHub Actions build. Original licenses and credits remain intact.
Personal handoffs, local editor configuration, captures, saves and caches are excluded. Historical results and their
limitations are described in [CHANGES-U59.md](CHANGES-U59.md).
