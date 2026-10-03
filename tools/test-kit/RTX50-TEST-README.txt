RTX 50 / masterSemaphore test kit 2
=================================

This package is the int4b release (Kyty U59) plus the swapchain fix from upstream pull
request #1001 (window minimize/resize/move/fullscreen; it also stops a pending semaphore
from being reused) and scripts that start the launcher with different settings. Start the game with one of the scripts below
instead of launcher.exe. Each script copies its settings into u59-preset.json;
"Normal (release settings).cmd" puts the release settings back.

Run the tests in this order (1-7). Stop at the first one where the game gets past its
startup, i.e. the intro video plays and the title screen appears:

  Test 1 - Diagnostics.cmd
      Release settings, plus recording of freezes and GPU crashes.
  Test 2 - Diagnostics, overlays off.cmd
      As test 1, with the Vulkan overlay layers (RTSS/Afterburner, Steam overlay,
      Rockstar Social Club, Nsight, ...) switched off for this launch only.
  Test 3 - Diagnostics, sparse off.cmd
      As test 1, with sparse texture residency off.
  Test 4 - Diagnostics, fresh shader cache.cmd
      As test 1, after moving the shader cache aside (it is kept as
      _PipelineCache-old-<number>, nothing is deleted). This start is slower:
      every shader is compiled again.
  Test 5 - Safe mode.cmd
      As test 1, with the newer features off (sparse residency, draw runs,
      function-array shrink, the VRAM budget collector, the CPU core reservation).

  Test 6 - Diagnostics, real GPU clock off.cmd
      As test 1, with the shader clock (S_MEMREALTIME) back to the fixed placeholder
      used before int3. This start compiles shaders again and is slower.
  Test 7 - Diagnostics, int3 shader changes off.cmd
      As test 6, and also without the other int3 shader changes (DPP inactive lanes,
      lane reductions, dispatcher cap). This start compiles shaders again.
After each test write down:
  - whether the game got past its startup;
  - if it froze or crashed: after how long, and what was on the screen;
  - the text of the black console window from a "Device loss" line on, if there
    is one (a photo or screenshot is fine).
and send:
  - the newest folder inside _HangTrace next to kyty_emulator.exe (zip it);
  - _device_fault.nv-gpudmp, if one was written next to kyty_emulator.exe;
  - the GPU model and NVIDIA driver version.