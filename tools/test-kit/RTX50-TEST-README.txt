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

Round 3: tests 8 to 12
======================

The RTX 5090 logs of kit 2 show the GPU failing on a memory write in one
particular shader at the title -> galaxy map step (test 5 recorded it as a GPU
fault; the other tests froze at the same moment). These tests check where that
write goes. Each one is a single start: title screen, then on into the galaxy
map, or until it freezes or crashes. Run them in this order.

Tests 9 to 12 also write a line to test-log.txt next to launcher.exe: the test,
the time, your GPU and its driver version.

  Test 8 - Diagnostics, frame fence off.cmd
      As sent before (the add-on). Skip it if you have already sent its result.
  Test 9 - Diagnostics, binding check.cmd
      As test 1, and the log (_kyty.txt) records where that shader's buffers
      are in GPU memory and any buffer that is used after it was freed.
      It needs the log: in the launcher, set the printf output to a file
      (_kyty.txt) as in your earlier logs.
  Test 10 - Diagnostics, delayed buffer erase.cmd
      As test 9, and freed GPU buffers are kept a few seconds longer before
      their memory is released (this uses more video memory). If test 10
      gets past the point where the others fail, tell us at once.
  Test 11 - Diagnostics, after an NVIDIA driver update.cmd
      Same settings as test 1. First install the newest NVIDIA Game Ready
      driver (616.92 or newer; NVIDIA's notes for 616.92 list a fix for
      Blackwell (RTX 50) GPUs). If your driver is already that new, run it
      anyway.
  Test 12 - Diagnostics, uniform lane reads.cmd
      As test 1, with a shader option that keeps loops from splitting up on
      the GPU. The first start compiles shaders again and is slower.

When a test freezes, leave the frozen game open and run
"While frozen - GPU load.cmd" (it takes 15 seconds): it notes whether the GPU
and the emulator's threads are still busy, in gpu-load-while-frozen.txt. Then
close the game.

For each test send the same as before (whether it got into the galaxy map,
after how long it froze or crashed, the newest _HangTrace folder zipped, the
console text from a "Device loss" line on, _device_fault.nv-gpudmp if one was
written), and also _kyty.txt, test-log.txt, gpu-load-while-frozen.txt and,
once, Kyty.ini (the launcher's settings, next to launcher.exe).
