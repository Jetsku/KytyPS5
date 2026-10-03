RTX 50 test kit 4 (3 tests)
===========================

This is the int4b release with the swapchain fix from upstream pull request #1001 and
debug options for the RTX 50 GPU crash. Earlier kits showed it always happens at the same
moment (title screen -> galaxy map). Three questions are left; each test answers one.

Start the game with a script instead of launcher.exe. Stop at the first test that gets
past the galaxy map:

  1. Test 2 - ReBAR staging off.cmd
     Turns off an upload path that is only used on PCs with Resizable BAR on.
  2. Test 3 - Delayed buffer freeing.cmd
     Keeps freed GPU buffers alive a little longer, and logs any buffer used after it
     was freed.
  3. Update the NVIDIA driver to the newest Game Ready driver (616.92 or newer; its notes
     list a fix for RTX 50 GPUs), then run Test 1 - Diagnostics.cmd.

After each test send: whether it got into the galaxy map, the newest folder in _HangTrace
(zipped), _kyty.txt, test-log.txt, and a screenshot of the black console window.
"Normal (release settings).cmd" puts the normal settings back.