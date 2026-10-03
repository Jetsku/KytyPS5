RTX 50 fix attempt
==================

The int4b release with the swapchain fix from upstream pull request #1001 and all three
workarounds for the RTX 50 GPU crash at the title screen -> galaxy map step switched on.

1. Update the NVIDIA driver to the newest Game Ready driver (616.92 or newer).
2. Start the game with "Start game (RTX 50 workarounds).cmd" instead of launcher.exe and play
   on into the galaxy map and a level. The first start compiles shaders again and is slower.

If it works, keep using that script. If it still freezes or crashes, send the newest folder in
_HangTrace (zipped), _kyty.txt, test-log.txt and a screenshot of the black console window:
these runs record exactly where it fails.

"Normal (release settings).cmd" puts the normal settings back.