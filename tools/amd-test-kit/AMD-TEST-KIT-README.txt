AMD test kit for the "device lost" right after the Astro Bot intro video
=========================================================================

What it does
------------
It starts Astro Bot about 19 times in a row, by itself, each time with one emulator setting
changed, and writes down how far each start gets (device lost, or still running 45 seconds after
the intro video). At the end it puts all console logs into one zip file. One session of about
35 minutes answers which part of the emulator makes your GPU stop responding.

Nothing is changed on your PC or in your game or emulator folders: every setting is passed only to
the emulator process the kit starts (the same way the launcher passes u59-preset.json). The kit
only reads your launcher settings (Kyty.ini) to start the game exactly as the launcher does, and it
reads Windows' own record of GPU resets for the time of the test. It does not send anything
anywhere; you send us the zip.

How to run it
-------------
1. Copy your working int16.1 folder (the whole folder, with its _Patches, _PipelineCache and
   _SaveData) to a new folder, then unzip this build into that new folder and let it overwrite the
   files. Your patch choices (_Patches\PPSA21564.json) must be there: the kit warns if they are
   not. The kit files are AMD-Test-Kit.cmd and AMD-Test-Kit.ps1, next to kyty_emulator.exe.
2. Start launcher.exe from the new folder once, check that Astro Bot is in the list with your
   usual settings and patches (both lighting patches and the fixed 1920x1080 resolution), then
   CLOSE the launcher again. Do not start the game from the launcher.
3. Double-click AMD-Test-Kit.cmd. A console window shows each run. The game window opens and
   closes by itself about 19 times. Please do not touch the keyboard, mouse or controller while it
   runs, and do not close the game windows. The screen may go black for a few seconds when the GPU
   resets; that is expected.
4. When it says "Send this file: ...AMD-Test-Results-<date>.zip", send us that zip.

If the console says it cannot find the game, run it from a command prompt in that folder with the
game folder:
    AMD-Test-Kit.cmd -GamePath "D:\Games\PPSA21564"
or let it copy the launcher's exact command line: start AMD-Test-Kit.cmd -CaptureFromLauncher, then
start Astro Bot from the launcher as usual; the kit takes the command line from that game window,
closes it, and goes on by itself.

To stop early, close the kit's console window; the zip is then not written, but the folder
AMD-Test-Results-<date> next to kyty_emulator.exe has the logs so far.

What the runs are
-----------------
baseline        as int16.1, with GPU breadcrumbs (what the GPU was running when it stopped)
all-safe        everything below off and every new fix on at once
fastfirst-off   pipelines built fully optimized from the start (no quick unoptimized first build)
helper-fix      new fix: pixel-shader helper lanes skip compare-exchange loops
wave64-split    wave64 compute shaders run the way they run on NVIDIA
loop-guard      every shader loop ends after 200000 iterations, and the log names the shader
sparse-off      no sparse textures / sparse page table
queues-off      no second graphics queue and no copy-engine queue
drawrun-off     no draw-run reuse (it starts at the title screen)
dma-off         no copy-engine (SDMA) queue for uploads
sidequeue-off   no second graphics queue
rebar-off       no texture staging ring in VRAM (Smart Access Memory)
live-exec-all   pixel shaders start without helper lanes
volatile-loads  shaders always reread memory they poll
lane-opts-off   two shader optimizations off
shrink-off      shader arrays not shrunk
submit-direct   no separate submission thread
recorder-off    no separate command recording thread
baseline-log    baseline again with the emulator's log file
(no-breadcrumbs runs only if the baseline does not crash)

Advanced options (normally not needed)
--------------------------------------
-Only baseline,helper-fix   run only these
-SecondsAfterVideo 45       how long a run goes on after the intro video
-TitleId PPSA21564          if the kit picks the wrong patch file
