AMD test kit 2 for the "device lost" right after the Astro Bot intro video
==========================================================================

What it does
------------
It starts Astro Bot 6 times in a row, by itself, each time with one emulator setting changed, and
writes down how far each start gets (device lost, or still running after the intro video). At the
end it puts all console logs into one zip file. One session of about 15 minutes.

The first AMD test showed what the GPU was running when it stopped: the title screen's particle
draw. This build contains the fix for it; the runs check the fix and confirm the cause.

Nothing is changed on your PC or in your game or emulator folders: every setting is passed only to
the emulator process the kit starts (the same way the launcher passes u59-preset.json). The kit
only reads your launcher settings (Kyty.ini) to start the game exactly as the launcher does, and it
reads Windows' own record of GPU resets for the time of the test. It does not send anything
anywhere; you send us the zip.

How to run it
-------------
1. Copy your working int16.1 folder (the whole folder, with its _Patches, _PipelineCache and
   _SaveData) to a new folder, then unzip this build into that new folder and let it overwrite the
   files. The kit files are AMD-Test-Kit.cmd and AMD-Test-Kit.ps1, next to kyty_emulator.exe.
2. Start launcher.exe from the new folder once, check that Astro Bot is in the list with your
   usual settings and patches, then CLOSE the launcher again. Do not start the game from the
   launcher.
3. Double-click AMD-Test-Kit.cmd. A console window shows each run. The game window opens and
   closes by itself 6 times. Please do not touch the keyboard, mouse or controller while it runs,
   and do not close the game windows. The screen may go black for a few seconds when the GPU
   resets; that is expected (two of the runs are meant to reproduce the old crash).
4. When it says "Send this file: ...AMD-Test-Results-<date>.zip", send us that zip. A short note
   on what the title screen looked like in the runs that did not crash (pink clouds or not) helps.

If the console says it cannot find the game, run it from a command prompt in that folder with the
game folder:
    AMD-Test-Kit.cmd -GamePath "D:\Games\PPSA21567"
or let it copy the launcher's exact command line: start AMD-Test-Kit.cmd -CaptureFromLauncher, then
start Astro Bot from the launcher as usual; the kit takes the command line from that game window,
closes it, and goes on by itself.

To stop early, close the kit's console window; the zip is then not written, but the folder
AMD-Test-Results-<date> next to kyty_emulator.exe has the logs so far.

What the runs are
-----------------
fix             this build as it is (the fix on); should keep running
barrier-old     the shader barrier as in int16.1; expected to crash like before
barrier-off     no shader barrier at all (the workaround from the GitHub issue; may show pink
                clouds on the title screen)
old-loop-guard  barrier-old with a loop limit on the crashing shaders (tells a stuck barrier
                from an endless loop)
clusters-off    the fix without the second new fix (wave32 shaders on AMD)
fix-long        the fix again, 2 minutes at the title screen, with the emulator's log file

Advanced options (normally not needed)
--------------------------------------
-Only fix,barrier-off       run only these
-SecondsAfterVideo 45       how long a run goes on after the intro video
-TitleId PPSA21567          if the kit picks the wrong patch file
