AMD test kit 4 for Astro Bot on AMD Radeon cards
================================================

What it does
------------
It starts Astro Bot 3 times in a row, each time with one emulator setting changed, and writes down
how far each start gets (device lost, crash, or still running). At the end it puts all console logs
into one zip file. One session of about 15 minutes; a run that crashes ends early.

In AMD test 3 every run crashed a minute or so into play, at the same place inside AMD's Vulkan
driver ("crash in amdvlk64.dll+0x22240fc"). The crash is in the part of the driver that compiles
a pipeline with full optimization; the same pipelines compiled without optimization work. This
build does two things about it:
- the few pixel shaders that were seen crashing the driver are never compiled with full
  optimization (everything else still is), and
- if the driver crashes while optimizing some other pipeline, the emulator catches the crash, uses
  that pipeline unoptimized instead and goes on. It also notes the pipeline in
  _PipelineCache\<title>.noopt.txt, so later starts skip the optimizer for it from the beginning.
A driver that crashed may be left in a bad state, so a session in which the guard caught a crash may
still hang or crash later; the second run tests the other way round: no optimized pipelines at all.
Two of the runs need you to play: start the game, Dive In, and play the first level for a few
minutes. The kit tells you before each of these runs and waits for Enter.

Nothing is changed on your PC or in your game folder: every setting is passed only to the emulator
process the kit starts (the same way the launcher passes u59-preset.json). The emulator itself
writes its usual _PipelineCache files, and now also the .noopt.txt file above when it catches a
driver crash. The kit only reads your launcher settings (Kyty.ini) to start the game exactly as the
launcher does, and it reads Windows' own record of GPU resets for the time of the test. It does not
send anything anywhere; you send us the zip.

How to run it
-------------
1. Copy your working int16.1 folder (the whole folder, with its _Patches, _PipelineCache and
   _SaveData) to a new folder, then unzip this build into that new folder and let it overwrite the
   files. The kit files are AMD-Test-Kit.cmd and AMD-Test-Kit.ps1, next to kyty_emulator.exe.
2. Start launcher.exe from the new folder once, check that Astro Bot is in the list with your
   usual settings and patches, then CLOSE the launcher again. Do not start the game from the
   launcher.
3. Double-click AMD-Test-Kit.cmd. A console window shows each run.
   - For a run marked "YOU PLAY": press Enter in the console, wait for the title screen, start the
     game, Dive In, and play the first level (near the water) until the kit closes the game, about
     4 minutes after the intro video. You may close the game window yourself earlier. Afterwards
     the console asks what you saw: press Enter if nothing looked wrong, or type a few words
     ("stretched blue triangles near the water", "crash after Dive In", "stutter").
   - The last run is hands off: the game opens at the title screen and closes by itself.
   The screen may go black for a few seconds if the GPU resets; the kit then goes on by itself.
4. When it says "Send this file: ...AMD-Test-Results-<date>.zip", send us that zip, also when every run
   went fine: the summary says whether the guard caught a driver crash ("driver fault(s) caught")
   and how many pipelines were kept unoptimized. If _PipelineCache\<title>.noopt.txt exists in the
   emulator folder afterwards, send it too. If you play another game with this build (for example
   Astro's Playroom) and see broken effects, send its _kyty.txt log too.

If the console says it cannot find the game, run it from a command prompt in that folder with the
game folder:
    AMD-Test-Kit.cmd -GamePath "D:\Games\PPSA21567"
or let it copy the launcher's exact command line: start AMD-Test-Kit.cmd -CaptureFromLauncher, then
start Astro Bot from the launcher as usual; the kit takes the command line from that game window,
closes it, and goes on by itself.

To stop early, close the kit's console window; the zip is then not written, but the folder
AMD-Test-Results-<date> next to kyty_emulator.exe has the logs so far.

Astro's Playroom: broken clouds or smoke
----------------------------------------
The kit only runs Astro Bot. If clouds or smoke in Astro's Playroom break up into blocks on your
AMD card, start Playroom twice by hand from this folder, once with each setting below, look at the
same clouds, and send us the _kyty.txt of each start with a note on whether the clouds changed.

1. Open a command prompt in this folder (type cmd in the Explorer address bar and press Enter) and
   type:
       set KYTY_DPP_SKIP_INACTIVE=0
       launcher.exe
   Start Playroom from the launcher, look at the clouds, close the game and the launcher, and keep
   a copy of _kyty.txt.
2. Close that command prompt, open a new one in this folder, and type:
       set KYTY_WAVE32_CLUSTERS=0
       launcher.exe
   Start Playroom again, look at the same clouds, close it, and keep this _kyty.txt too.

The setting only applies to the launcher started from that command prompt; nothing is changed on
your PC.

What the runs are
-----------------
noopt-list        this build as it is: the known crashing shaders compiled without
                  full optimization, and a driver crash in any other optimized build
                  caught (only these few pipelines skip the optimizer; the speed
                  should be the same as AMD test 3)                               (you play)
optimize-off      no pipeline compiled with full optimization at all
                  (KYTY_PIPELINE_OPTIMIZE=0). If the crash is gone here, it is the
                  driver's optimizer for sure. Unoptimized shader code can run slower
                  on the GPU: tell us if this run is slower or stutters more      (you play)
fix               this build at the title screen only, hands off (AMD test 2 check)

If a run crashes, the console log names the module the crash happened in (for example
"pc=0x00007ff9fc1640fc (amdvlk64.dll+0x...)") and lists the call chain. A crash the guard catches
does not end the run; the log says "Pipeline optimization: the driver faulted ...".

Advanced options (normally not needed)
--------------------------------------
-NoPlay                     only the hands-off runs
-PlaySeconds 240            how long a played run may go on after the intro video
-Only noopt-list,fix        run only these. Also available: noopt-none (no list and no learned
                            pipelines: the AMD test 3 behaviour, crash expected), guard-off (the
                            list without the crash guard), clusters-off, the AMD test 3 runs
                            (lds-fix, pervertex-off, overlays-off, fastfirst-off, lds-clamp),
                            shrink-off, clipguard-off, dpp-skip-off, barrier-old, barrier-off,
                            old-loop-guard, fix-long, lds-force and the AMD test 1 runs
-SecondsAfterVideo 45       how long a hands-off run goes on after the intro video
-TitleId PPSA21567          if the kit picks the wrong patch file
