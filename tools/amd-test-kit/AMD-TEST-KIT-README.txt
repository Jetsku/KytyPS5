AMD test kit 3 for Astro Bot on AMD Radeon cards
================================================

What it does
------------
It starts Astro Bot 6 times in a row, each time with one emulator setting changed, and writes down
how far each start gets (device lost, crash, or still running). At the end it puts all console logs
into one zip file. One session of about 35 minutes; a run that crashes ends early.

AMD test 2 fixed the "device lost" at the title screen. Its logs still show "Clamping LDS" once the
game is played: one of the game's compute shaders asks for 48 KiB of shared memory (LDS), and AMD
cards allow 32 KiB, so AMD test 2 cut it down. This build keeps that shader's shared memory in a
GPU buffer instead. The kit also checks a crash some players see a minute into play, inside a DLL
(a driver, or an overlay such as Steam, Epic, GOG Galaxy or OBS): one run starts the game without
any Vulkan overlay layer, one without the quick first pipeline builds, and one avoids a shader
feature of the pipeline that was being built when it crashed. Five of the runs need you to
play: start the game, Dive In, and play the first level for a few minutes. The kit tells you before
each of these runs and waits for Enter.

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
3. Double-click AMD-Test-Kit.cmd. A console window shows each run.
   - For a run marked "YOU PLAY": press Enter in the console, wait for the title screen, start the
     game, Dive In, and play the first level (near the water) until the kit closes the game, about
     4 minutes after the intro video. You may close the game window yourself earlier. Afterwards
     the console asks what you saw: press Enter if nothing looked wrong, or type a few words
     ("stretched blue triangles near the water", "crash after Dive In", "stutter").
   - The last run is hands off: the game opens at the title screen and closes by itself.
   The screen may go black for a few seconds if the GPU resets; the kit then goes on by itself.
4. When it says "Send this file: ...AMD-Test-Results-<date>.zip", send us that zip, also when every run
   went fine: the logs name the shaders that ask for more shared memory than the card has ("GPU
   note: game compute shader 0x... requests ... bytes of LDS"). If you play another game with this
   build (for example Astro's Playroom) and see broken effects, send its _kyty.txt log too.

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
lds-fix           this build as it is: the 48 KiB shader's LDS in a GPU buffer   (you play)
pervertex-off     pixel shaders without raw per-vertex inputs (the crash a minute
                  into play happened right after such a shader was compiled; some
                  shading may look slightly wrong in this run)                     (you play)
overlays-off      this build without the Vulkan overlay layers (Steam, Epic, GOG
                  Galaxy, OBS, fossilize; on a laptop with two GPUs also AMD's GPU
                  switching layer)                                                 (you play)
fastfirst-off     pipelines built fully optimized at once (no quick first build);
                  more stutter while shaders compile, but no second build          (you play)
lds-clamp         LDS cut to 32 KiB again, as in AMD test 2                       (you play)
fix               this build at the title screen only, hands off (AMD test 2 check)

If a run crashes, the console log now names the module the crash happened in (for example
"pc=0x00007ff9fc1640fc (amdvlk64.dll+0x...)") and lists the call chain.

Advanced options (normally not needed)
--------------------------------------
-NoPlay                     only the hands-off runs
-PlaySeconds 240            how long a played run may go on after the intro video
-Only lds-fix,lds-clamp     run only these (also: shrink-off, clipguard-off, dpp-skip-off, clusters-off,
                            barrier-old, barrier-off, old-loop-guard, fix-long, lds-force
                            and the AMD test 1 runs)
-SecondsAfterVideo 45       how long a hands-off run goes on after the intro video
-TitleId PPSA21567          if the kit picks the wrong patch file
