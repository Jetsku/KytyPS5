@echo off
rem Run this while the game is frozen (leave the frozen game open). Read-only. For 10 seconds it
rem reads the GPU load, clock and power once a second with nvidia-smi, then for 5 seconds the
rem processor time of each kyty_emulator thread, and adds both to gpu-load-while-frozen.txt next
rem to launcher.exe. Send that file with the results.
echo Reading the GPU load and the emulator threads for 15 seconds...
>>"%~dp0gpu-load-while-frozen.txt" echo %DATE% %TIME% while frozen
for /l %%i in (1,1,10) do (
  nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,clocks.sm,power.draw,pstate --format=csv,noheader >>"%~dp0gpu-load-while-frozen.txt" 2>&1
  "%SystemRoot%\System32\PING.EXE" -n 2 127.0.0.1 >nul
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test-presets\frozen-threads.ps1" -OutFile "%~dp0gpu-load-while-frozen.txt"
echo Done: gpu-load-while-frozen.txt
