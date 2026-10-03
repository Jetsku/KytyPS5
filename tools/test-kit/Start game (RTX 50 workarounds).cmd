@echo off
rem RTX 50 fix attempt: all three workarounds on (ReBAR staging off, delayed buffer freeing, uniform lane reads),
rem plus crash recording and a check that logs any GPU buffer used after it was freed.
>>"%~dp0test-log.txt" echo %DATE% %TIME% Start game (RTX 50 workarounds)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >>"%~dp0test-log.txt" 2>nul
copy /y "%~dp0test-presets\rtx50-workarounds.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
