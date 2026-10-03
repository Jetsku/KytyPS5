@echo off
rem Test 3: as test 1, with the binding check on and freed GPU buffers kept a few seconds longer (uses more video memory).
>>"%~dp0test-log.txt" echo %DATE% %TIME% Test 3 - Delayed buffer freeing
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >>"%~dp0test-log.txt" 2>nul
copy /y "%~dp0test-presets\9-diagnostics-delayed-buffer-erase.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
