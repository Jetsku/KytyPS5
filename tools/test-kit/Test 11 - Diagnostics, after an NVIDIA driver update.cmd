@echo off
rem As test 1 (same settings). Run it after installing the newest NVIDIA driver (616.92 or newer).
rem Notes the test, the time and the GPU driver in test-log.txt (send that file with the results).
>>"%~dp0test-log.txt" echo %DATE% %TIME% Test 11 - Diagnostics, after an NVIDIA driver update
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >>"%~dp0test-log.txt" 2>nul
copy /y "%~dp0test-presets\2-diagnostics.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
