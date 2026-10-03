@echo off
rem As test 1, with a shader option that keeps loops from splitting up on the GPU. The first
rem start compiles shaders again and is slower.
rem Notes the test, the time and the GPU driver in test-log.txt (send that file with the results).
>>"%~dp0test-log.txt" echo %DATE% %TIME% Test 12 - Diagnostics, uniform lane reads
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >>"%~dp0test-log.txt" 2>nul
copy /y "%~dp0test-presets\10-diagnostics-uniform-lane-reads.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
