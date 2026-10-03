@echo off
rem As test 1, and the log records the buffers of the shader that faulted on the RTX 5090 (and
rem of four shaders compiled just before), and any buffer that is used after it was freed.
rem Notes the test, the time and the GPU driver in test-log.txt (send that file with the results).
>>"%~dp0test-log.txt" echo %DATE% %TIME% Test 9 - Diagnostics, binding check
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >>"%~dp0test-log.txt" 2>nul
copy /y "%~dp0test-presets\8-diagnostics-binding-check.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
