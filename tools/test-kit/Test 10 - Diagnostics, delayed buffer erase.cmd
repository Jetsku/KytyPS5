@echo off
rem As test 9, and freed GPU buffers are kept a few seconds longer before their memory is
rem released (it uses more video memory).
rem Notes the test, the time and the GPU driver in test-log.txt (send that file with the results).
>>"%~dp0test-log.txt" echo %DATE% %TIME% Test 10 - Diagnostics, delayed buffer erase
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >>"%~dp0test-log.txt" 2>nul
copy /y "%~dp0test-presets\9-diagnostics-delayed-buffer-erase.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
