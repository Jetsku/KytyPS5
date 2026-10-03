@echo off
rem As test 1, with the ReBAR texture staging ring off (KYTY_TEXTURE_STAGING_REBAR=0). That ring is only used on PCs
rem with Resizable BAR on, which the test PCs without the problem do not have.
rem Notes the test, the time and the GPU driver in test-log.txt (send that file with the results).
>>"%~dp0test-log.txt" echo %DATE% %TIME% Test 13 - Diagnostics, ReBAR staging off
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >>"%~dp0test-log.txt" 2>nul
copy /y "%~dp0test-presets\11-diagnostics-rebar-staging-off.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
