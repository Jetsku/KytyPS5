@echo off
rem Test 2: as test 1, with the ReBAR texture staging ring off (KYTY_TEXTURE_STAGING_REBAR=0).
>>"%~dp0test-log.txt" echo %DATE% %TIME% Test 2 - ReBAR staging off
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >>"%~dp0test-log.txt" 2>nul
copy /y "%~dp0test-presets\11-diagnostics-rebar-staging-off.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
