@echo off
rem As test 1, with the shader clock (S_MEMREALTIME) back to the fixed placeholder used before int3.
copy /y "%~dp0test-presets\5-diagnostics-real-clock-off.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
