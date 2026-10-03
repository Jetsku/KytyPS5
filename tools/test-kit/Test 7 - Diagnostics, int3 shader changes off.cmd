@echo off
rem As test 6, and also without the other int3 shader changes (DPP inactive lanes, lane reductions, dispatcher cap).
copy /y "%~dp0test-presets\6-diagnostics-int3-shader-changes-off.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
