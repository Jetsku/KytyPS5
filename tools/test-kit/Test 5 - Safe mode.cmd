@echo off
rem As test 1, with the newer features off: sparse residency, draw runs, function-array shrink,
rem the VRAM budget collector and the CPU core reservation.
copy /y "%~dp0test-presets\4-safe-mode.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
