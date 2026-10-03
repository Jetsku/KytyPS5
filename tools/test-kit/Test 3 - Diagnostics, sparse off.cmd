@echo off
rem As test 1, with sparse residency (textures and the BDA page table) off.
copy /y "%~dp0test-presets\3-diagnostics-no-sparse.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
