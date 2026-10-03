@echo off
rem Release settings plus freeze/crash recording (_HangTrace folder, GPU fault diagnostics).
copy /y "%~dp0test-presets\2-diagnostics.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
