@echo off
rem Puts the release settings back in u59-preset.json and starts the launcher.
copy /y "%~dp0test-presets\1-release.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
