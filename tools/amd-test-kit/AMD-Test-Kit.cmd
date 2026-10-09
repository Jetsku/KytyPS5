@echo off
rem AMD test kit (see AMD-TEST-KIT-README.txt). Runs Astro Bot 6 times in a row, each time with one
rem emulator switch changed (five of them you play), and zips the console logs.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0AMD-Test-Kit.ps1" %*
echo.
pause
