@echo off
rem AMD device-lost test kit (see AMD-TEST-KIT-README.txt). Runs Astro Bot about 19 times in a row,
rem each time with one emulator switch changed, and zips the console logs.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0AMD-Test-Kit.ps1" %*
echo.
pause
